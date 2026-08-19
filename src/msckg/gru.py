import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from .constants import FEATURE_COMBINATIONS, FEATURE_GROUPS
from .evaluation import decode_labels, encode_labels, evaluate, group_split, make_xgboost
from .features import point_features
from .io import load_embedding, resolve_columns


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sequences(trajectory, embedding_path, columns, dimension=100):
    frame = pd.read_csv(trajectory)
    embedding = load_embedding(embedding_path, dimension)
    rows, resolved = point_features(frame, embedding, columns)
    time_columns = resolve_columns(frame, columns, ("timestamp",))
    metadata = pd.DataFrame(
        {
            "gr": [row[0] for row in rows],
            "label": [row[1] for row in rows],
            "timestamp": pd.to_numeric(frame[time_columns["timestamp"]], errors="raise"),
        }
    )
    vectors = np.stack([np.concatenate(row[2:]) for row in rows]).astype(np.float32)
    sequences, labels, groups = [], [], []
    for group, indices in metadata.groupby("gr", sort=True).groups.items():
        selected = metadata.loc[indices].copy()
        selected["row"] = np.asarray(indices)
        selected = selected.sort_values("timestamp")
        steps = []
        for _, time_rows in selected.groupby("timestamp", sort=True):
            steps.append(vectors[time_rows["row"].to_numpy()].mean(axis=0))
        sequences.append(np.stack(steps))
        labels.append(int(selected["label"].iloc[0]))
        groups.append(int(group))
    return sequences, np.asarray(labels), np.asarray(groups)


def scale_sequences(sequences, train):
    scaler = StandardScaler()
    scaler.fit(np.concatenate([sequences[index] for index in train], axis=0))
    return [scaler.transform(sequence).astype(np.float32) for sequence in sequences], scaler


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, mask):
        self.sequences = sequences
        self.labels = encode_labels(labels)
        self.mask = torch.as_tensor(mask, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        values = torch.as_tensor(self.sequences[index], dtype=torch.float32) * self.mask
        return values, int(self.labels[index]), index


def collate(batch):
    sequences, labels, indices = zip(*batch)
    lengths = torch.as_tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    return pad_sequence(sequences, batch_first=True), lengths, torch.as_tensor(labels), torch.as_tensor(indices)


class GRUModel(nn.Module):
    def __init__(self, input_dim=400, hidden_dim=400, classes=6):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, classes)

    def encode(self, values, lengths):
        packed = pack_padded_sequence(values, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return hidden[-1]

    def forward(self, values, lengths):
        return self.head(self.encode(values, lengths))


def loader(dataset, indices, batch_size, shuffle):
    return DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def train_epoch(model, batches, optimizer, device, gradient_clip):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total = 0.0
    for values, lengths, labels, _ in batches:
        values, labels = values.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(values, lengths), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total += float(loss.detach()) * len(labels)
    return total / len(batches.dataset)


def predict_head(model, batches, device):
    model.eval()
    predicted, truth = [], []
    with torch.no_grad():
        for values, lengths, labels, _ in batches:
            logits = model(values.to(device), lengths)
            predicted.extend(logits.argmax(dim=1).cpu().numpy())
            truth.extend(labels.numpy())
    return np.asarray(truth), np.asarray(predicted)


def extract_vectors(model, batches, device):
    model.eval()
    vectors, labels, indices = [], [], []
    with torch.no_grad():
        for values, lengths, current_labels, current_indices in batches:
            vectors.append(model.encode(values.to(device), lengths).cpu().numpy())
            labels.extend(current_labels.numpy())
            indices.extend(current_indices.numpy())
    order = np.argsort(indices)
    return np.concatenate(vectors)[order], np.asarray(labels)[order]


def mask_for_combination(combination, dimension=100):
    active = set(FEATURE_COMBINATIONS[combination])
    return np.concatenate([np.ones(dimension) if group in active else np.zeros(dimension) for group in FEATURE_GROUPS]).astype(np.float32)


def select_epochs(dataset, outer_train, labels, settings, device, seed):
    encoded = encode_labels(labels)
    train, validation = train_test_split(outer_train, test_size=settings["validation_fraction"], random_state=seed, stratify=encoded[outer_train])
    model = GRUModel(hidden_dim=settings["hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    train_loader = loader(dataset, train, settings["batch_size"], True)
    validation_loader = loader(dataset, validation, settings["batch_size"], False)
    best_epoch, best_score, waiting = 1, -np.inf, 0
    history = []
    for epoch in range(1, settings["max_epochs"] + 1):
        loss = train_epoch(model, train_loader, optimizer, device, settings["gradient_clip"])
        truth, predicted = predict_head(model, validation_loader, device)
        score = evaluate(decode_labels(truth), decode_labels(predicted))["macro_f1"]
        history.append({"epoch": epoch, "loss": loss, "validation_macro_f1": score})
        if score > best_score + 1e-8:
            best_epoch, best_score, waiting = epoch, score, 0
        else:
            waiting += 1
        if waiting >= settings["patience"]:
            break
    return best_epoch, history


def run_gru_experiment(trajectory, embedding_path, columns, parameters, settings, output, seed=5, jobs=8, test_size=0.1, device=None):
    set_seed(seed)
    sequences, labels, groups = build_sequences(trajectory, embedding_path, columns)
    indices = np.arange(len(groups))
    outer_train, outer_test = group_split(groups, labels, test_size, seed)
    sequences, _ = scale_sequences(sequences, outer_train)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([len(sequence) for sequence in sequences])
    pd.DataFrame(
        [
            {
                "trajectories": len(sequences),
                "mean_length": float(lengths.mean()),
                "median_length": float(np.median(lengths)),
                "maximum_length": int(lengths.max()),
                "single_step_fraction": float((lengths == 1).mean()),
            }
        ]
    ).to_csv(output / "sequence_summary.csv", index=False, encoding="utf-8-sig")
    rows = []
    for offset, combination in enumerate(FEATURE_COMBINATIONS):
        set_seed(seed + offset)
        dataset = SequenceDataset(sequences, labels, mask_for_combination(combination))
        best_epoch, history = select_epochs(dataset, outer_train, labels, settings, device, seed + offset)
        set_seed(seed + offset)
        model = GRUModel(hidden_dim=settings["hidden_dim"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        train_loader = loader(dataset, outer_train, settings["batch_size"], True)
        for _ in range(best_epoch):
            train_epoch(model, train_loader, optimizer, device, settings["gradient_clip"])
        train_vectors, train_labels = extract_vectors(model, loader(dataset, outer_train, settings["batch_size"], False), device)
        test_vectors, test_labels = extract_vectors(model, loader(dataset, outer_test, settings["batch_size"], False), device)
        classifier = make_xgboost(parameters, seed, jobs)
        classifier.fit(train_vectors, train_labels)
        prediction = classifier.predict(test_vectors)
        metrics = evaluate(decode_labels(test_labels), decode_labels(prediction))
        rows.append(
            {
                "feature_combination": combination,
                "selected_epochs": best_epoch,
                "train_count": len(outer_train),
                "test_count": len(outer_test),
                **{key: value for key, value in metrics.items() if key not in {"class_accuracy", "confusion_matrix"}},
            }
        )
        pd.DataFrame(history).to_csv(output / f"history_{combination.replace('+', '_')}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(output / "gru_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"gr": groups, "class": labels, "split": np.where(np.isin(indices, outer_train), "train", "test")}).to_csv(output / "split.csv", index=False, encoding="utf-8-sig")
    return rows
