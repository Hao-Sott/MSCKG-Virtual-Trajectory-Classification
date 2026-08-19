from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier

from .constants import CLASS_LABELS, CLASS_NAMES, FEATURE_COMBINATIONS
from .features import feature_matrix
from .io import save_json


def encode_labels(values, labels=CLASS_LABELS):
    lookup = {int(value): index for index, value in enumerate(labels)}
    return np.asarray([lookup[int(value)] for value in values], dtype=np.int64)


def decode_labels(values, labels=CLASS_LABELS):
    return np.asarray(labels, dtype=np.int64)[np.asarray(values, dtype=np.int64)]


def make_xgboost(parameters, seed=5, jobs=8):
    values = {
        **parameters,
        "objective": "multi:softprob",
        "num_class": len(CLASS_LABELS),
        "eval_metric": "mlogloss",
        "random_state": int(seed),
        "n_jobs": int(jobs),
        "verbosity": 0,
        "importance_type": "gain",
    }
    return XGBClassifier(**values)


def evaluate(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    class_accuracy = recall_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    return {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "class_accuracy": {str(label): float(value) for label, value in zip(CLASS_LABELS, class_accuracy)},
        "confusion_matrix": matrix.tolist(),
    }


def group_split(groups, labels, test_size=0.1, seed=5):
    order = pd.Series(np.arange(len(groups))).sample(frac=1, random_state=seed).to_numpy(dtype=np.int64)
    cut = int(len(order) * (1.0 - test_size))
    return order[:cut], order[cut:]


def fit_evaluate(x, y, train, test, parameters, seed=5, jobs=8):
    model = make_xgboost(parameters, seed, jobs)
    model.fit(x[train], encode_labels(y[train]))
    prediction = decode_labels(model.predict(x[test]))
    return model, prediction, evaluate(y[test], prediction)


def feature_block_gain(model, combination, dimension=100):
    raw = model.get_booster().get_score(importance_type="gain")
    vector = np.zeros(model.n_features_in_, dtype=float)
    for key, value in raw.items():
        vector[int(key[1:])] = value
    blocks = {}
    start = 0
    for group in FEATURE_COMBINATIONS[combination]:
        blocks[group] = float(vector[start : start + dimension].sum())
        start += dimension
    total = sum(blocks.values())
    shares = {key: value / total if total else 0.0 for key, value in blocks.items()}
    return vector, blocks, shares


def select_xgboost_parameters(features, fixed_parameters, output, seed=5, test_size=0.1, jobs=8):
    candidates = (
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05},
    )
    train, _ = group_split(features["gr"], features["classes"], test_size, seed)
    matrix = feature_matrix(features, "T+P+TC+PC")[train]
    labels = features["classes"][train]
    encoded = encode_labels(labels)
    cross_validation = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    rows = []
    for candidate_id, candidate in enumerate(candidates, 1):
        scores = []
        for fold, (fit, validation) in enumerate(cross_validation.split(matrix, encoded), 1):
            parameters = {**fixed_parameters, **candidate}
            model = make_xgboost(parameters, seed, jobs)
            model.fit(matrix[fit], encoded[fit])
            prediction = decode_labels(model.predict(matrix[validation]))
            score = float(f1_score(labels[validation], prediction, average="macro", zero_division=0))
            scores.append(score)
            rows.append({"candidate_id": candidate_id, "fold": fold, **candidate, "macro_f1": score})
        rows.append({"candidate_id": candidate_id, "fold": "mean", **candidate, "macro_f1": float(np.mean(scores)), "macro_f1_sd": float(np.std(scores, ddof=1))})
    frame = pd.DataFrame(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "xgboost_parameter_selection_cv.csv", index=False, encoding="utf-8-sig")
    summary = frame[frame["fold"] == "mean"].sort_values(["macro_f1", "candidate_id"], ascending=[False, True])
    summary.to_csv(output / "xgboost_parameter_selection_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def run_feature_combinations(features, parameters, output, seed=5, test_size=0.1, jobs=8, model_name="DistMult"):
    groups = features["gr"]
    labels = features["classes"]
    train, test = group_split(groups, labels, test_size, seed)
    results = []
    predictions = []
    importance = []
    for combination in FEATURE_COMBINATIONS:
        if not set(FEATURE_COMBINATIONS[combination]).issubset(features):
            continue
        matrix = feature_matrix(features, combination)
        model, predicted, metrics = fit_evaluate(matrix, labels, train, test, parameters, seed, jobs)
        vector, blocks, shares = feature_block_gain(model, combination, matrix.shape[1] // len(FEATURE_COMBINATIONS[combination]))
        results.append(
            {
                "embedding_model": model_name,
                "feature_combination": combination,
                "seed": seed,
                "train_count": len(train),
                "test_count": len(test),
                **metrics,
            }
        )
        for index, predicted_label in zip(test, predicted):
            predictions.append(
                {
                    "embedding_model": model_name,
                    "feature_combination": combination,
                    "gr": int(groups[index]),
                    "true_class": int(labels[index]),
                    "predicted_class": int(predicted_label),
                }
            )
        for index, value in enumerate(vector):
            importance.append({"feature_combination": combination, "dimension": index, "gain": value})
        for key in blocks:
            importance.append({"feature_combination": combination, "dimension": key, "gain": blocks[key], "share": shares[key]})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    flat = pd.DataFrame([{key: value for key, value in row.items() if key not in {"class_accuracy", "confusion_matrix"}} for row in results])
    per_class = []
    confusions = []
    for row in results:
        for label in CLASS_LABELS:
            per_class.append(
                {
                    "embedding_model": row["embedding_model"],
                    "feature_combination": row["feature_combination"],
                    "class_id": label,
                    "class_name": CLASS_NAMES[label],
                    "class_accuracy": row["class_accuracy"][str(label)],
                }
            )
        for i, true_label in enumerate(CLASS_LABELS):
            for j, predicted_label in enumerate(CLASS_LABELS):
                confusions.append(
                    {
                        "embedding_model": row["embedding_model"],
                        "feature_combination": row["feature_combination"],
                        "true_class": true_label,
                        "predicted_class": predicted_label,
                        "count": row["confusion_matrix"][i][j],
                    }
                )
    flat.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class).to_csv(output / "class_accuracy.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(confusions).to_csv(output / "confusion_matrices.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(importance).to_csv(output / "gain_importance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"gr": groups, "class": labels, "split": np.where(np.isin(np.arange(len(groups)), train), "train", "test")}).to_csv(
        output / "split.csv", index=False, encoding="utf-8-sig"
    )
    save_json(output / "results.json", results)
    return results


def repeated_splits(features, parameters, output, seeds, test_size=0.1, jobs=8):
    rows = []
    pooled = {name: np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=int) for name in FEATURE_COMBINATIONS}
    for seed in seeds:
        train, test = group_split(features["gr"], features["classes"], test_size, seed)
        for combination in FEATURE_COMBINATIONS:
            matrix = feature_matrix(features, combination)
            _, prediction, metrics = fit_evaluate(matrix, features["classes"], train, test, parameters, seed, jobs)
            pooled[combination] += np.asarray(metrics["confusion_matrix"], dtype=int)
            row = {"seed": seed, "feature_combination": combination, **metrics}
            for label in CLASS_LABELS:
                row[f"accuracy_{label}"] = metrics["class_accuracy"][str(label)]
            rows.append({key: value for key, value in row.items() if key not in {"class_accuracy", "confusion_matrix"}})
    frame = pd.DataFrame(rows)
    numeric = ["overall_accuracy", "weighted_precision", "weighted_recall", "weighted_f1", "macro_f1", "balanced_accuracy"] + [f"accuracy_{label}" for label in CLASS_LABELS]
    summary = []
    for combination, group in frame.groupby("feature_combination", sort=False):
        row = {"feature_combination": combination}
        for column in numeric:
            values = group[column].to_numpy(float)
            mean = values.mean()
            sd = values.std(ddof=1)
            half = float(stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))) if len(values) > 1 else 0.0
            row[f"{column}_mean"] = mean
            row[f"{column}_sd"] = sd
            row[f"{column}_ci_low"] = mean - half
            row[f"{column}_ci_high"] = mean + half
        summary.append(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(output / "summary_95ci.csv", index=False, encoding="utf-8-sig")
    records = []
    for combination, matrix in pooled.items():
        for i, true_label in enumerate(CLASS_LABELS):
            for j, predicted_label in enumerate(CLASS_LABELS):
                records.append({"feature_combination": combination, "true_class": true_label, "predicted_class": predicted_label, "count": matrix[i, j]})
    pd.DataFrame(records).to_csv(output / "pooled_confusion_matrices.csv", index=False, encoding="utf-8-sig")
    return frame, pd.DataFrame(summary)


def spatial_folds(metadata, labels, block_quantiles=5):
    x = pd.qcut(metadata["longitude"], block_quantiles, labels=False, duplicates="drop")
    y = pd.qcut(metadata["latitude"], block_quantiles, labels=False, duplicates="drop")
    blocks = x.astype(str) + "_" + y.astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=5)
    return list(splitter.split(np.zeros(len(labels)), labels, groups=blocks)), blocks.to_numpy()


def regional_folds(metadata):
    east = metadata["longitude"].to_numpy() >= metadata["longitude"].median()
    north = metadata["latitude"].to_numpy() >= metadata["latitude"].median()
    regions = np.select([north & ~east, north & east, ~north & ~east, ~north & east], ["NW", "NE", "SW", "SE"], default="SE")
    indices = np.arange(len(metadata))
    folds = [(indices[regions != region], indices[regions == region]) for region in ("NW", "NE", "SW", "SE")]
    return folds, regions
