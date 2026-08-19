import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_mapping


MODEL_NAMES = {
    "TransE_L1": "TransE_l1",
    "TransE_L2": "TransE_l2",
    "TransR": "TransR",
    "ComplEx": "ComplEx",
    "RotatE": "RotatE",
    "DistMult": "DistMult",
    "RESCAL": "RESCAL",
}


def count_lines(path):
    with Path(path).open("rb") as source:
        return sum(block.count(b"\n") for block in iter(lambda: source.read(8 * 1024 * 1024), b""))


def prepare_dglke_dataset(kg_dir, output):
    kg_dir = Path(kg_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entities = read_mapping(kg_dir / "entity2id.txt")
    relations = read_mapping(kg_dir / "relation2id.txt")
    (output / "entities.dict").write_text("\n".join(f"{index}\t{name}" for name, index in sorted(entities.items(), key=lambda item: item[1])) + "\n", encoding="utf-8")
    (output / "relations.dict").write_text("\n".join(f"{index}\t{name}" for name, index in sorted(relations.items(), key=lambda item: item[1])) + "\n", encoding="utf-8")
    chunks = pd.read_csv(kg_dir / "train.txt", sep="\t", header=None, names=["head", "tail", "relation"], dtype=str, chunksize=1_000_000)
    target = output / "train.ids.tsv"
    first = True
    for chunk in chunks:
        converted = pd.DataFrame(
            {
                "head": chunk["head"].map(entities),
                "tail": chunk["tail"].map(entities),
                "relation": chunk["relation"].map(relations),
            }
        )
        if converted.isna().any().any():
            raise ValueError("Triples contain names absent from the ID mappings")
        converted.astype(np.int64).to_csv(target, sep="\t", header=False, index=False, mode="w" if first else "a")
        first = False
    return target


def dglke_command(kg_dir, output, model, dimension, epochs, batch_size, negative_size, learning_rate, regularization, regularization_norm, gpu=None, jobs=8):
    kg_dir = Path(kg_dir)
    output = Path(output)
    prepared = output / "dglke_input"
    train_file = prepare_dglke_dataset(kg_dir, prepared)
    triples = count_lines(train_file)
    max_step = int(math.ceil(triples / batch_size) * epochs)
    command = [
        "dglke_train",
        "--model_name",
        MODEL_NAMES[model],
        "--dataset",
        f"MSCKG_{model}",
        "--data_path",
        str(prepared),
        "--format",
        "udd_htr",
        "--data_files",
        "entities.dict",
        "relations.dict",
        train_file.name,
        "--delimiter",
        "\t",
        "--hidden_dim",
        str(dimension),
        "--batch_size",
        str(batch_size),
        "--neg_sample_size",
        str(negative_size),
        "--lr",
        str(learning_rate),
        "--regularization_coef",
        str(regularization),
        "--regularization_norm",
        str(regularization_norm),
        "--max_step",
        str(max_step),
        "--num_thread",
        str(jobs),
        "--save_path",
        str(output / "checkpoint"),
    ]
    if gpu is not None:
        command.extend(("--gpu", str(gpu)))
    return command, {"triples": triples, "steps_per_epoch": math.ceil(triples / batch_size), "max_step": max_step}


def train_dglke(kg_dir, output, model, settings, gpu=None, jobs=8, dry_run=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    command, metadata = dglke_command(
        kg_dir,
        output,
        model,
        settings["embedding_dim"],
        settings["epochs"],
        settings["batch_size"],
        settings["negative_size"],
        settings["learning_rate"],
        settings["regularization"],
        settings["regularization_norm"],
        gpu,
        jobs,
    )
    (output / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    if dry_run:
        return command, metadata
    environment = os.environ.copy()
    environment.setdefault("DGLBACKEND", "pytorch")
    subprocess.run(command, check=True, env=environment)
    candidates = sorted((output / "checkpoint").rglob("*_entity.npy"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("DGL-KE did not produce an entity embedding file")
    shutil.copy2(candidates[-1], output / "entity_embeddings_100d.npy")
    return command, metadata


def align_embedding(source_embedding, source_mapping, target_mapping, output, fill_value=np.nan):
    source = np.load(source_embedding, mmap_mode="r")
    source_ids = read_mapping(source_mapping)
    target_ids = read_mapping(target_mapping)
    result = np.full((len(target_ids), source.shape[1]), fill_value, dtype=np.float32)
    for name, source_id in source_ids.items():
        target_id = target_ids.get(name)
        if target_id is not None:
            result[target_id] = source[source_id]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result)
    return result

