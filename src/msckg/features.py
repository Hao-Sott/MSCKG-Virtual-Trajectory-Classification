from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import FEATURE_COMBINATIONS, FEATURE_GROUPS
from .io import load_embedding, parse_ids, resolve_columns


def mean_rows(embedding, ids, empty_value):
    if not ids:
        return np.full(embedding.shape[1], empty_value, dtype=np.float32)
    index = np.asarray(ids, dtype=np.int64)
    if index.min() < 0 or index.max() >= len(embedding):
        raise IndexError("Entity ID is outside the embedding matrix")
    return np.asarray(embedding[index], dtype=np.float32).mean(axis=0)


def point_features(frame, embedding, columns):
    resolved = resolve_columns(frame, columns, ("group", "label", "tile_entity", "poi_entities", "tile_category", "poi_categories"))
    rows = []
    for values in frame[list(resolved.values())].itertuples(index=False, name=None):
        row = dict(zip(resolved, values))
        p_ids = parse_ids(row["poi_entities"])
        pc_ids = parse_ids(row["poi_categories"])
        rows.append(
            (
                int(row["group"]),
                int(row["label"]),
                mean_rows(embedding, (int(row["tile_entity"]),), 0),
                mean_rows(embedding, p_ids, -1),
                mean_rows(embedding, (int(row["tile_category"]),), 0),
                mean_rows(embedding, pc_ids, -1),
            )
        )
    return rows, resolved


def aggregate_trajectories(frame, embedding, columns, active_groups=FEATURE_GROUPS):
    resolved = resolve_columns(frame, columns, ("group", "label", "tile_entity", "poi_entities", "tile_category", "poi_categories"))
    grouped = OrderedDict()
    for values in frame[list(resolved.values())].itertuples(index=False, name=None):
        row = dict(zip(resolved, values))
        group = int(row["group"])
        label = int(row["label"])
        entry = grouped.setdefault(group, {"label": label, "T": [], "P": [], "TC": [], "PC": []})
        if entry["label"] != label:
            raise ValueError(f"Trajectory {group} has multiple labels")
        entry["T"].append(int(row["tile_entity"]))
        entry["P"].extend(parse_ids(row["poi_entities"]))
        entry["TC"].append(int(row["tile_category"]))
        entry["PC"].extend(parse_ids(row["poi_categories"]))
    result = {
        "gr": np.asarray(list(grouped), dtype=np.int64),
        "classes": np.asarray([entry["label"] for entry in grouped.values()], dtype=np.int64),
    }
    for key in active_groups:
        empty_value = -1 if key in {"P", "PC"} else 0
        result[key] = np.stack([mean_rows(embedding, entry[key], empty_value) for entry in grouped.values()]).astype(np.float32)
        if not np.isfinite(result[key]).all():
            raise ValueError(f"Missing aligned embeddings in active feature group {key}")
    return result


def build_feature_archive(trajectory, embedding_path, output, columns, dimension=100, active_groups=FEATURE_GROUPS):
    frame = pd.read_csv(trajectory)
    embedding = load_embedding(embedding_path, dimension)
    result = aggregate_trajectories(frame, embedding, columns, active_groups)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    return result


def feature_matrix(features, combination):
    keys = FEATURE_COMBINATIONS[combination]
    return np.concatenate([np.asarray(features[key], dtype=np.float32) for key in keys], axis=1)


def available_combinations(features):
    present = set(features)
    return [name for name, keys in FEATURE_COMBINATIONS.items() if set(keys).issubset(present)]
