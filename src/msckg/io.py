import json
from pathlib import Path

import numpy as np
import pandas as pd


ALIASES = {
    "group": ("gr", "group", "trajectory_id"),
    "label": ("class", "label", "domain"),
    "timestamp": ("restored_time", "timestamp", "time"),
    "longitude": ("lon", "longitude", "lng"),
    "latitude": ("lat", "latitude"),
    "tile_entity": ("T_entity_id", "entityid", "tile_entity_id"),
    "poi_entities": ("P_entity_ids", "poientityids", "poi_entity_ids"),
    "tile_category": ("TC_entity_id", "wmtsclassentityids", "tile_category_entity_id"),
    "poi_categories": ("PC_entity_ids", "poiclassentityids", "poi_category_entity_ids"),
}


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_columns(frame, configured, required):
    result = {}
    for key in required:
        candidates = [configured.get(key)] + list(ALIASES.get(key, ()))
        match = next((name for name in candidates if name and name in frame.columns), None)
        if match is None:
            raise ValueError(f"Missing column for {key}; checked {candidates}")
        result[key] = match
    return result


def parse_ids(value):
    if value is None or pd.isna(value):
        return ()
    text = str(value).strip()
    if not text or text.lower() in {"nopoi", "none", "nan", "null", "-1"}:
        return ()
    return tuple(int(float(item)) for item in text.replace(",", ";").split(";") if item.strip())


def read_mapping(path):
    mapping = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"Invalid mapping row {number} in {path}")
        if parts[0].lstrip("-").isdigit() and not parts[1].lstrip("-").isdigit():
            index, name = int(parts[0]), parts[1]
        else:
            name, index = parts[0], int(parts[1])
        mapping[name] = index
    ids = sorted(mapping.values())
    if ids != list(range(len(ids))):
        raise ValueError(f"IDs are not contiguous in {path}")
    return mapping


def write_mapping(path, mapping):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(f"{name}\t{index}" for name, index in sorted(mapping.items(), key=lambda item: item[1]))
    path.write_text(text + "\n", encoding="utf-8")


def load_embedding(path, expected_dim=100):
    values = np.load(path, mmap_mode="r")
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise ValueError(f"Expected an N x {expected_dim} embedding matrix, found {values.shape}")
    return values


def load_feature_archive(path):
    with np.load(path, allow_pickle=False) as source:
        result = {key: source[key].copy() for key in source.files}
    required = {"gr", "classes", "T", "P", "TC", "PC"}
    missing = required.difference(result)
    if missing:
        raise ValueError(f"Feature archive is missing {sorted(missing)}")
    return result


def read_triples(path):
    frame = pd.read_csv(path, sep="\t", header=None, names=["head", "tail", "relation"], dtype=str)
    if frame.isna().any().any():
        raise ValueError(f"Invalid triple file: {path}")
    return frame

