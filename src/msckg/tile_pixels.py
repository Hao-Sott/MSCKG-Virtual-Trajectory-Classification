from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def count_chunk(records, image_root, path_column, templates, features):
    lookup = {int(code): feature for code, feature in templates}
    rows = []
    for record in records:
        path = Path(record[path_column])
        if not path.is_absolute():
            path = Path(image_root) / path
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint32)
        codes = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        unique, counts = np.unique(codes, return_counts=True)
        result = {key: value for key, value in record.items() if key != path_column}
        result["pixel_total"] = int(codes.size)
        result.update({feature: 0 for feature in features})
        for code, count in zip(unique, counts):
            feature = lookup.get(int(code))
            if feature is not None:
                result[feature] += int(count)
        rows.append(result)
    return rows


def extract_tile_pixels(index_path, image_root, template_path, output, path_column="path", jobs=8, chunk_size=500):
    index = pd.read_csv(index_path)
    templates = pd.read_csv(template_path)
    if path_column not in index.columns:
        raise ValueError(f"Tile index is missing {path_column}")
    required = {"feature", "r", "g", "b"}
    if required.difference(templates.columns):
        raise ValueError("Colour template must contain feature, r, g, and b")
    templates["feature"] = templates["feature"].astype(str)
    templates["code"] = (templates["r"].astype(np.uint32) << 16) | (templates["g"].astype(np.uint32) << 8) | templates["b"].astype(np.uint32)
    pairs = list(templates[["code", "feature"]].itertuples(index=False, name=None))
    features = list(dict.fromkeys(templates["feature"].astype(str)))
    records = index.to_dict("records")
    chunks = [records[start : start + chunk_size] for start in range(0, len(records), chunk_size)]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    first = True
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for rows in executor.map(count_chunk, chunks, [str(image_root)] * len(chunks), [path_column] * len(chunks), [pairs] * len(chunks), [features] * len(chunks)):
            pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig" if first else "utf-8", mode="w" if first else "a", header=first)
            first = False
    return output
