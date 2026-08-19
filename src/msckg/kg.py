from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .constants import ABLATIONS
from .io import read_mapping, write_mapping


RELATIONS = (
    "tile_adjacent_tile",
    "tile_belongs_to_tile_category",
    "poi_located_in_tile",
    "poi_near_poi",
    "poi_belongs_to_poi_category_level2",
    "poi_category_level2_belongs_to_level1",
)


def entity_name(prefix, value):
    text = str(value).strip().replace("\t", " ").replace("\n", " ")
    return f"{prefix}{text}"


def build_knowledge_graph(tiles_path, pois_path, output, columns, proximity_m=50):
    tiles = pd.read_csv(tiles_path)
    pois = pd.read_csv(pois_path)
    tile_required = [columns["tile_id"], columns["tile_row"], columns["tile_col"], columns["tile_category_name"]]
    poi_required = [columns["poi_id"], columns["poi_category_level1"], columns["poi_category_level2"], columns["tile_id"], columns["longitude"], columns["latitude"]]
    if set(tile_required).difference(tiles.columns):
        raise ValueError("Tile table does not satisfy the configured schema")
    if set(poi_required).difference(pois.columns):
        raise ValueError("POI table does not satisfy the configured schema")
    tiles = tiles[tile_required].drop_duplicates(columns["tile_id"]).copy()
    pois = pois[poi_required].drop_duplicates(columns["poi_id"]).copy()
    tile_names = {value: entity_name("w_", value) for value in tiles[columns["tile_id"]]}
    tc_names = {value: entity_name("wc_", value) for value in tiles[columns["tile_category_name"]].drop_duplicates()}
    poi_names = {value: entity_name("p_", value) for value in pois[columns["poi_id"]]}
    pc1_names = {value: entity_name("pc1_", value) for value in pois[columns["poi_category_level1"]].drop_duplicates()}
    pc2_names = {value: entity_name("pc2_", value) for value in pois[columns["poi_category_level2"]].drop_duplicates()}
    entities = list(tile_names.values()) + list(poi_names.values()) + list(tc_names.values()) + list(pc1_names.values()) + list(pc2_names.values())
    entity_to_id = {name: index for index, name in enumerate(entities)}
    relation_to_id = {name: index for index, name in enumerate(RELATIONS)}
    tile_lookup = {(int(row), int(col)): tile_names[tile_id] for tile_id, row, col in tiles[[columns["tile_id"], columns["tile_row"], columns["tile_col"]]].itertuples(index=False, name=None)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_mapping(output / "entity2id.txt", entity_to_id)
    write_mapping(output / "relation2id.txt", relation_to_id)
    count = 0
    buffer = []
    target = (output / "train.txt").open("w", encoding="utf-8", newline="")

    def emit(head, tail, relation):
        nonlocal count
        buffer.append(f"{head}\t{tail}\t{relation}\n")
        count += 1
        if len(buffer) >= 100_000:
            target.writelines(buffer)
            buffer.clear()

    try:
        for tile_id, row, col, category in tiles.itertuples(index=False, name=None):
            tile = tile_names[tile_id]
            emit(tile, tc_names[category], RELATIONS[1])
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = tile_lookup.get((int(row) + dr, int(col) + dc))
                    if neighbor is not None:
                        emit(tile, neighbor, RELATIONS[0])
        pc2_to_pc1 = pois[[columns["poi_category_level2"], columns["poi_category_level1"]]].drop_duplicates()
        for pc2, pc1 in pc2_to_pc1.itertuples(index=False, name=None):
            emit(pc2_names[pc2], pc1_names[pc1], RELATIONS[5])
        for poi_id, _, pc2, tile_id, _, _ in pois.itertuples(index=False, name=None):
            emit(poi_names[poi_id], tile_names[tile_id], RELATIONS[2])
            emit(poi_names[poi_id], pc2_names[pc2], RELATIONS[4])
        radians = np.radians(pois[[columns["latitude"], columns["longitude"]]].to_numpy(float))
        tree = BallTree(radians, metric="haversine")
        poi_ids = pois[columns["poi_id"]].to_numpy()
        for start in range(0, len(radians), 100_000):
            neighbors = tree.query_radius(radians[start : start + 100_000], r=proximity_m / 6371008.8, return_distance=False)
            for offset, values in enumerate(neighbors):
                i = start + offset
                for j in values:
                    if i != j:
                        emit(poi_names[poi_ids[i]], poi_names[poi_ids[j]], RELATIONS[3])
        target.writelines(buffer)
    finally:
        target.close()
    return {"entities": len(entity_to_id), "relations": len(relation_to_id), "triples": count}


def make_ablation_graphs(full_graph, output_root):
    full_graph = Path(full_graph)
    output_root = Path(output_root)
    entities = read_mapping(full_graph / "entity2id.txt")
    names = [name for name, _ in sorted(entities.items(), key=lambda item: item[1])]
    reports = []
    for name, prefixes in ABLATIONS.items():
        if name == "full":
            continue
        target = output_root / name
        target.mkdir(parents=True, exist_ok=True)
        entity_count = 0
        with (target / "entity2id.txt").open("w", encoding="utf-8") as mapping_file:
            for entity in names:
                if not entity.startswith(prefixes):
                    mapping_file.write(f"{entity}\t{entity_count}\n")
                    entity_count += 1
        shutil.copy2(full_graph / "relation2id.txt", target / "relation2id.txt")
        triple_count = 0
        with (full_graph / "train.txt").open("r", encoding="utf-8-sig") as source, (target / "train.txt").open("w", encoding="utf-8") as destination:
            for line in source:
                head, tail, _ = line.rstrip("\r\n").split("\t")
                if not head.startswith(prefixes) and not tail.startswith(prefixes):
                    destination.write(line)
                    triple_count += 1
        reports.append({"variant": name, "entities": entity_count, "triples": triple_count, "removed_prefixes": ";".join(prefixes)})
    pd.DataFrame(reports).to_csv(output_root / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    return reports
