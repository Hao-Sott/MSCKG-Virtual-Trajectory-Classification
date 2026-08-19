from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .baselines import count_matrix
from .evaluation import evaluate, fit_evaluate, group_split, regional_folds, spatial_folds
from .features import feature_matrix
from .io import resolve_columns


def trajectory_metadata(trajectory, groups, columns, grid_size_degrees=0.02):
    frame = pd.read_csv(trajectory)
    resolved = resolve_columns(frame, columns, ("group", "label", "longitude", "latitude", "tile_entity"))
    group_col = resolved["group"]
    metadata = frame.groupby(group_col, sort=False).agg(
        label=(resolved["label"], "first"),
        longitude=(resolved["longitude"], "mean"),
        latitude=(resolved["latitude"], "mean"),
    )
    metadata = metadata.reindex(groups)
    if metadata.isna().any().any():
        raise ValueError("Trajectory groups and feature groups are not aligned")
    tile_counters = {int(group): Counter() for group in groups}
    grid_counters = {int(group): Counter() for group in groups}
    for group, tile in frame[[group_col, resolved["tile_entity"]]].itertuples(index=False, name=None):
        tile_counters[int(group)].update((int(tile),))
    for group, longitude, latitude in frame[[group_col, resolved["longitude"], resolved["latitude"]]].itertuples(index=False, name=None):
        grid_counters[int(group)].update(((int(np.floor(float(longitude) / grid_size_degrees)), int(np.floor(float(latitude) / grid_size_degrees))),))
    vocabulary = sorted({tile for counter in tile_counters.values() for tile in counter})
    tile_frequency = count_matrix([tile_counters[int(group)] for group in groups], vocabulary)
    totals = np.asarray(tile_frequency.sum(axis=1)).reshape(-1)
    tile_frequency = sp.diags(np.reciprocal(np.maximum(totals, 1))) @ tile_frequency
    grid_vocabulary = sorted({grid for counter in grid_counters.values() for grid in counter})
    grid_one_hot = count_matrix([grid_counters[int(group)] for group in groups], grid_vocabulary)
    grid_one_hot.data[:] = 1.0
    return metadata.reset_index(names="gr"), tile_frequency, grid_one_hot


def run_spatial_validation(trajectory, features, columns, parameters, output, combination="T+PC", jobs=8, block_quantiles=5, grid_size_degrees=0.02):
    metadata, tile_frequency, grid_one_hot = trajectory_metadata(trajectory, features["gr"], columns, grid_size_degrees)
    labels = features["classes"]
    matrices = {
        "DistMult-T": feature_matrix(features, "T"),
        f"DistMult-{combination}": feature_matrix(features, combination),
        "DistMult-All": feature_matrix(features, "T+P+TC+PC"),
        "tile-ID frequency": tile_frequency,
        "spatial-grid one-hot": grid_one_hot,
    }
    random_fold = [group_split(features["gr"], labels, 0.1, 5)]
    block_folds, blocks = spatial_folds(metadata, labels, block_quantiles)
    region_holdouts, regions = regional_folds(metadata)
    protocols = {"random_90_10": random_fold, "spatial_block_5fold": block_folds, "region_leave_one_out": region_holdouts}
    rows = []
    assignments = metadata.copy()
    assignments["spatial_block"] = blocks
    assignments["region"] = regions
    for protocol, folds in protocols.items():
        for method, matrix in matrices.items():
            pooled_true, pooled_pred = [], []
            for fold, (train, test) in enumerate(folds):
                _, prediction, metrics = fit_evaluate(matrix, labels, train, test, parameters, 5 + fold, jobs)
                rows.append(
                    {
                        "protocol": protocol,
                        "method": method,
                        "fold": fold,
                        "train_count": len(train),
                        "test_count": len(test),
                        **{key: value for key, value in metrics.items() if key not in {"class_accuracy", "confusion_matrix"}},
                    }
                )
                pooled_true.extend(labels[test])
                pooled_pred.extend(prediction)
            metrics = evaluate(np.asarray(pooled_true), np.asarray(pooled_pred))
            rows.append(
                {
                    "protocol": protocol,
                    "method": method,
                    "fold": "pooled",
                    "train_count": "",
                    "test_count": len(pooled_true),
                    **{key: value for key, value in metrics.items() if key not in {"class_accuracy", "confusion_matrix"}},
                }
            )
        majority_true, majority_pred = [], []
        for train, test in folds:
            majority = Counter(labels[train]).most_common(1)[0][0]
            majority_true.extend(labels[test])
            majority_pred.extend(np.repeat(majority, len(test)))
        metrics = evaluate(np.asarray(majority_true), np.asarray(majority_pred))
        rows.append({"protocol": protocol, "method": "majority", "fold": "pooled", "test_count": len(majority_true), **{key: value for key, value in metrics.items() if key not in {"class_accuracy", "confusion_matrix"}}})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "spatial_validation.csv", index=False, encoding="utf-8-sig")
    assignments.to_csv(output / "spatial_assignments.csv", index=False, encoding="utf-8-sig")
    return rows


def expand_training_trajectories(trajectory, tile_context, train_groups, columns, output, target_per_class=None, seed=5):
    frame = pd.read_csv(trajectory)
    context = pd.read_csv(tile_context)
    required = ["tile_entity", "neighbor_tile_entity", "neighbor_tile_category", "neighbor_poi_entities", "neighbor_poi_categories"]
    if set(required).difference(context.columns):
        raise ValueError("Tile context must contain current and neighbouring entity columns")
    group_col = columns["group"]
    label_col = columns["label"]
    training = frame[frame[group_col].isin(set(train_groups))].copy()
    labels = training.groupby(group_col)[label_col].first()
    counts = labels.value_counts()
    target = int(target_per_class or round(float(counts.median())))
    rng = np.random.default_rng(seed)
    selected_groups = []
    generated = []
    next_group = int(frame[group_col].max()) + 1
    context_by_tile = {int(tile): group for tile, group in context.groupby("tile_entity")}
    for label, count in counts.items():
        groups = labels[labels == label].index.to_numpy()
        if count >= target:
            selected_groups.extend(rng.choice(groups, target, replace=False).tolist())
            continue
        selected_groups.extend(groups.tolist())
        for source_group in rng.choice(groups, target - count, replace=True):
            copied = training[training[group_col] == source_group].copy()
            row_index = int(rng.choice(copied.index))
            tile = int(copied.loc[row_index, columns["tile_entity"]])
            candidates = context_by_tile.get(tile)
            if candidates is not None and len(candidates):
                replacement = candidates.iloc[int(rng.integers(len(candidates)))]
                copied.loc[row_index, columns["tile_entity"]] = replacement["neighbor_tile_entity"]
                copied.loc[row_index, columns["tile_category"]] = replacement["neighbor_tile_category"]
                copied.loc[row_index, columns["poi_entities"]] = replacement["neighbor_poi_entities"]
                copied.loc[row_index, columns["poi_categories"]] = replacement["neighbor_poi_categories"]
            copied[group_col] = next_group
            copied["source_gr"] = source_group
            copied["is_augmented"] = 1
            generated.append(copied)
            next_group += 1
    retained = training[training[group_col].isin(set(selected_groups))].copy()
    retained["source_gr"] = retained[group_col]
    retained["is_augmented"] = 0
    result = pd.concat([retained, *generated], ignore_index=True) if generated else retained
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return result
