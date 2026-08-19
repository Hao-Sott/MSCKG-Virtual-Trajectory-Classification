from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import cohen_kappa_score

from .constants import CLASS_LABELS, CLASS_NAMES
from .io import parse_ids, resolve_columns


def expert_validation(path, columns, output):
    frame = pd.read_excel(path)
    category = columns["expert_category"]
    expert1 = columns["expert_1"]
    expert2 = columns["expert_2"]
    if set((category, expert1, expert2)).difference(frame.columns):
        raise ValueError("Expert annotation columns do not match the configuration")
    frame = frame[[category, expert1, expert2]].dropna()
    frame[[expert1, expert2]] = frame[[expert1, expert2]].astype(int)
    frame["both_complete"] = frame[expert1].eq(1) & frame[expert2].eq(1)
    category_summary = frame.groupby(category).agg(
        tiles=(category, "size"),
        expert1_complete=(expert1, lambda values: int((values == 1).sum())),
        expert2_complete=(expert2, lambda values: int((values == 1).sum())),
        exact_agreement=(expert1, lambda values: 0),
    ).reset_index()
    agreements = frame[expert1].eq(frame[expert2])
    for index, name in enumerate(category_summary[category]):
        mask = frame[category].eq(name)
        category_summary.loc[index, "exact_agreement"] = int(agreements[mask].sum())
    summary = pd.DataFrame(
        [
            {
                "tiles": len(frame),
                "categories": frame[category].nunique(),
                "expert1_complete": int((frame[expert1] == 1).sum()),
                "expert2_complete": int((frame[expert2] == 1).sum()),
                "expert1_mismatch": int((frame[expert1] == -1).sum()),
                "expert2_mismatch": int((frame[expert2] == -1).sum()),
                "exact_agreement": int(agreements.sum()),
                "exact_agreement_rate": float(agreements.mean()),
                "cohen_kappa": float(cohen_kappa_score(frame[expert1], frame[expert2])),
                "categories_all_four_complete": int(((category_summary["expert1_complete"] == 2) & (category_summary["expert2_complete"] == 2)).sum()),
                "categories_with_mutual_complete_tile": int(frame.groupby(category)["both_complete"].any().sum()),
            }
        ]
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "expert_ratings.csv", index=False, encoding="utf-8-sig")
    category_summary.to_csv(output / "category_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "expert_validation_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def domain_indicators(trajectory, columns, output, class_accuracy=None, feature_combination="T+PC"):
    frame = pd.read_csv(trajectory)
    resolved = resolve_columns(frame, columns, ("group", "label", "longitude", "latitude", "poi_entities"))
    frame["poi_hit"] = frame[resolved["poi_entities"]].map(lambda value: bool(parse_ids(value)))
    hit = frame.groupby(resolved["label"])["poi_hit"].mean()
    points = frame.sort_values(resolved["group"]).groupby(resolved["group"], sort=True).first().reset_index()
    coordinates = points[[resolved["longitude"], resolved["latitude"]]].to_numpy(float)
    labels = points[resolved["label"]].to_numpy(int)
    scaled = np.column_stack((coordinates[:, 0] / 0.0032, coordinates[:, 1] / 0.0024))
    near = np.zeros(len(points), dtype=bool)
    for index in range(len(points)):
        within = np.max(np.abs(scaled - scaled[index]), axis=1) < 1
        near[index] = np.any(within & (labels != labels[index]))
    proximity = {label: float(near[labels == label].mean()) for label in CLASS_LABELS}
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    area = float(np.prod(upper - lower))
    radii = np.linspace(0, float(np.linalg.norm(upper - lower)), 100)
    theoretical = np.pi * radii**2
    concentration = {}
    ripley = []
    for label in CLASS_LABELS:
        selected = coordinates[labels == label]
        distances = cdist(selected, selected)
        np.fill_diagonal(distances, np.inf)
        observed = np.asarray([(distances < radius).sum() * area / (len(selected) ** 2) for radius in radii])
        concentration[label] = float(np.mean(observed - theoretical))
        ripley.extend({"class_id": label, "radius": radius, "observed_k": value, "theoretical_k": expected} for radius, value, expected in zip(radii, observed, theoretical))
    rows = []
    accuracy_lookup = {}
    if class_accuracy:
        accuracy_frame = pd.read_csv(class_accuracy)
        if "feature_combination" in accuracy_frame.columns:
            accuracy_frame = accuracy_frame[accuracy_frame["feature_combination"] == feature_combination]
        accuracy_lookup = dict(zip(accuracy_frame["class_id"].astype(int), accuracy_frame["class_accuracy"].astype(float)))
    for label in CLASS_LABELS:
        rows.append(
            {
                "class_id": label,
                "domain": CLASS_NAMES[label],
                "poi_hit_rate": float(hit.loc[label]),
                "heterogeneous_neighbour_proximity_rate": proximity[label],
                "spatial_concentration": concentration[label],
                "class_accuracy": accuracy_lookup.get(label, np.nan),
            }
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "domain_indicators.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ripley).to_csv(output / "ripley_curves.csv", index=False, encoding="utf-8-sig")
    return rows
