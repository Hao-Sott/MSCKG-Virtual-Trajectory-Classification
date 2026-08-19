from pathlib import Path

import pandas as pd

from .constants import ABLATION_LABELS, KGE_MODELS
from .evaluation import repeated_splits, run_feature_combinations
from .io import load_feature_archive


def embedding_comparison(feature_root, parameters, output, seed=5, test_size=0.1, jobs=8):
    output = Path(output)
    frames = []
    for model in KGE_MODELS:
        archive = Path(feature_root) / f"{model}.npz"
        if not archive.exists():
            raise FileNotFoundError(archive)
        target = output / model
        run_feature_combinations(load_feature_archive(archive), parameters, target, seed, test_size, jobs, model)
        frames.append(pd.read_csv(target / "metrics.csv"))
    result = pd.concat(frames, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "embedding_model_comparison.csv", index=False, encoding="utf-8-sig")
    return result


def component_ablation(feature_root, parameters, output, seed=5, test_size=0.1, jobs=8):
    output = Path(output)
    frames = []
    for variant in ABLATION_LABELS:
        archive = Path(feature_root).parent / "models" / "DistMult.npz" if variant == "full" else Path(feature_root) / f"{variant}.npz"
        if not archive.exists():
            raise FileNotFoundError(archive)
        target = output / variant
        run_feature_combinations(load_feature_archive(archive), parameters, target, seed, test_size, jobs, ABLATION_LABELS[variant])
        frame = pd.read_csv(target / "metrics.csv")
        frame["graph_variant"] = ABLATION_LABELS[variant]
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output / "component_ablation.csv", index=False, encoding="utf-8-sig")
    full = result[result["graph_variant"] == "Full"][["feature_combination", "overall_accuracy"]].rename(columns={"overall_accuracy": "full_accuracy"})
    differences = result.merge(full, on="feature_combination", how="left")
    differences["accuracy_change_from_full"] = differences["overall_accuracy"] - differences["full_accuracy"]
    differences.to_csv(output / "component_ablation_with_differences.csv", index=False, encoding="utf-8-sig")
    return differences


def spatial_expansion_comparison(expanded_archive, unexpanded_archive, parameters, output, seeds, test_size=0.1, jobs=8):
    output = Path(output)
    frames = []
    for condition, archive in (("spatial expansion", expanded_archive), ("without spatial expansion", unexpanded_archive)):
        target = output / condition.replace(" ", "_")
        repeated_splits(load_feature_archive(archive), parameters, target, seeds, test_size, jobs)
        frame = pd.read_csv(target / "summary_95ci.csv")
        frame["condition"] = condition
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output / "spatial_expansion_comparison.csv", index=False, encoding="utf-8-sig")
    return result
