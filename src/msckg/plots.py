from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import CLASS_NAMES, FEATURE_COMBINATIONS, KGE_MODELS


def style():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.unicode_minus": False,
        }
    )


def save(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=420, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def heatmap(axis, pivot, label, vmin, vmax, cmap):
    values = pivot.to_numpy(float)
    image = axis.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", fontsize=10.5)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label(label)


def embedding_heatmap(metrics, output):
    style()
    frame = pd.read_csv(metrics)
    column = "embedding_model" if "embedding_model" in frame.columns else "model"
    pivot = frame.pivot(index="feature_combination", columns=column, values="overall_accuracy").reindex(index=FEATURE_COMBINATIONS, columns=KGE_MODELS)
    figure, axis = plt.subplots(figsize=(11.5, 8.2))
    heatmap(axis, pivot, "Accuracy", 0.35, 0.80, "YlGnBu")
    axis.set_xlabel("Knowledge graph embedding model")
    axis.set_ylabel("Feature combination")
    save(figure, output)


def class_accuracy_heatmap(path, output):
    style()
    frame = pd.read_csv(path)
    pivot = frame.pivot_table(index="feature_combination", columns="class_name", values="class_accuracy", aggfunc="mean").reindex(index=FEATURE_COMBINATIONS, columns=list(CLASS_NAMES.values()))
    figure, axis = plt.subplots(figsize=(11, 8.2))
    heatmap(axis, pivot, "Class accuracy", 0, 1, "YlOrRd")
    axis.set_xlabel("Domain")
    axis.set_ylabel("Feature combination")
    save(figure, output)


def ablation_heatmap(path, output):
    style()
    frame = pd.read_csv(path)
    pivot = frame.pivot(index="feature_combination", columns="graph_variant", values="overall_accuracy").reindex(index=FEATURE_COMBINATIONS, columns=["Full", "No T/TC", "No TC", "No P/PC", "No PC"])
    figure, axis = plt.subplots(figsize=(9.2, 8.2))
    heatmap(axis, pivot, "Accuracy", 0.35, 0.80, "YlGnBu")
    axis.set_xlabel("Knowledge graph variant")
    axis.set_ylabel("Feature combination")
    save(figure, output)


def grouped_bars(path, output, name_column="method"):
    style()
    frame = pd.read_csv(path)
    x = np.arange(len(frame))
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(8.5, len(frame) * 1.5), 5.3))
    first = axis.bar(x - width / 2, frame["overall_accuracy"], width, label="Accuracy", color="#2F5597")
    second = axis.bar(x + width / 2, frame["macro_f1"], width, label="Macro-F1", color="#70AD47")
    axis.set_xticks(x, frame[name_column], rotation=25, ha="right")
    axis.set_ylabel("Score")
    axis.set_ylim(0, min(1.0, max(frame[["overall_accuracy", "macro_f1"]].max()) + 0.12))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.legend(frameon=False, ncol=2)
    for bars in (first, second):
        axis.bar_label(bars, fmt="%.3f", fontsize=11, padding=3)
    save(figure, output)


def gru_line(mean_metrics, gru_metrics, output):
    style()
    mean = pd.read_csv(mean_metrics).set_index("feature_combination").reindex(FEATURE_COMBINATIONS)
    gru = pd.read_csv(gru_metrics).set_index("feature_combination").reindex(FEATURE_COMBINATIONS)
    x = np.arange(len(FEATURE_COMBINATIONS))
    figure, axis = plt.subplots(figsize=(12, 5.4))
    axis.plot(x, mean["macro_f1"], color="#2878B5", marker="o", linewidth=2.6, markersize=8, label="Mean pooling + XGBoost")
    axis.plot(x, gru["macro_f1"], color="#D95F02", marker="s", linewidth=2.6, markersize=8, label="GRU + XGBoost")
    axis.set_xticks(x, FEATURE_COMBINATIONS, rotation=40, ha="right")
    axis.set_ylabel("Macro-F1")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    figure.subplots_adjust(bottom=0.25, top=0.83)
    save(figure, output)


def domain_scatter(indicators, output):
    style()
    frame = pd.read_csv(indicators)
    figure, axis = plt.subplots(figsize=(8.4, 6.2))
    axis.scatter(frame["heterogeneous_neighbour_proximity_rate"], frame["spatial_concentration"], s=115, color="#1F5D85", edgecolor="white", linewidth=1.0)
    for row in frame.itertuples(index=False):
        text = f"{row.domain}\n{row.class_accuracy:.3f}" if np.isfinite(row.class_accuracy) else row.domain
        axis.annotate(text, (row.heterogeneous_neighbour_proximity_rate, row.spatial_concentration), xytext=(6, 6), textcoords="offset points", fontsize=12)
    axis.set_xlabel("Heterogeneous-neighbour proximity rate")
    axis.set_ylabel("Spatial concentration")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(color="#E6E6E6", linewidth=0.7)
    save(figure, output)


def gain_importance(path, output, combination="T+P+TC+PC"):
    style()
    frame = pd.read_csv(path)
    selected = frame[frame["feature_combination"] == combination].copy()
    blocks = selected[selected["dimension"].isin(("T", "P", "TC", "PC"))].set_index("dimension")
    dimensions = selected[pd.to_numeric(selected["dimension"], errors="coerce").notna()].copy()
    dimensions["dimension"] = pd.to_numeric(dimensions["dimension"]).astype(int)
    top = dimensions.nlargest(20, "gain").sort_values("gain")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [0.8, 1.5]})
    axes[0].bar(blocks.index, blocks["share"], color=["#2F5597", "#70AD47", "#ED7D31", "#A5A5A5"][: len(blocks)])
    axes[0].set_ylabel("Share of gain importance")
    axes[0].set_title("(a) Feature blocks")
    axes[1].barh(top["dimension"].astype(str), top["gain"], color="#2F5597")
    axes[1].set_xlabel("Gain importance")
    axes[1].set_ylabel("Embedding dimension")
    axes[1].set_title("(b) Top 20 dimensions")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    save(figure, output)


def repeated_uncertainty(summary_path, confusion_path, output, combination="T+PC"):
    style()
    summary = pd.read_csv(summary_path).sort_values("overall_accuracy_mean", ascending=False).head(6)
    confusion = pd.read_csv(confusion_path)
    confusion = confusion[confusion["feature_combination"] == combination].pivot(index="true_class", columns="predicted_class", values="count").reindex(index=CLASS_NAMES, columns=CLASS_NAMES).fillna(0)
    normalized = confusion.div(confusion.sum(axis=1).replace(0, 1), axis=0)
    x = np.arange(len(summary))
    lower = summary["overall_accuracy_mean"] - summary["overall_accuracy_ci_low"]
    upper = summary["overall_accuracy_ci_high"] - summary["overall_accuracy_mean"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.05, 1.25]})
    axes[0].errorbar(x, summary["overall_accuracy_mean"], yerr=np.vstack((lower, upper)), fmt="o", color="#2F5597", capsize=5, markersize=8)
    axes[0].set_xticks(x, summary["feature_combination"], rotation=35, ha="right")
    axes[0].set_ylabel("Accuracy with 95% confidence interval")
    heatmap(axes[1], normalized, "Row-normalised proportion", 0, 1, "Blues")
    axes[1].set_xlabel("Predicted domain")
    axes[1].set_ylabel("True domain")
    save(figure, output)


def poi_hit(indicator_path, class_accuracy_path, output):
    style()
    indicators = pd.read_csv(indicator_path).set_index("class_id")
    accuracy = pd.read_csv(class_accuracy_path)
    if "embedding_model" in accuracy.columns:
        accuracy = accuracy[accuracy["embedding_model"] == "DistMult"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    for axis, feature, title in zip(axes, ("PC", "P"), ("(a) POI-category representation", "(b) POI-entity representation")):
        values = accuracy[accuracy["feature_combination"] == feature].set_index("class_id")["class_accuracy"]
        aligned = indicators.join(values.rename("accuracy"))
        axis.scatter(aligned["poi_hit_rate"], aligned["accuracy"], s=105, color="#2F5597", edgecolor="white")
        for row in aligned.itertuples():
            axis.annotate(row.domain, (row.poi_hit_rate, row.accuracy), xytext=(5, 5), textcoords="offset points", fontsize=11)
        axis.set_xlabel("POI hit rate")
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#E6E6E6", linewidth=0.7)
    axes[0].set_ylabel("Class accuracy")
    save(figure, output)


def ripley_curves(path, output):
    style()
    frame = pd.read_csv(path)
    figure, axis = plt.subplots(figsize=(9, 5.8))
    for label, group in frame.groupby("class_id"):
        axis.plot(group["radius"], group["observed_k"], linewidth=2.2, label=CLASS_NAMES[int(label)])
    reference = frame.drop_duplicates("radius")
    axis.plot(reference["radius"], reference["theoretical_k"], color="black", linestyle="--", linewidth=2, label="Theoretical reference")
    axis.set_xlabel("Radius")
    axis.set_ylabel("Ripley’s K(r)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2)
    axis.grid(color="#E6E6E6", linewidth=0.7)
    save(figure, output)


def spatial_expansion(path, output):
    style()
    frame = pd.read_csv(path)
    pivot = frame.pivot(index="feature_combination", columns="condition", values="overall_accuracy_mean").reindex(FEATURE_COMBINATIONS)
    figure, axis = plt.subplots(figsize=(12, 5.4))
    x = np.arange(len(pivot))
    for condition, marker in zip(pivot.columns, ("o", "s")):
        axis.plot(x, pivot[condition], marker=marker, linewidth=2.4, markersize=7.5, label=condition)
    axis.set_xticks(x, pivot.index, rotation=40, ha="right")
    axis.set_ylabel("Mean accuracy across five splits")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axis.legend(frameon=False)
    save(figure, output)
