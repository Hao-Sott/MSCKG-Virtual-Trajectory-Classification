# MSCKG-based virtual trajectory classification

This repository contains the end-to-end experimental code for multimodal spatial co-occurrence knowledge graph (MSCKG) construction, knowledge graph embedding, virtual-trajectory representation, XGBoost classification, sensitivity analysis, and interpretation.

No trajectory logs, POIs, map tiles, knowledge graph files, embeddings, annotations, or experimental results are included. All paths are configured locally in `configs/experiment.yaml`.

## Pipeline

1. Convert tile pixel composition into the 117-category semantic taxonomy.
2. Build the six-relation MSCKG from tiles, POIs, and their category attributes.
3. Construct No T/TC, No TC, No P/PC, and No PC component-ablation graphs.
4. Train 100-dimensional TransE-L1, TransE-L2, TransR, ComplEx, RotatE, DistMult, and RESCAL embeddings with DGL-KE.
5. Aggregate T, P, TC, and PC embeddings by trajectory group.
6. Select XGBoost parameters once on the DistMult four-feature training partition.
7. Run the predefined 90:10 split, embedding-model comparison, classifier comparison, repeated splits, semantic baselines, spatial validation, component ablation, spatial-expansion sensitivity, expert validation, GRU comparison, and domain-indicator analysis.
8. Export metrics, predictions, confusion matrices, gain importance, and manuscript-ready figures.

The six domain labels are `food supervision`, `tourism`, `mechanical service`, `weather monitoring`, `hotel`, and `sport`. T denotes tile-entity embeddings, P denotes POI-entity embeddings, TC denotes tile-category embeddings, and PC denotes second-level POI-category embeddings.

## Installation

Linux or WSL2 is recommended for DGL-KE.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install dgl dglke
```

For Windows classification-only runs:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

DGL-KE supports all seven embedding models used here and accepts user-defined `htr` data with explicit entity and relation mappings. The training wrapper converts the manuscript's 200 epochs into DGL-KE update steps and writes the exact command to each model directory. See the [DGL-KE user-data documentation](https://aws-dglke.readthedocs.io/en/latest/train_user_data.html) and [output format](https://aws-dglke.readthedocs.io/en/latest/format_out.html).

## Configuration

Edit `configs/experiment.yaml`. Paths may be absolute or relative to the repository root. The complete schema is in [docs/data_contract.md](docs/data_contract.md).

The entity embedding matrix must be an `N × 100` NumPy array whose row index is the entity ID. For an ablated graph, the code realigns retained entity vectors to the full-graph ID space before feature extraction. NoPOI produces an all-−1 P vector and an all-−1 PC vector.

The fixed XGBoost configuration is:

```text
n_estimators=400
max_depth=6
learning_rate=0.05
subsample=0.8
colsample_bytree=0.8
min_child_weight=1
reg_lambda=1.0
tree_method=hist
random_state=5
```

The main group-level split uses 90% of trajectories for training and 10% for testing. Five repeated splits use seeds 0–4. The original class IDs are retained throughout.

## Commands

Validate configured inputs:

```bash
msckg --config configs/experiment.yaml validate
```

Extract colour-template pixel counts, build tile categories, the full graph, and the four ablation graphs:

```bash
msckg --config configs/experiment.yaml extract-tile-pixels
msckg --config configs/experiment.yaml classify-tiles
msckg --config configs/experiment.yaml build-kg
msckg --config configs/experiment.yaml ablate-kg
```

Inspect a DGL-KE command without training:

```bash
msckg --config configs/experiment.yaml train-kge --model DistMult --variant full --gpu 0 --dry-run
```

Train the full graph with all seven models and the four DistMult ablation graphs:

```bash
msckg --config configs/experiment.yaml train-kge --model all --variant full --gpu 0
msckg --config configs/experiment.yaml train-kge --model DistMult --variant no_t_tc --gpu 0
msckg --config configs/experiment.yaml train-kge --model DistMult --variant no_tc --gpu 0
msckg --config configs/experiment.yaml train-kge --model DistMult --variant no_p_pc --gpu 0
msckg --config configs/experiment.yaml train-kge --model DistMult --variant no_pc --gpu 0
```

Prepare trajectory features:

```bash
msckg --config configs/experiment.yaml prepare-features --model DistMult --variant full
msckg --config configs/experiment.yaml prepare-features --model DistMult --variant no_tc
```

Repeat `prepare-features` for every embedding model and ablation variant required by a comparison.

Run downstream experiments:

```bash
msckg --config configs/experiment.yaml select-parameters
msckg --config configs/experiment.yaml main
msckg --config configs/experiment.yaml embedding-comparison
msckg --config configs/experiment.yaml classifier-comparison --combination all
msckg --config configs/experiment.yaml repeated-splits
msckg --config configs/experiment.yaml semantic-baselines
msckg --config configs/experiment.yaml spatial-validation --combination T+PC
msckg --config configs/experiment.yaml component-ablation
msckg --config configs/experiment.yaml expert-validation
msckg --config configs/experiment.yaml gru
msckg --config configs/experiment.yaml domain-indicators --class-accuracy outputs/main/class_accuracy.csv
```

`semantic-baselines` fits Word2Vec only on the training partition unless a precomputed aligned matrix is supplied with `--word2vec`. TF-IDF is also fitted only on the training partition.

Spatial expansion requires a private tile-context table and a text file containing training trajectory-group IDs:

```bash
msckg --config configs/experiment.yaml expand-training --tile-context /private/tile_context.csv --train-groups /private/train_groups.txt --output /private/expanded_training.csv
msckg --config configs/experiment.yaml spatial-expansion --expanded /private/expanded_features.npz --unexpanded /private/unexpanded_features.npz
```

Run the standard sequence of stages with restart indices:

```bash
python scripts/run_all.py --config configs/experiment.yaml --dry-run
python scripts/run_all.py --config configs/experiment.yaml --start 0 --stop 32 --gpu 0
```

Long-running KGE stages should be executed separately and checked before continuing.

## Figures

```bash
msckg plot --kind embedding-heatmap --input outputs/embedding_comparison/embedding_model_comparison.csv --output outputs/figures/embedding_heatmap.png
msckg plot --kind ablation-heatmap --input outputs/component_ablation/component_ablation.csv --output outputs/figures/ablation_heatmap.png
msckg plot --kind bars --input outputs/semantic_baselines/semantic_baselines.csv --name-column method --output outputs/figures/semantic_baselines.png
msckg plot --kind gru-line --input outputs/main/metrics.csv --second-input outputs/gru/gru_results.csv --output outputs/figures/gru_comparison.png
msckg plot --kind domain-scatter --input outputs/domain_indicators/domain_indicators.csv --output outputs/figures/domain_indicators.png
```

Figures use Times New Roman, large labels, compact layouts, and high-resolution raster export.

## Reproducibility notes

- Splits operate on trajectory groups, never on individual log rows.
- The predefined split follows the stored experimental procedure: a seed-5 permutation followed by a 90:10 cut.
- XGBoost hyperparameters are selected once by three-fold stratified cross-validation on the DistMult T+P+TC+PC training partition, then held fixed.
- Repeated random splits use seeds 0, 1, 2, 3, and 4 and report 95% Student-t confidence intervals.
- Spatial block validation uses a 5 × 5 longitude–latitude quantile grid and `StratifiedGroupKFold`; regional hold-out uses quadrants defined by median trajectory-centroid longitude and latitude.
- GRU uses a single unidirectional layer with 400 hidden units, AdamW, learning rate `8e-4`, weight decay `1e-4`, batch size 64, gradient clipping at 1.0, 15% internal validation, and early-stopping patience 10. The selected epoch count is refitted on the full outer training set before XGBoost evaluation.
- Component removal deletes the specified node families and every incident edge. Internal graph prefixes `w_`, `wc_`, `p_`, `pc1_`, and `pc2_` correspond to formal T, TC, P, and PC terminology.

## Repository contents

```text
configs/experiment.yaml        experiment settings and private paths
docs/data_contract.md          required input schemas
docs/manuscript_mapping.md     manuscript-to-command map
docs/github_upload.md          publication workflow
src/msckg/                     reusable implementation
scripts/run_all.py             restartable stage runner
```

## License

The code is released under the MIT License. Data remain subject to their original access and privacy restrictions.
