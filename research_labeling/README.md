# Research Labeling — Research-Mention Detection in School Board Transcripts

Trains a classifier to identify transcript chunks where research, data, reports, or studies are cited to inform school board policy decisions.

## Overview

This module takes labeled transcript CSVs (one per meeting) and trains a sentence-embedding-based classifier. Each text chunk is encoded with MPNet (`all-mpnet-base-v2`), then a logistic regression or XGBoost model is trained and evaluated. The best model and decision threshold are selected on a held-out validation set using the F2 score, which weights recall twice as heavily as precision — reflecting the sponsor's preference to flag extra chunks for review rather than miss genuine research mentions.

## Setup

```bash
cd research_labeling
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

## Directory Structure

```
research_labeling/
├── research_chunk_pipeline/          # Core ML pipeline
│   ├── config.py                     # All configuration dataclasses
│   ├── data_utils.py                 # CSV loading, validation, transcript-level splitting
│   ├── embedding_utils.py            # Sentence embedding and feature construction
│   ├── modeling.py                   # Model training, threshold selection, evaluation
│   ├── pipeline.py                   # Single train/val/test run + CLI
│   ├── cv_pipeline.py                # Grouped K-Fold cross-validation + CLI
│   ├── run_experiments.py            # Multi-seed experiment driver + aggregation
│   ├── analyze_errors.py             # Post-hoc false positive/negative analysis
│   ├── outputs/                      # Pipeline output artifacts
│   │   ├── no_feature_selection/     # Baseline LR results (10 seeds)
│   │   ├── feature_selection_bic/    # LASSO-BIC feature selection results
│   │   ├── feature_selection_aic/    # LASSO-AIC feature selection results
│   │   └── xgboost_gpu/              # XGBoost results
│   └── results_visualization_scripts/
│       ├── plot_embeddings_extended.py   # t-SNE + HDBSCAN cluster plots
│       ├── plot_fold_embeddings.py       # Per-fold embedding visualizations
│       └── plot_pr_curve.py              # Precision-recall curve comparison
├── Transcript Data/                  # Input: labeled CSVs (one per meeting)
└── requirements.txt
```

## Input Data Format

Each CSV in `Transcript Data/` represents one school board meeting. The pipeline expects these columns:

| Column | Type | Description |
|--------|------|-------------|
| `chunk_id` | int | Unique row ID within the transcript |
| `window_start` | str | Video timestamp where the chunk begins (MM:SS) |
| `window_end` | str | Video timestamp where the chunk ends (MM:SS) |
| `text` | str | Raw transcribed text for this chunk |
| `binary_hit` | int | `1` = research/data mention, `0` = not |

The dataset covers 25 meetings across Houston ISD, Katy ISD, and Spring Branch ISD.

## Pipeline Steps

| Step | Description |
|------|-------------|
| Load | One CSV per transcript is read and validated against the required schema |
| Split | Transcripts are assigned to train / val / test **at the transcript level** to prevent data leakage |
| Context | Chunk text is optionally enriched by joining neighboring chunks |
| Embed | Each chunk is encoded with `sentence-transformers/all-mpnet-base-v2` (768-d) |
| Features | Optional query embedding is concatenated to each chunk embedding (1536-d) |
| Select | Optional LASSO feature selection reduces the embedding dimension using AIC or BIC |
| Train | Logistic regression or XGBoost is fit with a grid search over hyperparameters |
| Tune | Best model + threshold is selected on validation by maximising the F2 score |
| Evaluate | Final metrics are computed once on the held-out test set |
| Artifacts | All outputs (predictions, metrics, error CSVs) are written to the output directory |

---

## Scripts

### `pipeline.py` — single train/val/test run

Runs one complete experiment: load → split → embed → train → evaluate → write artifacts.

```bash
cd research_chunk_pipeline
python pipeline.py --transcript-data-dir "../Transcript Data"
```

| Option | Default | Description |
|--------|---------|-------------|
| `--transcript-data-dir` | *(required)* | Directory containing transcript CSV files |
| `--output-dir` | `outputs` | Directory where all artifacts are written |
| `--context-window` | `0` | Neighboring chunks on each side to join before embedding |
| `--feature-mode` | `chunk_only` | `chunk_only` or `query_conditioned` |
| `--query-text` | *"How are research..."* | Guiding question for `query_conditioned` mode |
| `--save-embeddings` | off | Persist the full feature matrix as a `.npy` file |
| `--seed` | `42` | Random seed for split assignment and model training |
| `--no-stratify` | off | Use a pure random transcript shuffle instead of balancing positive rates |
| `--c` | *(grid search)* | Pin logistic regression C to this value instead of searching |
| `--threshold` | *(auto)* | Pin the decision threshold instead of auto-selecting on validation |
| `--class-weight` | *(grid search)* | `balanced` or `none` — pin class weight instead of sweeping |
| `--train-fraction` | `0.6` | Fraction of transcripts for training |
| `--val-fraction` | `0.2` | Fraction of transcripts for validation |
| `--test-fraction` | `0.2` | Fraction of transcripts for testing |
| `--full-train` | off | Train on all transcripts (no val/test). Requires `--c` and `--threshold` |

**Output artifacts** (written to `--output-dir`):

| File | Description |
|------|-------------|
| `all_transcript_predictions.csv` | Predicted probability and label for every chunk across all transcripts |
| `test_predictions.csv` | Predictions for test-set chunks only |
| `false_positives.csv` | Chunks incorrectly flagged as research mentions |
| `false_negatives.csv` | Research mentions the model missed |
| `metrics_summary.json` | Key metrics: recall, precision, F1, F2, average precision, confusion counts |
| `transcript_split_assignments.csv` | One row per transcript showing its assigned split |
| `validation_threshold_sweep.csv` | Metrics at every candidate threshold for each model configuration |
| `feature_matrix.npy` | Optional: raw feature matrix (only written with `--save-embeddings`) |

**Examples:**

```bash
# Query-conditioned features
python pipeline.py \
  --transcript-data-dir "../Transcript Data" \
  --feature-mode query_conditioned

# Context window: join previous + current + next chunk before embedding
python pipeline.py \
  --transcript-data-dir "../Transcript Data" \
  --context-window 1

# Pin C and threshold, skip grid search
python pipeline.py \
  --transcript-data-dir "../Transcript Data" \
  --c 0.5 --threshold 0.35

# Train on all data (no test split)
python pipeline.py \
  --transcript-data-dir "../Transcript Data" \
  --full-train --c 1.0 --threshold 0.30
```

---

### `cv_pipeline.py` — grouped K-fold cross-validation

Evaluates the pipeline with K-fold cross-validation. Transcripts are never split across folds — a whole meeting is always in exactly one fold's test set.

```bash
cd research_chunk_pipeline
python cv_pipeline.py --transcript-data-dir "../Transcript Data" --n-folds 5
```

| Option | Default | Description |
|--------|---------|-------------|
| `--transcript-data-dir` | *(required)* | Directory containing transcript CSV files |
| `--output-dir` | `outputs` | Directory where `cv_results.json` and fold CSVs are written |
| `--n-folds` | `5` | Number of CV folds (controls approximate train/test ratio) |
| `--val-fraction` | `0.2` | Fraction of each fold's training transcripts held out for selection |
| `--seed` | `42` | Random seed for inner val split and model training |
| `--feature-mode` | `chunk_only` | `chunk_only` or `query_conditioned` |
| `--query-text` | *"How are research..."* | Guiding question for `query_conditioned` mode |
| `--context-window` | `0` | Neighboring chunks to join before embedding |
| `--use-feature-selection` | off | Enable LASSO feature selection within each fold |
| `--lasso-criterion` | `bic` | Information criterion for LASSO C selection: `aic` or `bic` |
| `--lasso-c-values` | `0.001 ... 1.0` | LASSO C candidates swept during feature selection |
| `--model-type` | `logistic_regression` | `logistic_regression` or `xgboost` |
| `--xgb-n-estimators` | `100 300 500` | XGBoost n_estimators candidates |
| `--xgb-max-depth` | `3 5 7` | XGBoost max_depth candidates |
| `--xgb-learning-rate` | `0.05 0.1 0.3` | XGBoost learning_rate candidates |
| `--xgb-device` | `cpu` | Device for XGBoost: `cpu` or `cuda` |

**Train/test ratio by `--n-folds`:**

| `--n-folds` | Approx. train / test |
|-------------|----------------------|
| `3` | ~67 / 33 |
| `4` | ~75 / 25 |
| `5` (default) | ~80 / 20 |

**Output artifacts** (written to `--output-dir`):

| File | Description |
|------|-------------|
| `cv_results.json` | Per-fold metrics and aggregate mean ± std |
| `fold_N_test_predictions.csv` | Predictions for fold N's test set |
| `fold_N_false_positives.csv` | Fold N's incorrectly flagged chunks |
| `fold_N_false_negatives.csv` | Fold N's missed research mentions |

**Examples:**

```bash
# 5-fold CV with LASSO-BIC feature selection
python cv_pipeline.py \
  --transcript-data-dir "../Transcript Data" \
  --use-feature-selection --lasso-criterion bic

# XGBoost with GPU acceleration
python cv_pipeline.py \
  --transcript-data-dir "../Transcript Data" \
  --model-type xgboost --xgb-device cuda

# ~70/30 split with query-conditioned features
python cv_pipeline.py \
  --transcript-data-dir "../Transcript Data" \
  --n-folds 3 --feature-mode query_conditioned
```

---

### `run_experiments.py` — multi-seed experiment driver

Runs `cv_pipeline` over multiple random seeds and aggregates results across all seeds and folds. Use this to assess how sensitive results are to the random seed.

```bash
cd research_chunk_pipeline
python run_experiments.py \
    --experiment-name no_feature_selection \
    --transcript-data-dir "../Transcript Data" \
    --seeds 1 2 3 4 5 6 7 8 9 10
```

All `cv_pipeline` flags (`--feature-mode`, `--use-feature-selection`, `--model-type`, etc.) are passed through directly.

| Option | Default | Description |
|--------|---------|-------------|
| `--experiment-name` | *(required)* | Name for this run; used as a subfolder under `--base-output-dir` |
| `--seeds` | *(required)* | One or more random seeds (e.g., `--seeds 1 2 3 4 5`) |
| `--base-output-dir` | `outputs` | Root directory for outputs |
| `--transcript-data-dir` | *(required)* | Directory containing transcript CSV files |
| *(all cv_pipeline options)* | — | Passed through to `cv_pipeline.run_cv` |

**Output structure:**

```
outputs/<experiment_name>/
├── seed_1/
│   ├── cv_results.json
│   ├── fold_1_test_predictions.csv
│   ├── fold_1_false_positives.csv
│   └── fold_1_false_negatives.csv
├── seed_2/ ...
└── aggregate/
    ├── all_fold_results.csv          # One row per (seed, fold)
    ├── aggregate_summary.json        # Mean / std / min / max per metric
    ├── metrics_barchart.png          # Bar chart: mean ± std for 5 metrics
    ├── threshold_frequency.png       # How often each threshold was selected
    ├── c_frequency.png               # How often each C value was selected
    └── confusion_matrix.png          # Confusion matrix summed across all folds × seeds
```

**Examples:**

```bash
# Baseline: no feature selection, 10 seeds
python run_experiments.py \
    --experiment-name no_feature_selection \
    --transcript-data-dir "../Transcript Data" \
    --seeds 1 2 3 4 5 6 7 8 9 10

# LASSO-BIC feature selection
python run_experiments.py \
    --experiment-name feature_selection_bic \
    --transcript-data-dir "../Transcript Data" \
    --seeds 1 2 3 4 5 6 7 8 9 10 \
    --use-feature-selection --lasso-criterion bic

# XGBoost with GPU
python run_experiments.py \
    --experiment-name xgboost_gpu \
    --transcript-data-dir "../Transcript Data" \
    --seeds 1 2 3 4 5 6 7 8 9 10 \
    --model-type xgboost --xgb-device cuda
```

---

### `analyze_errors.py` — standalone error analysis

Reads a predictions CSV and splits it into false positives and false negatives. The pipeline already writes these automatically, but this script is useful for re-running error analysis on an existing predictions file without re-training.

```bash
cd research_chunk_pipeline
python analyze_errors.py --predictions outputs/all_transcript_predictions.csv
```

| Option | Default | Description |
|--------|---------|-------------|
| `--predictions` | *(required)* | Path to `all_transcript_predictions.csv` |
| `--output-dir` | same folder as predictions | Directory for output CSVs |

**Outputs:** `false_positives.csv`, `false_negatives.csv`, and per-transcript error breakdowns printed to stdout.

---

### `results_visualization_scripts/` — embedding and curve plots

These scripts visualize the embedding space and model performance. They import from the pipeline modules, so run them from `research_chunk_pipeline/results_visualization_scripts/`.

#### `plot_embeddings_extended.py` — t-SNE + cluster analysis

Computes t-SNE on all chunk embeddings and produces three plots plus a cluster CSV.

```bash
cd research_chunk_pipeline/results_visualization_scripts
python plot_embeddings_extended.py --transcript-data-dir "../../Transcript Data"
```

| Option | Default | Description |
|--------|---------|-------------|
| `--transcript-data-dir` | *(required)* | Directory containing transcript CSV files |
| `--district-prefix-parts` | `1` | Number of underscore-separated tokens used to identify the district from `transcript_id` |
| `--hdbscan-min-cluster-size` | `5` | Minimum cluster size for HDBSCAN (lower = more, smaller clusters) |

**Outputs** (written to `../plots/`):

| File | Description |
|------|-------------|
| `full_dataset_tsne.png` | t-SNE with binary labels (red = negative, green = positive) |
| `tsne_by_district.png` | t-SNE colored by school district; positives as stars |
| `tsne_positive_neighborhoods.png` | Positive chunks only, colored by HDBSCAN cluster |
| `positive_neighborhoods.csv` | One row per positive: cluster label, t-SNE coordinates, text preview |

#### `plot_pr_curve.py` — precision-recall curve

Plots mean PR curves across all folds × seeds for one or more experiments, with a shaded min-max range.

```bash
# Single experiment
python plot_pr_curve.py \
    --experiment-dirs "../outputs/no_feature_selection" \
    --output-path "../plots/pr_curve.png"

# Multiple experiments overlaid for comparison
python plot_pr_curve.py \
    --experiment-dirs "../outputs/no_feature_selection" "../outputs/xgboost_gpu" \
    --labels "Logistic Regression" "XGBoost" \
    --output-path "../plots/pr_curve_comparison.png"
```

#### `plot_fold_embeddings.py` — per-fold embedding visualizations

Plots t-SNE for each CV fold, showing how train/test transcripts are distributed in embedding space.

---

## Pipeline Design

### Transcript-level splitting

All chunks from a single meeting are always assigned to the same split. Chunks from the same meeting are highly correlated (same speakers, same agenda items, sometimes the same sentence spanning adjacent chunks), so mixing them across train/val/test would leak information and inflate metrics.

With 25 transcripts at 60/20/20 the default split gives 15 train / 5 val / 5 test transcripts.

### Class-imbalance handling

Positive chunks (research mentions) are rare. Two mechanisms address this:

1. **`class_weight="balanced"`** — re-weights the logistic loss so missing a positive is penalised proportionally to the imbalance ratio. Swept automatically alongside `class_weight=None`.
2. **F2-based threshold and model selection** — selects the threshold that maximises F2 score (recall weighted 2×). The same ordering ranks models across the hyperparameter grid.

### Feature modes

| Mode | Feature vector | Dimension |
|------|---------------|-----------|
| `chunk_only` (default) | `embed(chunk)` | 768 |
| `query_conditioned` | `[embed(chunk) ; embed(query)]` | 1 536 |

Default query: *"How are research, data, reports, or studies used to make informed decisions?"*

### Token truncation (chunk-and-pool)

MPNet has a hard 512-token limit. Chunks exceeding this are split into overlapping 512-token windows (stride 256), each embedded independently, then averaged into a single 768-d vector. This ensures the full text of every chunk is captured.

### Optional LASSO feature selection

When `--use-feature-selection` is enabled, an L1-penalised logistic regression is fit on the inner-training set. The regularisation strength C is chosen by AIC or BIC evaluated on the validation set, and the resulting feature mask is applied before the main hyperparameter grid search.

---

## Reproducing Reported Results

The labeled CSVs are already in `Transcript Data/`. All random seeds are fixed for determinism.

```bash
cd research_chunk_pipeline

# Baseline (no feature selection), 10 seeds
python run_experiments.py \
    --experiment-name no_feature_selection \
    --transcript-data-dir "../Transcript Data" \
    --seeds 1 2 3 4 5 6 7 8 9 10

# LASSO-BIC feature selection, 10 seeds
python run_experiments.py \
    --experiment-name feature_selection_bic \
    --transcript-data-dir "../Transcript Data" \
    --seeds 1 2 3 4 5 6 7 8 9 10 \
    --use-feature-selection --lasso-criterion bic

# Generate PR curve comparison
cd results_visualization_scripts
python plot_pr_curve.py \
    --experiment-dirs "../outputs/no_feature_selection" "../outputs/feature_selection_bic" \
    --labels "Baseline" "LASSO-BIC" \
    --output-path "../plots/pr_curve_comparison.png"
```
