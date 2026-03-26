# Research Labeling - Research-Mention Detection in Board Transcripts

Trains a logistic regression classifier to identify transcript chunks where research, data, reports, or studies are cited to inform school board decisions.

## Overview

This module takes labeled transcript CSVs (one per meeting) and trains a sentence-embedding-based classifier. The pipeline embeds each text chunk using MPNet, then trains and evaluates a logistic regression model. The best model and threshold are selected on the validation set using F2 score, which weights recall twice as heavily as precision.

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
├── research_chunk_pipeline/    # Core ML pipeline
│   ├── config.py               # All configuration dataclasses
│   ├── data_utils.py           # CSV loading, validation, transcript-level splitting
│   ├── embedding_utils.py      # Sentence embedding and feature construction
│   ├── modeling.py             # Logistic regression training and evaluation
│   ├── pipeline.py             # End-to-end orchestration and CLI
│   ├── analyze_errors.py       # Post-hoc false positive/negative analysis
│   ├── outputs/                # Pipeline output CSVs and JSON
│   └── plots/                  # Visualization scripts and PNG outputs
│       ├── visualize_results.py
│       ├── graphic.py
│       └── *.png
├── Transcript Data/            # Input: labeled CSVs (one per meeting)
└── requirements.txt
```

## What the Pipeline Does

| Step | Description |
|------|-------------|
| Load | One CSV per transcript/meeting is read and validated |
| Split | Transcripts are assigned to train / val / test **at the transcript level** to prevent data leakage |
| Context | Chunk text is optionally enriched with neighboring chunks |
| Embed | Each chunk is encoded with `sentence-transformers/all-mpnet-base-v2` |
| Features | Optional query embedding is appended to each chunk embedding |
| Train | Logistic regression is fit with a grid search over C and class_weight |
| Tune | Best model + threshold is selected on validation by maximising F2 score |
| Evaluate | Final metrics are computed once on the held-out test set |
| Artifacts | All outputs are written to a configurable output directory |

## Input Data Format

Each CSV in `Transcript Data/` represents one school board meeting and must contain:

| Column | Description |
|--------|-------------|
| `chunk_id` | Unique row ID within the transcript |
| `window_start` | Video timestamp where the chunk begins |
| `window_end` | Video timestamp where the chunk ends |
| `text` | Raw transcribed text |
| `binary_hit` | `1` = research/data mention, `0` = not |

## Usage

### Run the classification pipeline

```bash
cd research_chunk_pipeline
python pipeline.py --transcript-data-dir "../Transcript Data"
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--transcript-data-dir` | *(required)* | Directory containing transcript CSV files (one per meeting) |
| `--output-dir` | `outputs` | Directory where all pipeline artifacts will be written |
| `--context-window` | `0` | Number of neighboring chunks on each side to join before embedding (0 = current chunk only) |
| `--feature-mode` | `chunk_only` | `chunk_only` or `query_conditioned` (concatenates a query embedding) |
| `--query-text` | *"How are research..."* | Guiding question used when `--feature-mode=query_conditioned` |
| `--save-embeddings` | off | Persist the full feature matrix as a `.npy` file |

### Example: query-conditioned features

```bash
python pipeline.py \
  --transcript-data-dir "../Transcript Data" \
  --output-dir "../outputs_qc" \
  --feature-mode query_conditioned \
  --query-text "How are research, data, reports, or studies used to make informed decisions?"
```

### Example: context window (previous + current + next chunk)

```bash
python pipeline.py \
  --transcript-data-dir "../Transcript Data" \
  --output-dir "../outputs_ctx1" \
  --context-window 1
```

### Generate evaluation plots

```bash
cd research_chunk_pipeline/plots
python visualize_results.py   # produces confusion_matrix.png, threshold_sweep.png, metrics_summary.png
python graphic.py             # produces dataset_distribution.png
```

### Analyze model errors (standalone)

```bash
cd research_chunk_pipeline
python analyze_errors.py --predictions outputs/all_transcript_predictions.csv
```

Prints per-transcript error breakdowns to stdout. Error CSVs are also generated automatically by the pipeline.

## Output Artifacts

Written to `research_chunk_pipeline/outputs/`:

| File | Description |
|------|-------------|
| `all_transcript_predictions.csv` | Predicted probability and label for every chunk across all transcripts |
| `false_positives.csv` | Chunks the model incorrectly flagged as research mentions |
| `false_negatives.csv` | Research mentions the model missed |
| `test_predictions.csv` | Predictions for test-set chunks only |
| `metrics_summary.json` | Key metrics: recall, precision, F1, F2, average precision, confusion counts |
| `transcript_split_assignments.csv` | One row per transcript showing its assigned split |
| `validation_threshold_sweep.csv` | Metrics at every candidate threshold for each model configuration |
| `feature_matrix.npy` | Optional: raw feature matrix (requires `--save-embeddings`) |

Written to `research_chunk_pipeline/plots/`:

| File | Description |
|------|-------------|
| `confusion_matrix.png` | Confusion matrix heatmap |
| `threshold_sweep.png` | Threshold vs. metrics line chart |
| `metrics_summary.png` | Metrics bar chart |
| `dataset_distribution.png` | Dataset split and class distribution |

## Pipeline Design

### Data-split rationale

With **25 transcripts** at 60 / 20 / 20 you get **15 train / 5 val / 5 test**. Splitting is done at the **transcript level** (whole meetings, not individual chunks) because chunks from the same meeting are highly correlated and would leak information if split across sets.

### Class-imbalance strategy

Positive chunks (research mentions) are rare. Two levers handle this:

1. **`class_weight="balanced"`** — re-weights the logistic loss so that missing a positive is penalised proportionally to the imbalance ratio. This is swept automatically alongside `class_weight=None`.

2. **F2-based threshold and model selection** — the pipeline selects the threshold that maximises the F2 score on the validation set. F2 weights recall twice as heavily as precision, favouring models that miss fewer research mentions. The same F2-first ordering is used to pick the best model across the C x class_weight grid.

### Feature modes

| Mode | Feature vector | Dimension |
|------|---------------|-----------|
| `chunk_only` (default) | `embed(chunk)` | 768 |
| `query_conditioned` | `[embed(chunk) ; embed(query)]` | 1 536 |

Default query (when using `query_conditioned`):
> "How are research, data, reports, or studies used to make informed decisions?"

### Other design decisions

- **Optional query-conditioned features** can frame classification as "is this chunk relevant to the research question?" by concatenating chunk and query embeddings (pass `--feature-mode query_conditioned`).
- **Fixed random seed (42)** ensures reproducible splits and model training across runs.

## Reproducing Results

```bash
cd research_chunk_pipeline
python pipeline.py --transcript-data-dir "../Transcript Data"
cd plots
python visualize_results.py
python graphic.py
```

This will regenerate all predictions, metrics, and plots using the labeled CSVs already in the repository.
