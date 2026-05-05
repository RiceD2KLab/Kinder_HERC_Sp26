# Kinder HERC - School Board Meeting Analysis Pipeline

**Faculty Mentor:** Arko Barman
**Team Members:** Annabelle Du (ad110), Grant Thompson (gwt5035), Melody Dao (md82), Mirfat Maani (mam35), Nilda Jarero (nmj5)

## Project Description

This repository contains an end-to-end pipeline for collecting, transcribing, and analyzing school board meeting recordings from Houston-area school districts. The project was developed for the Kinder Institute for Urban Research at Rice University (HERC) as part of COMP 449 / DATA 435 (Spring 2026).

The goal is to identify segments of school board meetings where **research, data, reports, or studies are cited to inform policy decisions**. The pipeline consists of five standalone programs that are run sequentially:

1. **Web Scraping** — Download meeting videos from district websites (Swagit).
2. **Transcription** — Convert audio recordings to text using NVIDIA Parakeet ASR.
3. **Word Error Rate** — Evaluate transcription accuracy against human-labeled gold standards before downstream use.
4. **Transcript Chunking** — Split timestamped transcripts into fixed-duration chunks for labeling.
5. **Research Labeling** — Train a logistic regression classifier to detect research-mention chunks in transcripts.

## Repository Structure

```
Kinder_HERC_Sp26/
├── app/                        # Desktop GUI and Google Colab inference app
│   ├── main.py                 # Entry point (run: python main.py or frozen exe)
│   ├── gui.py                  # CustomTkinter GUI wiring all pipeline steps
│   ├── highlighter.py          # Builds highlighted .docx from predictions
│   ├── trained_model.py        # Inference wrapper: loads artifacts and scores chunks
│   ├── colab_runner.py         # Google Colab entry point (run())
│   ├── __init__.py
│   └── requirements.txt
├── web_scraping/               # Download meeting videos from Swagit district pages
│   ├── cli.py                  # Command-line entry point
│   ├── config.py               # Constants, regex patterns, HTTP headers
│   ├── dates.py                # Date parsing from video titles/URLs
│   ├── html_scrape.py          # HTML fetching and link extraction
│   ├── models.py               # Source dataclass
│   ├── pipeline.py             # Per-source orchestration logic
│   ├── ytdlp_runner.py         # yt-dlp download + ffmpeg audio extraction
│   ├── __init__.py
│   └── requirements.txt
├── transcription/              # Audio-to-text transcription
│   ├── parakeet_transcribe.py  # Chunked long-form transcription with Parakeet
│   └── requirements.txt
├── word_error_rate/            # Transcription quality evaluation
│   ├── wer_norm.py             # Normalized WER calculation
│   └── requirements.txt
├── transcript_chunking/        # Split transcripts into time-window chunks
│   ├── create_chunks.py        # Chunking script (standard library only)
│   └── mult_chunk.py           # Batch helper: chunks every .txt in a directory
├── research_labeling/          # Research-mention classification
│   ├── research_chunk_pipeline/  # Core ML pipeline modules
│   │   ├── config.py             # Pipeline configuration dataclasses
│   │   ├── data_utils.py         # Data loading and transcript-level splitting
│   │   ├── embedding_utils.py    # Sentence embedding and feature construction
│   │   ├── modeling.py           # Model training, threshold selection, evaluation
│   │   ├── pipeline.py           # Single train/val/test run + CLI
│   │   ├── cv_pipeline.py        # Grouped K-Fold cross-validation + CLI
│   │   ├── run_experiments.py    # Multi-seed experiment driver + aggregation
│   │   ├── analyze_errors.py     # False positive/negative error analysis
│   │   ├── outputs/              # Pipeline output CSVs, JSON, and plots
│   │   └── results_visualization_scripts/  # t-SNE, PR curve, fold embedding plots
│   ├── Transcript Data/          # Labeled CSV files (one per meeting)
│   └── requirements.txt
├── requirements.txt            # All Python dependencies (combined)
└── README.md                   # This file
```

## Installation

**Prerequisites:**
- Python 3.11.6
- `ffmpeg` installed and available on PATH (required by web scraping and transcription)

**Clone and set up a virtual environment:**

```bash
git clone git@github.com:RiceD2KLab/Kinder_HERC_Sp26.git
cd Kinder_HERC_Sp26
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Or install per-component (each subdirectory has its own `requirements.txt`):

| Component | Key Packages |
|-----------|-------------|
| Web Scraping | `yt-dlp`, `requests`, `beautifulsoup4`, `lxml`, `tqdm` |
| Transcription | `nemo_toolkit[asr]` (includes PyTorch, NVIDIA Parakeet) |
| Word Error Rate | `jiwer`, `contractions`, `num2words` |
| Transcript Chunking | *(standard library only — no extra dependencies)* |
| Research Labeling | `sentence-transformers`, `scikit-learn`, `pandas`, `numpy`, `matplotlib` |

## End-to-End Workflow

The five programs are run sequentially. Each step's output becomes the next step's input.

### Step 1: Download meeting videos

Create a `sources.txt` file with one `District|URL` entry per line:

```text
# sources.txt — one district per line
Spring Branch ISD|https://springbranchisdtx.new.swagit.com/videos/364571/download
Katy ISD|https://katyisd.new.swagit.com/videos/123456/download
```

Run:

```bash
python -m web_scraping.cli --sources-file sources.txt --cutoff 2024-09-01
```

**Output:** WAV audio files organized by district:

```
School Board Meetings/
├── Spring Branch ISD/
│   ├── Spring_Branch_ISD-2024-09-23-Board_Meetings.wav
│   └── Spring_Branch_ISD-2024-12-13-Special_Board_Meetings.wav
└── Katy ISD/
    └── Katy_ISD-2024-12-09-Board_Meetings.wav
```

See [web_scraping/README.md](web_scraping/README.md) for all CLI options.

### Step 2: Transcribe audio to text

```bash
python transcription/parakeet_transcribe.py \
    --input "School Board Meetings/" \
    --output_dir transcripts/
```

**Output:** One `.txt` transcript per audio file with `[MM:SS–MM:SS]` section headers:

```
transcripts/
├── Spring_Branch_ISD-2024-09-23-Board_Meetings.txt
├── Spring_Branch_ISD-2024-12-13-Special_Board_Meetings.txt
└── Katy_ISD-2024-12-09-Board_Meetings.txt
```

See [transcription/README.md](transcription/README.md) for all CLI options.

### Step 3: Evaluate transcription quality

Validate the ASR output before downstream use by comparing against a human reference transcript:

```bash
python word_error_rate/wer_norm.py reference.txt hypothesis.txt
```

**Output:** Printed to stdout — WER score, substitution/deletion/insertion counts, and top-10 errors per category.

See [word_error_rate/README.md](word_error_rate/README.md) for normalization details.

### Step 4: Chunk transcripts for labeling

Split each transcript into fixed-duration time windows (default 2 minutes):

```bash
python transcript_chunking/create_chunks.py \
    --input transcripts/Spring_Branch_ISD-2024-09-23-Board_Meetings.txt \
    --output chunks/Chunk_Spring_Branch_ISD-2024-09-23-Board_Meetings.csv
```

**Output:** One CSV per transcript with columns `chunk_id`, `window_start`, `window_end`, `text`.

See [transcript_chunking/README.md](transcript_chunking/README.md) for options.

### Step 5: Label transcripts and train the classifier

The labeled transcript CSVs in `research_labeling/Transcript Data/` are used as training data. To run a single experiment:

```bash
cd research_labeling/research_chunk_pipeline
python pipeline.py --transcript-data-dir "../Transcript Data"
```

To reproduce the full multi-seed experiments from the report:

```bash
# Baseline (no feature selection), 10 seeds
python run_experiments.py \
    --experiment-name no_feature_selection \
    --transcript-data-dir "../Transcript Data" \
    --seeds 1 2 3 4 5 6 7 8 9 10
```

**Output:** Predictions, metrics, error analysis, and aggregate plots written under `outputs/<experiment_name>/`:

```
outputs/no_feature_selection/
├── seed_1/
│   ├── cv_results.json
│   ├── fold_1_test_predictions.csv
│   ├── fold_1_false_positives.csv
│   └── fold_1_false_negatives.csv
├── seed_2/ ...
└── aggregate/
    ├── all_fold_results.csv
    ├── aggregate_summary.json
    ├── metrics_barchart.png
    ├── threshold_frequency.png
    └── confusion_matrix.png
```

To generate embedding visualizations:

```bash
cd results_visualization_scripts
python plot_embeddings_extended.py --transcript-data-dir "../../Transcript Data"
python plot_pr_curve.py --experiment-dirs "../outputs/no_feature_selection" \
    --output-path "../plots/pr_curve.png"
```

See [research_labeling/README.md](research_labeling/README.md) for complete documentation of all scripts and options.

### App: Desktop GUI and Google Colab runner

A pre-built graphical interface wraps all five pipeline steps into a single window.

**Desktop (Windows/macOS/Linux):**

```bash
cd app
pip install -r requirements.txt
python main.py
```

Enter a district name and either a Swagit URL or local audio files, choose an output folder, and click **Run Pipeline**.  A highlighted `.docx` report is written to the output folder on completion.

**Google Colab:**

```python
import sys
sys.path.insert(0, "/content/Kinder_HERC_Sp26")
from app.colab_runner import run
run(url="https://...", district="Houston ISD", out_dir="/content/output")
```

See [app/README.md](app/README.md) for full usage instructions.

## Data

Labeled transcript CSVs are stored in `research_labeling/Transcript Data/`. Each CSV represents one school board meeting and contains these columns:

| Column | Description |
|--------|-------------|
| `chunk_id` | Unique row ID within the transcript |
| `window_start` | Video timestamp where the chunk begins |
| `window_end` | Video timestamp where the chunk ends |
| `text` | Raw transcribed text for this chunk |
| `binary_hit` | `1` = research/data mention, `0` = not |

The dataset covers 25 meetings across Houston ISD, Katy ISD, and Spring Branch ISD.

Audio/video files are not included in the repository due to size. To obtain them, run the web scraping module (Step 1 above) with the appropriate district URLs from their Swagit pages.

## Reproducing Results

To reproduce the classification results from the report:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run multi-seed experiments (labeled CSVs already in the repo)
cd research_labeling/research_chunk_pipeline

python run_experiments.py \
    --experiment-name no_feature_selection \
    --transcript-data-dir "../Transcript Data" \
    --seeds 1 2 3 4 5 6 7 8 9 10

python run_experiments.py \
    --experiment-name feature_selection_bic \
    --transcript-data-dir "../Transcript Data" \
    --seeds 1 2 3 4 5 6 7 8 9 10 \
    --use-feature-selection --lasso-criterion bic

# 3. Generate embedding visualizations
cd results_visualization_scripts
python plot_embeddings_extended.py --transcript-data-dir "../../Transcript Data"
python plot_pr_curve.py \
    --experiment-dirs "../outputs/no_feature_selection" "../outputs/feature_selection_bic" \
    --labels "Baseline" "LASSO-BIC" \
    --output-path "../plots/pr_curve_comparison.png"
```

All random seeds are fixed for determinism. Running the above commands will reproduce the metrics in `aggregate_summary.json` and the plots under `outputs/<experiment_name>/aggregate/`.
