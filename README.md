# Kinder HERC - School Board Meeting Analysis Pipeline

**Faculty Mentor:** Arko Barman
**Team Members:** Annabelle Du (ad110), Grant Thompson (gwt5035), Melody Dao (md82), Mirfat Maani (mam35), Nilda Jarero (nmj5)

## Project Description

This repository contains an end-to-end pipeline for collecting, transcribing, and analyzing school board meeting recordings from Houston-area school districts. The project was developed for the Kinder Institute for Urban Research at Rice University (HERC) as part of COMP 449 / DATA 435 (Spring 2026).

The goal is to identify segments of school board meetings where **research, data, reports, or studies are cited to inform policy decisions**. The pipeline consists of four standalone programs that are run sequentially:

1. **Web Scraping** — Download meeting videos from district websites (Swagit).
2. **Transcription** — Convert audio recordings to text using NVIDIA Parakeet ASR.
3. **Transcript Chunking** — Split timestamped transcripts into fixed-duration chunks for labeling.
4. **Research Labeling** — Train a logistic regression classifier to detect research-mention chunks in transcripts.
5. **Word Error Rate** — Evaluate transcription accuracy against human-labeled gold standards.

## Repository Structure

```
Kinder_HERC_Sp26/
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
├── transcript_chunking/        # Split transcripts into time-window chunks
│   └── create_chunks.py        # Chunking script (standard library only)
├── research_labeling/          # Research-mention classification
│   ├── research_chunk_pipeline/  # Core ML pipeline modules
│   │   ├── config.py             # Pipeline configuration dataclasses
│   │   ├── data_utils.py         # Data loading and transcript-level splitting
│   │   ├── embedding_utils.py    # Sentence embedding and feature construction
│   │   ├── modeling.py           # Logistic regression training and evaluation
│   │   ├── pipeline.py           # End-to-end pipeline orchestration
│   │   ├── analyze_errors.py     # False positive/negative error analysis
│   │   ├── outputs/              # Pipeline output CSVs and JSON
│   │   └── plots/                # Visualization scripts and PNG outputs
│   ├── Transcript Data/          # Labeled CSV files (one per meeting)
│   └── requirements.txt
├── word_error_rate/            # Transcription quality evaluation
│   ├── wer_norm.py             # Normalized WER calculation
│   └── requirements.txt
├── requirements.txt            # All Python dependencies (combined)
└── README.md                   # This file
```

## Installation

**Prerequisites:**
- Python 3.10+
- `ffmpeg` installed and available on PATH (required by web scraping and transcription)

**Clone and set up a virtual environment:**

```bash
git clone <repository-url>
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
| Research Labeling | `sentence-transformers`, `scikit-learn`, `pandas`, `numpy`, `matplotlib` |
| Word Error Rate | `jiwer`, `contractions`, `num2words` |

## End-to-End Workflow

The four programs are run sequentially. Each step's output becomes the next step's input.

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

### Step 3: Chunk transcripts for labeling

Split each transcript into fixed-duration time windows (default 2 minutes):

```bash
python transcript_chunking/create_chunks.py \
    --input transcripts/Spring_Branch_ISD-2024-09-23-Board_Meetings.txt \
    --output chunks/Chunk_Spring_Branch_ISD-2024-09-23-Board_Meetings.csv
```

**Output:** One CSV per transcript with columns `chunk_id`, `window_start`, `window_end`, `text`.

See [transcript_chunking/README.md](transcript_chunking/README.md) for options.

### Step 4: Label transcripts and train the classifier

The labeled transcript CSVs in `research_labeling/Transcript Data/` are used as training data. To run the classification pipeline:

```bash
cd research_labeling/research_chunk_pipeline
python pipeline.py --transcript-data-dir "../Transcript Data"
```

**Output:** Predictions, metrics, and error analysis in `outputs/`:

```
outputs/
├── all_transcript_predictions.csv       # Predictions for every chunk
├── false_positives.csv                  # Incorrectly flagged chunks
├── false_negatives.csv                  # Missed research mentions
├── test_predictions.csv                 # Test-set predictions only
├── metrics_summary.json                 # Key metrics (recall, precision, F2, etc.)
├── transcript_split_assignments.csv     # Per-transcript split assignments
└── validation_threshold_sweep.csv       # Threshold sweep results
```

To generate evaluation plots:

```bash
cd plots
python visualize_results.py   # confusion matrix, threshold sweep, metrics bar chart
python graphic.py             # dataset distribution chart
```

See [research_labeling/README.md](research_labeling/README.md) for detailed documentation.

### Step 5: Evaluate transcription quality (optional)

Compare an ASR transcript against a human reference:

```bash
python word_error_rate/wer_norm.py reference.txt hypothesis.txt
```

**Output:** Printed to stdout — WER score, substitution/deletion/insertion counts, and top-10 errors per category.

See [word_error_rate/README.md](word_error_rate/README.md) for normalization details.

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

# 2. Run the pipeline (uses labeled CSVs already in the repo)
cd research_labeling/research_chunk_pipeline
python pipeline.py --transcript-data-dir "../Transcript Data"

# 3. Generate evaluation plots
cd plots
python visualize_results.py
python graphic.py
```

The pipeline uses fixed random seeds (`random_seed=42`) for deterministic train/val/test splitting and model training. Running the above commands will reproduce the metrics in `metrics_summary.json` and the plots in `plots/`.
