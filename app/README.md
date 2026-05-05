# App — Desktop GUI and Google Colab Runner

A graphical interface and headless Colab entry point that wrap the full five-step pipeline into a single call. Given a Swagit URL or local audio files and a district name, the app downloads (or accepts) audio, transcribes it, chunks it, runs the trained classifier, and writes a highlighted `.docx` report.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point — starts the CustomTkinter GUI |
| `gui.py` | GUI layout and pipeline orchestration |
| `highlighter.py` | Builds a highlighted `.docx` from model predictions |
| `trained_model.py` | Inference wrapper — loads `inference_artifacts.pkl` and scores chunks |
| `colab_runner.py` | Headless entry point for Google Colab |

## Desktop Usage

**Prerequisites:** Python 3.11.6, `ffmpeg` on PATH (or bundled via `imageio-ffmpeg`).

```bash
cd app
pip install -r requirements.txt
python main.py
```

The GUI opens with two input tabs:

- **Website URL** — paste a Swagit meeting URL; the app downloads audio automatically.
- **Upload File(s)** — select one or more local audio files (`.wav`, `.mp3`, `.mp4`, `.m4a`, `.flac`, `.ogg`, `.aac`) or a folder containing them.

Fill in the **District Name** and **Output Folder**, then click **Run Pipeline**. Progress is logged in the text box. On completion, a highlighted `.docx` report is written to the output folder.

## Google Colab Usage

Clone the repository into your Colab environment, then:

```python
import sys
sys.path.insert(0, "/content/Kinder_HERC_Sp26")

from app.colab_runner import run

run(
    url="https://springbranchisdtx.new.swagit.com/videos/364571/download",
    district="Spring Branch ISD",
    out_dir="/content/output",        # where the .docx report is saved
    cutoff_str="2024-09-01",          # skip meetings before this date
)
```

### `run()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | required | Swagit meeting or playlist URL |
| `district` | `str` | required | Human-readable district name |
| `out_dir` | `str` | `"/content/output"` | Output directory for the `.docx` report |
| `cutoff_str` | `str` | `"2024-09-01"` | ISO date; meetings before this date are skipped |
| `max_candidates` | `int` | `60` | Maximum Swagit candidate links to check |
| `asr_model_name` | `str` | `"nvidia/parakeet-tdt-0.6b-v3"` | NeMo ASR model identifier |

## Pipeline Steps

The app orchestrates four steps internally:

1. **Download** — Scrapes the Swagit page and downloads WAV audio via `yt-dlp`.
2. **Transcribe** — Runs NVIDIA Parakeet ASR to produce a timestamped `.txt` transcript.
3. **Chunk** — Splits the transcript into 2-minute windows and writes a `.csv`.
4. **Predict + Report** — Embeds chunks with sentence-transformers, scores with the trained logistic regression model, and writes a highlighted `.docx`.

## Output

A `.docx` report where chunks predicted as research/data mentions are **bolded and highlighted in yellow**. Each chunk entry shows:

- `Chunk ID`
- `Time window` (start → end)
- `Text`
- `Confidence` score

## Inference Artifacts

The classifier requires a pre-trained `inference_artifacts.pkl` file at:

```
research_labeling/outputs/inference_artifacts.pkl
```

This file is produced by the training pipeline (`research_labeling/research_chunk_pipeline/pipeline.py`). It bundles the fitted model, classification threshold, embedding model name, and feature mode.

## Dependencies

All dependencies are listed in [requirements.txt](requirements.txt). Key packages:

| Package | Purpose |
|---------|---------|
| `customtkinter` | Desktop GUI |
| `imageio-ffmpeg` | Bundled ffmpeg binary (used when system ffmpeg is unavailable) |
| `python-docx` | Build highlighted `.docx` reports |
| `sentence-transformers` | Chunk embedding for inference |
| `scikit-learn` | Logistic regression classifier |
| `xgboost` | XGBoost inference (alternative model artifact) |
| `joblib` | Load `inference_artifacts.pkl` |
| `yt-dlp` | Video/audio download |
| `nemo_toolkit[asr]` | NVIDIA Parakeet transcription |
