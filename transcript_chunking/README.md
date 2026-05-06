# Transcript Chunking - Split Transcripts into Time-Window Chunks

Splits timestamped transcripts into fixed-duration chunks for labeling and classification. Designed to process the `[MM:SS–MM:SS]` section headers produced by the transcription step.

## Setup

No additional dependencies required — uses only the Python standard library.

## Usage

```bash
python create_chunks.py --input transcript.txt --output chunks.csv
```

With a custom chunk duration:

```bash
python create_chunks.py --input transcript.txt --output chunks.csv --chunk-minutes 3
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--input` | Path to a timestamped transcript `.txt` file | Required |
| `--output` | Path for the output CSV file | Required |
| `--chunk-minutes` | Duration of each chunk in minutes | `2` |

## Input Format

The input transcript must contain `[MM:SS–MM:SS]` or `[HH:MM:SS–HH:MM:SS]` section headers, as produced by the transcription step:

```
[00:00–00:30]
Good evening everyone. Welcome to the regular meeting of the board of trustees.

[00:30–01:00]
First item on the agenda is the approval of the minutes from our last meeting.
```

## Output Format

A CSV with one row per chunk:

| Column | Description |
|--------|-------------|
| `chunk_id` | Sequential chunk index (0-based) |
| `window_start` | Start time of the chunk window (HH:MM:SS) |
| `window_end` | End time of the chunk window (HH:MM:SS) |
| `text` | Merged transcript text for all sections within the window |

## How It Works

1. **Parse** — Timestamp headers are extracted from the transcript using regex.
2. **Group** — Consecutive sections are grouped into fixed-duration windows (default 2 minutes).
3. **Write** — Each window becomes one row in the output CSV, with the section texts merged.

## Batch Processing

`mult_chunk.py` chunks every `.txt` transcript in a directory in a single run. Edit the `input_dir` and `output_dir` constants at the top of the file, then run:

```bash
python mult_chunk.py
```

One CSV is written to `output_dir` for each `.txt` file found in `input_dir`.
