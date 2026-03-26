# Transcription - Parakeet Long-Form ASR

Transcribes audio files to text using NVIDIA's Parakeet ASR model (NeMo), with chunked processing for long-form audio and overlap-based deduplication.

## Setup

```bash
cd transcription
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

**Also required:** `ffmpeg` and `ffprobe` on PATH.

**GPU:** The model automatically uses CUDA if available; falls back to CPU otherwise.

## Usage

Transcribe a single file:

```bash
python parakeet_transcribe.py --input meeting.wav --output_dir transcripts/
```

Transcribe a directory of audio files:

```bash
python parakeet_transcribe.py --input "School Board Meetings/" --output_dir transcripts/
```

### Key options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | HuggingFace model name | `nvidia/parakeet-tdt-0.6b-v3` |
| `--chunk_s` | Chunk length in seconds | `35.0` |
| `--overlap_s` | Overlap between chunks in seconds | `1.0` |
| `--section_s` | Timestamp section header interval | `30` |
| `--no_sections` | Disable `[MM:SS–MM:SS]` section headers | Off |
| `--no_highpass` | Disable 80 Hz high-pass filter | Off |
| `--no_lowpass` | Disable 7500 Hz low-pass filter | Off |
| `--batch_size` | Chunks per transcription batch | `8` |

Run `python parakeet_transcribe.py --help` for all options.

## Output

Each input audio file produces a `.txt` transcript in the output directory:

```
transcripts/
├── meeting_2024-09-23.txt
└── meeting_2024-12-13.txt
```

**Transcript format** (with default section headers enabled):

```
[00:00–00:30]
Good evening everyone. Welcome to the regular meeting of the board of trustees.
I'd like to call this meeting to order at 6:02 p.m.

[00:30–01:00]
First item on the agenda is the approval of the minutes from our last meeting.
Do I have a motion?
```

## How It Works

1. **Preprocessing** — Audio is converted to mono 16kHz PCM WAV via ffmpeg, with bandpass filtering (80 Hz–7500 Hz by default) to reduce noise outside the speech frequency range.
2. **Chunking** — Long audio is split into overlapping chunks (~35s each with 1s overlap). Files over 1 hour are split into hour blocks first.
3. **Transcription** — Chunks are batch-transcribed using the Parakeet model.
4. **Merge** — Overlapping text at chunk boundaries is deduplicated using suffix-prefix word matching. This avoids the repeated-word artifacts that naive concatenation produces.
5. **Output** — A readable `.txt` transcript is written with optional `[MM:SS–MM:SS]` section headers.

## Supported Audio Formats

WAV, MP3, M4A, FLAC, OGG, OPUS, AAC, WMA, MP4, MKV, WEBM (anything ffmpeg can decode).
