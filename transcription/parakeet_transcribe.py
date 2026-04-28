"""Parakeet long-form transcription with preprocessing, overlap chunking, and suffix-prefix merge.

Converts audio files to text using NVIDIA's Parakeet ASR model. Long audio is
split into overlapping chunks, transcribed in batches, then merged with
suffix-prefix deduplication to remove repeated words at chunk boundaries.

Usage:
    python parakeet_transcribe.py --input meeting.wav --output_dir transcripts/
    python parakeet_transcribe.py --input "School Board Meetings/" --output_dir transcripts/
"""

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import nemo.collections.asr as nemo_asr
import textwrap



###############################################################################
# Text formatting helpers
###############################################################################

def fmt_ts(seconds: float) -> str:
    """Format a duration in seconds as a human-readable timestamp.

    Inputs
    ------
    seconds : float
        Duration in seconds.

    Outputs
    -------
    str
        Formatted timestamp string (MM:SS or HH:MM:SS for durations >= 1 hour).
    """
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def wrap_paragraphs(text: str, width: int = 100) -> str:
    """Wrap text to a max line width while preserving paragraph breaks.

    Inputs
    ------
    text : str
        Input text with newline-separated paragraphs.
    width : int
        Maximum characters per line.

    Outputs
    -------
    str
        Text with each paragraph wrapped to the specified width.
    """
    paras = [p.strip() for p in text.split("\n")]
    wrapped = []
    for p in paras:
        if not p:
            wrapped.append("")
        else:
            wrapped.append(textwrap.fill(p, width=width))
    return "\n".join(wrapped)


###############################################################################
# Overlap deduplication — suffix/prefix matching
###############################################################################

def _normalize_word(w: str) -> str:
    """Lowercase and strip punctuation from a word for overlap comparison.

    Inputs
    ------
    w : str
        A single word token.

    Outputs
    -------
    str
        Lowercased word with leading/trailing punctuation removed.
    """
    return w.lower().strip(".,!?;:\"'()-")


def find_suffix_prefix_overlap(
    tail: List[str],
    head: List[str],
    min_len: int = 2,
    max_check: int = 40,
) -> int:
    """Find the longest suffix of `tail` that matches a prefix of `head`.

    Uses case-insensitive, punctuation-tolerant comparison.

    Inputs
    ------
    tail : List[str]
        Word list from the previous chunk.
    head : List[str]
        Word list from the next chunk.
    min_len : int
        Minimum overlap length to consider.
    max_check : int
        Maximum number of words to compare at the boundary.

    Outputs
    -------
    int
        Number of words to trim from the start of `head` (the overlap length).
        Returns 0 when no overlap of at least `min_len` is found.
    """
    tail_norm = [_normalize_word(w) for w in tail]
    head_norm = [_normalize_word(w) for w in head]

    max_overlap = min(len(tail), len(head), max_check)
    for length in range(max_overlap, min_len - 1, -1):
        if tail_norm[-length:] == head_norm[:length]:
            return length
    return 0


def merge_all_chunks_global(
    chunk_texts: List[str],
    min_overlap: int = 2,
    max_check: int = 40,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Sequentially merge all chunk transcripts, removing duplicated overlap words.

    Inputs
    ------
    chunk_texts : List[str]
        Ordered list of transcript strings, one per chunk.
    min_overlap : int
        Minimum word overlap to detect.
    max_check : int
        Maximum boundary window size.

    Outputs
    -------
    merged_words : List[str]
        Single deduplicated word list for the whole audio.
    chunk_word_ranges : List[Tuple[int, int]]
        Per-chunk (start_idx, end_idx) into merged_words.
    """
    if not chunk_texts:
        return [], []

    merged_words: List[str] = []
    chunk_word_ranges: List[Tuple[int, int]] = []

    for i, text in enumerate(chunk_texts):
        new_words = text.split()
        if not new_words:
            chunk_word_ranges.append((len(merged_words), len(merged_words)))
            continue

        if i == 0:
            start_idx = 0
            merged_words.extend(new_words)
        else:
            tail = merged_words[-max_check:] if len(merged_words) > max_check else merged_words[:]
            head = new_words[:max_check]
            trim = find_suffix_prefix_overlap(tail, head, min_len=min_overlap, max_check=max_check)
            start_idx = len(merged_words)
            merged_words.extend(new_words[trim:])

        chunk_word_ranges.append((start_idx, len(merged_words)))

    return merged_words, chunk_word_ranges


def merge_transcripts(
    transcripts: List[str],
    min_overlap: int = 2,
    max_check: int = 40,
) -> str:
    """Merge a list of transcript strings using suffix-prefix overlap detection.

    Used by the --no_sections code path to produce plain (unsectioned) output.

    Inputs
    ------
    transcripts : List[str]
        Ordered list of chunk transcript strings.
    min_overlap : int
        Minimum word overlap to detect.
    max_check : int
        Maximum boundary window size.

    Outputs
    -------
    str
        Single merged transcript string.
    """
    merged_words, _ = merge_all_chunks_global(
        transcripts, min_overlap=min_overlap, max_check=max_check
    )
    return " ".join(merged_words)


def build_sectioned_transcript(
    chunks: List["Chunk"],
    chunk_texts: List[str],
    section_s: int = 30,
    wrap_width: int = 100,
) -> str:
    """Build a readable transcript with [start-end] headers every `section_s` seconds.

    Steps:
      1. Merge ALL chunks globally (sequential suffix-prefix dedup).
      2. Assign each word to a time section using its originating chunk's start time.
      3. Emit sections in order.

    Inputs
    ------
    chunks : List[Chunk]
        List of Chunk objects with timing info.
    chunk_texts : List[str]
        Corresponding transcript strings, one per chunk.
    section_s : int
        Section duration in seconds for timestamp headers.
    wrap_width : int
        Maximum characters per line in output.

    Outputs
    -------
    str
        Formatted transcript string with section headers.
    """
    if not chunks or not chunk_texts:
        return ""

    merged_words, chunk_word_ranges = merge_all_chunks_global(chunk_texts)
    if not merged_words:
        return ""

    # Assign each word to a section via its originating chunk's time
    word_section: List[int] = [0] * len(merged_words)
    for ch, (start_idx, end_idx) in zip(chunks, chunk_word_ranges):
        sec_idx = int(ch.start_s // section_s)
        for j in range(start_idx, end_idx):
            word_section[j] = sec_idx

    # Bucket words by section and emit
    buckets: dict = {}
    for word, sec in zip(merged_words, word_section):
        buckets.setdefault(sec, []).append(word)

    lines: List[str] = []
    for sec_idx in sorted(buckets.keys()):
        start = sec_idx * section_s
        end = start + section_s
        header = f"[{fmt_ts(start)}–{fmt_ts(end)}]"
        text = wrap_paragraphs(" ".join(buckets[sec_idx]), width=wrap_width)
        lines.append(header)
        lines.append(text)
        lines.append("")

    return "\n".join(lines).rstrip()


###############################################################################
# Audio preprocessing via ffmpeg
###############################################################################

def require_ffmpeg() -> str:
    """Locate ffmpeg on PATH or raise RuntimeError.

    Inputs
    ------
    None.

    Outputs
    -------
    str
        Absolute path to the ffmpeg executable.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg and make sure it's available as `ffmpeg`."
        )
    return ffmpeg


def ffprobe_duration_seconds(path: Path) -> float:
    """Get audio duration in seconds using ffprobe.

    Inputs
    ------
    path : Path
        Path to an audio file.

    Outputs
    -------
    float
        Duration in seconds, or 0.0 if ffprobe is unavailable or fails.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0

    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    try:
        return float(out)
    except Exception:
        return 0.0


def preprocess_audio_to_wav(
    input_path: Path,
    output_wav: Path,
    sample_rate: int = 16000,
    mono: bool = True,
    pcm: str = "pcm_s16le",
    highpass_hz: Optional[int] = 80,
    lowpass_hz: Optional[int] = 7500,
) -> None:
    """Convert arbitrary audio to mono 16kHz PCM WAV with optional bandpass filtering.

    Inputs
    ------
    input_path : Path
        Path to source audio file (any format ffmpeg supports).
    output_wav : Path
        Path to write the standardized WAV.
    sample_rate : int
        Target sample rate in Hz.
    mono : bool
        If True, downmix to single channel.
    pcm : str
        PCM codec for output.
    highpass_hz : Optional[int]
        High-pass filter cutoff in Hz (None to disable).
    lowpass_hz : Optional[int]
        Low-pass filter cutoff in Hz (None to disable).

    Outputs
    -------
    None
        Writes the converted WAV file to output_wav.
    """
    ffmpeg = require_ffmpeg()

    filters = []
    if highpass_hz is not None:
        filters.append(f"highpass=f={int(highpass_hz)}")
    if lowpass_hz is not None:
        filters.append(f"lowpass=f={int(lowpass_hz)}")

    filter_chain = ",".join(filters) if filters else None

    cmd = [ffmpeg, "-y", "-i", str(input_path)]
    if filter_chain:
        cmd += ["-af", filter_chain]

    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-ar", str(sample_rate)]
    cmd += ["-c:a", pcm]
    cmd += ["-f", "wav", str(output_wav)]

    subprocess.check_call(cmd)


def extract_wav_segment(
    input_wav: Path,
    start_s: float,
    duration_s: float,
    output_wav: Path,
) -> None:
    """Extract a WAV segment (mono 16kHz PCM) using ffmpeg.

    Inputs
    ------
    input_wav : Path
        Source WAV file.
    start_s : float
        Start time in seconds.
    duration_s : float
        Duration to extract in seconds.
    output_wav : Path
        Path to write the extracted segment.

    Outputs
    -------
    None
        Writes the extracted WAV segment to output_wav.
    """
    ffmpeg = require_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration_s:.3f}",
        "-i", str(input_wav),
        "-c:a", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav),
    ]
    subprocess.check_call(cmd)


###############################################################################
# Chunk planning: hour blocks, then 30-40s chunks with 1s overlap
###############################################################################

@dataclass
class Chunk:
    """A single audio chunk to be transcribed."""
    start_s: float
    dur_s: float
    path: Path


def plan_chunks(
    wav_path: Path,
    chunk_s: float = 35.0,
    overlap_s: float = 1.0,
    hour_block_s: float = 3600.0,
    tmp_dir: Path = Path("."),
) -> List[List[Chunk]]:
    """Plan overlapping chunks for a WAV file, splitting into hour blocks if needed.

    Inputs
    ------
    wav_path : Path
        Path to the preprocessed WAV file.
    chunk_s : float
        Target chunk duration in seconds.
    overlap_s : float
        Overlap between consecutive chunks in seconds.
    hour_block_s : float
        Maximum block duration (splits very long audio).
    tmp_dir : Path
        Directory for temporary chunk WAV files.

    Outputs
    -------
    List[List[Chunk]]
        List of blocks, where each block is a list of Chunk objects.
    """
    total = ffprobe_duration_seconds(wav_path)
    if total <= 0:
        total = hour_block_s

    num_blocks = max(1, math.ceil(total / hour_block_s))
    blocks: List[List[Chunk]] = []

    for b in range(num_blocks):
        block_start = b * hour_block_s
        block_end = min((b + 1) * hour_block_s, total)
        block_len = max(0.0, block_end - block_start)

        chunks: List[Chunk] = []
        t = 0.0
        idx = 0
        step = max(0.01, chunk_s - overlap_s)

        while t < block_len:
            dur = min(chunk_s, block_len - t)
            out = tmp_dir / f"chunk_b{b:03d}_{idx:05d}.wav"
            chunks.append(Chunk(start_s=block_start + t, dur_s=dur, path=out))
            idx += 1
            t += step

        blocks.append(chunks)

    return blocks

##############################################################################
# Load asr model
############################################################################
def load_asr_model(model: str = "nvidia/parakeet-tdt-0.6b-v3"):
    print(f"Loading ASR Model: {model}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model)
    print(f"Model loaded: {type(asr_model).__name__}")
    return asr_model





###############################################################################
# Transcription
###############################################################################

def transcribe_chunks(
    model,
    chunks: List[Chunk],
    batch_size: int = 8,
) -> List[str]:
    """Batch-transcribe a list of audio chunks using the ASR model.

    Inputs
    ------
    model : nemo_asr.models.ASRModel
        Loaded NeMo ASR model.
    chunks : List[Chunk]
        List of Chunk objects with .path pointing to WAV files.
    batch_size : int
        Number of chunks to transcribe per model call.

    Outputs
    -------
    List[str]
        List of transcript strings, one per chunk.
    """
    texts: List[str] = []
    paths = [str(c.path) for c in chunks]

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        out = model.transcribe(batch_paths)
        for hyp in out:
            texts.append(getattr(hyp, "text", "") if hyp is not None else "")

    return texts


def is_audio_file(p: Path) -> bool:
    """Check if a path has a recognized audio/video file extension.

    Inputs
    ------
    p : Path
        File path to check.

    Outputs
    -------
    bool
        True if the file extension is a supported audio/video format.
    """
    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".webm"}
    return p.suffix.lower() in exts


###############################################################################
# Main
###############################################################################


#############################################################################
# Run Transcription -- Separates logic of transcribing the input file
############################################################################
def run_transcription(input_path: Path, output_path: Path, model:str, no_highpass: bool = False,
    no_lowpass: bool = False,
    highpass_hz: int = 80,
    lowpass_hz: int = 7500,
    chunk_s: float = 35.0,
    overlap_s: float = 1.0,
    hour_block_s: float = 3600.0,
    batch_size: int = 8,
    wrap_width: int = 100,
    section_s: int = 30,
    no_sections: bool = False):
    in_path = Path(input_path).expanduser().resolve()
    out_dir = Path(output_path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect input audio files
    if in_path.is_dir():
        audio_files = sorted([p for p in in_path.rglob("*") if p.is_file() and is_audio_file(p)])
    elif in_path.is_file():
        audio_files = [in_path]
    else:
        raise FileNotFoundError(f"Input not found: {in_path}")

    if not audio_files:
        print("No audio files found.")
        sys.exit(0)

    print(f"Loading model: {model}")
    if model is None:
        print(f"Loading model: {model}")
        asr_model = load_asr_model(model)

    with tempfile.TemporaryDirectory(prefix="parakeet_tmp_") as tmp:
        tmp_dir = Path(tmp)

        for src in audio_files:
            print(f"\n=== Processing: {src.name} ===")

            # Preprocess: convert to mono 16kHz WAV with bandpass filtering
            standardized = tmp_dir / f"{src.stem}__std.wav"
            preprocess_audio_to_wav(
                input_path=src,
                output_wav=standardized,
                sample_rate=16000,
                mono=True,
                pcm="pcm_s16le",
                highpass_hz=None if no_highpass else highpass_hz,
                lowpass_hz=None if no_lowpass else lowpass_hz,
            )

            # Plan and extract overlapping chunks
            blocks = plan_chunks(
                wav_path=standardized,
                chunk_s=chunk_s,
                overlap_s=overlap_s,
                hour_block_s=hour_block_s,
                tmp_dir=tmp_dir,
            )

            all_block_texts: List[str] = []

            for b_idx, chunks in enumerate(blocks):
                for ch in chunks:
                    extract_wav_segment(standardized, ch.start_s, ch.dur_s, ch.path)

                chunk_texts = transcribe_chunks(
                    asr_model, chunks, batch_size=batch_size
                )

                # Merge chunks with overlap deduplication
                if no_sections:
                    merged_text = merge_transcripts(chunk_texts)
                    merged_text = wrap_paragraphs(merged_text, width=wrap_width)
                else:
                    merged_text = build_sectioned_transcript(
                        chunks=chunks,
                        chunk_texts=chunk_texts,
                        section_s=section_s,
                        wrap_width=wrap_width,
                    )

                all_block_texts.append(merged_text)

            # Write final transcript
            final_text = "\n\n".join(all_block_texts).strip()
            txt_path = out_dir / f"{src.stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(final_text + "\n")
            return(f"Wrote transcript: {txt_path}")
            



def main():
    """Parse arguments, load the ASR model, and transcribe all input audio files.

    Inputs
    ------
    None. Reads from command-line arguments (sys.argv).

    Outputs
    -------
    None
        Writes .txt transcript files to the specified output directory.
    """
    ap = argparse.ArgumentParser(
        description="Parakeet long-form transcription with preprocessing + overlap chunking + suffix-prefix merge."
    )
    ap.add_argument("--input", required=True, help="Path to an audio file OR a directory of audio files.")
    ap.add_argument("--output_dir", required=True, help="Directory to write transcripts.")
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3", help="HF model name for NeMo ASRModel.")

    # Preprocessing toggles
    ap.add_argument("--no_highpass", action="store_true", help="Disable high-pass filter.")
    ap.add_argument("--no_lowpass", action="store_true", help="Disable low-pass filter.")
    ap.add_argument("--highpass_hz", type=int, default=80, help="High-pass cutoff Hz (default 80).")
    ap.add_argument("--lowpass_hz", type=int, default=7500, help="Low-pass cutoff Hz (default 7500).")

    # Chunking params
    ap.add_argument("--chunk_s", type=float, default=35.0, help="Chunk length seconds (default 35).")
    ap.add_argument("--overlap_s", type=float, default=1.0, help="Overlap seconds (default 1).")
    ap.add_argument("--hour_block_s", type=float, default=3600.0, help="Hour block seconds (default 3600).")
    ap.add_argument("--batch_size", type=int, default=8, help="Transcribe batch size (paths per call).")

    # Output formatting
    ap.add_argument("--wrap_width", type=int, default=100, help="Max characters per line in output transcript.")
    ap.add_argument("--section_s", type=int, default=30, help="Section size in seconds for timestamp headers.")
    ap.add_argument("--no_sections", action="store_true", help="Disable [MM:SS–MM:SS] section headers.")

    args = ap.parse_args()
    transcript_msg = run_transcription(args.input, args.output_dir, args.model, args.no_highpass, args.no_lowpass, args.highpass_hz, args.lowpass_hz,
                      args.chunk_s, args.overlap_s, args.hour_block_s, args.batch_size, args.wrap_width, args.section_s, args.no_sections)

    print(transcript_msg)

if __name__ == "__main__":
    main()
