"""Split a timestamped transcript into fixed-length time-window chunks.

Reads a transcript with [MM:SS-MM:SS] or [HH:MM:SS-HH:MM:SS] section headers
(as produced by parakeet_transcribe.py) and groups consecutive sections into
chunks of a configurable duration. Outputs a CSV suitable for labeling or
classification.

Usage:
    python create_chunks.py --input transcript.txt --output chunks.csv
    python create_chunks.py --input transcript.txt --output chunks.csv --chunk-minutes 3
"""

import argparse
import csv
import re
from pathlib import Path

# Matches timestamp headers like [00:00-00:30] or [01:00:00-01:00:30]
TS_PATTERN = re.compile(
    r"\[\s*"
    r"(?P<start>(?:\d{1,2}:)?\d{2}:\d{2})\s*[–-]\s*(?P<end>(?:\d{1,2}:)?\d{2}:\d{2})"
    r"\s*\]\s*"
)


def time_to_seconds(t: str) -> int:
    """Convert a timestamp string to total seconds.

    Inputs
    ------
    t : str
        Time string in "MM:SS" or "HH:MM:SS" format.

    Outputs
    -------
    int
        Total seconds represented by the timestamp.
    """
    parts = t.split(":")
    if len(parts) == 2:
        mm, ss = parts
        return int(mm) * 60 + int(ss)
    if len(parts) == 3:
        hh, mm, ss = parts
        return int(hh) * 3600 + int(mm) * 60 + int(ss)
    raise ValueError(f"Unrecognized time format: {t}")


def seconds_to_hhmmss(seconds: int) -> str:
    """Convert total seconds to HH:MM:SS format.

    Inputs
    ------
    seconds : int
        Total seconds.

    Outputs
    -------
    str
        Formatted time string, e.g. "01:05:30".
    """
    h = seconds // 3600
    rem = seconds % 3600
    m = rem // 60
    s = rem % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_segments(text: str) -> list[dict]:
    """Extract timestamped text segments from a transcript.

    Splits the transcript at each [start-end] timestamp header and returns
    a list of segments with their start/end times and raw text content.

    Inputs
    ------
    text : str
        Full transcript text with [MM:SS-MM:SS] section headers.

    Outputs
    -------
    list[dict]
        Each dict has keys "start" (int seconds), "end" (int seconds),
        and "raw" (str block text including the timestamp header).
    """
    matches = list(TS_PATTERN.finditer(text))
    if not matches:
        raise ValueError(
            "No timestamp blocks found. "
            "Expected format like [00:00-00:30] or [01:00:00-01:00:30]."
        )

    segments = []
    for i, m in enumerate(matches):
        start_sec = time_to_seconds(m.group("start"))
        end_sec = time_to_seconds(m.group("end"))

        block_start = m.start()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw_block = text[block_start:block_end].strip()

        segments.append({"start": start_sec, "end": end_sec, "raw": raw_block})

    return segments


def chunk_by_time(segments: list[dict], window_seconds: int) -> list[dict]:
    """Group segments into fixed-duration time-window chunks.

    Consecutive segments are merged into chunks of ``window_seconds`` duration.
    Each chunk contains all segments whose start time falls within the window.

    Inputs
    ------
    segments : list[dict]
        Parsed segments from ``parse_segments()``.
    window_seconds : int
        Duration of each output chunk in seconds.

    Outputs
    -------
    list[dict]
        Each dict has keys "chunk_id" (int), "window_start" (str HH:MM:SS),
        "window_end" (str HH:MM:SS), and "text" (str merged block text).
    """
    chunks = []

    current_window_start = (segments[0]["start"] // window_seconds) * window_seconds
    current_window_end = current_window_start + window_seconds
    current_blocks = []

    def flush():
        nonlocal current_blocks
        if not current_blocks:
            return
        chunks.append({
            "chunk_id": len(chunks),
            "window_start": seconds_to_hhmmss(current_window_start),
            "window_end": seconds_to_hhmmss(current_window_end),
            "text": "\n\n".join(current_blocks).strip()
        })
        current_blocks = []

    for seg in segments:
        while seg["start"] >= current_window_end:
            flush()
            current_window_start = current_window_end
            current_window_end = current_window_start + window_seconds

        current_blocks.append(seg["raw"])

    flush()
    return chunks


def chunk_transcript(input_path: Path, output_path: Path, chunk_minutes: int):
    text = input_path.read_text(encoding="utf-8")
    segments = parse_segments(text)
    window_seconds = chunk_minutes * 60
    chunks = chunk_by_time(segments, window_seconds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "window_start", "window_end", "text"])
        for ch in chunks:
            writer.writerow([ch["chunk_id"], ch["window_start"], ch["window_end"], ch["text"]])

    return f"Saved {len(chunks)} chunks to {output_path}"



def main():
    """Parse CLI arguments, chunk the transcript, and write the output CSV.

    Inputs
    ------
    --input : str
        Path to a timestamped transcript .txt file.
    --output : str
        Path for the output CSV file.
    --chunk-minutes : int
        Duration of each chunk in minutes (default: 2).

    Outputs
    -------
    None
        Writes a CSV with columns: chunk_id, window_start, window_end, text.
    """
    parser = argparse.ArgumentParser(
        description="Split a timestamped transcript into fixed-length time-window chunks."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to a timestamped transcript .txt file.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path for the output CSV file.",
    )
    parser.add_argument(
        "--chunk-minutes", type=int, default=2,
        help="Duration of each chunk in minutes (default: 2).",
    )
    args = parser.parse_args()
    final_message = chunk_transcript(args.input, args.output, args.chunk_minutes)
    print(final_message)


if __name__ == "__main__":
    main()
