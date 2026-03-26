"""Analyze model errors from all_transcript_predictions.csv.

Produces two CSVs:
    false_positives.csv  — model predicted 1, ground truth is 0
    false_negatives.csv  — model predicted 0, ground truth is 1

Both are sorted by predicted_probability descending.

Usage:
    python analyze_errors.py --predictions "path/to/all_transcript_predictions.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def analyze_errors(predictions_path: Path, output_dir: Path) -> None:
    """Split predictions into false positives and false negatives.

    Reads model predictions, separates incorrect classifications, writes
    each error type to its own CSV, and prints per-transcript breakdowns.

    Inputs
    ------
    predictions_path : Path
        Path to all_transcript_predictions.csv containing columns
        ``predicted_label``, ``binary_hit``, and ``predicted_probability``.
    output_dir : Path
        Directory to write ``false_positives.csv`` and ``false_negatives.csv``.

    Outputs
    -------
    None
        Results are written to disk and printed to stdout.
    """
    df = pd.read_csv(predictions_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Columns to include in output — no model_text, no split bookkeeping
    output_cols = [
        "transcript_id",
        "chunk_id",
        "source_file",
        "window_start",
        "window_end",
        "text",
        "binary_hit",
        "predicted_probability",
        "predicted_label",
    ]

    # False positives: model said 1, ground truth is 0
    # Sorted by confidence descending — highest confidence wrong answers first
    false_positives = (
        df[(df["predicted_label"] == 1) & (df["binary_hit"] == 0)]
        .sort_values("predicted_probability", ascending=False)
        .reset_index(drop=True)
        [output_cols]
    )

    # False negatives: model said 0, ground truth is 1
    # Sorted by confidence descending — closest misses first
    false_negatives = (
        df[(df["predicted_label"] == 0) & (df["binary_hit"] == 1)]
        .sort_values("predicted_probability", ascending=False)
        .reset_index(drop=True)
        [output_cols]
    )

    # Write outputs
    fp_path = output_dir / "false_positives.csv"
    fn_path = output_dir / "false_negatives.csv"

    false_positives.to_csv(fp_path, index=False)
    false_negatives.to_csv(fn_path, index=False)

    # Print summary
    print(f"Total chunks:      {len(df)}")
    print(f"False positives:   {len(false_positives)}  (predicted 1, actual 0)  → {fp_path}")
    print(f"False negatives:   {len(false_negatives)}  (predicted 0, actual 1)  → {fn_path}")

    # Per-transcript breakdown — useful to spot if errors cluster in one meeting
    print("\nFalse positives per transcript:")
    print(false_positives.groupby("transcript_id").size().sort_values(ascending=False).to_string())

    print("\nFalse negatives per transcript:")
    print(false_negatives.groupby("transcript_id").size().sort_values(ascending=False).to_string())


def main() -> None:
    """Run error analysis from the command line.

    Inputs
    ------
    sys.argv[1:] : parsed via argparse
        --predictions : Path to all_transcript_predictions.csv (required).
        --output-dir  : Directory for output CSVs (defaults to predictions dir).

    Outputs
    -------
    None
        Writes false_positives.csv and false_negatives.csv; prints summary to stdout.
    """
    parser = argparse.ArgumentParser(
        description="Analyze false positives and false negatives from model predictions."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to all_transcript_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write error CSVs. Defaults to same folder as predictions file.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.predictions.parent
    analyze_errors(predictions_path=args.predictions, output_dir=output_dir)


if __name__ == "__main__":
    main()