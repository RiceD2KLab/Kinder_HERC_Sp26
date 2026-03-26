"""Data loading, validation, and transcript-level splitting utilities.

CSV schema expected (one file per school-board meeting):
    chunk_id     – unique row identifier within the transcript
    window_start – video timestamp where the chunk begins
    window_end   – video timestamp where the chunk ends
    text         – raw transcribed text for this chunk
    binary_hit   – 1 if the chunk mentions research / data used for a
                   decision, 0 otherwise

Key design principle: transcript-level splitting
    All chunks from a single meeting must land in the same split.  Chunks
    from the same meeting are highly correlated (same speakers, same topics,
    sometimes the same sentence spanning two adjacent chunks), so mixing them
    across train / val / test would leak information and produce misleadingly
    high metrics.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: tuple[str, ...] = (
    "chunk_id",
    "window_start",
    "window_end",
    "text",
    "binary_hit",
)


# ---------------------------------------------------------------------------
# CSV discovery and loading
# ---------------------------------------------------------------------------

def find_transcript_csvs(transcript_data_dir: Path) -> list[Path]:
    """Return all CSV files in *transcript_data_dir*, sorted alphabetically.

    Inputs:
        transcript_data_dir: Directory containing one CSV per transcript.

    Outputs:
        Sorted list of CSV paths.

    Raises:
        FileNotFoundError: If no CSV files are present in the directory.
    """
    csv_paths = sorted(transcript_data_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in: {transcript_data_dir}"
        )
    return csv_paths


def _check_required_columns(df: pd.DataFrame, source_name: str) -> None:
    """Raise ValueError if any required column is absent.

    Inputs:
        df:          Dataframe to check.
        source_name: Label used in the error message (usually the filename).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Transcript '{source_name}' is missing required columns: {missing}"
        )


def _normalize_binary_hit(series: pd.Series, source_name: str) -> pd.Series:
    """Coerce the binary_hit column to a clean integer 0 / 1 series.

    Rules:
      - Values that are already 0 or 1 are kept as-is.
      - NaN / blank values are treated as 0 and a warning is emitted, so
        partially-labelled transcripts can still be loaded.
      - Any other numeric value (e.g. 2, -1) raises ValueError immediately.

    Inputs:
        series:      The raw binary_hit column from one transcript CSV.
        source_name: Label used in warnings / errors (usually the filename).

    Outputs:
        Integer Series containing only 0s and 1s.

    Raises:
        ValueError: If any non-NaN, non-binary value is present.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    # Warn about blanks / non-parseable entries, default them to 0.
    missing_count = int(numeric.isna().sum())
    if missing_count:
        warnings.warn(
            f"Transcript '{source_name}' has {missing_count} missing "
            "binary_hit value(s); defaulting to 0.",
            stacklevel=3,
        )

    # Reject anything that is numeric but not in {0, 1}.
    bad_mask = numeric.notna() & ~numeric.isin([0, 1])
    if bad_mask.any():
        bad_values = sorted(numeric.loc[bad_mask].unique().tolist())
        raise ValueError(
            f"Transcript '{source_name}' has invalid binary_hit values: "
            f"{bad_values}. Only 0 and 1 are allowed."
        )

    return numeric.fillna(0).astype(int)


def load_transcript_csv(
    csv_path: Path,
    transcript_id: str | None = None,
) -> pd.DataFrame:
    """Load one transcript CSV and add bookkeeping columns.

    Inputs:
        csv_path:      Path to the CSV file.
        transcript_id: Override for the transcript identifier.  Defaults to
                       the file stem (e.g. "meeting_01" from "meeting_01.csv").

    Outputs:
        DataFrame with validated columns plus ``transcript_id`` and
        ``source_file`` columns appended.
    """
    df = pd.read_csv(csv_path)
    _check_required_columns(df, source_name=csv_path.name)

    df = df.copy()
    df["transcript_id"] = transcript_id or csv_path.stem
    df["source_file"]   = csv_path.name
    df["binary_hit"]    = _normalize_binary_hit(df["binary_hit"], source_name=csv_path.name)
    df["text"]          = df["text"].fillna("").astype(str)
    return df


def load_all_transcripts(transcript_data_dir: Path) -> pd.DataFrame:
    """Load every transcript CSV and concatenate into one dataframe.

    Inputs:
        transcript_data_dir: Directory containing transcript CSV files.

    Outputs:
        Combined dataframe with rows from all transcripts, indexed from 0.
    """
    frames = [
        load_transcript_csv(csv_path)
        for csv_path in find_transcript_csvs(transcript_data_dir)
    ]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Transcript-level splitting
# ---------------------------------------------------------------------------

def summarize_transcripts(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-transcript statistics used to balance split assignments.

    Inputs:
        df: Combined chunk dataframe with a ``transcript_id`` column.

    Outputs:
        One-row-per-transcript dataframe with columns:
        ``transcript_id``, ``n_chunks``, ``n_positive``, ``positive_rate``.
    """
    summary = (
        df.groupby("transcript_id", as_index=False)
        .agg(
            n_chunks=("chunk_id", "size"),
            n_positive=("binary_hit", "sum"),
        )
        .sort_values("transcript_id")
        .reset_index(drop=True)
    )
    summary["positive_rate"] = summary["n_positive"] / summary["n_chunks"]
    return summary


def _target_split_counts(
    n_transcripts: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, int]:
    """Convert split fractions into integer transcript counts.

    Uses largest-remainder rounding so the total always equals n_transcripts.

    Inputs:
        n_transcripts:  Total number of unique transcripts.
        train_fraction: Desired training fraction.
        val_fraction:   Desired validation fraction.
        test_fraction:  Desired testing fraction.

    Outputs:
        Dict mapping split name → target transcript count.

    Raises:
        ValueError: If any split would receive zero transcripts.
    """
    raw    = {
        "train": n_transcripts * train_fraction,
        "val":   n_transcripts * val_fraction,
        "test":  n_transcripts * test_fraction,
    }
    counts = {name: int(v) for name, v in raw.items()}

    # Distribute any transcripts lost to floor() by largest fractional part.
    remainder = n_transcripts - sum(counts.values())
    for name in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[:remainder]:
        counts[name] += 1

    if any(v == 0 for v in counts.values()):
        raise ValueError(
            "Too few transcripts for non-empty train / val / test splits. "
            f"Computed counts: {counts}"
        )
    return counts


def assign_transcript_splits(
    transcript_summary: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Assign each transcript to train / val / test with a proportional deficit heuristic.

    The previous greedy version scored by absolute or relative positive gap,
    which caused val and test (small quotas) to win high-positive transcripts
    because 20 positives fills a quota of 28 more snugly than a quota of 84.

    This version scores by *proportional deficit* — which split is currently
    furthest behind its target share of all positives assigned so far. Train
    targets 60% of all positives; if it currently holds only 20%, it has a
    40% deficit and wins the next assignment. This treats all splits fairly
    regardless of their absolute target size.

    Assignment order:
        1. Transcripts with the most positives are assigned first.
        2. Each transcript goes to whichever eligible split has the largest
           proportional deficit in positives relative to its target fraction.
        3. Ties broken randomly (seeded for reproducibility).

    Inputs:
        transcript_summary: Output of :func:`summarize_transcripts`.
        train_fraction:     Fraction of transcripts for training.
        val_fraction:       Fraction of transcripts for validation.
        test_fraction:      Fraction of transcripts for testing.
        random_seed:        Seed for reproducible tie-breaking.

    Outputs:
        Copy of ``transcript_summary`` with an added ``split`` column.
    """
    summary = transcript_summary.copy()
    rng     = random.Random(random_seed)

    target_counts = _target_split_counts(
        n_transcripts=len(summary),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    # Target fraction of total positives each split should receive.
    target_fraction = {
        "train": train_fraction,
        "val":   val_fraction,
        "test":  test_fraction,
    }

    # Sort transcripts highest-positive first, shuffle within ties.
    summary = (
        summary
        .sample(frac=1.0, random_state=random_seed)
        .sort_values(
            by=["n_positive", "positive_rate", "n_chunks", "transcript_id"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )

    split_members:   dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_positives: dict[str, int]       = {"train": 0,  "val": 0,  "test": 0}
    assignments: list[str] = []

    for _, row in summary.iterrows():
        eligible = [s for s, m in split_members.items() if len(m) < target_counts[s]]
        if not eligible:
            raise RuntimeError("No eligible split during assignment — this is a bug.")

        # Total positives assigned so far across all splits.
        total_assigned = sum(split_positives.values()) + int(row["n_positive"])

        def _score(split_name: str) -> tuple[float, float, float]:
            """Lower score = better. Assign to the split most behind its
            proportional positive share.

            deficit = target_fraction - current_fraction
            Higher deficit means this split needs positives more urgently.
            We negate it so min() picks the largest deficit.
            """
            projected         = split_positives[split_name] + int(row["n_positive"])
            current_fraction  = projected / max(total_assigned, 1)
            deficit           = target_fraction[split_name] - current_fraction
            tie_break         = rng.random()
            # Negate deficit so min() picks the most-behind split.
            return (-deficit, tie_break)

        chosen = min(eligible, key=_score)
        split_members[chosen].append(str(row["transcript_id"]))
        split_positives[chosen] += int(row["n_positive"])
        assignments.append(chosen)

    summary["split"] = assignments
    return summary



def apply_split_assignments(
    df: pd.DataFrame,
    split_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Attach transcript-level split labels to every chunk row.

    Inputs:
        df:               Combined chunk dataframe.
        split_assignments: Output of :func:`assign_transcript_splits`,
                           must contain ``transcript_id`` and ``split``.

    Outputs:
        Copy of ``df`` with a ``split`` column added.

    Raises:
        ValueError: If any chunk did not receive a split label.
    """
    merged = df.merge(
        split_assignments[["transcript_id", "split"]],
        on="transcript_id",
        how="left",
        validate="many_to_one",
    )
    if merged["split"].isna().any():
        raise ValueError(
            "Some chunk rows did not receive a split assignment.  "
            "Check that all transcript_ids in df appear in split_assignments."
        )
    return merged


# ---------------------------------------------------------------------------
# Context-window text construction
# ---------------------------------------------------------------------------

def build_context_text(df: pd.DataFrame, context_window: int) -> pd.Series:
    """Build model-input text by optionally joining neighboring chunks.

    When ``context_window > 0`` the text for chunk *i* becomes the
    concatenation of chunks *i − window* through *i + window* (clamped to
    the transcript boundaries).  This is useful because a research mention
    sometimes begins in one chunk and concludes in the next.

    The output Series is indexed identically to ``df`` so it can be assigned
    back as a new column without alignment issues.

    Inputs:
        df:             Combined dataframe.  Rows must be ordered within each
                        transcript (i.e. consecutive rows = consecutive chunks).
        context_window: Number of neighboring chunks on *each side* to include.
                        0 = use the chunk's own text only.

    Outputs:
        String Series aligned with ``df.index``.

    Raises:
        ValueError: If ``context_window`` is negative.
    """
    if context_window < 0:
        raise ValueError("context_window must be non-negative.")

    # Fast path: no context needed.
    if context_window == 0:
        return df["text"].astype(str).copy()

    # Allocate the result Series up front, keyed by df's actual index.
    # This avoids the ordering bug that would arise from building a flat list
    # and then zipping it back with df.index — if groupby ever reorders groups
    # the flat list would be misaligned.
    result = pd.Series("", index=df.index, dtype=str)

    for _, group in df.groupby("transcript_id", sort=False):
        texts   = group["text"].astype(str).tolist()
        indices = group.index.tolist()   # original DataFrame indices for this group
        n       = len(texts)

        for pos, idx in enumerate(indices):
            start = max(0, pos - context_window)
            end   = min(n, pos + context_window + 1)
            result.at[idx] = "\n\n".join(texts[start:end])

    return result
