"""Visualize a single fold's test set in embedding space with prediction outcomes.

Embeds the FULL labeled dataset, reduces to 2-D with PCA and t-SNE, then
renders the test fold's chunks colored by prediction outcome on top of a
gray background of all other chunks.

This gives a stable embedding geometry that reflects the true data
distribution — not just the ~20 % slice in the test fold.

Color scheme
------------
    True Positive  (predicted 1, actual 1) → green   (#1A9641)
    False Positive (predicted 1, actual 0) → yellow  (#F1C40F)
    False Negative (predicted 0, actual 1) → red     (#E22808)
    True Negative  (predicted 0, actual 0) → blue    (#2C7BB6)
    Background (not in test fold)          → gray    (#AAAAAA)

Full data path
--------------
``--full-data-path`` may point to either:
  - a single CSV file, or
  - a directory of CSV files (all *.csv files are concatenated).

Embedding cache
---------------
Embeddings are cached as ``<full-data-path>.embeddings.npy`` (or
``<full-data-path>/_embeddings_cache.npy`` when a directory is given).
A paired ``.hashes.npy`` file fingerprints the data so the cache is
invalidated automatically if any source file changes.
Pass --no-cache to force re-embedding.

Usage
-----
    python plot_fold_embeddings.py \\
        --predictions-path "outputs/no_feature_selection/seed_2/fold_2_test_predictions.csv" \\
        --full-data-path   "../Transcript Data"

    python plot_fold_embeddings.py \\
        --predictions-path "outputs/no_feature_selection/seed_2/fold_2_test_predictions.csv" \\
        --full-data-path   "data/labeled_chunks.csv" \\
        --output-dir plots/ --seed 42 --no-cache
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from embedding_utils import encode_texts, load_embedder


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

WHITE = "#FFFFFF"
LIGHT = "#F7F7F7"

# Outcome → (color, display name, z-order, size, alpha)
OUTCOME_STYLE: dict[str, tuple[str, str, int, int, float]] = {
    "TP": ("#1A9641", "True Positive  (predicted 1, actual 1)",  5, 20, 0.95),
    "FP": ("#F1C40F", "False Positive (predicted 1, actual 0)",  4, 20, 0.90),
    "FN": ("#E22808", "False Negative (predicted 0, actual 1)",  4, 20, 0.90),
    "TN": ("#2C7BB6", "True Negative  (predicted 0, actual 0)",  3, 12, 0.70),
    "BG": ("#AAAAAA", "Background (not in test fold)",           1,  6, 0.30),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_full_data(full_data_path: Path) -> tuple[pd.DataFrame, Path, Path]:
    """Load the full dataset from a CSV file or a directory of CSV files.

    Inputs:
        full_data_path: Path to a CSV file or a directory of CSV files.

    Outputs:
        df:         Concatenated DataFrame with at least a 'text' column.
        cache_emb:  Path where the embedding .npy cache should be stored.
        cache_hash: Path where the hash .npy cache should be stored.
    """
    if full_data_path.is_dir():
        csv_files = sorted(full_data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in directory: {full_data_path}"
            )
        print(f"  Found {len(csv_files)} CSV files in {full_data_path}:")
        frames = []
        for p in csv_files:
            print(f"    {p.name}")
            frames.append(pd.read_csv(p))
        df = pd.concat(frames, ignore_index=True)
        # Cache lives inside the directory itself
        cache_emb  = full_data_path / "_embeddings_cache.npy"
        cache_hash = full_data_path / "_embeddings_cache.hashes.npy"
    else:
        df = pd.read_csv(full_data_path)
        cache_emb  = full_data_path.with_suffix(".embeddings.npy")
        cache_hash = full_data_path.with_suffix(".embeddings.hashes.npy")

    print(f"  {len(df):,} total chunks loaded from full dataset")
    if "text" not in df.columns:
        raise ValueError(
            f"Full data must contain a 'text' column. "
            f"Columns found: {list(df.columns)}"
        )
    return df, cache_emb, cache_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_outcome(df_test: pd.DataFrame) -> pd.Series:
    """Return a string Series with TP / FP / FN / TN per test chunk."""
    pred   = df_test["predicted_label"].astype(int)
    actual = pd.to_numeric(df_test["binary_hit"], errors="coerce").fillna(0).astype(int)
    outcome = pd.Series("TN", index=df_test.index)
    outcome[(pred == 1) & (actual == 1)] = "TP"
    outcome[(pred == 1) & (actual == 0)] = "FP"
    outcome[(pred == 0) & (actual == 1)] = "FN"
    return outcome


def _text_hash(text: str) -> str:
    """Stable SHA-256 hex digest of a text string (first 16 chars)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_or_embed(
    texts: list[str],
    cache_emb: Path,
    cache_hash: Path,
    use_cache: bool,
) -> np.ndarray:
    """Return embeddings for ``texts``, loading from cache if available."""
    if use_cache and cache_emb.exists() and cache_hash.exists():
        stored_hashes  = np.load(cache_hash, allow_pickle=True)
        current_hashes = np.array([_text_hash(t) for t in texts])
        if np.array_equal(stored_hashes, current_hashes):
            print(f"  Loading cached embeddings from {cache_emb}")
            return np.load(cache_emb)
        else:
            print("  Cache exists but data changed — re-embedding...")

    print(f"  Embedding {len(texts):,} chunks with MPNet (this may take a while)...")
    embedder   = load_embedder("sentence-transformers/all-mpnet-base-v2")
    embeddings = encode_texts(
        model=embedder,
        texts=texts,
        batch_size=32,
        normalize_embeddings=True,
    )

    if use_cache:
        np.save(cache_emb,  embeddings)
        np.save(cache_hash, np.array([_text_hash(t) for t in texts]))
        print(f"  Embeddings cached → {cache_emb}")

    return embeddings


def _reduce_pca(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    print("Running PCA → 2D on full dataset...")
    return PCA(n_components=2, random_state=seed).fit_transform(embeddings)


def _reduce_tsne(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    n     = embeddings.shape[0]
    n_pre = min(50, embeddings.shape[1], n - 1)
    print(f"Running PCA → {n_pre}D (pre-processing for t-SNE) on full dataset...")
    reduced    = PCA(n_components=n_pre, random_state=seed).fit_transform(embeddings)
    perplexity = min(40, n // 4)
    print(f"Running t-SNE → 2D  (n={n:,}, perplexity={perplexity})...")
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1000,
        random_state=seed,
        init="pca",
    ).fit_transform(reduced)


def _save_legend(output_path: Path) -> None:
    """Save a standalone legend PNG with all outcome swatches."""
    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    fig.patch.set_facecolor(WHITE)
    ax.set_visible(False)

    handles = []
    for outcome_key in ["TP", "FP", "FN", "TN", "BG"]:
        color, label_name, _, size, alpha = OUTCOME_STYLE[outcome_key]
        handles.append(
            plt.scatter([], [], c=color, s=size * 4, alpha=alpha,
                        linewidths=0, label=label_name)
        )

    legend = fig.legend(
        handles=handles,
        fontsize=11,
        framealpha=0.95,
        markerscale=1.5,
        loc="center",
        title="Prediction outcome",
        title_fontsize=12,
    )
    legend.get_frame().set_edgecolor("#CCCCCC")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved legend → {output_path}")


def _scatter_plot(
    coords: np.ndarray,
    outcomes: pd.Series,
    test_mask: np.ndarray,
    method: str,
    predictions_path: Path,
    output_path: Path,
    seed: int,
) -> None:
    """Render and save a 2-D scatter plot colored by prediction outcome."""
    n_test = int(test_mask.sum())
    counts = outcomes.value_counts()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LIGHT)

    # Background — all non-test chunks
    bg_mask = ~test_mask
    if bg_mask.any():
        color, _, zorder, size, alpha = OUTCOME_STYLE["BG"]
        ax.scatter(
            coords[bg_mask, 0], coords[bg_mask, 1],
            c=color, s=size, alpha=alpha, linewidths=0,
            zorder=zorder, rasterized=True,
        )

    # Test-fold chunks colored by outcome (TN first, TP last)
    test_coords  = coords[test_mask]
    outcomes_arr = outcomes.values

    for outcome_key in ["TN", "FN", "FP", "TP"]:
        color, _, zorder, size, alpha = OUTCOME_STYLE[outcome_key]
        mask = (outcomes_arr == outcome_key)
        if not mask.any():
            continue
        ax.scatter(
            test_coords[mask, 0], test_coords[mask, 1],
            c=color, s=size, alpha=alpha, linewidths=0,
            zorder=zorder, rasterized=True,
        )

    tp = int(counts.get("TP", 0))
    fp = int(counts.get("FP", 0))
    fn = int(counts.get("FN", 0))
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f2        = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0.0

    ax.set_title(
        f"Fold test set — prediction outcomes  ({method})\n"
        f"{predictions_path.name}  |  test n={n_test:,}  |  "
        f"full n={len(coords):,}  |  seed={seed}\n"
        f"Recall={recall:.3f}  Precision={precision:.3f}  F2={f2:.3f}",
        fontsize=14,
        fontweight="600",
        pad=14,
    )
    ax.set_xlabel(f"{method} dim 1", fontsize=11)
    ax.set_ylabel(f"{method} dim 2", fontsize=11)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if method == "t-SNE":
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a fold's test predictions in the full-dataset embedding space, "
            "colored by TP / FP / FN / TN."
        )
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        required=True,
        help="Path to a fold_X_test_predictions.csv file.",
    )
    parser.add_argument(
        "--full-data-path",
        type=Path,
        required=True,
        help=(
            "Path to the full labeled dataset — either a single CSV file or "
            "a directory of CSV files (all *.csv files are concatenated)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write PNG files.  "
            "Defaults to the same folder as --predictions-path."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PCA / t-SNE.  Default: 42.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache — always re-embed from scratch.",
    )
    args = parser.parse_args()

    if not args.predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {args.predictions_path}")
    if not args.full_data_path.exists():
        raise FileNotFoundError(f"Full data path not found: {args.full_data_path}")

    output_dir = args.output_dir or args.predictions_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.predictions_path.stem.replace("_test_predictions", "")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading predictions from {args.predictions_path}...")
    df_test = pd.read_csv(args.predictions_path)
    print(f"  {len(df_test):,} test chunks loaded")

    missing = {"text", "binary_hit", "predicted_label"} - set(df_test.columns)
    if missing:
        raise ValueError(f"Predictions CSV is missing columns: {missing}")

    print(f"Loading full dataset from {args.full_data_path}...")
    df_full, cache_emb, cache_hash = _load_full_data(args.full_data_path)

    # ------------------------------------------------------------------
    # Align test predictions onto full dataset by text match
    # ------------------------------------------------------------------
    test_lookup                = df_test.set_index("text")["predicted_label"]
    df_full["predicted_label"] = df_full["text"].map(test_lookup)
    test_mask                  = df_full["predicted_label"].notna().values

    n_matched = int(test_mask.sum())
    if n_matched == 0:
        raise ValueError(
            "No test chunks matched rows in the full dataset by 'text'.  "
            "Check that both CSVs share identical text column values."
        )
    if n_matched < len(df_test):
        print(
            f"  WARNING: {len(df_test) - n_matched} test chunks did not match "
            "any row in the full dataset and will be excluded from the plot."
        )
    print(f"  {n_matched:,} / {len(df_test):,} test chunks matched in full dataset")

    outcomes = _assign_outcome(
        df_full[test_mask].assign(
            predicted_label=df_full.loc[test_mask, "predicted_label"].astype(int)
        )
    )
    counts = outcomes.value_counts()
    print(
        f"  TP={counts.get('TP', 0)}  FP={counts.get('FP', 0)}  "
        f"FN={counts.get('FN', 0)}  TN={counts.get('TN', 0)}"
    )

    # ------------------------------------------------------------------
    # Embed full dataset (with optional cache)
    # ------------------------------------------------------------------
    texts      = df_full["text"].astype(str).tolist()
    embeddings = _load_or_embed(
        texts=texts,
        cache_emb=cache_emb,
        cache_hash=cache_hash,
        use_cache=not args.no_cache,
    )

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------
    pca_coords = _reduce_pca(embeddings, seed=args.seed)
    _scatter_plot(
        coords=pca_coords,
        outcomes=outcomes,
        test_mask=test_mask,
        method="PCA",
        predictions_path=args.predictions_path,
        output_path=output_dir / f"{stem}_pca.png",
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------------------
    tsne_coords = _reduce_tsne(embeddings, seed=args.seed)
    _scatter_plot(
        coords=tsne_coords,
        outcomes=outcomes,
        test_mask=test_mask,
        method="t-SNE",
        predictions_path=args.predictions_path,
        output_path=output_dir / f"{stem}_tsne.png",
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Standalone legend
    # ------------------------------------------------------------------
    _save_legend(output_dir / f"{stem}_legend.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
