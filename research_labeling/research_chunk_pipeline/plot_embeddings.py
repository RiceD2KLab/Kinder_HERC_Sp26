"""Visualize the full dataset in embedding space with true binary labels.

Loads all transcript CSVs, re-embeds every chunk with MPNet, then reduces
to 2-D with PCA and t-SNE and saves one scatter plot per method.

Color scheme
------------
    Label 0 (non-hit)       → red    (#E22808)
    Label 1 (research hit)  → green  (#1A9641)

t-SNE note
----------
Running t-SNE directly on 768-d embeddings for 3000+ chunks is slow.
This script first reduces to 50 components with PCA, then feeds those
50-d vectors into t-SNE.  This is standard practice and preserves the
structure t-SNE needs while cutting runtime significantly.

Usage
-----
    python plot_embeddings.py --transcript-data-dir "../Transcript Data"
    python plot_embeddings.py --transcript-data-dir "../Transcript Data" \\
        --output-dir plots/ --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data_utils import build_context_text, load_all_transcripts
from embedding_utils import encode_texts, load_embedder


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

WHITE = "#FFFFFF"
LIGHT = "#F7F7F7"

LABEL_COLORS = {0: "#E22808", 1: "#1A9641"}
LABEL_NAMES  = {0: "Non-hit (0)", 1: "Research hit (1)"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_and_embed(transcript_data_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load all transcripts, embed chunks, return features + labels + df.

    Outputs:
        embeddings : float32 array (n_chunks, 768)
        labels     : int array     (n_chunks,)
        df         : combined dataframe
    """
    print("Loading transcripts...")
    df = load_all_transcripts(transcript_data_dir)
    df["model_text"] = build_context_text(df, context_window=0)

    n_chunks    = len(df)
    n_pos       = int(df["binary_hit"].sum())
    print(f"  {n_chunks} chunks  |  {n_pos} positives  ({100 * n_pos / n_chunks:.1f}%)")

    print("Embedding chunks with MPNet (sentence-transformers/all-mpnet-base-v2)...")
    embedder   = load_embedder("sentence-transformers/all-mpnet-base-v2")
    embeddings = encode_texts(
        model=embedder,
        texts=df["model_text"].astype(str).tolist(),
        batch_size=32,
        normalize_embeddings=True,
    )
    labels = df["binary_hit"].to_numpy(dtype=int)
    return embeddings, labels, df


def _reduce_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    print(f"Running PCA → {n_components}D...")
    return PCA(n_components=n_components, random_state=42).fit_transform(embeddings)


def _reduce_tsne(
    embeddings: np.ndarray,
    pca_pre: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """PCA to pca_pre dims, then t-SNE to 2D."""
    n_pre = min(pca_pre, embeddings.shape[1], embeddings.shape[0] - 1)
    print(f"Running PCA → {n_pre}D  (pre-processing for t-SNE)...")
    reduced = PCA(n_components=n_pre, random_state=seed).fit_transform(embeddings)
    print("Running t-SNE → 2D  (this may take a minute)...")
    return TSNE(
        n_components=2,
        perplexity=40,
        max_iter=1000,
        random_state=seed,
        init="pca",
    ).fit_transform(reduced)


def _scatter_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    method: str,
    n_chunks: int,
    n_pos: int,
    output_path: Path,
    seed: int,
) -> None:
    """Render and save a 2-D scatter plot colored by true label.

    Inputs:
        coords:      (n, 2) array of 2-D coordinates.
        labels:      (n,)   int array of true labels (0 or 1).
        method:      Display name for the reduction method (e.g. "PCA").
        n_chunks:    Total chunk count for subtitle.
        n_pos:       Positive chunk count for subtitle.
        output_path: Destination PNG path.
        seed:        Random seed used (shown in subtitle).
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LIGHT)

    # Plot negatives first so positives render on top.
    for label in [0, 1]:
        mask  = labels == label
        color = LABEL_COLORS[label]
        name  = LABEL_NAMES[label]
        count = int(mask.sum())
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color,
            s=10 if label == 0 else 14,
            alpha=0.45 if label == 0 else 0.70,
            linewidths=0,
            label=f"{name}  (n={count:,})",
            zorder=3 if label == 1 else 2,
            rasterized=True,
        )

    ax.set_title(
        f"Full dataset — true labels  ({method})\n"
        f"{n_chunks:,} chunks  |  {n_pos:,} positives ({100 * n_pos / n_chunks:.1f}%)  "
        f"|  seed={seed}",
        fontsize=12, fontweight="600", pad=12,
    )
    ax.set_xlabel(f"{method} dim 1", fontsize=10)
    ax.set_ylabel(f"{method} dim 2", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=10, framealpha=0.9, markerscale=2.0, loc="best")
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
        description="Plot full-dataset embeddings colored by true binary label."
    )
    parser.add_argument(
        "--transcript-data-dir",
        type=Path,
        required=True,
        help="Directory containing transcript CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to write PNG files.  Default: plots/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PCA / t-SNE.  Default: 42.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, labels, df = _load_and_embed(args.transcript_data_dir)
    n_chunks = len(labels)
    n_pos    = int(labels.sum())

    # PCA
    pca_coords = _reduce_pca(embeddings, n_components=2)
    _scatter_plot(
        coords=pca_coords,
        labels=labels,
        method="PCA",
        n_chunks=n_chunks,
        n_pos=n_pos,
        output_path=args.output_dir / "full_dataset_pca.png",
        seed=args.seed,
    )

    # t-SNE (with PCA pre-processing)
    tsne_coords = _reduce_tsne(embeddings, pca_pre=50, seed=args.seed)
    _scatter_plot(
        coords=tsne_coords,
        labels=labels,
        method="t-SNE",
        n_chunks=n_chunks,
        n_pos=n_pos,
        output_path=args.output_dir / "full_dataset_tsne.png",
        seed=args.seed,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
