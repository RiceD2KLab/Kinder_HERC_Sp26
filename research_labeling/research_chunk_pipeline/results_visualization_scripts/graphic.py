"""Generate a dataset distribution bar chart showing split composition.

Produces one PNG file:
    dataset_distribution.png — stacked bar chart of positive/negative chunks
                               per split (train / val / test).

Usage:
    cd research_chunk_pipeline/plots
    python graphic.py
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs"))

# ── Color palette ─────────────────────────────────────────────────────────────
NEG_COLOR = "#5B9BD5"
POS_COLOR = "#E24B4A"
GRAY      = "#666666"
LIGHT     = "#F7F7F7"

SPLIT_ORDER  = ["train", "val", "test"]
SPLIT_LABELS = ["Train\n(15 meetings)", "Validation\n(5 meetings)", "Test\n(5 meetings)"]


def load_split_data():
    """Load transcript split assignments and compute per-split aggregates.

    Inputs:
        None.  Reads ``transcript_split_assignments.csv`` from the
        ``outputs/`` directory relative to this script.

    Outputs:
        DataFrame indexed by split with columns ``total_chunks``,
        ``total_positive``, ``total_negative``, and ``positive_rate``.
    """
    df = pd.read_csv(os.path.join(OUTPUTS_DIR, "transcript_split_assignments.csv"))
    df["n_negative"] = df["n_chunks"] - df["n_positive"]

    agg = df.groupby("split").agg(
        total_chunks   = ("n_chunks",   "sum"),
        total_positive = ("n_positive", "sum"),
        total_negative = ("n_negative", "sum"),
    ).reindex(SPLIT_ORDER)
    agg["positive_rate"] = agg["total_positive"] / agg["total_chunks"]
    return agg


def plot_dataset_distribution(agg):
    """Render a stacked bar chart of dataset composition and save as PNG.

    Inputs:
        agg: DataFrame from :func:`load_split_data` with per-split chunk
             counts and positive rates.

    Outputs:
        Saves ``dataset_distribution.png`` to the plots directory.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(LIGHT)
    ax.grid(axis="y", color="white", linewidth=1.8, zorder=0)

    x     = np.arange(len(SPLIT_ORDER))
    bar_w = 0.45
    max_y = int(agg["total_chunks"].max())
    ylim  = max_y * 1.40

    for i, split in enumerate(SPLIT_ORDER):
        neg  = int(agg.loc[split, "total_negative"])
        pos  = int(agg.loc[split, "total_positive"])
        tot  = neg + pos
        rate = agg.loc[split, "positive_rate"]

        ax.bar(i, neg, width=bar_w, color=NEG_COLOR, zorder=3)
        ax.bar(i, pos, width=bar_w, bottom=neg, color=POS_COLOR, zorder=3)

        ax.text(i, neg / 2, f"{neg:,}\nnon-hits",
                ha="center", va="center", fontsize=10,
                fontweight="600", color="white", zorder=5)

        annotation_y = tot + ylim * 0.07
        ax.annotate(
            f"{pos} hits",
            xy=(i, tot),
            xytext=(i, annotation_y),
            ha="center", va="bottom",
            fontsize=11, fontweight="700", color=POS_COLOR,
            arrowprops=dict(arrowstyle="-|>", color=POS_COLOR, lw=1.4),
            zorder=6,
        )

        ax.text(i, annotation_y + ylim * 0.07,
                f"{rate:.1%} positive",
                ha="center", va="bottom", fontsize=12, color=GRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(SPLIT_LABELS, fontsize=11)
    ax.set_ylabel("Number of chunks", fontsize=11)
    ax.set_ylim(0, ylim)
    ax.tick_params(axis="y", labelsize=9)

    total_chunks = int(agg["total_chunks"].sum())
    total_pos    = int(agg["total_positive"].sum())
    overall_rate = total_pos / total_chunks
    ax.set_title(
        f"Dataset composition: ({overall_rate:.1%} overall positive rate)",
        fontsize=12, fontweight="600", pad=12, color="#222222",
    )

    neg_patch = mpatches.Patch(color=NEG_COLOR, label="Non-hits")
    pos_patch = mpatches.Patch(color=POS_COLOR, label="Hits (research mentions)")
    ax.legend(handles=[neg_patch, pos_patch], loc="upper right",
              fontsize=10, framealpha=0.95, edgecolor="#CCCCCC")

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    path = os.path.join(SCRIPT_DIR, "dataset_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    split_data = load_split_data()
    plot_dataset_distribution(split_data)
