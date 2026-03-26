"""Generate presentation-ready plots from actual pipeline output files.

Produces three PNG files:
    1. confusion_matrix.png  — TP/FP/FN/TN from test set
    2. threshold_sweep.png   — recall, precision, F2 across thresholds (validation)
    3. metrics_summary.png   — bar chart of key test metrics

Usage:
    cd research_chunk_pipeline/plots
    python visualize_results.py
"""

import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs"))
SAVE_DIR    = SCRIPT_DIR

# ── Color palette ─────────────────────────────────────────────────────────────
BLUE  = "#2C7BB6"
GREEN = "#1A9641"
RED   = "#E22808"
GRAY  = "#888888"
LIGHT = "#F7F7F7"


def load_pipeline_outputs():
    """Load metrics and threshold sweep data from pipeline output files.

    Inputs:
        None.  Reads ``metrics_summary.json`` and
        ``validation_threshold_sweep.csv`` from the ``outputs/`` directory
        relative to this script.

    Outputs:
        Tuple ``(summary, sweep)`` where *summary* is the parsed JSON dict
        and *sweep* is a DataFrame filtered to the best model configuration.
    """
    with open(os.path.join(OUTPUTS_DIR, "metrics_summary.json")) as f:
        summary = json.load(f)

    best_c  = summary["best_c"]
    best_cw = summary["best_class_weight"]

    sweep = pd.read_csv(
        os.path.join(OUTPUTS_DIR, "validation_threshold_sweep.csv")
    )
    sweep = (
        sweep[(sweep["c_value"] == best_c) & (sweep["class_weight"] == best_cw)]
        .sort_values("threshold")
        .reset_index(drop=True)
    )
    return summary, sweep


def plot_confusion_matrix(summary):
    """Render a confusion-matrix card for the test set and save as PNG.

    Inputs:
        summary: Parsed ``metrics_summary.json`` dict containing
                 ``test_metrics`` and ``best_threshold``.

    Outputs:
        Saves ``confusion_matrix.png`` to the plots directory.
    """
    tm        = summary["test_metrics"]
    tp        = tm["true_positive"]
    fp        = tm["false_positive"]
    fn        = tm["false_negative"]
    tn        = tm["true_negative"]
    recall    = tm["recall"]
    precision = tm["precision"]
    f2        = tm["f2"]
    threshold = summary["best_threshold"]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    cells = [
        (0, 0, tp, "True positive",  "Correctly flagged",  "#D4EDDA", "#155724"),
        (0, 1, fp, "False positive", "Wrongly flagged",    "#FFF3CD", "#856404"),
        (1, 0, fn, "False negative", "Missed mentions",    "#F8D7DA", "#721C24"),
        (1, 1, tn, "True negative",  "Correctly ignored",  "#D1ECF1", "#0C5460"),
    ]

    for row, col, val, label, sublabel, bg, fg in cells:
        x = col * 0.5 + 0.02
        y = 0.95 - row * 0.47
        rect = mpatches.FancyBboxPatch(
            (x, y - 0.42), 0.46, 0.42,
            boxstyle="round,pad=0.02",
            linewidth=0, facecolor=bg,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(x + 0.23, y - 0.10, str(val),
                ha="center", va="center", fontsize=36, fontweight="bold",
                color=fg, transform=ax.transAxes)
        ax.text(x + 0.23, y - 0.25, label,
                ha="center", va="center", fontsize=11, fontweight="600",
                color=fg, transform=ax.transAxes)
        ax.text(x + 0.23, y - 0.35, sublabel,
                ha="center", va="center", fontsize=9,
                color=fg, alpha=0.75, transform=ax.transAxes)

    total_chunks   = tp + fp + fn + tn
    flagged_chunks = tp + fp
    ax.text(0.5, 1.08, "Confusion matrix - test set (5 unseen meetings)",
            ha="center", va="bottom", fontsize=15, fontweight="600",
            transform=ax.transAxes)
    ax.text(0.5, 1.01,
            f"Model flagged {flagged_chunks} of {total_chunks} chunks for review  "
            f"({flagged_chunks/total_chunks:.0%} of total)",
            ha="center", va="bottom", fontsize=12, color=GRAY,
            transform=ax.transAxes)
    stats = [
        ("Threshold", f"{threshold}"),
        ("Recall",    f"{recall:.1%}"),
        ("Precision", f"{precision:.1%}"),
        ("F2",        f"{f2:.3f}"),
    ]
    for i, (label, val) in enumerate(stats):
        x = 0.125 + i * 0.25
        ax.text(x, -0.06, label,
                ha="center", va="top", fontsize=10, color=GRAY,
                transform=ax.transAxes)
        ax.text(x, -0.16, val,
                ha="center", va="top", fontsize=16, fontweight="600",
                color="#222222", transform=ax.transAxes)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def plot_threshold_sweep(summary, sweep):
    """Plot recall, precision, and F2 across candidate thresholds.

    Inputs:
        summary: Parsed ``metrics_summary.json`` dict (used for ``best_c``).
        sweep:   DataFrame of validation threshold sweep rows for the best
                 model configuration.

    Outputs:
        Saves ``threshold_sweep.png`` to the plots directory.
    """
    best_c     = summary["best_c"]
    thresholds = sweep["threshold"].tolist()
    recall_sw  = sweep["recall"].tolist()
    prec_sw    = sweep["precision"].tolist()
    f2_sw      = sweep["f2"].tolist()

    best_idx = int(np.argmax(f2_sw))
    best_t   = thresholds[best_idx]
    best_f2  = f2_sw[best_idx]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(LIGHT)
    ax.grid(color="white", linewidth=1.2, zorder=0)

    ax.plot(thresholds, recall_sw, color=BLUE,  lw=2.2, marker="o", ms=5,
            label="Recall", zorder=3)
    ax.plot(thresholds, prec_sw,   color=GREEN, lw=2.2, marker="o", ms=5,
            label="Precision", zorder=3)
    ax.plot(thresholds, f2_sw,     color=RED, lw=2.8, marker="o", ms=6,
            label="F2", zorder=4)

    ax.axvline(best_t, color=RED, lw=1.5, ls="--", alpha=0.7, zorder=2)
    ax.scatter([best_t], [best_f2], color=RED, s=100, zorder=5)
    ax.annotate(
        f"Selected threshold = {best_t}\nF2 = {best_f2:.3f}",
        xy=(best_t, best_f2),
        xytext=(best_t + 0.18, best_f2 + 0.12),
        fontsize=9, color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    )

    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Validation threshold sweep: recall / precision / F2 tradeoff\n"
        f"(C={best_c})",
        fontsize=11, fontweight="600", pad=10,
    )
    ax.set_xlim(min(thresholds) - 0.02, max(thresholds) + 0.02)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(thresholds)
    ax.tick_params(labelsize=9)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "threshold_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def plot_metrics_summary(summary):
    """Render a bar chart of key test-set metrics and save as PNG.

    Inputs:
        summary: Parsed ``metrics_summary.json`` dict containing
                 ``test_metrics``, ``best_c``, and ``best_threshold``.

    Outputs:
        Saves ``metrics_summary.png`` to the plots directory.
    """
    tm        = summary["test_metrics"]
    best_c    = summary["best_c"]
    threshold = summary["best_threshold"]

    metrics = {
        "Recall\n(test)":       tm["recall"],
        "Precision\n(test)":    tm["precision"],
        "F2\n(test)":           tm["f2"],
        "F1\n(test)":           tm["f1"],
        "Avg. precision\n(AP)": tm["average_precision"],
    }
    colors = [BLUE, GREEN, RED, GRAY, "#9B59B6"]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(LIGHT)
    ax.grid(axis="y", color="white", linewidth=1.5, zorder=0)

    bars = ax.bar(
        metrics.keys(), metrics.values(),
        color=colors, width=0.55, zorder=3,
        edgecolor="white", linewidth=1.5,
    )
    for bar, val in zip(bars, metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.018,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="600",
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Test set performance - 5 Transcripts\n"
        f"C={best_c}, threshold={threshold}",
        fontsize=11, fontweight="600", pad=10,
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "metrics_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    pipeline_summary, threshold_sweep = load_pipeline_outputs()
    plot_confusion_matrix(pipeline_summary)
    plot_threshold_sweep(pipeline_summary, threshold_sweep)
    plot_metrics_summary(pipeline_summary)
    print("\nAll plots saved.")
