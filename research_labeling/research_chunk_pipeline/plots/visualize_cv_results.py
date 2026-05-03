"""Generate plots from cv_results.json produced by cv_pipeline.py.

Produces three PNG files in this directory:
    1. cv_fold_metrics.png     — recall, precision, F2 per fold + mean lines
    2. cv_aggregate_metrics.png — bar chart of aggregate mean ± std
    3. cv_confusion_summary.png — confusion matrix summed across all folds

Usage:
    cd research_chunk_pipeline/plots
    python visualize_cv_results.py
"""

import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs"))
SAVE_DIR    = SCRIPT_DIR

# ── Color palette (matches visualize_results.py) ──────────────────────────────
BLUE   = "#2C7BB6"
GREEN  = "#1A9641"
RED    = "#E22808"
GRAY   = "#888888"
PURPLE = "#9B59B6"
LIGHT  = "#F7F7F7"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_cv_results() -> dict:
    """Load cv_results.json from the outputs directory.

    Outputs:
        Parsed dict with ``config``, ``fold_results``, and ``aggregate`` keys.
    """
    path = os.path.join(OUTPUTS_DIR, "cv_results.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── Plot 1: per-fold metrics ──────────────────────────────────────────────────

def plot_fold_metrics(cv: dict) -> None:
    """Line chart showing recall, precision, and F2 for every fold.

    A horizontal dashed line marks the mean for each metric, and a shaded
    band covers ±1 std.

    Inputs:
        cv: Parsed cv_results.json dict.

    Outputs:
        Saves ``cv_fold_metrics.png`` to the plots directory.
    """
    folds     = [r["fold"]                      for r in cv["fold_results"]]
    recall    = [r["test_metrics"]["recall"]    for r in cv["fold_results"]]
    precision = [r["test_metrics"]["precision"] for r in cv["fold_results"]]
    f2        = [r["test_metrics"]["f2"]        for r in cv["fold_results"]]

    agg = cv["aggregate"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(LIGHT)
    ax.grid(color="white", linewidth=1.2, zorder=0)

    # Lines + markers
    ax.plot(folds, recall,    color=BLUE,  lw=2.2, marker="o", ms=7,
            label="Recall",    zorder=3)
    ax.plot(folds, precision, color=GREEN, lw=2.2, marker="o", ms=7,
            label="Precision", zorder=3)
    ax.plot(folds, f2,        color=RED,   lw=2.8, marker="o", ms=8,
            label="F2",        zorder=4)

    # Mean lines + std bands
    for values, mean_key, std_key, color in [
        (recall,    "recall_mean",    "recall_std",    BLUE),
        (precision, "precision_mean", "precision_std", GREEN),
        (f2,        "f2_mean",        "f2_std",        RED),
    ]:
        mean = agg[mean_key]
        std  = agg[std_key]
        ax.axhline(mean, color=color, lw=1.4, ls="--", alpha=0.65, zorder=2)
        ax.fill_between(
            folds,
            mean - std,
            mean + std,
            color=color,
            alpha=0.08,
            zorder=1,
        )

    n_folds      = cv["config"]["n_folds"]
    n_transcripts = cv["config"]["n_transcripts"]
    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {f}" for f in folds], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title(
        f"Per-fold test metrics  ({n_folds}-fold grouped CV, {n_transcripts} transcripts)\n"
        f"Dashed lines = mean  |  Shaded bands = ±1 std",
        fontsize=11, fontweight="600", pad=10,
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "cv_fold_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 2: aggregate metrics bar chart ───────────────────────────────────────

def plot_aggregate_metrics(cv: dict) -> None:
    """Bar chart of aggregate mean ± std for each key metric.

    Inputs:
        cv: Parsed cv_results.json dict.

    Outputs:
        Saves ``cv_aggregate_metrics.png`` to the plots directory.
    """
    agg = cv["aggregate"]

    labels = ["Recall", "Precision", "F2", "F1", "Avg. Precision"]
    keys   = ["recall", "precision", "f2", "f1", "average_precision"]
    colors = [BLUE, GREEN, RED, GRAY, PURPLE]

    means = [agg[f"{k}_mean"] for k in keys]
    stds  = [agg[f"{k}_std"]  for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(LIGHT)
    ax.grid(axis="y", color="white", linewidth=1.5, zorder=0)

    bars = ax.bar(
        labels, means,
        color=colors, width=0.55, zorder=3,
        edgecolor="white", linewidth=1.5,
        yerr=stds, capsize=6,
        error_kw={"elinewidth": 1.8, "ecolor": "#333333", "capthick": 1.8},
    )

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.025,
            f"{mean:.3f}\n±{std:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="600",
            linespacing=1.4,
        )

    n_folds       = cv["config"]["n_folds"]
    n_transcripts = cv["config"]["n_transcripts"]
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Aggregate test metrics: mean ± std\n"
        f"({n_folds}-fold grouped CV, {n_transcripts} transcripts)",
        fontsize=11, fontweight="600", pad=10,
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "cv_aggregate_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 3: summed confusion matrix ──────────────────────────────────────────

def plot_confusion_summary(cv: dict) -> None:
    """Confusion matrix card with counts summed across all folds.

    Inputs:
        cv: Parsed cv_results.json dict.

    Outputs:
        Saves ``cv_confusion_summary.png`` to the plots directory.
    """
    tp = sum(r["test_metrics"]["true_positive"]  for r in cv["fold_results"])
    fp = sum(r["test_metrics"]["false_positive"] for r in cv["fold_results"])
    fn = sum(r["test_metrics"]["false_negative"] for r in cv["fold_results"])
    tn = sum(r["test_metrics"]["true_negative"]  for r in cv["fold_results"])

    total     = tp + fp + fn + tn
    flagged   = tp + fp
    recall    = cv["aggregate"]["recall_mean"]
    precision = cv["aggregate"]["precision_mean"]
    f2        = cv["aggregate"]["f2_mean"]

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

    n_folds = cv["config"]["n_folds"]
    ax.text(0.5, 1.08,
            f"Confusion matrix: summed across {n_folds} folds",
            ha="center", va="bottom", fontsize=15, fontweight="600",
            transform=ax.transAxes)
    ax.text(0.5, 1.01,
            f"Model flagged {flagged} of {total} chunks  ({flagged / total:.0%} of total)",
            ha="center", va="bottom", fontsize=12, color=GRAY,
            transform=ax.transAxes)

    stats = [
        ("Recall",    f"{recall:.1%}"),
        ("Precision", f"{precision:.1%}"),
        ("F2",        f"{f2:.3f}"),
    ]
    for i, (lbl, val) in enumerate(stats):
        x = 0.165 + i * 0.335
        ax.text(x, -0.06, lbl,
                ha="center", va="top", fontsize=10, color=GRAY,
                transform=ax.transAxes)
        ax.text(x, -0.16, val,
                ha="center", va="top", fontsize=16, fontweight="600",
                color="#222222", transform=ax.transAxes)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "cv_confusion_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cv = load_cv_results()
    plot_fold_metrics(cv)
    plot_aggregate_metrics(cv)
    plot_confusion_summary(cv)
    print("\nAll CV plots saved.")
