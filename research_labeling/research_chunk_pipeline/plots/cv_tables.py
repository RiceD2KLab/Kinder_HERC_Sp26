"""Render CV hyperparam and test-metrics tables as a single PNG.

Usage:
    cd research_chunk_pipeline/plots
    python cv_tables.py
"""

import json
import os

import matplotlib.pyplot as plt

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs"))
SAVE_DIR    = SCRIPT_DIR

BLUE  = "#2C7BB6"
LIGHT = "#F0F4F8"
WHITE = "#FFFFFF"
GRAY  = "#888888"


def load_cv() -> dict:
    with open(os.path.join(OUTPUTS_DIR, "cv_results.json"), encoding="utf-8") as fh:
        return json.load(fh)


def render_tables(cv: dict) -> None:
    folds = cv["fold_results"]
    agg   = cv["aggregate"]

    # ── Table data ────────────────────────────────────────────────────────────
    hyper_rows = [
        [f"Fold {r['fold']}", str(r['best_c']), str(r['best_threshold'])]
        for r in folds
    ]
    metric_rows = [
        [
            f"Fold {r['fold']}",
            f"{r['test_metrics']['recall']:.3f}",
            f"{r['test_metrics']['precision']:.3f}",
            f"{r['test_metrics']['f2']:.3f}",
        ]
        for r in folds
    ]
    metric_rows.append([
        "Mean ± std",
        f"{agg['recall_mean']:.3f} ± {agg['recall_std']:.3f}",
        f"{agg['precision_mean']:.3f} ± {agg['precision_std']:.3f}",
        f"{agg['f2_mean']:.3f} ± {agg['f2_std']:.3f}",
    ])

    hyper_cols  = ["Fold", "Best C", "Threshold"]
    metric_cols = ["Fold", "Recall", "Precision", "F2"]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_h, ax_m) = plt.subplots(
        1, 2, figsize=(11, 3.6),
        gridspec_kw={"width_ratios": [1, 1.6]},
    )
    fig.patch.set_facecolor(WHITE)

    for ax in (ax_h, ax_m):
        ax.axis("off")

    def _draw_table(ax, col_labels, rows, title):
        n_rows = len(rows)
        n_cols = len(col_labels)

        # Alternate row colors; last row of metric table gets a distinct shade
        row_colors = []
        for i, row in enumerate(rows):
            if row[0].startswith("Mean"):
                row_colors.append([LIGHT] * n_cols)
            else:
                row_colors.append([WHITE if i % 2 == 0 else "#EEF2F7"] * n_cols)

        tbl = ax.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            cellColours=row_colors,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1, 1.7)

        # Header style
        for col in range(n_cols):
            cell = tbl[0, col]
            cell.set_facecolor(BLUE)
            cell.set_text_props(color=WHITE, fontweight="bold")
            cell.set_edgecolor(WHITE)

        # Body cell borders
        for (row, col), cell in tbl.get_celld().items():
            if row > 0:
                cell.set_edgecolor("#DDDDDD")
                # Bold the Mean row
                if rows[row - 1][0].startswith("Mean"):
                    cell.set_text_props(fontweight="bold")

        ax.set_title(title, fontsize=12, fontweight="600", pad=12, color="#222222")

    _draw_table(
        ax_h, hyper_cols, hyper_rows,
        "Hyperparameters selected per fold\n(tuned by F2 on val set)",
    )
    _draw_table(
        ax_m, metric_cols, metric_rows,
        "Test metrics per fold",
    )

    plt.tight_layout(pad=1.5)
    path = os.path.join(SAVE_DIR, "cv_tables.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    render_tables(load_cv())
