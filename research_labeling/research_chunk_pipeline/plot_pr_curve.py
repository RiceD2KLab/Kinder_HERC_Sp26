"""Plot a mean precision-recall curve with shaded range across all folds × seeds.

Crawls an experiment output directory, reads every fold_X_test_predictions.csv,
computes a per-fold PR curve, interpolates all curves onto a shared recall axis,
then plots:
    - One thin translucent line per fold (optional, off by default)
    - A solid mean curve
    - A lightly shaded band showing the min-max range across all folds × seeds
    - A dashed horizontal line at the no-skill baseline (positive rate)
    - The mean average precision annotated in the legend

Optionally overlays multiple experiments on the same axes for direct comparison.

Usage
-----
Single experiment:
    python plot_pr_curve.py \\
        --experiment-dirs "outputs/no_feature_selection" \\
        --output-path "plots/pr_curve.png"

Multiple experiments (overlay):
    python plot_pr_curve.py \\
        --experiment-dirs "outputs/logistic_regression" "outputs/xgboost_gpu" \\
        --labels "Logistic Regression" "XGBoost" \\
        --output-path "plots/pr_curve_comparison.png"

Show individual fold lines:
    python plot_pr_curve.py \\
        --experiment-dirs "outputs/no_feature_selection" \\
        --show-folds \\
        --output-path "plots/pr_curve.png"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

WHITE = "#FFFFFF"
LIGHT = "#F7F7F7"

# Color cycle for multiple experiments
EXPERIMENT_COLORS = [
    "#2C7BB6",  # blue
    "#E22808",  # red
    "#1A9641",  # green
    "#9B59B6",  # purple
    "#E67E22",  # orange
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _collect_fold_predictions(experiment_dir: Path) -> list[pd.DataFrame]:
    """Crawl experiment_dir/seed_*/fold_*_test_predictions.csv and return all.

    Inputs:
        experiment_dir: Root experiment folder containing seed_* subfolders.

    Outputs:
        List of DataFrames, one per fold per seed, each containing
        ``binary_hit`` and ``predicted_probability`` columns.

    Raises:
        FileNotFoundError: If no prediction CSVs are found.
    """
    csvs = sorted(experiment_dir.glob("seed_*/fold_*_test_predictions.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No fold prediction CSVs found under {experiment_dir}.  "
            "Run run_experiments.py first."
        )

    dfs = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path, usecols=["binary_hit", "predicted_probability"])
        df = df.dropna(subset=["binary_hit", "predicted_probability"])
        dfs.append(df)

    print(f"  {experiment_dir.name}: {len(dfs)} fold CSVs loaded")
    return dfs


# ---------------------------------------------------------------------------
# PR curve computation
# ---------------------------------------------------------------------------

def _compute_pr_curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute precision-recall curve and average precision for one fold.

    Inputs:
        df: DataFrame with ``binary_hit`` and ``predicted_probability`` columns.

    Outputs:
        Tuple of (recall_arr, precision_arr, avg_precision) where arrays are
        sorted by recall ascending for interpolation.
    """
    y_true = df["binary_hit"].astype(int).to_numpy()
    y_prob = df["predicted_probability"].to_numpy()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = float(average_precision_score(y_true, y_prob))

    # precision_recall_curve returns arrays with recall descending;
    # reverse so recall is ascending for np.interp.
    recall    = recall[::-1]
    precision = precision[::-1]

    return recall, precision, ap


def _interpolate_to_common_axis(
    fold_curves: list[tuple[np.ndarray, np.ndarray]],
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate all fold curves onto a shared recall axis.

    Inputs:
        fold_curves: List of (recall, precision) arrays per fold.
        n_points:    Number of points on the shared recall axis.

    Outputs:
        Tuple of (recall_axis, mean_precision, std_minus, std_plus).
        std_minus = mean - 1std, std_plus = mean + 1std, clipped to [0, 1].
    """
    recall_axis = np.linspace(0.0, 1.0, n_points)
    interp_matrix = np.zeros((len(fold_curves), n_points))

    for i, (recall, precision) in enumerate(fold_curves):
        interp_matrix[i] = np.interp(recall_axis, recall, precision)

    mean_prec = interp_matrix.mean(axis=0)
    std_prec  = interp_matrix.std(axis=0)

    std_minus = np.clip(mean_prec - std_prec, 0, 1)
    std_plus  = np.clip(mean_prec + std_prec, 0, 1)

    return recall_axis, mean_prec, std_minus, std_plus


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pr_curves(
    experiment_dirs: list[Path],
    labels: list[str],
    output_path: Path,
    show_folds: bool = False,
    n_points: int = 200,
    selected_recall: float | None = None,
) -> None:
    """Build and save the precision-recall curve plot.

    Inputs:
        experiment_dirs:  One or more experiment root directories to overlay.
        labels:           Display name for each experiment.
        output_path:      Destination PNG path.
        show_folds:       If True, draw a thin translucent line per fold.
        n_points:         Resolution of the shared recall axis.
        selected_recall:  If set, draws a vertical dotted red line at this
                          recall value marking the chosen operating point.

    Outputs:
        None — saves PNG to output_path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LIGHT)
    ax.grid(color=WHITE, linewidth=1.2, zorder=0)

    all_baselines: list[float] = []

    for exp_idx, (exp_dir, label) in enumerate(zip(experiment_dirs, labels)):
        color = EXPERIMENT_COLORS[exp_idx % len(EXPERIMENT_COLORS)]

        fold_dfs = _collect_fold_predictions(exp_dir)

        fold_curves: list[tuple[np.ndarray, np.ndarray]] = []
        ap_scores:   list[float] = []
        baselines:   list[float] = []

        for df in fold_dfs:
            recall, precision, ap = _compute_pr_curve(df)
            fold_curves.append((recall, precision))
            ap_scores.append(ap)
            baselines.append(float(df["binary_hit"].mean()))

        all_baselines.extend(baselines)
        mean_ap = float(np.mean(ap_scores))
        std_ap  = float(np.std(ap_scores))

        # Optional: individual fold lines
        if show_folds:
            for recall, precision in fold_curves:
                ax.plot(
                    recall, precision,
                    color=color, lw=0.8, alpha=0.20, zorder=2,
                )

        # Interpolate to common axis — returns ±1 std band
        recall_axis, mean_prec, std_minus, std_plus = _interpolate_to_common_axis(
            fold_curves, n_points=n_points
        )

        # ±1 std shaded band
        ax.fill_between(
            recall_axis, std_minus, std_plus,
            color=color, alpha=0.18, zorder=3,
            label=f"_nolegend_",
        )

        # Mean line
        n_folds_total = len(fold_curves)
        ax.plot(
            recall_axis, mean_prec,
            color=color, lw=2.5, zorder=4,
            label=f"{label}  (AP = {mean_ap:.3f} ± {std_ap:.3f},  n={n_folds_total})",
        )

    # No-skill baseline
    baseline = float(np.mean(all_baselines))
    ax.axhline(
        baseline, color="#888888", lw=1.5, ls="--", zorder=1,
        label=f"No-skill baseline  ({baseline:.3f})",
    )

    # Selected operating point — vertical dotted red line
    if selected_recall is not None:
        ax.axvline(
            selected_recall, color="#E22808", lw=1.8, ls=":", zorder=5,
            label=f"Selected operating point  (recall = {selected_recall:.3f})",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)

    n_seeds = len(list(experiment_dirs[0].glob("seed_*")))
    n_folds = len(list(experiment_dirs[0].glob("seed_*/fold_*_test_predictions.csv"))) // max(n_seeds, 1)
    ax.set_title(
        f"Precision-Recall Curve  —  mean ± 1 std\n"
        f"({n_seeds} seeds × {n_folds} folds = {n_seeds * n_folds} total folds per experiment)",
        fontsize=12, fontweight="600", pad=12,
    )
    ax.legend(fontsize=10, framealpha=0.95, loc="upper right")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"\nSaved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean precision-recall curve with shaded range across all "
            "folds × seeds of one or more experiments."
        )
    )
    parser.add_argument(
        "--experiment-dirs",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more experiment root directories "
            "(e.g. outputs/no_feature_selection).  "
            "Multiple dirs are overlaid on the same axes."
        ),
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Display labels for each experiment.  "
            "Defaults to the directory name of each experiment."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("plots/pr_curve.png"),
        help="Destination PNG path.  Default: plots/pr_curve.png",
    )
    parser.add_argument(
        "--show-folds",
        action="store_true",
        default=False,
        help="Draw a thin translucent line for every individual fold.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=200,
        help="Resolution of the shared recall axis.  Default: 200.",
    )
    parser.add_argument(
        "--selected-recall",
        type=float,
        default=None,
        help=(
            "If set, draws a vertical dotted red line at this recall value "
            "marking the chosen operating point.  Example: --selected-recall 0.896"
        ),
    )
    args = parser.parse_args()

    # Default labels to directory names
    labels = args.labels or [d.name for d in args.experiment_dirs]

    if len(labels) != len(args.experiment_dirs):
        raise ValueError(
            f"--labels has {len(labels)} entries but "
            f"--experiment-dirs has {len(args.experiment_dirs)}."
        )

    for d in args.experiment_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Experiment directory not found: {d}")

    plot_pr_curves(
        experiment_dirs=args.experiment_dirs,
        labels=labels,
        output_path=args.output_path,
        show_folds=args.show_folds,
        n_points=args.n_points,
        selected_recall=args.selected_recall,
    )


if __name__ == "__main__":
    main()
