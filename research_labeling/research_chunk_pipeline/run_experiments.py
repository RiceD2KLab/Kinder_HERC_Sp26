"""Driver script: run cv_pipeline over multiple seeds and aggregate results.

Runs the grouped K-fold CV pipeline once per seed, collecting every fold's
metrics across all seeds.  After all seeds complete it writes:

    <base_output_dir>/<experiment_name>/
        seed_<n>/
            cv_results.json
            fold_<i>_test_predictions.csv
            fold_<i>_false_positives.csv
            fold_<i>_false_negatives.csv
        aggregate/
            all_fold_results.csv      — one row per (seed, fold)
            aggregate_summary.json    — mean / std / min / max per metric
            metrics_barchart.png      — bar chart: mean ± std for all 5 metrics

All cv_pipeline flags (feature mode, feature selection, n-folds, etc.) are
passed through directly, so you can run the same experiment config across seeds
by changing only the driver-level arguments.

Usage examples
--------------
# Baseline — no feature selection, seeds 1-10
python run_experiments.py \\
    --experiment-name no_feature_selection \\
    --transcript-data-dir "../Transcript Data" \\
    --seeds 1 2 3 4 5 6 7 8 9 10

# With LASSO-BIC feature selection
python run_experiments.py \\
    --experiment-name lasso_bic \\
    --transcript-data-dir "../Transcript Data" \\
    --seeds 1 2 3 4 5 6 7 8 9 10 \\
    --use-feature-selection --lasso-criterion bic

# With LASSO-AIC, query-conditioned features, 3 seeds
python run_experiments.py \\
    --experiment-name lasso_aic_query \\
    --transcript-data-dir "../Transcript Data" \\
    --seeds 42 99 7 \\
    --use-feature-selection --lasso-criterion aic \\
    --feature-mode query_conditioned
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PipelineConfig
from cv_pipeline import run_cv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRIC_KEYS   = ("recall", "precision", "f1", "f2", "average_precision")
METRIC_LABELS = ("Recall", "Precision", "F1", "F2", "Avg. Precision")

# Color palette — matches visualize_cv_results.py / cv_tables.py
BLUE   = "#2C7BB6"
GREEN  = "#1A9641"
RED    = "#E22808"
GRAY   = "#888888"
PURPLE = "#9B59B6"
ORANGE = "#E67E22"
LIGHT  = "#F7F7F7"
WHITE  = "#FFFFFF"

METRIC_COLORS = [BLUE, GREEN, GRAY, RED, PURPLE]


# ---------------------------------------------------------------------------
# Config builder (mirrors cv_pipeline.main())
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace, output_dir: Path) -> PipelineConfig:
    """Construct a PipelineConfig from parsed CLI arguments.

    Inputs:
        args:       Parsed argparse namespace (from _build_argument_parser).
        output_dir: Seed-specific output directory.

    Outputs:
        Fully populated PipelineConfig ready for run_cv.
    """
    config = PipelineConfig(transcript_data_dir=args.transcript_data_dir)
    config.output.output_dir             = output_dir
    config.embedding.feature_mode        = args.feature_mode
    config.embedding.query_text          = args.query_text
    config.embedding.context_window      = args.context_window
    config.model.use_feature_selection   = args.use_feature_selection
    config.model.lasso_criterion         = args.lasso_criterion
    config.model.lasso_c_values          = args.lasso_c_values
    config.model.model_type                = args.model_type
    config.model.xgb_n_estimators_options  = args.xgb_n_estimators
    config.model.xgb_max_depth_options     = args.xgb_max_depth
    config.model.xgb_learning_rate_options = args.xgb_learning_rate
    config.model.xgb_device                = args.xgb_device
    return config


# ---------------------------------------------------------------------------
# Aggregation and plotting
# ---------------------------------------------------------------------------

def _aggregate_results(
    all_fold_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute mean, std, min, max for each test metric across all folds × seeds.

    Inputs:
        all_fold_rows: List of per-fold result dicts (each has test_{metric} keys).

    Outputs:
        Dict mapping metric name → {mean, std, min, max}.
    """
    df = pd.DataFrame(all_fold_rows)
    summary: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        col = f"test_{key}"
        summary[key] = {
            "mean": float(df[col].mean()),
            "std":  float(df[col].std()),
            "min":  float(df[col].min()),
            "max":  float(df[col].max()),
        }
    return summary


def _plot_metrics(
    summary: dict[str, dict[str, float]],
    output_dir: Path,
    experiment_name: str,
    n_seeds: int,
    n_folds: int,
) -> None:
    """Bar chart of mean ± std for all five metrics, styled to match project plots.

    Inputs:
        summary:         Output of _aggregate_results.
        output_dir:      Directory to write metrics_barchart.png.
        experiment_name: Used in the chart title.
        n_seeds:         Number of seeds run.
        n_folds:         Number of CV folds per seed.

    Outputs:
        None — writes metrics_barchart.png to output_dir.
    """
    means = [summary[k]["mean"] for k in METRIC_KEYS]
    stds  = [summary[k]["std"]  for k in METRIC_KEYS]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LIGHT)
    ax.grid(axis="y", color=WHITE, linewidth=1.5, zorder=0)

    bars = ax.bar(
        METRIC_LABELS, means,
        color=METRIC_COLORS,
        width=0.55,
        zorder=3,
        edgecolor=WHITE,
        linewidth=1.5,
        yerr=stds,
        capsize=6,
        error_kw={"elinewidth": 1.8, "ecolor": "#333333", "capthick": 1.8},
    )

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.025,
            f"{mean:.3f}\n±{std:.3f}",
            ha="center", va="bottom",
            fontsize=9, fontweight="600",
            linespacing=1.4,
        )

    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Aggregate test metrics: mean ± std\n"
        f"({n_seeds} seeds × {n_folds} folds = {n_seeds * n_folds} total folds"
        f"  |  {experiment_name})",
        fontsize=11, fontweight="600", pad=10,
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    path = output_dir / "metrics_barchart.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Metrics bar chart saved → {path}")


def _plot_frequencies(
    all_fold_rows: list[dict[str, Any]],
    output_dir: Path,
    experiment_name: str,
) -> dict[str, Any]:
    """Frequency bar charts for threshold and C selection across all folds × seeds.

    Includes vertical lines for mean and median of the raw (non-aggregated)
    selections, so the most stable choice can be read directly off the chart.
    Annotations are placed inside tall bars and above short ones to avoid
    overlap.

    Inputs:
        all_fold_rows:   Per-fold result dicts (must contain best_threshold, best_c).
        output_dir:      Directory to write the two PNG files.
        experiment_name: Used in chart titles.

    Outputs:
        Dict with threshold_counts and c_counts sub-dicts (value → count),
        plus mean and median for each.
    """
    df      = pd.DataFrame(all_fold_rows)
    n_total = len(df)

    def _bar_chart(
        series: pd.Series,
        title: str,
        xlabel: str,
        color: str,
        filename: str,
    ) -> dict[str, Any]:
        counts     = series.value_counts().sort_index()
        raw_values = series.astype(float)
        mean_val   = float(raw_values.mean())
        median_val = float(raw_values.median())

        # x positions are categorical strings; we need numeric index positions
        # to draw mean/median lines that may fall between bars.
        cat_values = [float(v) for v in counts.index]   # sorted numeric values
        x_labels   = [str(v) for v in counts.index]
        n_cats     = len(cat_values)

        # Interpolate a float x-position for a given numeric value.
        def _x_pos(val: float) -> float:
            if val <= cat_values[0]:
                return 0.0
            if val >= cat_values[-1]:
                return float(n_cats - 1)
            for i in range(len(cat_values) - 1):
                if cat_values[i] <= val <= cat_values[i + 1]:
                    span = cat_values[i + 1] - cat_values[i]
                    frac = (val - cat_values[i]) / span if span else 0.0
                    return i + frac
            return float(n_cats - 1)

        max_count = int(counts.max())
        ylim      = max_count * 1.55   # generous headroom so annotations don't
                                       # collide with the title

        fig, ax = plt.subplots(figsize=(max(8, n_cats * 1.2), 6))
        fig.patch.set_facecolor(WHITE)
        ax.set_facecolor(LIGHT)
        ax.grid(axis="y", color=WHITE, linewidth=1.5, zorder=0)

        bars = ax.bar(
            x_labels,
            counts.values,
            color=color,
            width=0.55,
            zorder=3,
            edgecolor=WHITE,
            linewidth=1.5,
        )

        # Annotate bars — place text inside tall bars, above short ones.
        inside_threshold = ylim * 0.18   # bars taller than this get inside text
        for bar, count in zip(bars, counts.values):
            pct      = 100 * count / n_total
            label    = f"{count}\n({pct:.0f}%)"
            cx       = bar.get_x() + bar.get_width() / 2
            bar_top  = bar.get_height()
            if bar_top >= inside_threshold:
                ax.text(cx, bar_top * 0.55, label,
                        ha="center", va="center",
                        fontsize=9, fontweight="600",
                        color=WHITE, linespacing=1.4, zorder=5)
            else:
                ax.text(cx, bar_top + ylim * 0.02, label,
                        ha="center", va="bottom",
                        fontsize=9, fontweight="600",
                        color="#333333", linespacing=1.4, zorder=5)

        # Mean and median vertical lines.
        mean_x   = _x_pos(mean_val)
        median_x = _x_pos(median_val)

        ax.axvline(mean_x,   color="#C0392B", lw=2.0, ls="--",
                   zorder=6, label=f"Mean = {mean_val:.3f}")
        ax.axvline(median_x, color="#27AE60", lw=2.0, ls=":",
                   zorder=6, label=f"Median = {median_val:.3f}")

        ax.set_ylim(0, ylim)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Times selected", fontsize=11)
        ax.set_title(
            f"{title}\n"
            f"({experiment_name}  |  {n_total} total folds)",
            fontsize=11, fontweight="600", pad=12,
        )
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.legend(fontsize=10, framealpha=0.9, loc="upper right")
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        path = output_dir / filename
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=WHITE)
        plt.close()
        print(f"  Frequency chart saved → {path}")
        return {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "mean":   mean_val,
            "median": median_val,
        }

    threshold_result = _bar_chart(
        series=df["best_threshold"],
        title="Threshold selection frequency",
        xlabel="Decision threshold",
        color=ORANGE,
        filename="threshold_frequency.png",
    )
    c_result = _bar_chart(
        series=df["best_c"],
        title="Classifier C selection frequency",
        xlabel="C  (inverse regularisation strength)",
        color=BLUE,
        filename="c_frequency.png",
    )
    return {
        "threshold_counts": threshold_result["counts"],
        "threshold_mean":   threshold_result["mean"],
        "threshold_median": threshold_result["median"],
        "c_counts":         c_result["counts"],
        "c_mean":           c_result["mean"],
        "c_median":         c_result["median"],
    }


def _plot_confusion_matrix(
    all_fold_rows: list[dict[str, Any]],
    summary: dict[str, dict[str, float]],
    output_dir: Path,
    experiment_name: str,
    n_seeds: int,
    n_folds: int,
) -> None:
    """Confusion matrix card with counts summed across all folds × seeds.

    Inputs:
        all_fold_rows:   Per-fold result dicts containing tp/fp/fn/tn counts.
        summary:         Aggregated metrics from _aggregate_results.
        output_dir:      Directory to write confusion_matrix.png.
        experiment_name: Used in the chart title.
        n_seeds:         Number of seeds run.
        n_folds:         Number of CV folds per seed.

    Outputs:
        None — writes confusion_matrix.png to output_dir.
    """
    import matplotlib.patches as mpatches

    df  = pd.DataFrame(all_fold_rows)
    tp  = int(df["test_true_positive"].sum())
    fp  = int(df["test_false_positive"].sum())
    fn  = int(df["test_false_negative"].sum())
    tn  = int(df["test_true_negative"].sum())

    total   = tp + fp + fn + tn
    flagged = tp + fp
    recall    = summary["recall"]["mean"]
    precision = summary["precision"]["mean"]
    f2        = summary["f2"]["mean"]

    fig, ax = plt.subplots(figsize=(6, 5.2))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
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

    ax.text(0.5, 1.08,
            f"Confusion matrix: summed across {n_seeds} seeds × {n_folds} folds",
            ha="center", va="bottom", fontsize=13, fontweight="600",
            transform=ax.transAxes)
    ax.text(0.5, 1.01,
            f"{experiment_name}  |  "
            f"Model flagged {flagged:,} of {total:,} chunks  ({flagged / total:.0%})",
            ha="center", va="bottom", fontsize=10, color=GRAY,
            transform=ax.transAxes)

    for i, (lbl, val) in enumerate([
        ("Recall",    f"{recall:.1%}"),
        ("Precision", f"{precision:.1%}"),
        ("F2",        f"{f2:.3f}"),
    ]):
        x = 0.165 + i * 0.335
        ax.text(x, -0.06, lbl,
                ha="center", va="top", fontsize=10, color=GRAY,
                transform=ax.transAxes)
        ax.text(x, -0.16, val,
                ha="center", va="top", fontsize=16, fontweight="600",
                color="#222222", transform=ax.transAxes)

    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Confusion matrix saved → {path}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiments(args: argparse.Namespace) -> None:
    """Run cv_pipeline over all specified seeds and write aggregated outputs.

    Inputs:
        args: Parsed argparse namespace.

    Outputs:
        None — all results written to disk under
               <base_output_dir>/<experiment_name>/.
    """
    base_dir = Path(args.base_output_dir) / args.experiment_name
    base_dir.mkdir(parents=True, exist_ok=True)

    all_fold_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        print(f"\n{'=' * 60}")
        print(f"Experiment: {args.experiment_name}  |  Seed: {seed}")
        print(f"{'=' * 60}")

        seed_dir = base_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        config = _build_config(args, seed_dir)
        config.model.random_seed = seed

        results = run_cv(
            config=config,
            n_folds=args.n_folds,
            val_fraction=args.val_fraction,
            seed=seed,
        )

        for fold_result in results["fold_results"]:
            row: dict[str, Any] = {
                "seed":                seed,
                "fold":                fold_result["fold"],
                "best_c":              fold_result["best_c"],
                "best_class_weight":   fold_result["best_class_weight"],
                "best_threshold":      fold_result["best_threshold"],
                "n_features_selected": fold_result.get("n_features_selected"),
                "lasso_c_chosen":      fold_result.get("lasso_c_chosen"),
                "n_test_transcripts":  fold_result["n_test_transcripts"],
                # Confusion matrix counts for summing across folds × seeds
                "test_true_positive":  fold_result["test_metrics"]["true_positive"],
                "test_false_positive": fold_result["test_metrics"]["false_positive"],
                "test_false_negative": fold_result["test_metrics"]["false_negative"],
                "test_true_negative":  fold_result["test_metrics"]["true_negative"],
            }
            for key in METRIC_KEYS:
                row[f"test_{key}"] = fold_result["test_metrics"][key]
                row[f"val_{key}"]  = fold_result["val_metrics"][key]
            all_fold_rows.append(row)

    # ------------------------------------------------------------------
    # Aggregate across all seeds × folds
    # ------------------------------------------------------------------
    agg_dir = base_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    all_folds_df = pd.DataFrame(all_fold_rows)
    all_folds_df.to_csv(agg_dir / "all_fold_results.csv", index=False)
    print(f"\nAll fold results saved → {agg_dir / 'all_fold_results.csv'}")

    summary = _aggregate_results(all_fold_rows)

    # Frequency charts first — returns counts to embed in the JSON.
    freq_counts = _plot_frequencies(
        all_fold_rows=all_fold_rows,
        output_dir=agg_dir,
        experiment_name=args.experiment_name,
    )

    experiment_summary = {
        "experiment":    args.experiment_name,
        "seeds":         args.seeds,
        "n_folds":       args.n_folds,
        "n_total_folds": len(all_fold_rows),
        "feature_mode":  args.feature_mode,
        "use_feature_selection": args.use_feature_selection,
        "lasso_criterion": args.lasso_criterion if args.use_feature_selection else None,
        "metrics": summary,
        "threshold_selection_counts": freq_counts["threshold_counts"],
        "threshold_mean":             freq_counts["threshold_mean"],
        "threshold_median":           freq_counts["threshold_median"],
        "c_selection_counts":         freq_counts["c_counts"],
        "c_mean":                     freq_counts["c_mean"],
        "c_median":                   freq_counts["c_median"],
    }
    summary_path = agg_dir / "aggregate_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(experiment_summary, fh, indent=2)
    print(f"Aggregate summary saved → {summary_path}")

    _plot_metrics(
        summary=summary,
        output_dir=agg_dir,
        experiment_name=args.experiment_name,
        n_seeds=len(args.seeds),
        n_folds=args.n_folds,
    )

    _plot_confusion_matrix(
        all_fold_rows=all_fold_rows,
        summary=summary,
        output_dir=agg_dir,
        experiment_name=args.experiment_name,
        n_seeds=len(args.seeds),
        n_folds=args.n_folds,
    )

    # ------------------------------------------------------------------
    # Print final summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Experiment summary: {args.experiment_name}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Total folds evaluated: {len(all_fold_rows)}")
    print(f"{'=' * 60}")
    for key in METRIC_KEYS:
        m = summary[key]
        print(f"  {key:20s}  {m['mean']:.3f} ± {m['std']:.3f}  "
              f"[{m['min']:.3f} – {m['max']:.3f}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the grouped K-fold CV pipeline over multiple seeds and "
            "aggregate results into summary CSVs, JSON, and a bar chart."
        )
    )

    # ------------------------------------------------------------------
    # Experiment-level arguments (driver-specific)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help=(
            "Name for this experiment run.  Used as a top-level subfolder "
            "under --base-output-dir.  Examples: 'no_feature_selection', "
            "'lasso_bic', 'query_conditioned'."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="One or more random seeds to run.  Example: --seeds 1 2 3 4 5.",
    )
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=Path("outputs"),
        help=(
            "Root directory for all experiment outputs.  "
            "Results go to <base_output_dir>/<experiment_name>/.  "
            "Default: outputs/"
        ),
    )

    # ------------------------------------------------------------------
    # cv_pipeline pass-through arguments (identical defaults)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--transcript-data-dir",
        type=Path,
        required=True,
        help="Directory containing transcript CSV files (one per meeting).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds.  Default: 5.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of each fold's training transcripts held out for "
            "validation.  Default: 0.2."
        ),
    )
    parser.add_argument(
        "--feature-mode",
        choices=["chunk_only", "query_conditioned"],
        default="chunk_only",
        help="chunk_only (default) or query_conditioned.",
    )
    parser.add_argument(
        "--query-text",
        type=str,
        default=(
            "How are research, data, reports, or studies used to make informed decisions?"
        ),
        help="Guiding question used when --feature-mode=query_conditioned.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=0,
        help="Neighboring chunks on each side to join before embedding.  Default: 0.",
    )
    parser.add_argument(
        "--use-feature-selection",
        action="store_true",
        default=False,
        help="Enable LASSO feature selection within each fold.",
    )
    parser.add_argument(
        "--lasso-criterion",
        choices=["aic", "bic"],
        default="bic",
        help="Information criterion for LASSO C selection.  Default: bic.",
    )
    parser.add_argument(
        "--lasso-c-values",
        type=float,
        nargs="+",
        default=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        help="LASSO C candidates for feature selection.  Default: 0.001 0.005 0.01 0.05 0.1 0.5 1.0",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic_regression", "xgboost"],
        default="logistic_regression",
        help="Model to train: logistic_regression (default) or xgboost.",
    )
    parser.add_argument(
        "--xgb-n-estimators",
        type=int,
        nargs="+",
        default=[100, 300, 500],
        help="XGBoost n_estimators candidates.  Default: 100 300 500.",
    )
    parser.add_argument(
        "--xgb-max-depth",
        type=int,
        nargs="+",
        default=[3, 5, 7],
        help="XGBoost max_depth candidates.  Default: 3 5 7.",
    )
    parser.add_argument(
        "--xgb-learning-rate",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.3],
        help="XGBoost learning_rate candidates.  Default: 0.05 0.1 0.3.",
    )
    parser.add_argument(
        "--xgb-device",
        type=str,
        default="cpu",
        help="Device for XGBoost training: 'cpu' (default) or 'cuda' for GPU.",
    )

    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()

    if not args.transcript_data_dir.exists():
        raise FileNotFoundError(
            f"Transcript data directory not found: {args.transcript_data_dir}"
        )

    run_experiments(args)


if __name__ == "__main__":
    main()
