"""Grouped K-Fold cross-validation pipeline for research-chunk detection.

Pipeline steps
--------------
1.  Load all transcript CSVs from the configured directory.
2.  Optionally enrich chunk text with neighboring-chunk context.
3.  Embed ALL chunks once with MPNet (the slow step — done only once).
4.  Run GroupKFold: each fold's test set is a disjoint set of whole transcripts,
    so no transcript ever leaks between train and test.
5.  Within each fold's training transcripts, hold out a validation slice
    (GroupShuffleSplit, also transcript-level) for hyperparameter and threshold
    selection.
6.  Train + select the best logistic model on inner-train/val.
7.  Evaluate the selected model once on the held-out test transcripts.
8.  Report per-fold metrics and aggregate (mean ± std) to stdout and JSON.

Train / test split ratios by --n-folds
---------------------------------------
    --n-folds 3  →  ~67 / 33  (closest to 70/30)
    --n-folds 4  →  ~75 / 25
    --n-folds 5  →  ~80 / 20  (default)

Exact percentages depend on the number of transcripts available.

Usage (CLI)
-----------
    python cv_pipeline.py --transcript-data-dir "../Transcript Data"
    python cv_pipeline.py --transcript-data-dir "../Transcript Data" --n-folds 3
    python cv_pipeline.py --transcript-data-dir "../Transcript Data" \\
        --n-folds 5 --seed 7 --feature-mode query_conditioned

See ``python cv_pipeline.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from config import EmbeddingConfig, ModelConfig, PipelineConfig, ThresholdConfig
from data_utils import build_context_text, load_all_transcripts
from embedding_utils import (
    build_query_conditioned_features,
    encode_single_text,
    encode_texts,
    load_embedder,
)
from modeling import (
    evaluate_predictions,
    predict_positive_probabilities,
    select_best_logistic_model,
    select_best_model,
    select_features_lasso,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_feature_matrix(
    config: PipelineConfig,
    combined_df: pd.DataFrame,
) -> np.ndarray:
    """Embed all chunk texts and optionally append a query embedding.

    Inputs:
        config:      Full pipeline configuration.
        combined_df: Combined dataframe with a ``model_text`` column already set.

    Outputs:
        Feature matrix of shape ``(n_chunks, feature_dim)``.
    """
    embedder = load_embedder(config.embedding.model_name)

    print(f"Embedding {len(combined_df)} chunks with {config.embedding.model_name} ...")
    chunk_embeddings = encode_texts(
        model=embedder,
        texts=combined_df["model_text"].astype(str).tolist(),
        batch_size=config.embedding.batch_size,
        normalize_embeddings=config.embedding.normalize_embeddings,
        truncate=config.embedding.truncate_embeddings,
    )

    if config.embedding.feature_mode == "chunk_only":
        print(f"  Feature matrix shape: {chunk_embeddings.shape}")
        return chunk_embeddings

    query_embedding = encode_single_text(
        model=embedder,
        text=config.embedding.query_text,
        normalize_embeddings=config.embedding.normalize_embeddings,
    )
    feature_matrix = build_query_conditioned_features(chunk_embeddings, query_embedding)
    print(f"  Feature matrix shape: {feature_matrix.shape}")
    return feature_matrix


def _save_fold_predictions(
    combined_df: pd.DataFrame,
    test_chunk_idx: np.ndarray,
    y_test_prob: np.ndarray,
    threshold: float,
    fold_idx: int,
    output_dir: Path,
) -> None:
    """Write per-fold test predictions, false positives, and false negatives to CSV.

    Inputs:
        combined_df:    Full dataframe of all chunks (all transcripts).
        test_chunk_idx: Integer indices into combined_df for this fold's test set.
        y_test_prob:    Predicted positive-class probabilities for the test set.
        threshold:      Decision threshold selected for this fold.
        fold_idx:       Fold number (1-based), used in output filenames.
        output_dir:     Directory to write the three CSVs.

    Outputs:
        None — writes three CSVs to output_dir:
            fold_{i}_test_predictions.csv  — all test chunks with labels + probs
            fold_{i}_false_positives.csv   — predicted 1, actual 0
            fold_{i}_false_negatives.csv   — predicted 0, actual 1
    """
    fold_df = combined_df.iloc[test_chunk_idx].copy().reset_index(drop=True)
    fold_df["predicted_probability"] = y_test_prob
    fold_df["predicted_label"] = (y_test_prob >= threshold).astype(int)

    # Keep only columns that exist — guards against optional columns being absent.
    desired_cols = [
        "transcript_id", "chunk_id", "source_file",
        "window_start", "window_end",
        "text", "binary_hit",
        "predicted_probability", "predicted_label",
    ]
    output_cols = [c for c in desired_cols if c in fold_df.columns]
    fold_df = fold_df[output_cols]

    fold_df.to_csv(output_dir / f"fold_{fold_idx}_test_predictions.csv", index=False)

    fp = (
        fold_df[(fold_df["predicted_label"] == 1) & (fold_df["binary_hit"] == 0)]
        .sort_values("predicted_probability", ascending=False)
        .reset_index(drop=True)
    )
    fp.to_csv(output_dir / f"fold_{fold_idx}_false_positives.csv", index=False)

    fn = (
        fold_df[(fold_df["predicted_label"] == 0) & (fold_df["binary_hit"] == 1)]
        .sort_values("predicted_probability", ascending=False)
        .reset_index(drop=True)
    )
    fn.to_csv(output_dir / f"fold_{fold_idx}_false_negatives.csv", index=False)

    print(
        f"  Saved fold {fold_idx} predictions  "
        f"({len(fp)} FP, {len(fn)} FN)  → {output_dir}"
    )


def _aggregate_fold_metrics(
    fold_results: list[dict[str, Any]],
    metric_keys: tuple[str, ...],
) -> dict[str, float]:
    """Compute mean and std for each metric across all folds.

    Inputs:
        fold_results: List of per-fold result dicts, each containing a
                      ``test_metrics`` sub-dict.
        metric_keys:  Metric names to aggregate.

    Outputs:
        Flat dict with ``{key}_mean`` and ``{key}_std`` entries.
    """
    aggregate: dict[str, float] = {}
    for key in metric_keys:
        values = [float(r["test_metrics"][key]) for r in fold_results]
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"]  = float(np.std(values))
    return aggregate


# ---------------------------------------------------------------------------
# Main CV entry point
# ---------------------------------------------------------------------------

def run_cv(
    config: PipelineConfig,
    n_folds: int,
    val_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Run grouped K-fold cross-validation and write results to disk.

    Inputs:
        config:       Fully populated :class:`~config.PipelineConfig`.
                      ``config.split`` fractions are not used; splits are
                      controlled by ``n_folds`` and ``val_fraction`` instead.
        n_folds:      Number of CV folds.  Each fold uses 1/n_folds of all
                      transcripts as the test set.
        val_fraction: Fraction of each fold's training transcripts to hold out
                      for hyperparameter and threshold selection.
        seed:         Random seed for inner val split and model training.

    Outputs:
        Dict with per-fold results and aggregate statistics (also written
        to ``cv_results.json`` in the output directory).
    """
    config.output.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------
    print("Loading transcripts...")
    combined_df = load_all_transcripts(config.transcript_data_dir)

    n_transcripts = combined_df["transcript_id"].nunique()
    n_chunks      = len(combined_df)
    n_positives   = int(combined_df["binary_hit"].sum())
    print(
        f"  {n_chunks} chunks  |  {n_transcripts} transcripts  |  "
        f"{n_positives} positives  ({100 * n_positives / max(n_chunks, 1):.1f} %)"
    )

    if n_folds > n_transcripts:
        raise ValueError(
            f"n_folds ({n_folds}) cannot exceed the number of transcripts "
            f"({n_transcripts})."
        )

    # Build model_text (with optional context enrichment).
    combined_df["model_text"] = build_context_text(
        combined_df, config.embedding.context_window
    )

    # ------------------------------------------------------------------
    # Embed ALL chunks once — this is the expensive step
    # ------------------------------------------------------------------
    feature_matrix = _build_feature_matrix(config, combined_df)

    # Chunk-aligned arrays for sklearn splitters.
    labels = combined_df["binary_hit"].to_numpy(dtype=int)
    groups = combined_df["transcript_id"].to_numpy()          # one group per chunk

    # Unique transcript IDs (for GroupKFold which works on sample-level groups).
    # GroupKFold.split() needs sample-level groups, not unique groups.

    # ------------------------------------------------------------------
    # K-Fold loop
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    unique_transcripts = list(dict.fromkeys(groups))          # deduplicated, order-preserving
    shuffled_transcripts = rng.permutation(unique_transcripts)
    transcript_to_rank = {t: i for i, t in enumerate(shuffled_transcripts)}
    shuffled_groups = np.array([transcript_to_rank[g] for g in groups])

    gkf = GroupKFold(n_splits=n_folds)
    fold_results: list[dict[str, Any]] = []

    for fold_idx, (train_chunk_idx, test_chunk_idx) in enumerate(
        gkf.split(feature_matrix, labels, groups=shuffled_groups), start=1
    ):
        # Transcript IDs in this fold's training pool.
        train_transcripts = list(dict.fromkeys(groups[train_chunk_idx]))
        test_transcripts  = list(dict.fromkeys(groups[test_chunk_idx]))

        print(
            f"\nFold {fold_idx}/{n_folds}  —  "
            f"train pool: {len(train_transcripts)} transcripts  |  "
            f"test: {len(test_transcripts)} transcripts"
        )

        # --------------------------------------------------------------
        # Inner train / val split (transcript-level, for model selection)
        # --------------------------------------------------------------
        # Map chunk indices back to a transcript-level array so
        # GroupShuffleSplit respects transcript boundaries.
        train_groups = groups[train_chunk_idx]

        inner_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=val_fraction,
            random_state=seed,
        )
        inner_train_pos, inner_val_pos = next(
            inner_splitter.split(
                feature_matrix[train_chunk_idx],
                labels[train_chunk_idx],
                groups=train_groups,
            )
        )

        # Convert relative positions back to absolute chunk indices.
        inner_train_idx = train_chunk_idx[inner_train_pos]
        inner_val_idx   = train_chunk_idx[inner_val_pos]

        inner_train_transcripts = list(dict.fromkeys(groups[inner_train_idx]))
        inner_val_transcripts   = list(dict.fromkeys(groups[inner_val_idx]))

        x_train = feature_matrix[inner_train_idx]
        y_train = labels[inner_train_idx]
        x_val   = feature_matrix[inner_val_idx]
        y_val   = labels[inner_val_idx]
        x_test  = feature_matrix[test_chunk_idx]
        y_test  = labels[test_chunk_idx]

        # --------------------------------------------------------------
        # Optional: LASSO feature selection
        # Fit L1-penalised LR on x_train, choose sparsity level by
        # AIC/BIC on x_val, then mask all three splits consistently.
        # This happens BEFORE C/threshold tuning so the downstream grid
        # search operates on the already-reduced feature space.
        # --------------------------------------------------------------
        n_features_selected: int   = x_train.shape[1]
        lasso_c_chosen:      float | None = None

        if config.model.use_feature_selection:
            feature_mask, lasso_c_chosen, n_features_selected = select_features_lasso(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                lasso_c_values=config.model.lasso_c_values,
                max_iter=config.model.max_iter,
                random_seed=seed,
                criterion=config.model.lasso_criterion,
            )
            x_train = x_train[:, feature_mask]
            x_val   = x_val[:,   feature_mask]
            x_test  = x_test[:,  feature_mask]
            print(
                f"  Feature selection ({config.model.lasso_criterion.upper()}): "
                f"{n_features_selected}/{feature_matrix.shape[1]} features kept  "
                f"(LASSO C={lasso_c_chosen})"
            )

        n_train_pos = int(y_train.sum())
        n_val_pos   = int(y_val.sum())
        n_test_pos  = int(y_test.sum())
        print(
            f"  inner-train: {len(inner_train_transcripts)} transcripts  "
            f"{len(x_train)} chunks  {n_train_pos} positives  "
            f"({100 * n_train_pos / max(len(x_train), 1):.1f} %)"
        )
        print(
            f"  inner-val  : {len(inner_val_transcripts)} transcripts  "
            f"{len(x_val)} chunks  {n_val_pos} positives  "
            f"({100 * n_val_pos / max(len(x_val), 1):.1f} %)"
        )
        print(
            f"  test       : {len(test_transcripts)} transcripts  "
            f"{len(x_test)} chunks  {n_test_pos} positives  "
            f"({100 * n_test_pos / max(len(x_test), 1):.1f} %)"
        )

        # --------------------------------------------------------------
        # Train and select best model on inner-train / val
        # --------------------------------------------------------------
        selection = select_best_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            config=config.model,
            thresholds=config.threshold.candidate_thresholds,
            random_seed=seed,
        )
        print(
            f"  Best params={selection.best_params}  "
            f"threshold={selection.best_threshold:.2f}"
        )

        # --------------------------------------------------------------
        # Evaluate on held-out test transcripts
        # --------------------------------------------------------------
        y_test_prob  = predict_positive_probabilities(selection.model, x_test)
        test_metrics = evaluate_predictions(y_test, y_test_prob, selection.best_threshold)

        print(
            f"  Test  — recall={test_metrics.recall:.3f}  "
            f"precision={test_metrics.precision:.3f}  "
            f"F2={test_metrics.f2:.3f}"
        )

        # Save per-fold predictions, FP, FN to disk.
        _save_fold_predictions(
            combined_df=combined_df,
            test_chunk_idx=test_chunk_idx,
            y_test_prob=y_test_prob,
            threshold=selection.best_threshold,
            fold_idx=fold_idx,
            output_dir=config.output.output_dir,
        )

        fold_results.append({
            "fold":                     fold_idx,
            "n_inner_train_transcripts": len(inner_train_transcripts),
            "n_inner_val_transcripts":  len(inner_val_transcripts),
            "n_test_transcripts":       len(test_transcripts),
            "test_transcripts":         test_transcripts,
            "model_type":               config.model.model_type,
            "best_params":              selection.best_params,
            "best_c":                   selection.best_c,
            "best_class_weight":        str(selection.best_class_weight),
            "best_threshold":           selection.best_threshold,
            "n_features_selected":      n_features_selected,
            "lasso_c_chosen":           lasso_c_chosen,
            "val_metrics":              selection.validation_metrics.to_dict(),
            "test_metrics":             test_metrics.to_dict(),
        })

    # ------------------------------------------------------------------
    # Aggregate across folds
    # ------------------------------------------------------------------
    metric_keys = ("recall", "precision", "f1", "f2", "average_precision")
    aggregate = _aggregate_fold_metrics(fold_results, metric_keys)

    print("\n" + "=" * 60)
    print(f"Cross-validation summary ({n_folds} folds)")
    print("=" * 60)
    for key in metric_keys:
        mean = aggregate[f"{key}_mean"]
        std  = aggregate[f"{key}_std"]
        print(f"  {key:20s}  {mean:.3f} ± {std:.3f}")

    # ------------------------------------------------------------------
    # Write artifacts
    # ------------------------------------------------------------------
    output: dict[str, Any] = {
        "config": {
            "n_folds":      n_folds,
            "val_fraction": val_fraction,
            "seed":         seed,
            "feature_mode": config.embedding.feature_mode,
            "query_text": (
                config.embedding.query_text
                if config.embedding.feature_mode == "query_conditioned"
                else None
            ),
            "context_window":   config.embedding.context_window,
            "n_transcripts":    n_transcripts,
            "n_chunks":         n_chunks,
            "n_positives":      n_positives,
        },
        "fold_results": fold_results,
        "aggregate":    aggregate,
    }

    results_path = config.output.output_dir / "cv_results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults written to {results_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argument_parser() -> argparse.ArgumentParser:
    """Define all command-line arguments for the CV pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Grouped K-Fold cross-validation for research-mention detection "
            "in school board transcripts.  Transcripts are never split across "
            "folds, preventing data leakage."
        )
    )
    parser.add_argument(
        "--transcript-data-dir",
        type=Path,
        required=True,
        help="Directory containing transcript CSV files (one per meeting).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where cv_results.json will be written.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help=(
            "Number of CV folds.  Determines the approximate train/test ratio "
            "per fold (1/k of transcripts become the test set): "
            "3 ≈ 67/33, 4 ≈ 75/25, 5 ≈ 80/20.  Default: 5."
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of each fold's training transcripts to hold out as a "
            "validation set for hyperparameter and threshold selection. "
            "Default: 0.2."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for inner val split and model training.  Default: 42.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["chunk_only", "query_conditioned"],
        default="chunk_only",
        help=(
            "chunk_only: embed chunk text only.  "
            "query_conditioned: concatenate chunk and query embeddings."
        ),
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
        help=(
            "Number of neighboring chunks on each side to join before embedding. "
            "0 = current chunk only."
        ),
    )
    parser.add_argument(
        "--use-feature-selection",
        action="store_true",
        default=False,
        help=(
            "Enable LASSO feature selection within each fold.  A L1-penalised "
            "logistic regression is fit on the inner-train set; the sparsity level "
            "(LASSO C) is chosen by AIC/BIC on the inner-val set.  The resulting "
            "feature mask is applied to all three splits before C/threshold tuning."
        ),
    )
    parser.add_argument(
        "--lasso-criterion",
        choices=["aic", "bic"],
        default="bic",
        help="Information criterion used to choose the LASSO C value.  Default: bic.",
    )
    parser.add_argument(
        "--lasso-c-values",
        type=float,
        nargs="+",
        default=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        help=(
            "Candidate LASSO C values (inverse regularisation strength) swept "
            "during feature selection.  Smaller C → more aggressive sparsity.  "
            "Default: 0.001 0.005 0.01 0.05 0.1 0.5 1.0"
        ),
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
    """Parse CLI arguments, run cross-validation, and print the summary."""
    args = _build_argument_parser().parse_args()

    # Build a PipelineConfig reusing all the shared sub-configs.
    # Split fractions are not used by cv_pipeline so we leave them at defaults.
    config = PipelineConfig(transcript_data_dir=args.transcript_data_dir)
    config.output.output_dir        = args.output_dir
    config.embedding.feature_mode   = args.feature_mode
    config.embedding.query_text     = args.query_text
    config.embedding.context_window = args.context_window
    config.model.random_seed        = args.seed
    config.model.use_feature_selection    = args.use_feature_selection
    config.model.lasso_criterion          = args.lasso_criterion
    config.model.lasso_c_values           = args.lasso_c_values
    config.model.model_type               = args.model_type
    config.model.xgb_n_estimators_options = args.xgb_n_estimators
    config.model.xgb_max_depth_options    = args.xgb_max_depth
    config.model.xgb_learning_rate_options = args.xgb_learning_rate
    config.model.xgb_device               = args.xgb_device

    if not config.transcript_data_dir.exists():
        raise FileNotFoundError(
            f"Transcript data directory does not exist: {config.transcript_data_dir}"
        )

    results = run_cv(
        config=config,
        n_folds=args.n_folds,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(json.dumps(results["aggregate"], indent=2))


if __name__ == "__main__":
    main()
