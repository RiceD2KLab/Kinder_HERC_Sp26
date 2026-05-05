"""End-to-end supervised pipeline for research-chunk detection.

Pipeline steps
--------------
1.  Load all transcript CSVs from the configured directory.
2.  Compute per-transcript statistics and assign transcripts to
    train / val / test splits at the *transcript level* (no leakage).
3.  Optionally enrich each chunk's text with neighboring-chunk context.
4.  Embed chunk text with MPNet (all-mpnet-base-v2).
5.  Optionally concatenate a fixed query embedding to every chunk embedding
    (query_conditioned mode).
6.  Grid-search logistic regression over C and class_weight on the training
    set; select the best model and threshold on the validation set using a
    F2-based policy.
7.  Evaluate the winning model once on the held-out test set.
8.  Write reproducible artifacts to the configured output directory.

Usage (CLI)
-----------
    python pipeline.py --transcript-data-dir "../Transcript Data"

See ``python pipeline.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from config import PipelineConfig
from data_utils import (
    apply_split_assignments,
    assign_transcript_splits,
    build_context_text,
    load_all_transcripts,
    summarize_transcripts,
)
from embedding_utils import (
    build_query_conditioned_features,
    encode_single_text,
    encode_texts,
    load_embedder,
    save_embeddings,
)
from modeling import (
    ModelSelectionResult,
    evaluate_predictions,
    predict_positive_probabilities,
    select_best_logistic_model,
    train_logistic_regression,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_split_arrays(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    split_name: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Slice the combined dataframe and feature matrix for one split.

    Inputs:
        df:             Combined chunk dataframe with a ``split`` column.
        feature_matrix: Row-aligned feature matrix (same row order as ``df``).
        split_name:     One of ``"train"``, ``"val"``, or ``"test"``.

    Outputs:
        Tuple of ``(split_df, X, y)`` where X is the feature sub-matrix and
        y is the integer label array.
    """
    mask     = df["split"] == split_name
    split_df = df.loc[mask].copy().reset_index(drop=True)
    x        = feature_matrix[mask.to_numpy()]
    y        = split_df["binary_hit"].to_numpy(dtype=int)
    return split_df, x, y


def _build_feature_matrix(
    config: PipelineConfig,
    combined_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Embed chunk texts and optionally append a query embedding.

    Inputs:
        config:      Full pipeline configuration.
        combined_df: Combined dataframe with a ``model_text`` column already set.

    Outputs:
        Tuple ``(feature_matrix, chunk_embeddings, query_embedding)``.
        ``query_embedding`` is ``None`` when chunk-only mode is active.
    """
    embedder = load_embedder(config.embedding.model_name)

    chunk_embeddings = encode_texts(
        model=embedder,
        texts=combined_df["model_text"].astype(str).tolist(),
        batch_size=config.embedding.batch_size,
        normalize_embeddings=config.embedding.normalize_embeddings,
        truncate=config.embedding.truncate_embeddings,
    )

    if config.embedding.feature_mode == "chunk_only":
        return chunk_embeddings, chunk_embeddings, None

    # Query-conditioned: embed the fixed guiding question once and concatenate.
    query_embedding = encode_single_text(
        model=embedder,
        text=config.embedding.query_text,
        normalize_embeddings=config.embedding.normalize_embeddings,
    )
    feature_matrix = build_query_conditioned_features(
        chunk_embeddings=chunk_embeddings,
        query_embedding=query_embedding,
    )
    return feature_matrix, chunk_embeddings, query_embedding


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Execute the full baseline experiment and write artifacts to disk.

    Inputs:
        config: Fully populated :class:`~config.PipelineConfig`.

    Outputs:
        Dictionary summarising key experiment results (also written to
        ``metrics_summary.json`` in the output directory).
    """
    config.validate()
    config.output.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and split data
    # ------------------------------------------------------------------
    print("Loading transcripts...")
    combined_df = load_all_transcripts(config.transcript_data_dir)

    transcript_summary = summarize_transcripts(combined_df)
    split_assignments  = assign_transcript_splits(
        transcript_summary=transcript_summary,
        train_fraction=config.split.train_fraction,
        val_fraction=config.split.val_fraction,
        test_fraction=config.split.test_fraction,
        random_seed=config.split.random_seed,
        stratify=config.split.stratify,
    )
    combined_df = apply_split_assignments(combined_df, split_assignments)

    # Print a quick split summary so the user can verify balance.
    for split_name in ("train", "val", "test"):
        mask = combined_df["split"] == split_name
        n_pos = combined_df.loc[mask, "binary_hit"].sum()
        n_tot = mask.sum()
        print(
            f"  {split_name:5s}: {n_tot:5d} chunks  "
            f"({split_assignments[split_assignments['split']==split_name].shape[0]} transcripts)  "
            f"{n_pos} positives  ({100*n_pos/max(n_tot,1):.1f} %)"
        )

    # ------------------------------------------------------------------
    # Build model input text (with optional context window)
    # ------------------------------------------------------------------
    combined_df["model_text"] = build_context_text(
        df=combined_df,
        context_window=config.embedding.context_window,
    )

    # ------------------------------------------------------------------
    # Embed and (optionally) build query-conditioned features
    # ------------------------------------------------------------------
    print(f"\nEmbedding chunks with {config.embedding.model_name} ...")
    feature_matrix, chunk_embeddings, query_embedding = _build_feature_matrix(
        config, combined_df
    )
    print(f"  Feature matrix shape: {feature_matrix.shape}")

    if config.output.save_embeddings:
        save_embeddings(
            embeddings=feature_matrix,
            output_path=config.output.output_dir / config.output.embeddings_filename,
        )

    # ------------------------------------------------------------------
    # Slice into train / val / test arrays
    # ------------------------------------------------------------------
    train_df, x_train, y_train = _extract_split_arrays(combined_df, feature_matrix, "train")
    val_df,   x_val,   y_val   = _extract_split_arrays(combined_df, feature_matrix, "val")
    test_df,  x_test,  y_test  = _extract_split_arrays(combined_df, feature_matrix, "test")

    # ------------------------------------------------------------------
    # Grid search — train on train, select on val
    # When val is empty (manual C + threshold run) skip the sweep and
    # train directly with the pinned values.
    # ------------------------------------------------------------------
    manual_mode = (
        config.model.fixed_threshold is not None
        and len(config.model.c_values) == 1
        and len(x_val) == 0
        and len(x_test) == 0
    )

    if manual_mode:
        c_value      = config.model.c_values[0]
        class_weight = config.model.class_weight_options[0]
        threshold    = config.model.fixed_threshold
        print(
            f"\nManual mode — training directly with "
            f"C={c_value}, class_weight={class_weight}, threshold={threshold}"
        )
        model = train_logistic_regression(
            x_train=x_train,
            y_train=y_train,
            c_value=c_value,
            class_weight=class_weight,
            max_iter=config.model.max_iter,
            random_seed=config.model.random_seed,
            solver=config.model.solver,
        )
        selection = ModelSelectionResult(
            model=model,
            best_c=c_value,
            best_class_weight=class_weight,
            best_threshold=threshold,
            validation_metrics=None,
            validation_sweep_rows=[],
        )
    else:
        print("\nRunning model grid search...")
        selection = select_best_logistic_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            c_values=config.model.c_values,
            class_weight_options=config.model.class_weight_options,
            thresholds=config.threshold.candidate_thresholds,
            max_iter=config.model.max_iter,
            random_seed=config.model.random_seed,
            solver=config.model.solver,
        )
        # Allow the user to override the auto-selected threshold.
        if config.model.fixed_threshold is not None:
            selection = ModelSelectionResult(
                model=selection.model,
                best_c=selection.best_c,
                best_class_weight=selection.best_class_weight,
                best_threshold=config.model.fixed_threshold,
                validation_metrics=selection.validation_metrics,
                validation_sweep_rows=selection.validation_sweep_rows,
            )
            print(f"  [threshold overridden by user: {config.model.fixed_threshold}]")

    print(
        f"  Best C={selection.best_c}, "
        f"class_weight={selection.best_class_weight}, "
        f"threshold={selection.best_threshold:.2f}"
    )

    # ------------------------------------------------------------------
    # Evaluate on all three splits at the selected threshold
    # ------------------------------------------------------------------
    y_train_prob = predict_positive_probabilities(selection.model, x_train)
    train_metrics = evaluate_predictions(y_train, y_train_prob, selection.best_threshold)

    if len(x_val) > 0:
        y_val_prob  = predict_positive_probabilities(selection.model, x_val)
        val_metrics = evaluate_predictions(y_val, y_val_prob, selection.best_threshold)
    else:
        y_val_prob  = np.array([])
        val_metrics = None

    if len(x_test) > 0:
        y_test_prob  = predict_positive_probabilities(selection.model, x_test)
        test_metrics = evaluate_predictions(y_test, y_test_prob, selection.best_threshold)
    else:
        y_test_prob  = np.array([])
        test_metrics = None

    print(f"\n  Train  — recall={train_metrics.recall:.3f}  precision={train_metrics.precision:.3f}  F2={train_metrics.f2:.3f}")
    print(f"  Val    — {f'recall={val_metrics.recall:.3f}  precision={val_metrics.precision:.3f}  F2={val_metrics.f2:.3f}' if val_metrics else '(no val split)'}")
    print(f"  Test   — {f'recall={test_metrics.recall:.3f}  precision={test_metrics.precision:.3f}  F2={test_metrics.f2:.3f}' if test_metrics else '(no test split)'}")

    # ------------------------------------------------------------------
    # Build test-set prediction dataframe
    # ------------------------------------------------------------------
    test_predictions = test_df.copy()
    test_predictions["predicted_probability"] = y_test_prob
    test_predictions["predicted_label"]       = (y_test_prob >= selection.best_threshold).astype(int)
    test_predictions = test_predictions.sort_values(
        by=["transcript_id", "predicted_probability"],
        ascending=[True, False],
    ).reset_index(drop=True)

    test_predictions = test_predictions[[
        "transcript_id",
        "chunk_id",
        "window_start",
        "window_end",
        "text",
        "binary_hit",
        "predicted_probability",
        "predicted_label",
    ]]

    # ------------------------------------------------------------------
    # Run model on ALL transcripts and save full predictions
    # ------------------------------------------------------------------
    y_all_prob = predict_positive_probabilities(selection.model, feature_matrix)

    all_predictions = combined_df.copy()
    all_predictions["predicted_probability"] = y_all_prob
    all_predictions["predicted_label"]       = (y_all_prob >= selection.best_threshold).astype(int)
    all_predictions = all_predictions.sort_values(
        by=["transcript_id", "predicted_probability"],
        ascending=[True, False],
    ).reset_index(drop=True)

    # Keep only human-readable columns, source_file first.
    all_predictions = all_predictions[[
        "source_file",
        "transcript_id",
        "chunk_id",
        "window_start",
        "window_end",
        "text",
        "split",           # useful to know if this chunk was train/val/test
        "binary_hit",
        "predicted_probability",
        "predicted_label",
    ]]

    # ------------------------------------------------------------------
    # Write output artifacts
    # ------------------------------------------------------------------
    print(f"\nWriting artifacts to {config.output.output_dir} ...")

    split_assignments.to_csv(
        config.output.output_dir / config.output.transcript_split_filename,
        index=False,
    )
    if selection.validation_sweep_rows:
        pd.DataFrame(selection.validation_sweep_rows).to_csv(
            config.output.output_dir / config.output.validation_sweep_filename,
            index=False,
        )
    if len(test_df) > 0:
        test_predictions.to_csv(
            config.output.output_dir / config.output.predictions_filename,
            index=False,
        )
    all_predictions.to_csv(
        config.output.output_dir / "all_transcript_predictions.csv",
        index=False,
    )

    # Error analysis — false positives and false negatives
    error_cols = [
        "transcript_id", "chunk_id", "source_file",
        "window_start", "window_end", "text",
        "binary_hit", "predicted_probability", "predicted_label",
    ]
    false_positives = (
        all_predictions[
            (all_predictions["predicted_label"] == 1) & (all_predictions["binary_hit"] == 0)
        ]
        .sort_values("predicted_probability", ascending=False)
        .reset_index(drop=True)
        [error_cols]
    )
    false_negatives = (
        all_predictions[
            (all_predictions["predicted_label"] == 0) & (all_predictions["binary_hit"] == 1)
        ]
        .sort_values("predicted_probability", ascending=False)
        .reset_index(drop=True)
        [error_cols]
    )
    false_positives.to_csv(
        config.output.output_dir / "false_positives.csv", index=False,
    )
    false_negatives.to_csv(
        config.output.output_dir / "false_negatives.csv", index=False,
    )
    print(f"  False positives: {len(false_positives)}  |  False negatives: {len(false_negatives)}")

    # ------------------------------------------------------------------
    # Full-dataset metrics (--full-train mode only)
    # ------------------------------------------------------------------
    full_metrics = None
    if config.model.full_train_eval:
        y_all_true = combined_df["binary_hit"].to_numpy(dtype=int)
        full_metrics = evaluate_predictions(y_all_true, y_all_prob, selection.best_threshold)
        print(
            f"\n  Full dataset — recall={full_metrics.recall:.3f}  "
            f"precision={full_metrics.precision:.3f}  F2={full_metrics.f2:.3f}  "
            f"TP={full_metrics.true_positive}  FP={full_metrics.false_positive}  "
            f"FN={full_metrics.false_negative}  TN={full_metrics.true_negative}"
        )

    metrics_summary: dict[str, Any] = {
        "feature_mode":             config.embedding.feature_mode,
        "query_text":               (
            config.embedding.query_text
            if config.embedding.feature_mode == "query_conditioned"
            else None
        ),
        "best_c":                   selection.best_c,
        "best_class_weight":        str(selection.best_class_weight),
        "best_threshold":           selection.best_threshold,
        "feature_dimension":        int(feature_matrix.shape[1]),
        "chunk_embedding_dimension": int(chunk_embeddings.shape[1]),
        "query_embedding_dimension": (
            int(query_embedding.shape[0]) if query_embedding is not None else None
        ),
        "train_metrics":            train_metrics.to_dict(),
        "validation_metrics":       val_metrics.to_dict() if val_metrics is not None else None,
        "test_metrics":             test_metrics.to_dict() if test_metrics is not None else None,
        "full_dataset_metrics":     full_metrics.to_dict() if full_metrics is not None else None,
        "dataset_summary": {
            "n_total_chunks":       int(len(combined_df)),
            "n_total_positives":    int(combined_df["binary_hit"].sum()),
            "n_transcripts":        int(combined_df["transcript_id"].nunique()),
            "train_transcripts":    int((split_assignments["split"] == "train").sum()),
            "val_transcripts":      int((split_assignments["split"] == "val").sum()),
            "test_transcripts":     int((split_assignments["split"] == "test").sum()),
        },
    }

    #saving important model for app
    
    artifacts = {
        "model":           selection.model,
        "threshold":       selection.best_threshold,
        "feature_mode":    config.embedding.feature_mode,
        "query_text":      config.embedding.query_text,
        "query_embedding": query_embedding,   # None if chunk_only mode
        "embedding_model": config.embedding.model_name,
    }
    joblib.dump(artifacts, config.output.output_dir / "inference_artifacts.pkl")
    print("Saved inference_artifacts.pkl")



    metrics_path = config.output.output_dir / config.output.metrics_filename
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_summary, fh, indent=2)

    print("Done.")
    return metrics_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argument_parser() -> argparse.ArgumentParser:
    """Define all command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Recall-first MPNet + logistic regression baseline for "
            "research-mention detection in school board transcripts."
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
        help="Directory where all pipeline artifacts will be written.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=0,
        help=(
            "Number of neighboring chunks on each side to join before embedding. "
            "0 = current chunk only.  Try 1 to capture mentions that span chunk boundaries."
        ),
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
        "--save-embeddings",
        action="store_true",
        help="Persist the full feature matrix as a .npy file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split assignment and model training (default: 42).",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help=(
            "Assign transcripts to splits by pure random shuffle instead of "
            "balancing the positive rate across splits."
        ),
    )

    # --- Manual override: fix C instead of grid-searching ---
    parser.add_argument(
        "--c",
        type=float,
        default=None,
        metavar="C",
        help=(
            "Pin the logistic regression C to this single value instead of "
            "grid-searching.  E.g. --c 0.5"
        ),
    )

    # --- Manual override: fix the decision threshold ---
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="THRESH",
        help=(
            "Use this probability threshold for classification instead of "
            "auto-selecting on the validation set.  E.g. --threshold 0.35"
        ),
    )

    # --- Manual override: class weight ---
    parser.add_argument(
        "--class-weight",
        type=str,
        default=None,
        choices=["balanced", "none"],
        help=(
            "Pin the class_weight instead of sweeping.  "
            "Pass 'balanced' (default grid value) or 'none' for uniform weighting."
        ),
    )

    # --- Split fraction control ---
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=None,
        help="Fraction of transcripts for training (default: 0.6).  Must sum to 1 with val/test.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Fraction of transcripts for validation (default: 0.2).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help="Fraction of transcripts for testing (default: 0.2).",
    )
    parser.add_argument(
        "--full-train",
        action="store_true",
        help=(
            "Train on all transcripts (no val/test split) and report metrics "
            "on the full dataset.  Requires --c and --threshold to be set."
        ),
    )

    return parser


def main() -> None:
    """Parse CLI arguments, run the pipeline, and print the metrics summary."""
    args = _build_argument_parser().parse_args()

    config = PipelineConfig(transcript_data_dir=args.transcript_data_dir)
    config.output.output_dir          = args.output_dir
    config.embedding.context_window   = args.context_window
    config.embedding.feature_mode     = args.feature_mode
    config.embedding.query_text       = args.query_text
    config.output.save_embeddings     = args.save_embeddings
    config.split.random_seed          = args.seed
    config.model.random_seed          = args.seed
    config.split.stratify             = not args.no_stratify

    # --- Fix C value (skip grid search) ---
    if args.c is not None:
        config.model.c_values = [args.c]

    # --- Fix threshold (skip auto-selection) ---
    if args.threshold is not None:
        config.model.fixed_threshold = args.threshold

    # --- Fix class weight ---
    if args.class_weight is not None:
        cw = None if args.class_weight == "none" else args.class_weight
        config.model.class_weight_options = [cw]

    # --- Split fractions (resolve any that were left unset) ---
    train_f = args.train_fraction
    val_f   = args.val_fraction
    test_f  = args.test_fraction

    # --full-train: train on everything, report metrics on full dataset.
    if args.full_train:
        if args.c is None or args.threshold is None:
            raise SystemExit("--full-train requires --c and --threshold to be set.")
        if train_f is None and val_f is None and test_f is None:
            train_f, val_f, test_f = 1.0, 0.0, 0.0
        config.model.full_train_eval = True
    # Manual mode (C + threshold both pinned, no --full-train): default to 80/0/20.
    elif args.c is not None and args.threshold is not None:
        if train_f is None and val_f is None and test_f is None:
            train_f, val_f, test_f = 0.8, 0.0, 0.2

    n_specified = sum(x is not None for x in (train_f, val_f, test_f))
    if n_specified == 1:
        raise SystemExit(
            "Specify either none or at least two of --train-fraction / "
            "--val-fraction / --test-fraction."
        )
    if n_specified == 2:
        # Infer the missing fraction so the three always sum to 1.
        if train_f is None:
            train_f = round(1.0 - val_f - test_f, 10)
        elif val_f is None:
            val_f = round(1.0 - train_f - test_f, 10)
        else:
            test_f = round(1.0 - train_f - val_f, 10)
    if n_specified >= 2:
        config.split.train_fraction = train_f
        config.split.val_fraction   = val_f
        config.split.test_fraction  = test_f

    results = run_pipeline(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
