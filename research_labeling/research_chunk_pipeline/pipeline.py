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
    evaluate_predictions,
    predict_positive_probabilities,
    select_best_logistic_model,
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
    # ------------------------------------------------------------------
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
    print(
        f"  Best C={selection.best_c}, "
        f"class_weight={selection.best_class_weight}, "
        f"threshold={selection.best_threshold:.2f}"
    )

    # ------------------------------------------------------------------
    # Evaluate on all three splits at the selected threshold
    # ------------------------------------------------------------------
    y_train_prob = predict_positive_probabilities(selection.model, x_train)
    y_val_prob   = predict_positive_probabilities(selection.model, x_val)
    y_test_prob  = predict_positive_probabilities(selection.model, x_test)

    train_metrics = evaluate_predictions(y_train, y_train_prob, selection.best_threshold)
    val_metrics   = evaluate_predictions(y_val,   y_val_prob,   selection.best_threshold)
    test_metrics  = evaluate_predictions(y_test,  y_test_prob,  selection.best_threshold)

    print(f"\n  Train  — recall={train_metrics.recall:.3f}  precision={train_metrics.precision:.3f}  F2={train_metrics.f2:.3f}")
    print(f"  Val    — recall={val_metrics.recall:.3f}  precision={val_metrics.precision:.3f}  F2={val_metrics.f2:.3f}")
    print(f"  Test   — recall={test_metrics.recall:.3f}  precision={test_metrics.precision:.3f}  F2={test_metrics.f2:.3f}")

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
    pd.DataFrame(selection.validation_sweep_rows).to_csv(
        config.output.output_dir / config.output.validation_sweep_filename,
        index=False,
    )
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
        "validation_metrics":       val_metrics.to_dict(),
        "test_metrics":             test_metrics.to_dict(),
        "dataset_summary": {
            "n_total_chunks":       int(len(combined_df)),
            "n_total_positives":    int(combined_df["binary_hit"].sum()),
            "n_transcripts":        int(combined_df["transcript_id"].nunique()),
            "train_transcripts":    int((split_assignments["split"] == "train").sum()),
            "val_transcripts":      int((split_assignments["split"] == "val").sum()),
            "test_transcripts":     int((split_assignments["split"] == "test").sum()),
        },
    }

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
        default="query_conditioned",
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

    results = run_pipeline(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
