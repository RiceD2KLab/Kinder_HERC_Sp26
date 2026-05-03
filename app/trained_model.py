"""
app/trained_model.py

Inference-only predictor. Loads the saved artifacts from training
(inference_artifacts.pkl) and scores a chunked transcript CSV.

The artifacts file must contain:
    model           — fitted sklearn LogisticRegression
    threshold       — float, classification threshold
    feature_mode    — "chunk_only" or "query_conditioned"
    query_text      — str (used only in query_conditioned mode)
    query_embedding — np.ndarray or None
    embedding_model — str, sentence-transformers model name
"""

from pathlib import Path
from typing import Callable, List, Dict, Optional
import csv

# Path to the saved artifacts 
# Expected location:  Kinder_HERC_Sp26/research_labeling/outputs/inference_artifacts.pkl
_DEFAULT_ARTIFACTS = (
    Path(__file__).resolve().parent.parent / "research_labeling"  / "outputs"/ "inference_artifacts.pkl"
)



def _load_artifacts(artifacts_path: Path):
    """Load and cache the inference artifacts from disk."""
    import joblib
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"Inference artifacts not found at: {artifacts_path}\n"
            "Run the training pipeline first and copy inference_artifacts.pkl to the app directory."
        )
    return joblib.load(artifacts_path)


def run_predictions(
    chunks_csv: Path,
    artifacts_path: Optional[Path] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    """
    Score each chunk in a CSV using the trained LR model.

    Parameters
    ----------
    chunks_csv : Path
        CSV with columns: chunk_id, window_start, window_end, text
    artifacts_path : Path, optional
        Path to inference_artifacts.pkl. Defaults to model/inference_artifacts.pkl
        relative to the project root.
    log_fn : callable, optional
        Function to call with log messages.

    Returns
    -------
    list of dict
        Each dict: chunk_id, window_start, window_end, text,
                   flagged (bool), confidence (float 0-1)
    """
    import numpy as np

    def _log(msg):
        if log_fn:
            log_fn(f"      {msg}")

    artifacts_path = artifacts_path or _DEFAULT_ARTIFACTS
    _log(f"Loading model artifacts from: {artifacts_path.name}")
    arts = _load_artifacts(artifacts_path)

    model           = arts["model"]
    threshold       = arts["threshold"]
    feature_mode    = arts["feature_mode"]
    embedding_model = arts["embedding_model"]
    query_embedding = arts.get("query_embedding")   # None for chunk_only

    # ── Read chunks ────────────────────────────────────────────────────────
    rows = []
    with chunks_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        _log("Warning: no chunks found in CSV.")
        return []

    _log(f"Scoring {len(rows)} chunks  [mode={feature_mode}]")

    texts = [r["text"] for r in rows]

    # ── Embed ──────────────────────────────────────────────────────────────
    from research_labeling.research_chunk_pipeline.embedding_utils import load_embedder, encode_texts, build_query_conditioned_features

    embedder = load_embedder(embedding_model)
    chunk_embeddings = encode_texts(
        model=embedder,
        texts=texts,
        batch_size=32,
        normalize_embeddings=True,
    )

    if feature_mode == "query_conditioned" and query_embedding is not None:
        feature_matrix = build_query_conditioned_features(
            chunk_embeddings=chunk_embeddings,
            query_embedding=query_embedding,
        )
    else:
        feature_matrix = chunk_embeddings

    # ── Predict ────────────────────────────────────────────────────────────
    from research_labeling.research_chunk_pipeline.modeling import predict_positive_probabilities
    probabilities = predict_positive_probabilities(model, feature_matrix)

    # ── Build results ──────────────────────────────────────────────────────
    results = []
    for row, prob in zip(rows, probabilities):
        results.append({
            "chunk_id":     row["chunk_id"],
            "window_start": row["window_start"],
            "window_end":   row["window_end"],
            "text":         row["text"],
            "predicted_label": row["predicted_label"],
            "confidence":   round(float(prob), 4),
        })

    _log(f"(threshold={threshold:.2f})")
    return results