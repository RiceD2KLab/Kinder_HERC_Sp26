"""Sentence-transformer embedding helpers and query-conditioned feature construction.

Two feature modes are supported by this pipeline:

chunk_only
    Each example is represented by the 768-d MPNet embedding of its chunk
    text.  Simple and fast.

query_conditioned  (default)
    Each example is represented by the concatenation of its chunk embedding
    and a *shared* query embedding:

        feature = [embed(chunk_text) ; embed(query_text)]   shape: (1536,)

    The query embedding is computed once and broadcast across all chunks.
    This mirrors the mentor-provided demo and frames the task explicitly as
    "how relevant is this chunk to the guiding question?" rather than asking
    the model to infer relevance on its own.

    Default query:
        "How are research, data, reports, or studies used to make informed
        decisions?"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_embedder(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer model by name or local path.

    Inputs:
        model_name: HuggingFace model identifier or local directory, e.g.
                    ``"sentence-transformers/all-mpnet-base-v2"``.

    Outputs:
        Loaded ``SentenceTransformer`` instance ready for inference.
    """
    return SentenceTransformer(model_name)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 32,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """Encode a list of strings into a 2-D embedding matrix.

    Inputs:
        model:                Loaded sentence-transformer model.
        texts:                Input strings to encode.
        batch_size:           Number of strings processed per forward pass.
        normalize_embeddings: If True, each row is L2-normalised to unit length.
                              Required for cosine-similarity comparisons and
                              generally recommended for logistic regression too.

    Outputs:
        Float32 NumPy array of shape ``(len(texts), embedding_dim)``.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def encode_single_text(
    model: SentenceTransformer,
    text: str,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """Encode a single string into a 1-D embedding vector.

    Used to embed the fixed guiding query once before building the feature
    matrix for the full dataset.

    Inputs:
        model:                Loaded sentence-transformer model.
        text:                 Single string to embed.
        normalize_embeddings: Whether to L2-normalise the output.

    Outputs:
        Float32 NumPy array of shape ``(embedding_dim,)``.
    """
    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    return np.asarray(embedding, dtype=np.float32)


# ---------------------------------------------------------------------------
# Query-conditioned feature construction
# ---------------------------------------------------------------------------

def build_query_conditioned_features(
    chunk_embeddings: np.ndarray,
    query_embedding: np.ndarray,
) -> np.ndarray:
    """Concatenate each chunk embedding with a shared query embedding.

    The resulting features encode both *what the chunk says* and *how it
    relates to the guiding question*.  The logistic regression classifier
    can then learn a decision boundary that considers both dimensions.

    Inputs:
        chunk_embeddings: 2-D array of shape ``(n_chunks, embedding_dim)``.
        query_embedding:  1-D array of shape ``(embedding_dim,)``.

    Outputs:
        2-D array of shape ``(n_chunks, 2 * embedding_dim)``.

    Raises:
        ValueError: If input shapes are incompatible.
    """
    if chunk_embeddings.ndim != 2:
        raise ValueError(
            f"chunk_embeddings must be 2-D, got shape {chunk_embeddings.shape}."
        )
    if query_embedding.ndim != 1:
        raise ValueError(
            f"query_embedding must be 1-D, got shape {query_embedding.shape}."
        )
    if chunk_embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError(
            f"Embedding dimensions must match before concatenation.  "
            f"chunk dim={chunk_embeddings.shape[1]}, query dim={query_embedding.shape[0]}."
        )

    # Repeat the query vector once per chunk, then concatenate column-wise.
    query_tiled = np.tile(query_embedding, (chunk_embeddings.shape[0], 1))
    return np.concatenate([chunk_embeddings, query_tiled], axis=1)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_embeddings(embeddings: np.ndarray, output_path: Path) -> None:
    """Save an embedding or feature matrix to a NumPy binary file.

    Inputs:
        embeddings:  Array to persist.
        output_path: Destination path (typically ``*.npy``).
                     Parent directories are created automatically.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
