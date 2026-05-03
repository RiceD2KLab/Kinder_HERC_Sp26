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

Token truncation handling
--------------------------
MPNet (all-mpnet-base-v2) has a hard 512-token limit.  Chunks that exceed
this limit are handled via *chunk-and-pool*: the text is split into
overlapping sub-chunks of at most ``max_tokens`` tokens with a stride of
``stride`` tokens, each sub-chunk is embedded independently, and the
resulting vectors are averaged into a single 768-d representation.  This
ensures the full text of every chunk is captured, not just the first 512
tokens.  Chunks within the token limit are embedded normally with no
overhead.
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
# Chunk-and-pool for oversized texts
# ---------------------------------------------------------------------------

def _embed_long_text(
    model: SentenceTransformer,
    text: str,
    max_tokens: int = 512,
    stride: int = 256,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """Embed a text that may exceed the model's token limit via chunk-and-pool.

    The text is tokenized, split into overlapping windows of at most
    ``max_tokens`` tokens with a step size of ``stride`` tokens, each window
    is decoded back to a string and embedded independently, and the resulting
    vectors are averaged into a single embedding.

    For texts within the token limit this is equivalent to normal encoding
    (one window = the full text).

    Inputs:
        model:                Loaded sentence-transformer model.
        text:                 Input string (may be arbitrarily long).
        max_tokens:           Maximum tokens per sub-chunk window.
        stride:               Step size between window starts (overlap =
                              max_tokens - stride tokens).
        normalize_embeddings: Whether to L2-normalise each sub-chunk embedding
                              before averaging.  The averaged vector is also
                              L2-normalised before returning.

    Outputs:
        Float32 NumPy array of shape ``(embedding_dim,)``.
    """
    tokenizer = model.tokenizer

    # Tokenize without special tokens so we can slice cleanly.
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= max_tokens:
        # Fast path — no splitting needed.
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(embedding, dtype=np.float32)

    # Split into overlapping windows and embed each.
    sub_embeddings: list[np.ndarray] = []
    for start in range(0, len(token_ids), stride):
        window_ids = token_ids[start : start + max_tokens]
        window_text = tokenizer.decode(window_ids, skip_special_tokens=True)
        sub_emb = model.encode(
            window_text,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )
        sub_embeddings.append(np.asarray(sub_emb, dtype=np.float32))
        if start + max_tokens >= len(token_ids):
            break

    # Average sub-chunk embeddings and L2-normalise the result.
    averaged = np.mean(sub_embeddings, axis=0)
    norm = np.linalg.norm(averaged)
    if norm > 0:
        averaged = averaged / norm
    return averaged.astype(np.float32)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    max_tokens: int = 512,
    stride: int = 256,
) -> np.ndarray:
    """Encode a list of strings into a 2-D embedding matrix.

    Texts within the model's token limit are encoded in fast batched mode.
    Texts exceeding ``max_tokens`` tokens are handled via chunk-and-pool:
    split into overlapping windows, embed each, average the results.

    Inputs:
        model:                Loaded sentence-transformer model.
        texts:                Input strings to encode.
        batch_size:           Number of strings processed per forward pass
                              (applies to the normal batched path only).
        normalize_embeddings: If True, each row is L2-normalised to unit length.
        max_tokens:           Token limit above which chunk-and-pool is used.
                              Defaults to 512 (MPNet's hard limit).
        stride:               Step size in tokens between chunk-and-pool windows.
                              Smaller stride = more overlap = slower but
                              smoother coverage.  Default: 256 (50% overlap).

    Outputs:
        Float32 NumPy array of shape ``(len(texts), embedding_dim)``.
    """
    tokenizer = model.tokenizer

    # Partition texts into normal (≤ max_tokens) and long (> max_tokens).
    normal_indices: list[int] = []
    long_indices:   list[int] = []

    for i, text in enumerate(texts):
        n_tokens = len(tokenizer.encode(str(text), add_special_tokens=False))
        if n_tokens <= max_tokens:
            normal_indices.append(i)
        else:
            long_indices.append(i)

    if long_indices:
        print(
            f"  chunk-and-pool: {len(long_indices)} oversized chunks "
            f"(>{max_tokens} tokens) will be split and averaged."
        )

    # Allocate output matrix.
    # Embed one normal text to get the embedding dimension.
    sample_text = texts[normal_indices[0]] if normal_indices else texts[long_indices[0]]
    sample_emb  = model.encode(
        str(sample_text),
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    embedding_dim = len(sample_emb)
    result = np.zeros((len(texts), embedding_dim), dtype=np.float32)

    # Batch-encode normal texts.
    if normal_indices:
        normal_texts = [str(texts[i]) for i in normal_indices]
        normal_embs  = model.encode(
            normal_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=True,
        )
        for idx, emb in zip(normal_indices, normal_embs):
            result[idx] = emb

    # Chunk-and-pool long texts individually.
    for i, idx in enumerate(long_indices):
        if i % 50 == 0 and i > 0:
            print(f"    chunk-and-pool: {i}/{len(long_indices)} done...")
        result[idx] = _embed_long_text(
            model=model,
            text=str(texts[idx]),
            max_tokens=max_tokens,
            stride=stride,
            normalize_embeddings=normalize_embeddings,
        )

    return result


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
