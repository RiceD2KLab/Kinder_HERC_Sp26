"""Extended t-SNE visualisation with district coloring and positive-cluster analysis.

Produces four outputs
---------------------
1. full_dataset_tsne.png          – original binary-label plot (red/green)
2. tsne_by_district.png           – points colored by school district;
                                     positives drawn as stars, negatives as circles
3. tsne_positive_neighborhoods.png – positives only, HDBSCAN clusters circled,
                                     each cluster labeled with an ID letter
4. positive_neighborhoods.csv     – one row per positive chunk:
                                     transcript_id, chunk_id, neighborhood label,
                                     t-SNE x/y, and the first 120 chars of text

Usage
-----
    python plot_embeddings_extended.py \\
        --transcript-data-dir "../Transcript Data"

    # Control how district names are extracted from transcript_id
    # (default: first token before the first underscore)
    python plot_embeddings_extended.py \\
        --transcript-data-dir "../Transcript Data" \\
        --district-prefix-parts 2

    # Tune HDBSCAN sensitivity (lower = more, smaller clusters)
    python plot_embeddings_extended.py \\
        --transcript-data-dir "../Transcript Data" \\
        --hdbscan-min-cluster-size 5

Notes
-----
- t-SNE is computed once and reused for all three plots.
- HDBSCAN label -1 means "noise" (unclustered positives); those points are shown
  as small grey dots in plot 3 and marked neighborhood="noise" in the CSV.
- If hdbscan is not installed the script falls back to DBSCAN automatically.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Ellipse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data_utils import build_context_text, load_all_transcripts
from embedding_utils import load_embedder


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

WHITE   = "#FFFFFF"
LIGHT   = "#F5F5F5"

# Binary label colours (plot 1)
LABEL_COLORS = {0: "#E22808", 1: "#1A9641"}
LABEL_NAMES  = {0: "Non-hit (0)", 1: "Research hit (1)"}

# Neighbourhood circle colours (plot 3) — up to 12 distinct clusters
CLUSTER_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]
NOISE_COLOR = "#AAAAAA"

# Legend badge palette for neighbourhood rows — avoids blue, orange, and green
# so it cannot be confused with the tab10 district colors (blue / orange / green).
NEIGHBORHOOD_PALETTE = [
    "#7C3AED",  # violet
    "#EC4899",  # pink
    "#EF4444",  # red
    "#0D9488",  # teal
    "#6366F1",  # indigo
    "#F43F5E",  # rose
    "#C026D3",  # fuchsia
    "#DC2626",  # crimson
    "#8B5CF6",  # purple
    "#0F766E",  # dark teal
    "#DB2777",  # deep pink
    "#4F46E5",  # deep indigo
    "#E11D48",  # rose-red
    "#A855F7",  # amethyst
    "#BE185D",  # deep rose
    "#7E22CE",  # deep purple
    "#0E7490",  # cyan-teal
]

# Human-readable labels for each neighbourhood letter (A–Q)
NEIGHBORHOOD_NAMES: dict[str, str] = {
    "A": "Election Turnout Statistics",
    "B": "Board Governance and Process",
    "C": "District Achievement Recognition and Milestones",
    "D": "Process Transparency & Decision Justification",
    "E": "CTE and Workforce Development / Labor Market Data",
    "F": "Academic Performance Metrics and Assessment Targets",
    "G": "TEA Compliance Hearings",
    "H": "District Finance, Budgets, and Funding",
    "I": "Public Testimony Data (HISD Library and Program Cuts)",
    "J": "Community Partnerships and Literacy Curriculum Advocacy",
    "K": "Enrollment Decline and Teacher Retention Rates",
    "L": "Effective Schools Framework",
    "M": "Adversarial Public Testimony",
    "N": "Board Academic Progress Narratives",
    "O": "Stakeholder Perception Surveys and Engagement Data",
    "P": "Data-Driven Public Counter-Testimony",
    "Q": "Public, Evidence-Based, Policy Critiques",
}


# ---------------------------------------------------------------------------
# District extraction
# ---------------------------------------------------------------------------

KNOWN_DISTRICTS = ["Katy_ISD", "Spring_Branch_ISD", "Houston_ISD"]

def extract_district(transcript_id: str) -> str:
    """Return the district name found in *transcript_id* via regex.

    Matches against the three known district names (case-insensitive).
    Returns 'Unknown' if none match, so the script never crashes on an
    unexpected filename.

    Inputs:
        transcript_id: Transcript filename stem (e.g. ``"Houston_ISD_2024_09"``).

    Outputs:
        One of the known district strings (e.g. ``"Houston_ISD"``) or
        ``"Unknown"`` if no known district is found.
    """
    import re
    for district in KNOWN_DISTRICTS:
        if re.search(re.escape(district), transcript_id, re.IGNORECASE):
            return district
    return "Unknown"


# ---------------------------------------------------------------------------
# Embedding / reduction helpers  (identical to original script)
# ---------------------------------------------------------------------------

def _load_and_embed(
    transcript_data_dir: Path,
    label_context_window: int = 2,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load all transcript CSVs, embed chunk text with MPNet, and return arrays.

    Inputs:
        transcript_data_dir: Directory containing one labeled CSV per meeting.
        label_context_window: Number of neighboring chunks on each side to join
                              for the CSV ``label_context_text`` column. Does not
                              affect the embeddings (which use chunk-only text).

    Outputs:
        Tuple of:
            embeddings — Float32 array of shape ``(n_chunks, 768)``.
            labels     — Int array of shape ``(n_chunks,)`` with 0/1 binary labels.
            df         — Combined DataFrame with all chunk rows and a
                         ``label_context_text`` column added.
    """
    print("Loading transcripts…")
    df = load_all_transcripts(transcript_data_dir)

    # Text used for embedding — no context, keeps each chunk's own signal clean.
    df["model_text"] = build_context_text(df, context_window=0)

    # Text shown in the CSV — reconstructs the wider window visible during
    # labeling (default ±2 neighbours = 5 × 30sec chunks ≈ 2.5 min).
    df["label_context_text"] = build_context_text(df, context_window=label_context_window)
    total_chunks = 2 * label_context_window + 1
    approx_sec   = total_chunks * 30
    print(f"  Label context window: ±{label_context_window} chunks "
          f"({total_chunks} × 30 sec ≈ {approx_sec // 60}m{approx_sec % 60:02d}s per row)")

    n_chunks = len(df)
    n_pos    = int(df["binary_hit"].sum())
    print(f"  {n_chunks:,} chunks  |  {n_pos:,} positives  ({100 * n_pos / n_chunks:.1f}%)")

    print("Embedding with MPNet (truncate at 512 tokens)…")
    embedder   = load_embedder("sentence-transformers/all-mpnet-base-v2")
    # Call model.encode() directly so sentence-transformers silently truncates
    # any text over 512 tokens.  encode_texts() from embedding_utils would
    # instead split and average oversized chunks, which changes the embedding
    # space and was found to perform worse for this visualisation.
    embeddings = embedder.encode(
        df["model_text"].astype(str).tolist(),
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    labels = df["binary_hit"].to_numpy(dtype=int)
    return embeddings, labels, df


def _reduce_tsne(embeddings: np.ndarray, pca_pre: int = 50, seed: int = 42) -> np.ndarray:
    """Reduce embeddings to 2-D using PCA pre-processing followed by t-SNE.

    Inputs:
        embeddings: Float array of shape ``(n_chunks, embedding_dim)``.
        pca_pre:    Number of PCA components to compute before t-SNE.
                    Clamped to ``min(pca_pre, embedding_dim, n_chunks - 1)``.
        seed:       Random seed for PCA and t-SNE.

    Outputs:
        Float array of shape ``(n_chunks, 2)`` with t-SNE 2-D coordinates.
    """
    n_pre = min(pca_pre, embeddings.shape[1], embeddings.shape[0] - 1)
    print(f"PCA → {n_pre}D  (t-SNE pre-processing)…")
    reduced = PCA(n_components=n_pre, random_state=seed).fit_transform(embeddings)
    print("t-SNE → 2D  (this may take a minute)…")
    return TSNE(
        n_components=2,
        perplexity=40,
        max_iter=1000,
        random_state=seed,
        init="pca",
    ).fit_transform(reduced)


# ---------------------------------------------------------------------------
# Plot 1 – original binary-label scatter
# ---------------------------------------------------------------------------

def plot_binary_labels(
    coords: np.ndarray,
    labels: np.ndarray,
    n_chunks: int,
    n_pos: int,
    output_path: Path,
    seed: int,
) -> None:
    """Scatter plot of all chunks colored red (non-hit) or green (research hit).

    Inputs:
        coords:      2-D t-SNE coordinates, shape ``(n_chunks, 2)``.
        labels:      Binary label array, shape ``(n_chunks,)``.
        n_chunks:    Total chunk count (used in the title).
        n_pos:       Positive chunk count (used in the title).
        output_path: Destination PNG path.
        seed:        Random seed shown in the title for reproducibility reference.

    Outputs:
        None — saves PNG to output_path.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LIGHT)

    for label in [0, 1]:
        mask  = labels == label
        count = int(mask.sum())
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=LABEL_COLORS[label],
            s=10 if label == 0 else 14,
            alpha=0.45 if label == 0 else 0.70,
            linewidths=0,
            label=f"{LABEL_NAMES[label]}  (n={count:,})",
            zorder=3 if label == 1 else 2,
            rasterized=True,
        )

    ax.set_title(
        f"Full dataset — true labels  (t-SNE)\n"
        f"{n_chunks:,} chunks  |  {n_pos:,} positives ({100 * n_pos / n_chunks:.1f}%)  "
        f"|  seed={seed}",
        fontsize=12, fontweight="600", pad=12,
    )
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=10, framealpha=0.9, markerscale=2.0, loc="best")
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Plot 2 – colored by school district; shape encodes pos/neg
# ---------------------------------------------------------------------------

def plot_by_district(
    coords: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
    n_chunks: int,
    n_pos: int,
    output_path: Path,
    seed: int,
) -> None:
    """Scatter plot colored by school district; ● for non-hits, ★ for research hits.

    Inputs:
        coords:      2-D t-SNE coordinates, shape ``(n_chunks, 2)``.
        labels:      Binary label array, shape ``(n_chunks,)``.
        df:          Combined chunk DataFrame with a ``transcript_id`` column.
        n_chunks:    Total chunk count (used in the title).
        n_pos:       Positive chunk count (used in the title).
        output_path: Destination PNG path.
        seed:        Random seed shown in the title for reproducibility reference.

    Outputs:
        None — saves PNG to output_path.
    """

    df = df.copy()
    df["_district"] = df["transcript_id"].apply(extract_district)

    districts     = sorted(df["_district"].unique())
    n_districts   = len(districts)

    # Build a colour map.  Use a qualitative palette; cycle if > 10.
    cmap = plt.cm.get_cmap("tab10" if n_districts <= 10 else "tab20")
    district_color = {d: cmap(i % cmap.N) for i, d in enumerate(districts)}

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LIGHT)

    colors_all = np.array([district_color[d] for d in df["_district"]])

    # --- negatives: small translucent circles ---
    neg_mask = labels == 0
    ax.scatter(
        coords[neg_mask, 0], coords[neg_mask, 1],
        c=colors_all[neg_mask],
        s=10, alpha=0.30, linewidths=0,
        marker="o", zorder=2, rasterized=True,
        label="_nolegend_",
    )

    # --- positives: larger stars ---
    pos_mask = labels == 1
    ax.scatter(
        coords[pos_mask, 0], coords[pos_mask, 1],
        c=colors_all[pos_mask],
        s=60, alpha=0.85, linewidths=0.4,
        marker="*", zorder=4, rasterized=True,
        label="_nolegend_",
    )

    # Legend: one patch per district (color) + shape legend
    legend_patches = [
        mpatches.Patch(color=district_color[d], label=d)
        for d in districts
    ]
    # shape legend entries
    shape_neg = plt.Line2D(
        [], [], marker="o", linestyle="None", color="grey",
        markersize=6, alpha=0.6, label="Non-hit (●)",
    )
    shape_pos = plt.Line2D(
        [], [], marker="*", linestyle="None", color="grey",
        markersize=10, alpha=0.9, label="Research hit (★)",
    )
    ax.legend(
        handles=legend_patches + [shape_neg, shape_pos],
        fontsize=8,
        framealpha=0.85,
        loc="best",
        ncol=max(1, n_districts // 12 + 1),
        title="District",
        title_fontsize=9,
    )

    ax.set_title(
        f"t-SNE — colored by school district\n"
        f"{n_chunks:,} chunks  |  {n_pos:,} positives  |  {n_districts} districts  "
        f"|  seed={seed}",
        fontsize=12, fontweight="600", pad=12,
    )
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Neighbourhood clustering helpers
# ---------------------------------------------------------------------------

def _cluster_positives(
    pos_coords: np.ndarray,
    min_cluster_size: int = 8,
    min_samples: int | None = None,
) -> np.ndarray:
    """Cluster positive chunks in 2-D t-SNE space using HDBSCAN or DBSCAN.

    Tries hdbscan first; falls back to sklearn DBSCAN if hdbscan is not installed.

    Inputs:
        pos_coords:       2-D coordinates of positive-class chunks only,
                          shape ``(n_positives, 2)``.
        min_cluster_size: Minimum cluster size for HDBSCAN / DBSCAN.
                          Lower values produce more, smaller clusters.
        min_samples:      HDBSCAN min_samples parameter. Defaults to
                          ``max(3, min_cluster_size // 3)``.

    Outputs:
        Int array of shape ``(n_positives,)`` with cluster labels.
        ``-1`` indicates noise (point not assigned to any cluster).
    """
    try:
        import hdbscan
        min_s = min_samples or max(3, min_cluster_size // 3)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_s,
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(pos_coords)
        algo   = "HDBSCAN"
    except ImportError:
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        scaled = StandardScaler().fit_transform(pos_coords)
        eps    = 0.8
        labels = DBSCAN(eps=eps, min_samples=min_cluster_size // 2).fit_predict(scaled)
        algo   = "DBSCAN"

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    print(f"  {algo} → {n_clusters} neighbourhoods, {n_noise} noise points")
    return labels


def _confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    n_std: float = 2.0,
    color: str = "blue",
    alpha: float = 0.18,
    lw: float = 1.8,
) -> None:
    """Draw a covariance ellipse enclosing points (x, y) on a matplotlib Axes.

    Inputs:
        x:     X-coordinates of the points to enclose.
        y:     Y-coordinates of the points to enclose.
        ax:    Matplotlib Axes to draw on.
        n_std: Radius of the ellipse in units of standard deviations.
               2.0 ≈ 95% confidence interval for a bivariate normal.
        color: Fill and edge color of the ellipse (hex string or named color).
        alpha: Transparency of the filled ellipse body.
        lw:    Line width of the ellipse edge.

    Outputs:
        None — draws the ellipse patch directly onto ax.
    """
    if len(x) < 2:
        return
    cov    = np.cov(x, y)
    mean_x, mean_y = x.mean(), y.mean()
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort by largest eigenvalue
    order  = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    angle  = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width  = 2 * n_std * np.sqrt(eigvals[0])
    height = 2 * n_std * np.sqrt(eigvals[1])

    ellipse = Ellipse(
        xy=(mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=lw,
        zorder=5,
    )
    ax.add_patch(ellipse)


# ---------------------------------------------------------------------------
# Plot 3 + CSV – positive neighbourhoods
# ---------------------------------------------------------------------------

def plot_positive_neighborhoods(
    coords: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
    min_cluster_size: int,
    output_plot: Path,
    output_csv: Path,
    label_context_window: int = 2,
) -> None:
    """Scatter positive chunks colored by HDBSCAN cluster, with ellipses and a CSV.

    Inputs:
        coords:              2-D t-SNE coordinates for all chunks, shape ``(n_chunks, 2)``.
        labels:              Binary label array, shape ``(n_chunks,)``.
        df:                  Combined chunk DataFrame (used for transcript_id, chunk_id, text).
        min_cluster_size:    Minimum cluster size passed to ``_cluster_positives``.
        output_plot:         Destination PNG path for the scatter plot.
        output_csv:          Destination CSV path for the per-positive neighborhood table.
        label_context_window: Context window used when loading ``df``; stored in the CSV
                              for reference but does not change the embeddings.

    Outputs:
        None — saves PNG to output_plot and CSV to output_csv; also prints a
        per-neighborhood text summary to stdout.

    Side effects:
        Returns ``(cluster_labels, cluster_name)`` as a tuple so the caller can
        pass them to subsequent plots that overlay neighborhood ellipses.
    """

    pos_mask   = labels == 1
    pos_coords = coords[pos_mask]
    pos_df     = df[pos_mask].copy().reset_index(drop=True)

    cluster_labels = _cluster_positives(pos_coords, min_cluster_size=min_cluster_size)
    pos_df["neighborhood"] = cluster_labels

    unique_clusters = sorted(c for c in set(cluster_labels) if c != -1)
    n_clusters      = len(unique_clusters)

    # Map cluster int → letter label  (A, B, C …)
    cluster_name = {c: chr(65 + i) for i, c in enumerate(unique_clusters)}
    cluster_name[-1] = "noise"
    pos_df["neighborhood_label"] = pos_df["neighborhood"].map(cluster_name)

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LIGHT)

    # noise points
    noise_mask = cluster_labels == -1
    if noise_mask.any():
        ax.scatter(
            pos_coords[noise_mask, 0], pos_coords[noise_mask, 1],
            c=NOISE_COLOR, s=18, alpha=0.45, linewidths=0,
            marker="o", zorder=3, label=f"Noise  (n={noise_mask.sum()})",
            rasterized=True,
        )

    legend_handles = []
    for c in unique_clusters:
        mask  = cluster_labels == c
        color = CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)]
        name  = cluster_name[c]
        n     = int(mask.sum())

        ax.scatter(
            pos_coords[mask, 0], pos_coords[mask, 1],
            c=color, s=30, alpha=0.85, linewidths=0,
            marker="*", zorder=4, rasterized=True,
        )

        # Confidence ellipse
        _confidence_ellipse(
            pos_coords[mask, 0], pos_coords[mask, 1],
            ax=ax, n_std=2.0, color=color, alpha=0.15, lw=2.0,
        )

        # Centroid label
        cx, cy = pos_coords[mask, 0].mean(), pos_coords[mask, 1].mean()
        ax.text(
            cx, cy, name,
            fontsize=13, fontweight="bold", color=color,
            ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.8, lw=1.2),
        )

        legend_handles.append(
            plt.Line2D(
                [], [], marker="*", linestyle="None",
                color=color, markersize=9,
                label=f"Neighbourhood {name}  (n={n})",
            )
        )

    if noise_mask.any():
        legend_handles.append(
            plt.Line2D(
                [], [], marker="o", linestyle="None",
                color=NOISE_COLOR, markersize=6, alpha=0.6,
                label=f"Noise  (n={int(noise_mask.sum())})",
            )
        )

    ax.legend(
        handles=legend_handles,
        fontsize=9, framealpha=0.9, loc="best",
        title="Positive neighbourhoods", title_fontsize=9,
    )
    ax.set_title(
        f"t-SNE — positive-chunk neighbourhoods\n"
        f"{int(pos_mask.sum()):,} positives  |  {n_clusters} neighbourhoods  "
        f"(shaded = 2-σ ellipse)",
        fontsize=12, fontweight="600", pad=12,
    )
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved → {output_plot}")

    # ------------------------------------------------------------------ CSV
    pos_df["tsne_x"] = pos_coords[:, 0]
    pos_df["tsne_y"] = pos_coords[:, 1]

    out_cols = [
        "neighborhood_label", "neighborhood",
        "transcript_id", "chunk_id",
        "tsne_x", "tsne_y",
        "text",   # full original chunk text, exactly as labelled
    ]
    # Keep any extra columns that exist
    for col in ["window_start", "window_end"]:
        if col in pos_df.columns:
            out_cols.append(col)

    csv_df = pos_df[out_cols].sort_values(["neighborhood_label", "transcript_id", "chunk_id"])
    csv_df.to_csv(output_csv, index=False)
    print(f"  Saved → {output_csv}")

    # ------------------------------------------------------------------ summary
    _print_neighborhood_summary(pos_df, cluster_name)

    return cluster_labels, cluster_name


def _print_neighborhood_summary(pos_df: pd.DataFrame, cluster_name: dict) -> None:
    """Print a readable per-neighbourhood breakdown to stdout.

    Inputs:
        pos_df:       DataFrame of positive chunks with ``neighborhood_label``,
                      ``transcript_id``, ``chunk_id``, and ``text`` columns.
        cluster_name: Dict mapping integer cluster ID → letter label (e.g. ``{0: "A"}``).

    Outputs:
        None — prints to stdout.
    """
    print("\n" + "=" * 70)
    print("NEIGHBOURHOOD SUMMARY")
    print("=" * 70)
    for label in sorted(pos_df["neighborhood_label"].unique()):
        if label == "noise":
            continue
        sub = pos_df[pos_df["neighborhood_label"] == label]
        print(f"\n── Neighbourhood {label}  ({len(sub)} positives) ──")
        print(f"   Transcripts: {', '.join(sorted(sub['transcript_id'].unique()))}")
        # Print a short preview of each chunk's text for console readability
        for _, row in sub.iterrows():
            preview = textwrap.shorten(str(row["text"]), width=120, placeholder="…")
            print(f"   [{row['transcript_id']} / chunk {row['chunk_id']}]  {preview}")

    noise_sub = pos_df[pos_df["neighborhood_label"] == "noise"]
    if len(noise_sub):
        print(f"\n── Noise  ({len(noise_sub)} points — not in any neighbourhood) ──")


# ---------------------------------------------------------------------------
# Plot 4 – neighbourhoods with negatives shown in background
# ---------------------------------------------------------------------------

def plot_neighborhoods_with_negatives(
    coords: np.ndarray,
    labels: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_name: dict,
    df: pd.DataFrame,
    n_chunks: int,
    n_pos: int,
    output_path: Path,
    output_csv: Path,
) -> None:
    """Full dataset scatter with district colors, neighborhood ellipses, and a full CSV.

    Every point is colored by its school district. Shape encodes label: circles
    for non-hits, stars for research hits. Neighborhood ellipses and letter labels
    are drawn over the positives.

    Inputs:
        coords:         2-D t-SNE coordinates for all chunks, shape ``(n_chunks, 2)``.
        labels:         Binary label array, shape ``(n_chunks,)``.
        cluster_labels: Cluster assignments for positive chunks only,
                        shape ``(n_positives,)``.
        cluster_name:   Dict mapping cluster int ID → letter label.
        df:             Combined chunk DataFrame with ``transcript_id`` column.
        n_chunks:       Total chunk count (used in the title).
        n_pos:          Positive chunk count (used in the title).
        output_path:    Destination PNG path for the scatter plot.
        output_csv:     Destination CSV path; one row per chunk with
                        neighborhood_label, transcript_id, chunk_id, binary_hit,
                        tsne coordinates, and text.

    Outputs:
        None — saves PNG to output_path and CSV to output_csv.
    """
    pos_mask   = labels == 1
    neg_mask   = labels == 0
    pos_coords = coords[pos_mask]
    pos_df     = df[pos_mask].copy().reset_index(drop=True)
    neg_df     = df[neg_mask].copy().reset_index(drop=True)

    unique_clusters = sorted(c for c in set(cluster_labels) if c != -1)
    n_clusters      = len(unique_clusters)

    # Build district color arrays for pos and neg separately
    districts   = sorted(df["transcript_id"].apply(extract_district).unique())
    cmap        = plt.cm.get_cmap("tab10" if len(districts) <= 10 else "tab20")
    dist_color  = {d: cmap(i % cmap.N) for i, d in enumerate(districts)}

    neg_colors = [dist_color[extract_district(t)] for t in neg_df["transcript_id"]]
    pos_colors = [dist_color[extract_district(t)] for t in pos_df["transcript_id"]]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LIGHT)

    # --- negatives: circles, district color, faded ---
    ax.scatter(
        coords[neg_mask, 0], coords[neg_mask, 1],
        c=neg_colors, s=10, alpha=0.25, linewidths=0,
        marker="o", zorder=2, rasterized=True,
    )

    # --- noise positives: stars, district color, semi-transparent ---
    noise_mask = cluster_labels == -1
    if noise_mask.any():
        noise_colors = [pos_colors[i] for i in np.where(noise_mask)[0]]
        ax.scatter(
            pos_coords[noise_mask, 0], pos_coords[noise_mask, 1],
            c=noise_colors, s=30, alpha=0.45, linewidths=0,
            marker="*", zorder=3, rasterized=True,
        )

    # --- clustered positives: stars, district color, ellipses, letter labels ---
    for c in unique_clusters:
        mask  = cluster_labels == c
        name  = cluster_name[c]
        c_coords = pos_coords[mask]
        c_colors = [pos_colors[i] for i in np.where(mask)[0]]

        ax.scatter(
            c_coords[:, 0], c_coords[:, 1],
            c=c_colors, s=55, alpha=0.90, linewidths=0,
            marker="*", zorder=5, rasterized=True,
        )

        # Draw ellipse in a neutral dark color so it doesn't clash with district colors
        _confidence_ellipse(
            c_coords[:, 0], c_coords[:, 1],
            ax=ax, n_std=2.0, color="#333333", alpha=0.08, lw=1.8,
        )

        cx, cy = c_coords[:, 0].mean(), c_coords[:, 1].mean()
        ax.text(
            cx, cy, name,
            fontsize=12, fontweight="bold", color="#222222",
            ha="center", va="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#555555",
                      alpha=0.85, lw=1.2),
        )

    ax.set_title(
        f"t-SNE — neighbourhood ellipses, colored by district\n"
        f"{n_chunks:,} chunks  |  {n_pos:,} positives (★)  |  "
        f"{int(neg_mask.sum()):,} non-hits (●)  |  {n_clusters} neighbourhoods",
        fontsize=16, fontweight="600", pad=12,
    )
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved → {output_path}")

    # ------------------------------------------------------------------ CSV
    # Every chunk gets a neighbourhood_label:
    #   positives → their cluster letter (A–Q), "noise", or cluster letter
    #   negatives → "negative"
    full_df = df.copy().reset_index(drop=True)
    full_df["tsne_x"] = coords[:, 0]
    full_df["tsne_y"] = coords[:, 1]

    pos_indices = np.where(labels == 1)[0]
    neighborhood_col = np.array(["negative"] * len(full_df), dtype=object)
    for i, global_idx in enumerate(pos_indices):
        neighborhood_col[global_idx] = cluster_name[cluster_labels[i]]
    full_df["neighborhood_label"] = neighborhood_col

    out_cols = [
        "neighborhood_label",
        "transcript_id", "chunk_id",
        "binary_hit",
        "tsne_x", "tsne_y",
        "text",
    ]
    for col in ["window_start", "window_end"]:
        if col in full_df.columns:
            out_cols.append(col)

    csv_df = full_df[out_cols].sort_values(
        ["neighborhood_label", "transcript_id", "chunk_id"]
    )
    csv_df.to_csv(output_csv, index=False)
    print(f"  Saved → {output_csv}")


def save_district_legend(df: pd.DataFrame, output_path: Path) -> None:
    """Save a standalone PNG showing the district color legend.

    Includes one swatch per district plus a shape key (● non-hit, ★ hit).
    Sized to fit the content — no wasted whitespace.

    Inputs:
        df:          Combined chunk DataFrame with a ``transcript_id`` column.
        output_path: Destination PNG path.

    Outputs:
        None — saves PNG to output_path.
    """
    districts  = sorted(df["transcript_id"].apply(extract_district).unique())
    n_districts = len(districts)
    cmap       = plt.cm.get_cmap("tab10" if n_districts <= 10 else "tab20")
    dist_color = {d: cmap(i % cmap.N) for i, d in enumerate(districts)}

    # Layout: one row per district + one separator row + two shape-key rows
    n_rows     = n_districts + 1 + 2
    row_h      = 0.40   # inches per row
    fig_h      = n_rows * row_h + 0.5
    fig_w      = 3.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows)
    ax.axis("off")

    y = n_rows - 0.7   # start from top

    # Title
    ax.text(0.08, y + 0.25, "Legend", fontsize=11, fontweight="bold", va="center")
    y -= 0.2

    # District swatches
    ax.text(0.08, y, "District (color)", fontsize=8.5,
            fontweight="600", va="center", color="#444444")
    y -= row_h

    for district in districts:
        color = dist_color[district]
        # Color swatch rectangle
        rect = mpatches.FancyBboxPatch(
            (0.08, y - 0.12), 0.18, 0.26,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="none",
        )
        ax.add_patch(rect)
        ax.text(0.32, y + 0.01, district, fontsize=9, va="center")
        y -= row_h

    # Separator
    ax.axhline(y + row_h * 0.6, xmin=0.05, xmax=0.95,
               color="#CCCCCC", linewidth=0.8)
    y -= row_h * 0.3

    # Shape key header
    ax.text(0.08, y, "Point type (shape)", fontsize=8.5,
            fontweight="600", va="center", color="#444444")
    y -= row_h

    # Non-hit circle
    ax.scatter([0.17], [y + 0.02], marker="o", s=55,
               c="#888888", linewidths=0, zorder=3)
    ax.text(0.32, y + 0.01, "Non-hit (0)", fontsize=9, va="center")
    y -= row_h

    # Research-hit star
    ax.scatter([0.17], [y + 0.02], marker="*", s=130,
               c="#888888", linewidths=0, zorder=3)
    ax.text(0.32, y + 0.01, "Research hit (1)", fontsize=9, va="center")

    plt.tight_layout(pad=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved → {output_path}")


def save_neighborhood_legend(
    cluster_labels: np.ndarray,
    cluster_name: dict,
    output_path: Path,
) -> None:
    """Save a standalone PNG card showing the top-5 neighborhoods by positive count.

    Inputs:
        cluster_labels: Cluster assignments for positive chunks, shape ``(n_positives,)``.
                        ``-1`` indicates noise.
        cluster_name:   Dict mapping cluster int ID → letter label (e.g. ``{0: "A"}``).
        output_path:    Destination PNG path.

    Outputs:
        None — saves PNG to output_path.
    """
    TOP5_COLORS = ["#EC4899", "#8B5CF6", "#EF4444", "#F59E0B", "#0D9488"]

    unique_clusters = sorted(c for c in set(cluster_labels) if c != -1)
    n_noise        = int((cluster_labels == -1).sum())
    total_assigned = int((cluster_labels != -1).sum())

    all_rows: list[tuple[str, str, int]] = []
    for c in unique_clusters:
        letter = cluster_name[c]
        desc   = NEIGHBORHOOD_NAMES.get(letter, "")
        count  = int((cluster_labels == c).sum())
        all_rows.append((letter, desc, count))
    all_rows.sort(key=lambda r: r[2], reverse=True)

    rows = [(letter, TOP5_COLORS[i], desc, count)
            for i, (letter, desc, count) in enumerate(all_rows[:5])]

    # ── Measure longest description at target font size ───────────────────────
    DESC_FONTSIZE  = 10
    SUBLABEL_FONTSIZE = 7.5
    COUNT_FONTSIZE = 9

    fig_probe = plt.figure(figsize=(20, 2))
    ax_probe  = fig_probe.add_axes([0, 0, 1, 1])
    fig_probe.canvas.draw()
    renderer  = fig_probe.canvas.get_renderer()

    max_desc_w = max(
        ax_probe.text(0, 0, desc, fontsize=DESC_FONTSIZE, fontweight="bold")
                .get_window_extent(renderer).width / fig_probe.dpi
        for _, _, desc, _ in rows
    )
    count_w = (
        ax_probe.text(0, 0, "n = 9999", fontsize=COUNT_FONTSIZE)
                .get_window_extent(renderer).width / fig_probe.dpi
    )
    plt.close(fig_probe)

    # ── Fixed layout constants (inches) ──────────────────────────────────────
    MARGIN   = 0.22   # outer page margin
    PAD      = 0.16   # inner card padding
    BADGE_D  = 0.30   # badge diameter
    GAP      = 0.12   # gap between badge and description
    COL_SEP  = 0.18   # gap between description and count

    LEFT     = MARGIN + PAD
    BADGE_CX = LEFT + BADGE_D / 2
    DESC_X0  = BADGE_CX + BADGE_D / 2 + GAP
    FIG_W    = DESC_X0 + max_desc_w + COL_SEP + count_w + PAD + MARGIN
    COUNT_X  = FIG_W - MARGIN - PAD  # right-aligned anchor

    ROW_H    = 0.52
    HEAD_H   = 0.74
    FOOT_H   = 0.30
    V_PAD    = 0.16
    FIG_H    = V_PAD + HEAD_H + len(rows) * ROW_H + FOOT_H + V_PAD

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # Card border (light purple tint, matching screenshot)
    ax.add_patch(mpatches.FancyBboxPatch(
        (MARGIN, V_PAD), FIG_W - 2 * MARGIN, FIG_H - 2 * V_PAD,
        boxstyle="round,pad=0.06",
        facecolor=WHITE, edgecolor="#C4B5FD", linewidth=1.4, zorder=0,
    ))

    # ── Header ───────────────────────────────────────────────────────────────
    top_y = FIG_H - V_PAD
    ax.text(LEFT, top_y - 0.20,
            "Top Neighborhoods",
            fontsize=13, fontweight="bold", color="#111827",
            va="center", ha="left", zorder=2)
    ax.text(LEFT, top_y - 0.48,
            f"5 largest groups  ·  {total_assigned:,} positives total  ·  {n_noise} unassigned",
            fontsize=7.5, color="#9CA3AF", va="center", ha="left", zorder=2)

    div_y = top_y - HEAD_H + 0.06
    ax.plot([MARGIN + 0.06, FIG_W - MARGIN - 0.06], [div_y, div_y],
            color="#E9EAEE", linewidth=0.8, zorder=2)

    # ── Rows ─────────────────────────────────────────────────────────────────
    for i, (letter, color, desc, count) in enumerate(rows):
        row_top = div_y - i * ROW_H
        row_bot = row_top - ROW_H
        row_mid = (row_top + row_bot) / 2

        # Alternating stripe
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle(
                (MARGIN + 0.06, row_bot + 0.03),
                FIG_W - 2 * MARGIN - 0.12, ROW_H - 0.05,
                facecolor="#F5F3FF", edgecolor="none", zorder=1,
            ))

        # Badge
        ax.add_patch(Circle(
            (BADGE_CX, row_mid), BADGE_D / 2,
            facecolor=color, edgecolor="none", zorder=3,
        ))
        ax.text(BADGE_CX, row_mid, letter,
                fontsize=10, fontweight="bold", color="white",
                ha="center", va="center", zorder=4)

        # Description (bold, dark)
        ax.text(DESC_X0, row_mid + 0.046, desc,
                fontsize=DESC_FONTSIZE, fontweight="bold", color="#111827",
                ha="left", va="center", zorder=2)

        # "Group X" sub-label
        ax.text(DESC_X0, row_mid - 0.082, f"Group {letter}",
                fontsize=SUBLABEL_FONTSIZE, color="#9CA3AF",
                ha="left", va="center", zorder=2)

        # Count — pinned to right edge
        ax.text(COUNT_X, row_mid, f"n = {count}",
                fontsize=COUNT_FONTSIZE, color="#9CA3AF",
                ha="right", va="center", zorder=2)

    # ── Footer ───────────────────────────────────────────────────────────────
    foot_top = div_y - len(rows) * ROW_H
    ax.plot([MARGIN + 0.06, FIG_W - MARGIN - 0.06], [foot_top, foot_top],
            color="#E9EAEE", linewidth=0.7, zorder=2)
    ax.text(FIG_W / 2, foot_top - FOOT_H / 2,
            f"{len(all_rows) - 5} additional groups not shown  ·  {n_noise} unassigned (noise)",
            fontsize=7, color="#C4C9D4", ha="center", va="center", zorder=2)

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved → {output_path}")




# ---------------------------------------------------------------------------
# Transcript color helpers
# ---------------------------------------------------------------------------

def _build_transcript_colormap(df: pd.DataFrame) -> dict[str, tuple]:
    """Map each unique transcript_id to a distinct RGBA colour.

    Draws from tab20 → tab20b → tab20c in sequence (up to 60 distinct
    colours) so even a dataset with ~50 transcripts gets a unique swatch.

    Inputs:
        df: Combined chunk DataFrame with a ``transcript_id`` column.

    Outputs:
        Dict mapping each unique transcript_id string to an RGBA 4-tuple.
    """
    transcripts = sorted(df["transcript_id"].unique())
    palette: list = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.cm.get_cmap(cmap_name)
        palette.extend(cmap(i) for i in range(cmap.N))
    return {tid: palette[i % len(palette)] for i, tid in enumerate(transcripts)}


def _short_transcript_name(transcript_id: str) -> str:
    """Strip leading district tokens and return a readable short name.

    ``'Katy_ISD_2023_03_Board_Meeting'`` → ``'2023 03 Board Meeting'``
    Truncates to 38 characters so it fits a legend column.

    Inputs:
        transcript_id: Full transcript identifier string (filename stem).

    Outputs:
        Shortened display string with district tokens removed.
    """
    parts = transcript_id.split("_")
    # Drop tokens that match known district name fragments
    skip = {"Katy", "ISD", "Spring", "Branch", "Houston"}
    trimmed = [p for p in parts if p not in skip]
    name = " ".join(trimmed)
    if len(name) > 38:
        name = name[:36] + "…"
    return name


# ---------------------------------------------------------------------------
# Plot 5 – neighbourhood ellipses, colored by transcript
# ---------------------------------------------------------------------------

def plot_neighborhoods_by_transcript(
    coords: np.ndarray,
    labels: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_name: dict,
    df: pd.DataFrame,
    transcript_color: dict,
    n_chunks: int,
    n_pos: int,
    output_path: Path,
) -> None:
    """Full dataset t-SNE with each point colored by its individual transcript.

    Neighborhood ellipses and letter labels are drawn over the positives,
    identical to ``plot_neighborhoods_with_negatives``, but color encodes
    individual transcript rather than district.

    Inputs:
        coords:           2-D t-SNE coordinates for all chunks, shape ``(n_chunks, 2)``.
        labels:           Binary label array, shape ``(n_chunks,)``.
        cluster_labels:   Cluster assignments for positive chunks only,
                          shape ``(n_positives,)``.
        cluster_name:     Dict mapping cluster int ID → letter label.
        df:               Combined chunk DataFrame with ``transcript_id`` column.
        transcript_color: Dict mapping transcript_id → RGBA colour tuple (from
                          ``_build_transcript_colormap``).
        n_chunks:         Total chunk count (used in the title).
        n_pos:            Positive chunk count (used in the title).
        output_path:      Destination PNG path.

    Outputs:
        None — saves PNG to output_path.
    """
    pos_mask   = labels == 1
    neg_mask   = labels == 0
    pos_coords = coords[pos_mask]
    pos_df     = df[pos_mask].copy().reset_index(drop=True)
    neg_df     = df[neg_mask].copy().reset_index(drop=True)

    unique_clusters = sorted(c for c in set(cluster_labels) if c != -1)
    n_clusters      = len(unique_clusters)

    neg_colors = [transcript_color[t] for t in neg_df["transcript_id"]]
    pos_colors = [transcript_color[t] for t in pos_df["transcript_id"]]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LIGHT)

    # Negatives — small faded circles
    ax.scatter(
        coords[neg_mask, 0], coords[neg_mask, 1],
        c=neg_colors, s=10, alpha=0.22, linewidths=0,
        marker="o", zorder=2, rasterized=True,
    )

    # Noise positives — stars, district color, semi-transparent
    noise_mask = cluster_labels == -1
    if noise_mask.any():
        noise_colors = [pos_colors[i] for i in np.where(noise_mask)[0]]
        ax.scatter(
            pos_coords[noise_mask, 0], pos_coords[noise_mask, 1],
            c=noise_colors, s=30, alpha=0.40, linewidths=0,
            marker="*", zorder=3, rasterized=True,
        )

    # Clustered positives — stars + ellipses + letter labels
    for c in unique_clusters:
        mask     = cluster_labels == c
        name     = cluster_name[c]
        c_coords = pos_coords[mask]
        c_colors = [pos_colors[i] for i in np.where(mask)[0]]

        ax.scatter(
            c_coords[:, 0], c_coords[:, 1],
            c=c_colors, s=55, alpha=0.92, linewidths=0,
            marker="*", zorder=5, rasterized=True,
        )

        _confidence_ellipse(
            c_coords[:, 0], c_coords[:, 1],
            ax=ax, n_std=2.0, color="#333333", alpha=0.07, lw=1.8,
        )

        cx, cy = c_coords[:, 0].mean(), c_coords[:, 1].mean()
        ax.text(
            cx, cy, name,
            fontsize=12, fontweight="bold", color="#1A1A1A",
            ha="center", va="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#666666",
                      alpha=0.88, lw=1.2),
        )

    ax.set_title(
        f"t-SNE — neighbourhood ellipses, colored by transcript\n"
        f"{n_chunks:,} chunks  |  {n_pos:,} positives (★)  |  "
        f"{int(neg_mask.sum()):,} non-hits (●)  |  {n_clusters} neighbourhoods",
        fontsize=12, fontweight="600", pad=12,
    )
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Transcript legend (standalone PNG)
# ---------------------------------------------------------------------------

def save_transcript_legend(
    df: pd.DataFrame,
    transcript_color: dict,
    output_path: Path,
) -> None:
    """Save a standalone legend PNG mapping each transcript to its colour.

    Transcripts are grouped by district, with a bold section header per
    district. Within each district entries are split into two columns so
    the card stays compact even with ~50 transcripts.

    Inputs:
        df:               Combined chunk DataFrame with a ``transcript_id`` column.
        transcript_color: Dict mapping transcript_id → RGBA colour tuple (from
                          ``_build_transcript_colormap``).
        output_path:      Destination PNG path.

    Outputs:
        None — saves PNG to output_path.
    """
    # Group transcripts by district, sorted
    df2 = df[["transcript_id"]].copy()
    df2["_district"] = df2["transcript_id"].apply(extract_district)
    df2 = df2.drop_duplicates("transcript_id").sort_values(
        ["_district", "transcript_id"]
    )
    districts = df2["_district"].unique()

    # Build section blocks: list of (is_header, text, color_or_None)
    # Each district → 1 header row + N transcript rows split into 2 columns
    # We'll render each district as: header spanning full width, then 2-col grid
    Section = list  # just a list of (transcript_id, short_name, color)
    sections: list[tuple[str, list]] = []
    for dist in sorted(districts):
        tids = df2[df2["_district"] == dist]["transcript_id"].tolist()
        entries = [(tid, _short_transcript_name(tid), transcript_color[tid]) for tid in tids]
        sections.append((dist.replace("_", " "), entries))

    # ── Figure geometry ───────────────────────────────────────────────────────
    SWATCH_W = 0.18   # colour swatch width
    SWATCH_H = 0.14   # colour swatch height
    ROW_H    = 0.30   # height per transcript row
    HEAD_H   = 0.42   # height of district header row
    COL_GAP  = 0.22   # gap between the two entry columns
    SEC_GAP  = 0.18   # extra gap between district sections

    FIG_W    = 8.2
    CARD_PAD = 0.22
    COL_W    = (FIG_W - 2 * CARD_PAD - COL_GAP) / 2  # width of each entry column

    # Calculate total height
    total_rows = 0
    for _, entries in sections:
        n_rows_in_sec = -(-len(entries) // 2)  # ceiling div → rows in 2-col grid
        total_rows   += HEAD_H + n_rows_in_sec * ROW_H + SEC_GAP

    GLOBAL_HEAD = 0.90
    FOOT_H      = 0.44
    V_PAD       = 0.20
    FIG_H       = V_PAD + GLOBAL_HEAD + total_rows + FOOT_H + V_PAD

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("#F8F9FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # Outer card
    ax.add_patch(mpatches.FancyBboxPatch(
        (CARD_PAD - 0.06, CARD_PAD - 0.06),
        FIG_W - 2 * (CARD_PAD - 0.06),
        FIG_H - 2 * (CARD_PAD - 0.06),
        boxstyle="round,pad=0.08",
        facecolor=WHITE, edgecolor="#E2E4EA", linewidth=1.0, zorder=0,
    ))

    n_transcripts = len(transcript_color)

    # Global header
    top_y = FIG_H - V_PAD
    ax.text(
        CARD_PAD, top_y - 0.24,
        "Transcripts",
        fontsize=13.5, fontweight="bold", color="#111827",
        va="center", ha="left", zorder=2,
    )
    ax.text(
        CARD_PAD, top_y - 0.60,
        f"{n_transcripts} transcripts  ·  {len(districts)} districts  ·  each point colored by source transcript",
        fontsize=8, color="#9CA3AF",
        va="center", ha="left", zorder=2,
    )
    div_y = top_y - GLOBAL_HEAD + 0.08
    ax.plot([CARD_PAD, FIG_W - CARD_PAD], [div_y, div_y],
            color="#E9EAEE", linewidth=0.9, zorder=2)

    cursor_y = div_y  # tracks current y as we draw downward

    for dist_name, entries in sorted(sections, key=lambda s: s[0]):
        cursor_y -= SEC_GAP * 0.5

        # District header
        ax.text(
            CARD_PAD, cursor_y - HEAD_H * 0.5,
            dist_name,
            fontsize=9.5, fontweight="bold", color="#374151",
            va="center", ha="left", zorder=2,
        )
        ax.plot(
            [CARD_PAD, FIG_W - CARD_PAD],
            [cursor_y - HEAD_H + 0.06, cursor_y - HEAD_H + 0.06],
            color="#F0F1F4", linewidth=0.7, zorder=2,
        )
        cursor_y -= HEAD_H

        # 2-column grid of transcript entries
        col_starts = [CARD_PAD, CARD_PAD + COL_W + COL_GAP]
        n_rows_sec = -(-len(entries) // 2)

        for row_i in range(n_rows_sec):
            row_top = cursor_y - row_i * ROW_H
            row_mid = row_top - ROW_H / 2

            for col_i in range(2):
                idx = row_i * 2 + col_i
                if idx >= len(entries):
                    break
                _, short_name, color = entries[idx]
                x0 = col_starts[col_i]

                # Swatch rectangle
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x0, row_mid - SWATCH_H / 2),
                    SWATCH_W, SWATCH_H,
                    boxstyle="round,pad=0.02",
                    facecolor=color, edgecolor="none", zorder=3,
                ))

                # Transcript name
                ax.text(
                    x0 + SWATCH_W + 0.10, row_mid,
                    short_name,
                    fontsize=7.8, color="#374151",
                    ha="left", va="center", zorder=2,
                )

        cursor_y -= n_rows_sec * ROW_H + SEC_GAP * 0.5

    # Footer
    ax.plot([CARD_PAD, FIG_W - CARD_PAD], [cursor_y, cursor_y],
            color="#E9EAEE", linewidth=0.8, zorder=2)
    ax.text(
        FIG_W / 2, cursor_y - FOOT_H * 0.5,
        "★ = research hit   ●  = non-hit   color encodes individual transcript",
        fontsize=7.5, color="#C4C9D4",
        ha="center", va="center", zorder=2,
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="#F8F9FC")
    plt.close()
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run the full extended t-SNE visualization pipeline.

    Inputs:
        --transcript-data-dir:    Directory containing transcript CSV files (required).
        --output-dir:             Directory to write all PNG and CSV outputs (default: plots/).
        --seed:                   Random seed for PCA / t-SNE (default: 42).
        --context-window:         Label context window size in chunks (default: 2).
        --hdbscan-min-cluster-size: Minimum cluster size for HDBSCAN (default: 8).

    Outputs:
        None — writes the following files to --output-dir:
            full_dataset_tsne.png, tsne_by_district.png,
            tsne_positive_neighborhoods.png, positive_neighborhoods.csv,
            tsne_neighborhoods_with_negatives.png, all_chunks_neighborhoods.csv,
            tsne_neighborhoods_by_transcript.png,
            legend_district_colors.png, legend_neighborhoods.png,
            legend_transcripts.png.
    """
    parser = argparse.ArgumentParser(
        description="Extended t-SNE plots: district coloring + positive neighbourhoods."
    )
    parser.add_argument(
        "--transcript-data-dir",
        type=Path, required=True,
        help="Directory containing transcript CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path, default=Path("plots"),
        help="Directory to write outputs.  Default: plots/",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed for PCA / t-SNE.  Default: 42.",
    )
    parser.add_argument(
        "--context-window",
        type=int, default=2,
        help=(
            "Number of neighbouring 30-sec chunks on each side to include in "
            "the CSV text column, matching the window used during labeling.  "
            "Default: 2  (= 5 chunks × 30sec ≈ 2.5min)."
        ),
    )
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int, default=8,
        help=(
            "Minimum number of positives to form a neighbourhood (HDBSCAN / DBSCAN).  "
            "Lower = more, smaller clusters.  Default: 8."
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & embed ──────────────────────────────────────────────────────
    embeddings, labels, df = _load_and_embed(args.transcript_data_dir, label_context_window=args.context_window)
    n_chunks = len(labels)
    n_pos    = int(labels.sum())

    # ── 2. t-SNE (computed once, reused for all plots) ───────────────────────
    tsne_coords = _reduce_tsne(embeddings, pca_pre=50, seed=args.seed)

    # ── 3. Plot 1: binary labels ─────────────────────────────────────────────
    print("\n[Plot 1] Binary labels…")
    plot_binary_labels(
        coords=tsne_coords,
        labels=labels,
        n_chunks=n_chunks,
        n_pos=n_pos,
        output_path=args.output_dir / "full_dataset_tsne.png",
        seed=args.seed,
    )

    # ── 4. Plot 2: district coloring ─────────────────────────────────────────
    print("\n[Plot 2] District coloring…")
    plot_by_district(
        coords=tsne_coords,
        labels=labels,
        df=df,
        n_chunks=n_chunks,
        n_pos=n_pos,
        output_path=args.output_dir / "tsne_by_district.png",
        seed=args.seed,
    )

    # ── 5. Plot 3 + CSV: positive neighbourhoods ─────────────────────────────
    print("\n[Plot 3] Positive neighbourhoods…")
    cluster_labels, cluster_name = plot_positive_neighborhoods(
        coords=tsne_coords,
        labels=labels,
        df=df,
        min_cluster_size=args.hdbscan_min_cluster_size,
        output_plot=args.output_dir / "tsne_positive_neighborhoods.png",
        output_csv=args.output_dir  / "positive_neighborhoods.csv",
        label_context_window=args.context_window,
    )

    # ── 6. Plot 4: neighbourhoods + negatives ────────────────────────────────
    print("\n[Plot 4] Neighbourhoods with negatives…")
    plot_neighborhoods_with_negatives(
        coords=tsne_coords,
        labels=labels,
        cluster_labels=cluster_labels,
        cluster_name=cluster_name,
        df=df,
        n_chunks=n_chunks,
        n_pos=n_pos,
        output_path=args.output_dir / "tsne_neighborhoods_with_negatives.png",
        output_csv=args.output_dir  / "all_chunks_neighborhoods.csv",
    )

    # ── 7. District legend (standalone PNG) ──────────────────────────────────
    print("\n[Legend] District color legend…")
    save_district_legend(
        df=df,
        output_path=args.output_dir / "legend_district_colors.png",
    )

    # ── 8. Neighbourhood legend (standalone PNG) ─────────────────────────────
    print("\n[Legend] Neighbourhood legend…")
    save_neighborhood_legend(
        cluster_labels=cluster_labels,
        cluster_name=cluster_name,
        output_path=args.output_dir / "legend_neighborhoods.png",
    )

    # ── 9. Build transcript colormap (shared by plot + legend) ───────────────
    transcript_color = _build_transcript_colormap(df)

    # ── 10. Plot 5: neighbourhoods colored by transcript ─────────────────────
    print("\n[Plot 5] Neighbourhoods by transcript…")
    plot_neighborhoods_by_transcript(
        coords=tsne_coords,
        labels=labels,
        cluster_labels=cluster_labels,
        cluster_name=cluster_name,
        df=df,
        transcript_color=transcript_color,
        n_chunks=n_chunks,
        n_pos=n_pos,
        output_path=args.output_dir / "tsne_neighborhoods_by_transcript.png",
    )

    # ── 11. Transcript legend (standalone PNG) ────────────────────────────────
    print("\n[Legend] Transcript color legend…")
    save_transcript_legend(
        df=df,
        transcript_color=transcript_color,
        output_path=args.output_dir / "legend_transcripts.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()