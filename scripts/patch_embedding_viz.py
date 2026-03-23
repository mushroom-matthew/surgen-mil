"""
Crude patch embedding cluster analysis.

Samples a small number of patches per slide, reduces to 3D via PCA + UMAP,
and applies k-means for k in [k_min, k_max]. Outputs:
  - silhouette scores printed to stdout
  - interactive 3D Plotly HTML with colour toggles

The goal is to estimate how many natural tissue phenotype clusters exist in
the UNI embedding space, to inform the head count for HybridAttentionMIL.

Usage:
    python scripts/patch_embedding_viz.py \
        --data_root /mnt/data-surgen \
        --output outputs/patch_viz.html \
        --patches_per_slide 64 \
        --n_slides 40
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import umap

from src.data.feature_provider import UniFeatureProvider


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_patches(
    provider: UniFeatureProvider,
    slide_indices: list[int],
    patches_per_slide: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        features : (N_total, D)
        labels   : (N_total,)  — slide MSI/MMR label
        cohorts  : (N_total,)  — cohort string per patch
    """
    rng = np.random.default_rng(seed)
    all_features, all_labels, all_cohorts = [], [], []

    for idx in slide_indices:
        item = provider.load_slide(idx)
        feats = item["features"]          # (N_patches, D)
        n = len(feats)
        k = min(patches_per_slide, n)
        chosen = rng.choice(n, size=k, replace=False)
        all_features.append(feats[chosen])
        all_labels.append(np.full(k, item["label"], dtype=np.int32))
        all_cohorts.append(np.array([item["cohort"]] * k))

    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_cohorts, axis=0),
    )


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce(features: np.ndarray, pca_components: int = 50, umap_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        pca50  : (N, pca_components)  — used for k-means
        umap3d : (N, 3)               — used for visualisation
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    pca = PCA(n_components=pca_components, random_state=umap_seed)
    pca50 = pca.fit_transform(scaled)

    reducer = umap.UMAP(n_components=3, random_state=umap_seed, verbose=True)
    umap3d = reducer.fit_transform(pca50)

    return pca50, umap3d


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def run_kmeans(
    pca50: np.ndarray,
    k_min: int,
    k_max: int,
    seed: int = 42,
) -> dict[int, tuple[np.ndarray, float]]:
    """
    Returns dict: k -> (cluster_labels, silhouette_score)
    """
    results = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(pca50)
        sil = silhouette_score(pca50, labels, sample_size=min(5000, len(pca50)), random_state=seed)
        results[k] = (labels, sil)
        print(f"  k={k}  silhouette={sil:.4f}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_LABEL_MAP = {0: "MSS/pMMR", 1: "MSI-H/dMMR"}
_LABEL_COLOR = {0: "royalblue", 1: "tomato"}
_COHORT_COLOR = {"SR1482": "darkorange", "SR386": "mediumseagreen"}


def _scatter3d(x, y, z, color_vals, color_map: dict, name: str, visible: bool) -> go.Scatter3d:
    """One Scatter3d trace coloured by a categorical variable."""
    unique_vals = sorted(set(color_vals))
    traces = []
    for v in unique_vals:
        mask = np.array(color_vals) == v
        traces.append(
            go.Scatter3d(
                x=x[mask], y=y[mask], z=z[mask],
                mode="markers",
                marker=dict(size=2, color=color_map.get(v, None), opacity=0.6),
                name=f"{name}: {v}",
                visible=visible,
            )
        )
    return traces


def build_figure(
    umap3d: np.ndarray,
    labels: np.ndarray,
    cohorts: np.ndarray,
    kmeans_results: dict[int, tuple[np.ndarray, float]],
) -> go.Figure:
    x, y, z = umap3d[:, 0], umap3d[:, 1], umap3d[:, 2]

    all_traces = []
    button_groups: list[tuple[str, int, int]] = []  # (label, start_idx, n_traces)

    # --- MSI label colouring ---
    label_strs = [_LABEL_MAP[int(l)] for l in labels]
    label_traces = _scatter3d(x, y, z, label_strs,
                               {"MSS/pMMR": "royalblue", "MSI-H/dMMR": "tomato"},
                               "Label", visible=True)
    button_groups.append(("MSI/MMR label", len(all_traces), len(label_traces)))
    all_traces.extend(label_traces)

    # --- Cohort colouring ---
    cohort_traces = _scatter3d(x, y, z, list(cohorts), _COHORT_COLOR, "Cohort", visible=False)
    button_groups.append(("Cohort", len(all_traces), len(cohort_traces)))
    all_traces.extend(cohort_traces)

    # --- k-means colourings ---
    cmap = [
        "#e41a1c","#377eb8","#4daf4a","#984ea3",
        "#ff7f00","#a65628","#f781bf","#999999",
    ]
    for k, (cluster_labels, sil) in sorted(kmeans_results.items()):
        cluster_strs = [str(c) for c in cluster_labels]
        k_cmap = {str(i): cmap[i % len(cmap)] for i in range(k)}
        k_traces = _scatter3d(x, y, z, cluster_strs, k_cmap,
                               f"k={k}", visible=False)
        button_groups.append((f"k-means k={k}  (sil={sil:.3f})", len(all_traces), len(k_traces)))
        all_traces.extend(k_traces)

    total = len(all_traces)

    # Build dropdown buttons — each makes exactly one group visible
    buttons = []
    for btn_label, start, n in button_groups:
        visibility = [False] * total
        for i in range(start, start + n):
            visibility[i] = True
        buttons.append(dict(label=btn_label, method="update",
                            args=[{"visible": visibility}]))

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title="UNI patch embeddings — UMAP 3D (PCA-50 → UMAP-3)",
        scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"),
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        )],
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",         default="/mnt/data-surgen")
    parser.add_argument("--output",            default="outputs/patch_viz.html")
    parser.add_argument("--patches_per_slide", type=int, default=64)
    parser.add_argument("--n_slides",          type=int, default=None,
                        help="Randomly sample this many slides (default: all)")
    parser.add_argument("--pca_components",    type=int, default=50)
    parser.add_argument("--k_min",             type=int, default=2)
    parser.add_argument("--k_max",             type=int, default=8)
    parser.add_argument("--seed",              type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    provider = UniFeatureProvider(args.data_root)
    all_indices = list(range(len(provider)))

    if args.n_slides is not None and args.n_slides < len(all_indices):
        slide_indices = random.sample(all_indices, args.n_slides)
    else:
        slide_indices = all_indices

    print(f"Slides: {len(slide_indices)}  patches_per_slide: {args.patches_per_slide}")
    print(f"Expected patches: ~{len(slide_indices) * args.patches_per_slide}")

    print("\nSampling patches...")
    features, labels, cohorts = sample_patches(provider, slide_indices, args.patches_per_slide, args.seed)
    print(f"Sampled: {features.shape}")

    print("\nReducing dimensions (PCA → UMAP)...")
    pca50, umap3d = reduce(features, pca_components=args.pca_components, umap_seed=args.seed)

    print("\nRunning k-means...")
    kmeans_results = run_kmeans(pca50, args.k_min, args.k_max, seed=args.seed)

    best_k = max(kmeans_results, key=lambda k: kmeans_results[k][1])
    print(f"\nBest k by silhouette: k={best_k}  ({kmeans_results[best_k][1]:.4f})")

    print("\nBuilding figure...")
    fig = build_figure(umap3d, labels, cohorts, kmeans_results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs=True, full_html=True)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
