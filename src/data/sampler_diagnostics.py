from __future__ import annotations

import numpy as np


def _grid_bounds(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = coords.astype(np.float32)
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.clip(hi - lo, a_min=1.0, a_max=None)
    return lo, span


def occupied_grid_cells(
    coords: np.ndarray,
    grid_size: int,
    lo: np.ndarray | None = None,
    span: np.ndarray | None = None,
) -> set[int]:
    if len(coords) == 0:
        return set()

    xy = coords.astype(np.float32)
    if lo is None or span is None:
        lo, span = _grid_bounds(coords)

    bins = ((xy - lo) / span * grid_size).astype(np.int64)
    bins = np.clip(bins, 0, grid_size - 1)
    cell_ids = bins[:, 1] * grid_size + bins[:, 0]
    return {int(x) for x in np.unique(cell_ids)}


def grid_coverage_ratio(
    sampled_coords: np.ndarray,
    full_coords: np.ndarray,
    grid_size: int,
) -> float:
    lo, span = _grid_bounds(full_coords)
    full_cells = occupied_grid_cells(full_coords, grid_size, lo=lo, span=span)
    if not full_cells:
        return 0.0
    sampled_cells = occupied_grid_cells(sampled_coords, grid_size, lo=lo, span=span)
    return len(sampled_cells) / len(full_cells)


def mean_pairwise_cosine_distance(features: np.ndarray) -> float:
    n = len(features)
    if n < 2:
        return 0.0

    x = features.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(norms, a_min=1e-8, a_max=None)
    sim = x @ x.T
    upper = sim[np.triu_indices(n, k=1)]
    return float(np.mean(1.0 - upper))


def mean_pairwise_cosine_distance_capped(
    features: np.ndarray,
    max_points: int = 512,
    seed: int = 0,
) -> float:
    if len(features) <= max_points:
        return mean_pairwise_cosine_distance(features)

    state = np.random.get_state()
    try:
        np.random.seed(seed)
        idx = np.random.choice(len(features), size=max_points, replace=False)
    finally:
        np.random.set_state(state)
    return mean_pairwise_cosine_distance(features[idx])


def _sample_once(sampler, features: np.ndarray, coords: np.ndarray, seed: int):
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return sampler(features, coords)
    finally:
        np.random.set_state(state)


def diagnose_sampler(
    provider,
    indices: list[int],
    sampler,
    grid_size: int = 8,
    repeats: int = 1,
    base_seed: int = 0,
    max_slides: int | None = None,
    full_reference_max_points: int = 512,
) -> tuple[list[dict], dict]:
    if max_slides is not None:
        indices = indices[:max_slides]

    rows: list[dict] = []
    for slide_offset, idx in enumerate(indices):
        item = provider.load_slide(idx)
        features = item["features"]
        coords = item["coords"]

        lo, span = _grid_bounds(coords)
        full_cells = occupied_grid_cells(coords, grid_size, lo=lo, span=span)
        full_mean_cosine = mean_pairwise_cosine_distance_capped(
            features,
            max_points=full_reference_max_points,
            seed=base_seed + slide_offset,
        )

        for repeat_idx in range(repeats):
            seed = base_seed + slide_offset * max(repeats, 1) + repeat_idx
            sampled_features, sampled_coords = _sample_once(sampler, features, coords, seed)
            rows.append({
                "slide_id": item["slide_id"],
                "label": int(item["label"]),
                "repeat": repeat_idx,
                "n_full_patches": int(len(features)),
                "n_sampled_patches": int(len(sampled_features)),
                "full_occupied_cells": int(len(full_cells)),
                "sampled_occupied_cells": int(len(occupied_grid_cells(sampled_coords, grid_size, lo=lo, span=span))),
                "grid_coverage_ratio": grid_coverage_ratio(sampled_coords, coords, grid_size),
                "mean_pairwise_cosine_distance": mean_pairwise_cosine_distance(sampled_features),
                "full_mean_pairwise_cosine_distance": full_mean_cosine,
            })

    if not rows:
        return rows, {
            "n_rows": 0,
            "n_slides": 0,
            "mean_grid_coverage_ratio": None,
            "mean_pairwise_cosine_distance": None,
            "mean_full_pairwise_cosine_distance": None,
            "mean_sampled_fraction": None,
        }

    summary = {
        "n_rows": len(rows),
        "n_slides": len({r["slide_id"] for r in rows}),
        "mean_grid_coverage_ratio": float(np.mean([r["grid_coverage_ratio"] for r in rows])),
        "mean_pairwise_cosine_distance": float(np.mean([r["mean_pairwise_cosine_distance"] for r in rows])),
        "mean_full_pairwise_cosine_distance": float(np.mean([r["full_mean_pairwise_cosine_distance"] for r in rows])),
        "mean_sampled_fraction": float(np.mean([
            r["n_sampled_patches"] / max(r["n_full_patches"], 1) for r in rows
        ])),
    }
    return rows, summary
