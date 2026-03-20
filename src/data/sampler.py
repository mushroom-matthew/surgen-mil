from __future__ import annotations

import numpy as np


class RandomPatchSampler:
    def __init__(self, max_patches: int):
        self.max_patches = max_patches

    def __call__(self, features: np.ndarray, coords: np.ndarray):
        n = len(features)
        if n <= self.max_patches:
            return features, coords
        idx = np.random.choice(n, size=self.max_patches, replace=False)
        return features[idx], coords[idx]


class SpatialBalancedPatchSampler:
    def __init__(self, max_patches: int, grid_size: int = 8):
        self.max_patches = max_patches
        self.grid_size = grid_size

    def __call__(self, features: np.ndarray, coords: np.ndarray):
        n = len(features)
        if n <= self.max_patches:
            return features, coords

        xy = coords.astype(np.float32)
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
        span = np.clip(hi - lo, a_min=1.0, a_max=None)

        bins = ((xy - lo) / span * self.grid_size).astype(np.int64)
        bins = np.clip(bins, 0, self.grid_size - 1)
        cell_ids = bins[:, 1] * self.grid_size + bins[:, 0]

        unique_cells = np.unique(cell_ids)
        chosen: list[int] = []

        # First pass: ensure geographic coverage by drawing at most one patch per occupied cell.
        shuffled_cells = np.random.permutation(unique_cells)
        for cell_id in shuffled_cells:
            cell_idx = np.flatnonzero(cell_ids == cell_id)
            pick = int(np.random.choice(cell_idx))
            chosen.append(pick)
            if len(chosen) == self.max_patches:
                idx = np.array(chosen, dtype=np.int64)
                return features[idx], coords[idx]

        remaining = self.max_patches - len(chosen)
        if remaining > 0:
            all_idx = np.arange(n, dtype=np.int64)
            available = all_idx[~np.isin(all_idx, np.array(chosen, dtype=np.int64))]
            extra = np.random.choice(available, size=remaining, replace=False)
            chosen.extend(extra.tolist())

        idx = np.array(chosen, dtype=np.int64)
        return features[idx], coords[idx]


class FeatureDiversePatchSampler:
    def __init__(
        self,
        max_patches: int,
        proj_dim: int = 32,
        candidate_pool_size: int = 2048,
    ):
        self.max_patches = max_patches
        self.proj_dim = proj_dim
        self.candidate_pool_size = candidate_pool_size

    def _project_features(self, features: np.ndarray) -> np.ndarray:
        dim = features.shape[1]
        proj_dim = min(self.proj_dim, dim)
        proj = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(proj_dim), size=(dim, proj_dim))
        reduced = features.astype(np.float32) @ proj.astype(np.float32)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        return reduced / np.clip(norms, a_min=1e-8, a_max=None)

    def __call__(self, features: np.ndarray, coords: np.ndarray):
        n = len(features)
        if n <= self.max_patches:
            return features, coords

        pool_size = min(n, max(self.max_patches, self.candidate_pool_size))
        candidate_idx = np.random.choice(n, size=pool_size, replace=False)
        reduced = self._project_features(features[candidate_idx])

        start = int(np.random.randint(pool_size))
        selected = [start]
        min_dist = np.sum((reduced - reduced[start]) ** 2, axis=1)
        min_dist[start] = -np.inf

        while len(selected) < self.max_patches:
            next_local = int(np.argmax(min_dist))
            selected.append(next_local)
            dist = np.sum((reduced - reduced[next_local]) ** 2, axis=1)
            min_dist = np.minimum(min_dist, dist)
            min_dist[selected] = -np.inf

        idx = candidate_idx[np.array(selected, dtype=np.int64)]
        return features[idx], coords[idx]


class FullBagSampler:
    def __call__(self, features: np.ndarray, coords: np.ndarray):
        return features, coords


def build_patch_sampler(data_cfg: dict):
    sampler_cfg = data_cfg.get("sampler")
    max_patches = data_cfg.get("max_patches")

    if sampler_cfg is None:
        return FullBagSampler() if not max_patches else RandomPatchSampler(max_patches=max_patches)

    sampler_name = sampler_cfg.get("name", "random")
    sampler_max_patches = sampler_cfg.get("max_patches", max_patches)
    if not sampler_max_patches:
        return FullBagSampler()

    if sampler_name == "random":
        return RandomPatchSampler(max_patches=sampler_max_patches)
    if sampler_name == "spatial_balanced":
        return SpatialBalancedPatchSampler(
            max_patches=sampler_max_patches,
            grid_size=sampler_cfg.get("grid_size", 8),
        )
    if sampler_name == "feature_diverse":
        return FeatureDiversePatchSampler(
            max_patches=sampler_max_patches,
            proj_dim=sampler_cfg.get("proj_dim", 32),
            candidate_pool_size=sampler_cfg.get("candidate_pool_size", 2048),
        )

    raise ValueError(f"Unknown patch sampler: {sampler_name}")
