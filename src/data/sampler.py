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


class FullBagSampler:
    def __call__(self, features: np.ndarray, coords: np.ndarray):
        return features, coords
