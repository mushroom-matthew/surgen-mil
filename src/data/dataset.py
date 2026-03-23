from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .feature_provider import UniFeatureProvider
from .sampler import RandomPatchSampler, FullBagSampler


class SurgenBagDataset(Dataset):
    def __init__(
        self,
        provider: UniFeatureProvider,
        indices: list[int] | None = None,
        sampler=None,
    ):
        self.provider = provider
        self.indices = indices if indices is not None else list(range(len(provider)))
        self.sampler = sampler if sampler is not None else FullBagSampler()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        provider_idx = self.indices[i]
        item = self.provider.load_slide(provider_idx)
        features, coords = self.sampler(item["features"], item["coords"])

        return {
            "slide_id": item["slide_id"],
            "cohort": item["cohort"],
            "features": torch.tensor(features, dtype=torch.float32),
            "coords": torch.tensor(coords, dtype=torch.float32),
            "label": torch.tensor(item["label"], dtype=torch.float32),
        }
