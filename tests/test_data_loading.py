"""Test UniFeatureProvider, SurgenBagDataset, and splits with synthetic data."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

from src.data.dataset import SurgenBagDataset
from src.data.feature_provider import UniFeatureProvider
from src.data.sampler import FullBagSampler, RandomPatchSampler
from src.data.splits import case_grouped_stratified_split


def make_fake_dataset(root: Path, n_slides: int = 12, n_patches: int = 20) -> None:
    emb_root = root / "embeddings"
    emb_root.mkdir(parents=True)

    label_rows = []
    for case_id in range(1, n_slides + 1):
        slide_id = f"SR1482_40X_HE_T{case_id}_0"
        z = zarr.open(str(emb_root / f"{slide_id}.zarr"), mode="w")
        z["features"] = np.random.randn(n_patches, 1024).astype(np.float32)
        z["coords"] = np.random.randint(0, 100, (n_patches, 2)).astype(np.float32)

        msi = "MSI HIGH" if case_id <= 4 else "NO MSI"
        mmr = "MMR loss" if case_id <= 4 else "No loss"
        label_rows.append({"case_id": case_id, "MSI": msi, "MMR": mmr})

    pd.DataFrame(label_rows).to_csv(root / "SR1482_labels.csv", index=False)
    pd.DataFrame(columns=["case_id", "mmr_loss_binary"]).to_csv(root / "SR386_labels.csv", index=False)


def test_feature_provider_loads():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_fake_dataset(root)
        provider = UniFeatureProvider(str(root))
        assert len(provider) == 12


def test_feature_provider_record_fields():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_fake_dataset(root)
        provider = UniFeatureProvider(str(root))
        record = provider.get_record(0)
        assert hasattr(record, "slide_id")
        assert hasattr(record, "label")


def test_surgen_bag_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_fake_dataset(root, n_patches=20)
        provider = UniFeatureProvider(str(root))
        sampler = FullBagSampler()
        ds = SurgenBagDataset(provider, indices=list(range(len(provider))), sampler=sampler)

        item = ds[0]
        assert "features" in item
        assert "coords" in item
        assert "label" in item
        assert "slide_id" in item
        assert item["features"].shape[1] == 1024


def test_random_patch_sampler_limits():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_fake_dataset(root, n_patches=20)
        provider = UniFeatureProvider(str(root))
        sampler = RandomPatchSampler(max_patches=5)
        ds = SurgenBagDataset(provider, indices=list(range(len(provider))), sampler=sampler)

        item = ds[0]
        assert item["features"].shape[0] <= 5


def test_case_grouped_split_non_overlapping():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_fake_dataset(root)
        provider = UniFeatureProvider(str(root))
        indices = list(range(len(provider)))
        train, val, test = case_grouped_stratified_split(provider, indices, train_frac=0.6, val_frac=0.2, seed=0)

        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0
        assert sorted(train + val + test) == sorted(indices)
