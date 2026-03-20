"""Test UniFeatureProvider, SurgenBagDataset, and splits with synthetic data."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

from src.data.dataset import SurgenBagDataset
from src.data.feature_provider import UniFeatureProvider
from src.data.sampler import (
    FeatureDiversePatchSampler,
    FullBagSampler,
    RandomPatchSampler,
    SpatialBalancedPatchSampler,
    build_patch_sampler,
)
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


def test_build_patch_sampler_defaults_to_random():
    sampler = build_patch_sampler({"max_patches": 16})
    assert isinstance(sampler, RandomPatchSampler)


def test_build_patch_sampler_supports_spatial_balanced():
    sampler = build_patch_sampler({
        "max_patches": 16,
        "sampler": {"name": "spatial_balanced", "grid_size": 4},
    })
    assert isinstance(sampler, SpatialBalancedPatchSampler)


def test_build_patch_sampler_supports_feature_diverse():
    sampler = build_patch_sampler({
        "max_patches": 16,
        "sampler": {"name": "feature_diverse", "proj_dim": 8, "candidate_pool_size": 32},
    })
    assert isinstance(sampler, FeatureDiversePatchSampler)


def test_spatial_balanced_sampler_spreads_across_cells():
    np.random.seed(0)
    coords = []
    features = []
    # Four well-separated spatial quadrants with dense local redundancy inside each.
    for cx, cy in [(0, 0), (100, 0), (0, 100), (100, 100)]:
        for _ in range(25):
            coords.append([cx + np.random.randn() * 2, cy + np.random.randn() * 2])
            features.append(np.random.randn(8))

    coords = np.array(coords, dtype=np.float32)
    features = np.array(features, dtype=np.float32)

    def quadrant_coverage(sampled_coords: np.ndarray) -> int:
        x_bin = (sampled_coords[:, 0] > 50).astype(int)
        y_bin = (sampled_coords[:, 1] > 50).astype(int)
        return len(set(zip(x_bin, y_bin)))

    random_coverages = []
    balanced_coverages = []
    for seed in range(5):
        np.random.seed(seed)
        _, random_coords = RandomPatchSampler(max_patches=4)(features, coords)
        np.random.seed(seed)
        _, balanced_coords = SpatialBalancedPatchSampler(max_patches=4, grid_size=2)(features, coords)
        random_coverages.append(quadrant_coverage(random_coords))
        balanced_coverages.append(quadrant_coverage(balanced_coords))

    assert balanced_coverages == [4, 4, 4, 4, 4]
    assert min(balanced_coverages) > min(random_coverages)
    assert sum(balanced_coverages) > sum(random_coverages)


def test_feature_diverse_sampler_improves_feature_separation():
    np.random.seed(0)
    centers = np.eye(4, dtype=np.float32)
    features = []
    coords = []
    # Four feature clusters with many near-duplicates; a diverse sampler should avoid
    # selecting multiple almost-identical points from the same cluster.
    for center in centers:
        for _ in range(25):
            features.append(center + 0.05 * np.random.randn(4))
            coords.append(np.random.randn(2))

    features = np.array(features, dtype=np.float32)
    coords = np.array(coords, dtype=np.float32)

    def min_pairwise_distance(x: np.ndarray) -> float:
        dists = []
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                dists.append(float(np.linalg.norm(x[i] - x[j])))
        return min(dists)

    random_spreads = []
    diverse_spreads = []
    for seed in range(5):
        np.random.seed(seed)
        random_features, _ = RandomPatchSampler(max_patches=4)(features, coords)
        np.random.seed(seed)
        diverse_features, diverse_coords = FeatureDiversePatchSampler(
            max_patches=4, proj_dim=4, candidate_pool_size=100
        )(features, coords)
        random_spreads.append(min_pairwise_distance(random_features))
        diverse_spreads.append(min_pairwise_distance(diverse_features))
        assert diverse_features.shape[0] == 4
        assert diverse_coords.shape[0] == 4

    assert sum(diverse_spreads) > sum(random_spreads)
    assert min(diverse_spreads) > min(random_spreads)


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
