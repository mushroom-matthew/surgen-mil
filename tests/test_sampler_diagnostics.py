import numpy as np

from src.data.sampler import RandomPatchSampler, SpatialBalancedPatchSampler
from src.data.sampler_diagnostics import (
    diagnose_sampler,
    grid_coverage_ratio,
    mean_pairwise_cosine_distance,
    mean_pairwise_cosine_distance_capped,
    occupied_grid_cells,
)


class DummyProvider:
    def __init__(self, slides: list[dict]):
        self.slides = slides

    def load_slide(self, idx: int) -> dict:
        return self.slides[idx]


def test_occupied_grid_cells_counts_bins():
    coords = np.array([[0, 0], [1, 1], [9, 0], [9, 9]], dtype=np.float32)
    cells = occupied_grid_cells(coords, grid_size=2)
    assert cells == {0, 1, 3}


def test_grid_coverage_ratio_is_fraction_of_full_coverage():
    full_coords = np.array([[0, 0], [9, 0], [0, 9], [9, 9]], dtype=np.float32)
    sampled_coords = np.array([[0, 0], [9, 9]], dtype=np.float32)
    ratio = grid_coverage_ratio(sampled_coords, full_coords, grid_size=2)
    assert ratio == 0.5


def test_mean_pairwise_cosine_distance_matches_orthogonal_vectors():
    features = np.array([[1, 0], [0, 1]], dtype=np.float32)
    assert mean_pairwise_cosine_distance(features) == 1.0


def test_mean_pairwise_cosine_distance_capped_matches_full_when_under_cap():
    features = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    assert mean_pairwise_cosine_distance_capped(features, max_points=10, seed=0) == (
        mean_pairwise_cosine_distance(features)
    )


def test_diagnose_sampler_reports_expected_keys():
    slides = [{
        "slide_id": "SR1482_40X_HE_T1_0",
        "label": 1,
        "features": np.random.randn(10, 4).astype(np.float32),
        "coords": np.random.randn(10, 2).astype(np.float32),
    }]
    provider = DummyProvider(slides)
    rows, summary = diagnose_sampler(
        provider=provider,
        indices=[0],
        sampler=RandomPatchSampler(max_patches=4),
        grid_size=2,
        repeats=2,
        base_seed=0,
    )
    assert len(rows) == 2
    assert summary["n_rows"] == 2
    assert "mean_grid_coverage_ratio" in summary
    assert "mean_pairwise_cosine_distance" in summary


def test_diagnose_sampler_detects_better_spatial_coverage_than_random():
    np.random.seed(0)
    coords = []
    features = []
    for cx, cy in [(0, 0), (100, 0), (0, 100), (100, 100)]:
        for _ in range(25):
            coords.append([cx + np.random.randn() * 2, cy + np.random.randn() * 2])
            features.append(np.random.randn(8))

    provider = DummyProvider([{
        "slide_id": "SR1482_40X_HE_T1_0",
        "label": 0,
        "features": np.array(features, dtype=np.float32),
        "coords": np.array(coords, dtype=np.float32),
    }])

    _, random_summary = diagnose_sampler(
        provider=provider,
        indices=[0],
        sampler=RandomPatchSampler(max_patches=4),
        grid_size=2,
        repeats=5,
        base_seed=0,
    )
    _, balanced_summary = diagnose_sampler(
        provider=provider,
        indices=[0],
        sampler=SpatialBalancedPatchSampler(max_patches=4, grid_size=2),
        grid_size=2,
        repeats=5,
        base_seed=0,
    )

    assert balanced_summary["mean_grid_coverage_ratio"] > random_summary["mean_grid_coverage_ratio"]
