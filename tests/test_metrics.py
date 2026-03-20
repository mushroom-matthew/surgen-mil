"""Test evaluate() and compute_case_level_metrics() functions."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import evaluate, compute_case_level_metrics


def make_fake_loader(n_pos: int = 3, n_neg: int = 3, cohort: str = "SR1482"):
    """Create a list of fake batches that evaluate() can iterate over."""
    batches = []
    for i in range(n_pos + n_neg):
        label = 1 if i < n_pos else 0
        slide_id = f"{cohort}_40X_HE_T{i+1}_0"
        batches.append({
            "features": torch.randn(10, 1024),
            "coords": torch.zeros(10, 2),
            "label": torch.tensor(label, dtype=torch.float32),
            "slide_id": slide_id,
        })
    return batches


def test_evaluate_returns_metrics():
    from src.models.build import build_model
    cfg = {
        "model": {"name": "mean_pool", "input_dim": 1024, "hidden_dim": 64, "dropout": 0.0},
        "optimizer": {}, "loss": {}, "training": {},
    }
    model = build_model(cfg)
    loader = make_fake_loader(n_pos=3, n_neg=3)
    device = torch.device("cpu")
    metrics, rows = evaluate(model, loader, device, split="test")

    assert "auroc" in metrics
    assert "auprc" in metrics
    assert len(rows) == 6


def test_evaluate_single_class_returns_none():
    from src.models.build import build_model
    cfg = {
        "model": {"name": "mean_pool", "input_dim": 1024, "hidden_dim": 64, "dropout": 0.0},
        "optimizer": {}, "loss": {}, "training": {},
    }
    model = build_model(cfg)
    loader = make_fake_loader(n_pos=0, n_neg=4)
    device = torch.device("cpu")
    metrics, rows = evaluate(model, loader, device, split="test")

    assert metrics["auroc"] is None
    assert metrics["auprc"] is None


def test_case_level_metrics_aggregation():
    rows = [
        {"slide_id": "SR1482_40X_HE_T1_0", "label": 1, "prob": 0.8, "split": "test"},
        {"slide_id": "SR1482_40X_HE_T1_1", "label": 1, "prob": 0.7, "split": "test"},
        {"slide_id": "SR1482_40X_HE_T2_0", "label": 0, "prob": 0.2, "split": "test"},
        {"slide_id": "SR1482_40X_HE_T3_0", "label": 1, "prob": 0.9, "split": "test"},
        {"slide_id": "SR1482_40X_HE_T4_0", "label": 0, "prob": 0.1, "split": "test"},
    ]
    result = compute_case_level_metrics(rows)
    assert "max" in result
    assert "mean" in result
    assert "noisy_or" in result

    for agg in ("max", "mean", "noisy_or"):
        if result[agg]["auroc"] is not None:
            assert 0.0 <= result[agg]["auroc"] <= 1.0
        if result[agg]["auprc"] is not None:
            assert 0.0 <= result[agg]["auprc"] <= 1.0
