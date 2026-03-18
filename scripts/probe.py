"""
Two experiments that together answer: embedding bottleneck or aggregation failure?

Experiment A — Linear probe on pooled bag descriptors
  Features: [mean, max, std] per slide → 3072-dim vector
  Classifier: logistic regression (sklearn)
  Split: same case-grouped split used in training (seed=42)

Experiment B — Top-k vs random-k patch restriction
  For the best attention model, re-predict each test slide using:
    - all patches
    - top-k patches (by attention)
    - random-k patches (5 random seeds, averaged)
  k values: 8, 16, 32, 64, 128

Usage:
    python scripts/probe.py \
        --config configs/uni_attention.yaml \
        --checkpoint outputs/uni_attention/model.pt
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.feature_provider import UniFeatureProvider
from src.data.splits import case_grouped_stratified_split
from src.models.aggregators.attention_mil import AttentionMIL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pool_slide(features: np.ndarray) -> np.ndarray:
    """Concatenate mean, max, std pooling → [3*D]."""
    return np.concatenate([
        features.mean(axis=0),
        features.max(axis=0),
        features.std(axis=0),
    ])


def load_attention_model(cfg: dict, checkpoint: Path, device: torch.device) -> AttentionMIL:
    model = AttentionMIL(
        input_dim=cfg["model"]["input_dim"],
        attention_dim=cfg["model"]["attention_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )
    state = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_restricted(
    model: AttentionMIL,
    features: np.ndarray,
    patch_indices: np.ndarray,
    device: torch.device,
) -> float:
    x = torch.tensor(features[patch_indices], dtype=torch.float32).to(device)
    out = model(x)
    return torch.sigmoid(out["logit"].view(())).item()


@torch.no_grad()
def get_attention_order(
    model: AttentionMIL,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Return patch indices sorted by attention weight descending."""
    x = torch.tensor(features, dtype=torch.float32).to(device)
    out = model(x)
    attn = out["attention_weights"].cpu().numpy()
    return np.argsort(attn)[::-1]


def metrics(y_true, y_score) -> dict:
    if len(set(y_true)) < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": round(float(roc_auc_score(y_true, y_score)), 4),
        "auprc": round(float(average_precision_score(y_true, y_score)), 4),
    }


# ---------------------------------------------------------------------------
# Experiment A: linear probe
# ---------------------------------------------------------------------------

def run_linear_probe(provider: UniFeatureProvider, train_idx, test_idx):
    print("\n=== Experiment A: Linear probe (mean + max + std pooling) ===")

    def build_xy(indices):
        X, y = [], []
        for idx in indices:
            item = provider.load_slide(idx)
            X.append(pool_slide(item["features"]))
            y.append(item["label"])
        return np.array(X), np.array(y)

    print("  Building train features...")
    X_train, y_train = build_xy(train_idx)
    print("  Building test features...")
    X_test, y_test = build_xy(test_idx)

    clf = LogisticRegression(
        max_iter=1000, C=0.01, solver="lbfgs", class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    train_scores = clf.predict_proba(X_train)[:, 1]
    test_scores  = clf.predict_proba(X_test)[:, 1]

    train_m = metrics(y_train.tolist(), train_scores.tolist())
    test_m  = metrics(y_test.tolist(),  test_scores.tolist())

    print(f"  Train: AUROC={train_m['auroc']}  AUPRC={train_m['auprc']}")
    print(f"  Test:  AUROC={test_m['auroc']}   AUPRC={test_m['auprc']}")
    return test_m


# ---------------------------------------------------------------------------
# Experiment B: top-k vs random-k
# ---------------------------------------------------------------------------

def run_topk_experiment(
    provider: UniFeatureProvider,
    test_idx: list[int],
    model: AttentionMIL,
    device: torch.device,
    k_values: list[int],
    n_random_seeds: int = 5,
):
    print("\n=== Experiment B: Top-k vs random-k patch restriction ===")

    results = {}  # k -> {"topk": metrics, "random": metrics, "full": metrics}

    # full-bag baseline (already the model's default behaviour)
    y_true, y_full = [], []
    for idx in test_idx:
        item = provider.load_slide(idx)
        n = len(item["features"])
        order = get_attention_order(model, item["features"], device)
        prob_full = predict_restricted(model, item["features"], np.arange(n), device)
        y_true.append(item["label"])
        y_full.append(prob_full)

    print(f"  Full bag: {metrics(y_true, y_full)}")

    for k in k_values:
        y_topk = []
        y_random = []

        for idx in test_idx:
            item = provider.load_slide(idx)
            features = item["features"]
            n = len(features)
            actual_k = min(k, n)

            # top-k by attention
            order = get_attention_order(model, features, device)
            topk_idx = order[:actual_k]
            y_topk.append(predict_restricted(model, features, topk_idx, device))

            # random-k averaged over seeds
            random_probs = []
            for seed in range(n_random_seeds):
                rng = np.random.default_rng(seed)
                rand_idx = rng.choice(n, size=actual_k, replace=False)
                random_probs.append(predict_restricted(model, features, rand_idx, device))
            y_random.append(float(np.mean(random_probs)))

        m_topk   = metrics(y_true, y_topk)
        m_random = metrics(y_true, y_random)

        print(f"  k={k:4d}  top-k: AUROC={m_topk['auroc']}  AUPRC={m_topk['auprc']}"
              f"   |   random-k: AUROC={m_random['auroc']}  AUPRC={m_random['auprc']}")
        results[k] = {"topk": m_topk, "random": m_random}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--k_values",   default="8,16,32,64,128")
    parser.add_argument("--seeds",      type=int, default=5)
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    provider = UniFeatureProvider(cfg["data"]["root"])
    all_indices = list(range(len(provider)))

    train_idx, val_idx, test_idx = case_grouped_stratified_split(
        provider, all_indices,
        train_frac=cfg["data"].get("train_frac", 0.7),
        val_frac=cfg["data"].get("val_frac", 0.15),
        seed=cfg["training"]["seed"],
    )

    # Experiment A — no model needed
    run_linear_probe(provider, train_idx, test_idx)

    # Experiment B — needs attention model
    if cfg["model"]["name"] != "attention_mil":
        print("\nSkipping experiment B: config is not attention_mil.")
        return

    model = load_attention_model(cfg, Path(args.checkpoint), device)
    run_topk_experiment(provider, test_idx, model, device, k_values, n_random_seeds=args.seeds)


if __name__ == "__main__":
    main()
