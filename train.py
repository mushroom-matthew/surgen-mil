from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SurgenBagDataset
from src.data.feature_provider import UniFeatureProvider
from src.data.sampler import FullBagSampler, RandomPatchSampler
from src.data.splits import case_grouped_stratified_split
from src.losses import build_loss
from src.models.build import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_one(batch):
    assert len(batch) == 1, "Use batch_size=1 for variable-size bags"
    return batch[0]



def build_loaders(cfg):
    provider = UniFeatureProvider(cfg["data"]["root"])
    all_indices = list(range(len(provider)))

    train_idx, val_idx, test_idx = case_grouped_stratified_split(
        provider,
        all_indices,
        train_frac=cfg["data"].get("train_frac", 0.7),
        val_frac=cfg["data"].get("val_frac", 0.15),
        seed=cfg["training"]["seed"],
    )

    max_patches = cfg["data"]["max_patches"]
    train_sampler = RandomPatchSampler(max_patches=max_patches)
    eval_sampler = FullBagSampler()

    train_ds = SurgenBagDataset(provider, indices=train_idx, sampler=train_sampler)
    val_ds = SurgenBagDataset(provider, indices=val_idx, sampler=eval_sampler)
    test_ds = SurgenBagDataset(provider, indices=test_idx, sampler=eval_sampler)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=collate_one,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_one,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, collate_fn=collate_one,
        num_workers=2, pin_memory=True,
    )

    return provider, train_loader, val_loader, test_loader, train_idx, val_idx, test_idx


@torch.no_grad()
def evaluate(model, loader, device, split: str = "", temperature: float = 1.0):
    model.eval()
    rows = []

    for batch in loader:
        x = batch["features"].to(device)
        y = batch["label"].to(device)
        coords = batch["coords"].to(device)

        out = model(x, coords=coords)
        logit = out["logit"].view(())
        prob = torch.sigmoid(logit / temperature).item()

        rows.append({
            "slide_id": batch["slide_id"],
            "label": int(y.item()),
            "prob": prob,
            "split": split,
        })

    y_true = [r["label"] for r in rows]
    y_score = [r["prob"] for r in rows]

    if len(set(y_true)) < 2:
        metrics = {"auroc": None, "auprc": None}
    else:
        metrics = {
            "auroc": float(roc_auc_score(y_true, y_score)),
            "auprc": float(average_precision_score(y_true, y_score)),
        }

    return metrics, rows


@torch.no_grad()
def _collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        x = batch["features"].to(device)
        coords = batch["coords"].to(device)
        out = model(x, coords=coords)
        logits.append(out["logit"].view(()).cpu().item())
        labels.append(float(batch["label"].item()))
    return torch.tensor(logits), torch.tensor(labels)


def find_temperature(model, loader, device) -> float:
    """Find scalar temperature T that minimises NLL on the given loader."""
    from scipy.optimize import minimize_scalar

    logits, labels = _collect_logits(model, loader, device)

    def nll(t):
        probs = torch.sigmoid(logits / max(float(t), 1e-6))
        return torch.nn.functional.binary_cross_entropy(probs, labels).item()

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    provider, train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = build_loaders(cfg)

    model = build_model(cfg).to(device)

    train_labels = [provider.get_record(i).label for i in train_idx]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos

    loss_cfg = cfg.get("loss", {"name": "bce", "weighted": True})
    if loss_cfg.get("weighted", False):
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
        print(f"Train positives: {n_pos}, negatives: {n_neg}, pos_weight: {pos_weight.item():.3f}")
    else:
        pos_weight = None
        print(f"Train positives: {n_pos}, negatives: {n_neg}")
    criterion = build_loss(loss_cfg, device, pos_weight=pos_weight)
    print(f"Loss: {loss_cfg['name']}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    patience = cfg["training"].get("early_stopping_patience", None)
    selection_metric = cfg["training"].get("selection_metric", "val_auprc")

    best_val_metric = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

        for batch in pbar:
            x = batch["features"].to(device)
            y = batch["label"].to(device)
            coords = batch["coords"].to(device)

            optimizer.zero_grad()
            out = model(x, coords=coords)
            logit = out["logit"].view(())
            loss = criterion(logit.unsqueeze(0), y.unsqueeze(0))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / max(len(train_loader), 1)
        val_metrics, _ = evaluate(model, val_loader, device, split="val")

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
        }
        history.append(record)
        print(record)

        current_metric = val_metrics.get(selection_metric.replace("val_", ""))
        if current_metric is not None and current_metric > best_val_metric:
            best_val_metric = current_metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement in {selection_metric} for {patience} epochs)")
            break

    print(f"Best {selection_metric}: {best_val_metric:.4f} — loading best checkpoint")
    if best_state is not None:
        model.load_state_dict(best_state)

    temperature = find_temperature(model, val_loader, device)
    print(f"Temperature (val): {temperature:.4f}")

    test_metrics, test_rows = evaluate(model, test_loader, device, split="test", temperature=temperature)
    _, val_rows   = evaluate(model, val_loader,  device, split="val",   temperature=temperature)
    _, train_rows = evaluate(model, train_loader, device, split="train", temperature=temperature)
    all_preds = train_rows + val_rows + test_rows

    print("Test metrics (temperature-scaled):", test_metrics)

    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"test": test_metrics, "temperature": temperature}, f, indent=2)
    pd.DataFrame(all_preds).to_csv(out_dir / "predictions.csv", index=False)
    print(f"Predictions saved to {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
