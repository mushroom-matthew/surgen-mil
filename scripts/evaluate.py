"""Standalone evaluation from a saved checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import build_loaders, evaluate, find_temperature


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on a data split.")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--out", default=None, help="Optional path to save predictions CSV")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature scalar (default: load from metrics.json or fit on val)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.models.build import build_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    provider, train_loader, val_loader, test_loader, *_ = build_loaders(cfg)

    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Determine temperature
    temperature = args.temperature
    if temperature is None:
        metrics_path = Path(args.checkpoint).parent / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                saved = json.load(f)
            temperature = saved.get("temperature", 1.0)
            print(f"Temperature from metrics.json: {temperature:.4f}")
        else:
            print("Fitting temperature on validation set...")
            temperature = find_temperature(model, val_loader, device)
            print(f"Temperature (fit): {temperature:.4f}")

    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loader_map[args.split]

    metrics, rows = evaluate(model, loader, device, split=args.split, temperature=temperature)
    print(f"\n{args.split} metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: None")

    if args.out:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nPredictions saved to {args.out}")


if __name__ == "__main__":
    main()
