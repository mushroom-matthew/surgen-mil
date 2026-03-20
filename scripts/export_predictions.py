"""Export per-slide predictions from a trained checkpoint to CSV."""
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
    parser = argparse.ArgumentParser(
        description="Export per-slide predictions from a trained checkpoint."
    )
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"],
                        help="Which split(s) to export (default: test)")
    parser.add_argument("--out", default="predictions.csv",
                        help="Output CSV path (default: predictions.csv)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature scalar (default: load from metrics.json)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.models.build import build_model
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    provider, train_loader, val_loader, test_loader, *_ = build_loaders(cfg)

    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

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

    all_rows = []
    if args.split == "all":
        for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            _, rows = evaluate(model, loader, device, split=split_name, temperature=temperature)
            all_rows.extend(rows)
    else:
        loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
        _, rows = evaluate(model, loader_map[args.split], device, split=args.split, temperature=temperature)
        all_rows = rows

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Predictions ({len(all_rows)} slides) saved to {out_path}")


if __name__ == "__main__":
    main()
