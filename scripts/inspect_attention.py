"""Inspect attention weight distribution for attention-MIL models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import build_loaders, compute_attention_diagnostics


def main():
    parser = argparse.ArgumentParser(
        description="Inspect attention weight statistics for attention-MIL models."
    )
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Split to run diagnostics on (default: test)")
    parser.add_argument("--out", default=None,
                        help="Optional CSV path to save per-slide attention stats")
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

    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loader_map[args.split]

    diag = compute_attention_diagnostics(model, loader, device)

    if not diag:
        print("No attention weights found — is this an attention-MIL model?")
        return

    print(f"\nAttention diagnostics ({args.split}):")
    for k, v in diag.items():
        print(f"  {k}: {v:.4f}")

    if args.out:
        import pandas as pd
        pd.DataFrame([diag]).to_csv(args.out, index=False)
        print(f"\nDiagnostics saved to {args.out}")


if __name__ == "__main__":
    main()
