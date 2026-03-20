"""
Quantify train-time sampler behaviour on actual slides.

Metrics:
  - occupied-grid coverage ratio: sampled occupied cells / full occupied cells
  - mean pairwise cosine distance among sampled patch embeddings

Usage:
    python scripts/sampler_diagnostics.py \
        --configs configs/appendix/phase1_mean_random.yaml \
                  configs/appendix/phase1_mean_spatial.yaml \
                  configs/appendix/phase1_mean_feature_diverse.yaml \
        --split train \
        --repeats 3 \
        --out outputs/sampler_diagnostics
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.feature_provider import UniFeatureProvider
from src.data.sampler import build_patch_sampler
from src.data.sampler_diagnostics import diagnose_sampler
from src.data.splits import case_grouped_stratified_split


def resolve_indices(cfg: dict, provider: UniFeatureProvider, split: str) -> list[int]:
    all_indices = list(range(len(provider)))
    train_idx, val_idx, test_idx = case_grouped_stratified_split(
        provider,
        all_indices,
        train_frac=cfg["data"].get("train_frac", 0.7),
        val_frac=cfg["data"].get("val_frac", 0.15),
        seed=cfg["data"].get("split_seed", cfg["training"]["seed"]),
    )
    if split == "train":
        return train_idx
    if split == "val":
        return val_idx
    if split == "test":
        return test_idx
    if split == "all":
        return train_idx + val_idx + test_idx
    raise ValueError(f"Unknown split: {split}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure sampler coverage/diversity on real slides.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--max_slides", type=int, default=None)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--full_reference_max_points", type=int, default=512)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for config_path in args.configs:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        provider = UniFeatureProvider(cfg["data"]["root"])
        indices = resolve_indices(cfg, provider, args.split)
        sampler = build_patch_sampler(cfg["data"])

        rows, summary = diagnose_sampler(
            provider=provider,
            indices=indices,
            sampler=sampler,
            grid_size=args.grid_size,
            repeats=args.repeats,
            base_seed=args.base_seed,
            max_slides=args.max_slides,
            full_reference_max_points=args.full_reference_max_points,
        )

        config_name = Path(config_path).stem
        detail_path = out_dir / f"{config_name}_{args.split}_details.csv"
        pd.DataFrame(rows).to_csv(detail_path, index=False)

        summary_row = {
            "config": config_name,
            "split": args.split,
            "grid_size": args.grid_size,
            "repeats": args.repeats,
            **summary,
        }
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows).sort_values("config")
    summary_path = out_dir / f"summary_{args.split}.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Sampler diagnostics summary ===")
    if summary_df.empty:
        print("No rows generated.")
    else:
        cols = [
            "config",
            "n_slides",
            "mean_sampled_fraction",
            "mean_grid_coverage_ratio",
            "mean_pairwise_cosine_distance",
            "mean_full_pairwise_cosine_distance",
        ]
        print(summary_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
