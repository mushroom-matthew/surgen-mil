"""Smoke test: create synthetic dataset, run one short training pass, verify outputs."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import zarr


def make_fake_dataset(root: Path, n_patches: int = 20) -> None:
    """Create a synthetic SurGen-like dataset in root."""
    emb_root = root / "embeddings"
    emb_root.mkdir(parents=True)

    # 4 positive (MSI HIGH), 8 negative — enough for stratified split to have >=2 of each class in test
    label_rows = []
    for case_id in range(1, 13):
        slide_id = f"SR1482_40X_HE_T{case_id}_0"
        z = zarr.open(str(emb_root / f"{slide_id}.zarr"), mode="w")
        z["features"] = np.random.randn(n_patches, 1024).astype(np.float32)
        z["coords"] = np.random.randint(0, 100, (n_patches, 2)).astype(np.float32)

        msi = "MSI HIGH" if case_id <= 4 else "NO MSI"
        mmr = "MMR loss" if case_id <= 4 else "No loss"
        label_rows.append({"case_id": case_id, "MSI": msi, "MMR": mmr})

    pd.DataFrame(label_rows).to_csv(root / "SR1482_labels.csv", index=False)
    pd.DataFrame(columns=["case_id", "mmr_loss_binary"]).to_csv(root / "SR386_labels.csv", index=False)


def make_config(root: Path) -> Path:
    cfg = {
        "data": {
            "root": str(root),
            "split_seed": 0,
            "max_patches": 10,
            "train_frac": 0.6,
            "val_frac": 0.2,
            "train_num_workers": 0,
            "eval_num_workers": 0,
            "pin_memory": False,
        },
        "model": {
            "name": "mean_pool",
            "input_dim": 1024,
            "hidden_dim": 64,
            "dropout": 0.0,
        },
        "optimizer": {
            "lr": 0.001,
            "weight_decay": 0.0001,
        },
        "loss": {
            "name": "bce",
            "weighted": True,
        },
        "training": {
            "epochs": 2,
            "seed": 42,
            "min_epochs": 1,
            "early_stopping_patience": 5,
            "selection_metric": "val_auprc",
            "ema_alpha": None,
        },
        "output": {
            "dir": str(root / "outputs" / "smoke"),
        },
    }
    config_path = root / "smoke_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return config_path


def main() -> int:
    repo_root = Path(__file__).parent.parent

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        print("Creating synthetic dataset...")
        make_fake_dataset(tmp_root)
        config_path = make_config(tmp_root)

        print("Running training pass...")
        result = subprocess.run(
            [sys.executable, str(repo_root / "train.py"), "--config", str(config_path)],
            capture_output=False,
        )

        if result.returncode != 0:
            print("\nFAIL: training exited with non-zero code")
            return 1

        out_dir = tmp_root / "outputs" / "smoke" / "runs" / "001"
        missing = []
        for fname in ("metrics.json", "predictions.csv"):
            if not (out_dir / fname).exists():
                missing.append(fname)

        if missing:
            print(f"\nFAIL: missing output files: {missing}")
            return 1

        print("\nPASS: smoke test complete")
        return 0


if __name__ == "__main__":
    sys.exit(main())
