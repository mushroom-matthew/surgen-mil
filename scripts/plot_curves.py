"""
Generate ROC and PR curves for one or more model runs.

Usage:
    python scripts/plot_curves.py \
        outputs/uni_mean/predictions.csv \
        outputs/uni_attention/predictions.csv \
        --split test \
        --out outputs/curves.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
)


def load(path: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["split"] == split].reset_index(drop=True)


def label_from_path(path: str) -> str:
    return Path(path).parent.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+", help="predictions.csv files")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default="outputs/curves.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_roc, ax_pr = axes

    for path in args.predictions:
        df = load(path, args.split)
        if df.empty:
            print(f"WARNING: no rows for split={args.split} in {path}")
            continue

        y_true = df["label"].values
        y_score = df["prob"].values
        name = label_from_path(path)

        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

        RocCurveDisplay.from_predictions(
            y_true, y_score,
            name=f"{name} (AUROC={auroc:.3f})",
            ax=ax_roc,
        )
        PrecisionRecallDisplay.from_predictions(
            y_true, y_score,
            name=f"{name} (AUPRC={auprc:.3f})",
            ax=ax_pr,
        )

    ax_roc.set_title(f"ROC Curve ({args.split})")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)

    ax_pr.set_title(f"Precision-Recall Curve ({args.split})")

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
