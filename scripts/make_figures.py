"""Generate ROC/PR/variance plots from a comparison summary CSV or config list."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures from comparison outputs or config list."
    )
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Config YAMLs (alternative to --summary + --preds)")
    parser.add_argument("--summary", default=None,
                        help="Path to summary.csv produced by compare_models.py")
    parser.add_argument("--out", default="outputs/figures", help="Output directory for figures")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.configs:
        # Delegate to compare_models logic
        import subprocess, sys
        cmd = [sys.executable, str(Path(__file__).parent / "compare_models.py"),
               "--configs"] + args.configs + ["--out", str(out_dir)]
        subprocess.run(cmd, check=True)
        return

    if args.summary:
        import pandas as pd
        import matplotlib.pyplot as plt
        summary_df = pd.read_csv(args.summary)

        fig, ax = plt.subplots(figsize=(6, 4))
        stds = summary_df["auroc_std"].fillna(0)
        ax.bar(summary_df["model"], stds, color="steelblue")
        ax.set_ylabel("AUROC std across seeds")
        ax.set_title("Seed Variance by Model")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        out_path = out_dir / "seed_variance.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Variance chart saved to {out_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
