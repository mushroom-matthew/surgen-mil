"""
Build a master failure manifest across all available model runs.

For each slide in any split, records:
  slide_id, cohort, case_id, true_label, split,
  prob_{model_key}, is_fp_{model_key}, is_fn_{model_key}   (one set per model)
  n_models_fp, n_models_fn   (how many models make this error)

Usage:
    python scripts/failures/export_failure_manifest.py \
        --threshold 0.5 \
        --split test \
        --out outputs/failure_manifest.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Fair-comparison runs (split_seed=0, seeds 42/123/456) — the canonical set.
# Each model's runs/ directory is scanned for all numbered seeds; predictions
# are averaged across seeds before computing FP/FN flags.
RUNS = {
    "mean_pool":      "outputs/uni_mean_fair",
    "attention_mil":  "outputs/uni_attention_fair",
    "transformer":    "outputs/paper_reproduction_fair",
}


def extract_cohort(slide_id: str) -> str:
    return slide_id.split("_")[0]


def extract_case_id(slide_id: str) -> int:
    m = re.match(r"^SR\d+_40X_HE_T(\d+)_\d+$", slide_id)
    return int(m.group(1)) if m else -1


def youden_threshold(val_df: pd.DataFrame, n: int = 500) -> float:
    """Threshold that maximises Youden's J (sensitivity + specificity - 1) on val_df."""
    import numpy as np
    y_true  = val_df["label"].values
    y_score = val_df["prob"].values
    if len(set(y_true)) < 2:
        return 0.5
    best_j, best_t = -1.0, 0.5
    for t in np.linspace(0, 1, n):
        y_pred = (y_score >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        j = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t


def load_averaged_predictions(base_dir: Path, split: str) -> pd.DataFrame | None:
    """Load predictions.csv from all numbered seed runs and average prob per slide.

    Returns a DataFrame with columns [slide_id, label, prob, split, n_seeds],
    where prob is the mean predicted probability across all seeds.  If the runs/
    directory does not exist or contains no valid predictions, returns None.
    """
    runs_dir = base_dir / "runs"
    frames = []
    if runs_dir.is_dir():
        for d in sorted(runs_dir.iterdir()):
            if not (d.is_dir() and d.name.isdigit()):
                continue
            p = d / "predictions.csv"
            if p.exists():
                frames.append(pd.read_csv(p))

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    if split != "all":
        combined = combined[combined["split"] == split]

    if combined.empty:
        return None

    averaged = (
        combined.groupby("slide_id")
        .agg(label=("label", "first"), prob=("prob", "mean"),
             split=("split", "first"), n_seeds=("prob", "count"))
        .reset_index()
    )
    return averaged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="outputs/failure_manifest.csv")
    parser.add_argument(
        "--min_models", type=int, default=None,
        help="Minimum number of models that must make the error to appear in summary "
             "(default: ceil(n_models / 2))",
    )
    args = parser.parse_args()

    frames: dict[str, pd.DataFrame] = {}     # test predictions, indexed by slide_id
    thresholds: dict[str, float] = {}         # per-model optimal val threshold
    for key, run_dir in RUNS.items():
        test_df = load_averaged_predictions(Path(run_dir), args.split)
        if test_df is None:
            print(f"  SKIP {key} (no runs found in {run_dir})")
            continue
        n_seeds = int(test_df["n_seeds"].max())

        val_df = load_averaged_predictions(Path(run_dir), "val")
        if val_df is not None and val_df["label"].nunique() >= 2:
            t = youden_threshold(val_df)
            source = "Youden J (val)"
        else:
            t = args.threshold
            source = f"fallback ({args.threshold})"

        thresholds[key] = t
        frames[key] = test_df.set_index("slide_id")
        print(f"  loaded {key}  ({len(test_df)} slides, {n_seeds} seeds averaged)"
              f"  threshold={t:.3f}  [{source}]")

    if not frames:
        print("No runs found.")
        return

    # Union of all slide_ids present across any run
    all_slides = sorted(set.union(*[set(df.index) for df in frames.values()]))

    rows = []
    for slide_id in all_slides:
        # base info from whichever run has this slide
        ref = next(df.loc[slide_id] for df in frames.values() if slide_id in df.index)
        row: dict = {
            "slide_id": slide_id,
            "cohort": extract_cohort(slide_id),
            "case_id": extract_case_id(slide_id),
            "label": int(ref["label"]),
            "split": ref["split"],
        }
        n_fp = n_fn = 0
        for key, df in frames.items():
            if slide_id not in df.index:
                row[f"prob_{key}"] = None
                row[f"is_fp_{key}"] = None
                row[f"is_fn_{key}"] = None
                continue
            r = df.loc[slide_id]
            prob = float(r["prob"])
            label = int(r["label"])
            pred = int(prob >= thresholds[key])
            is_fp = int(label == 0 and pred == 1)
            is_fn = int(label == 1 and pred == 0)
            row[f"prob_{key}"] = round(prob, 4)
            row[f"is_fp_{key}"] = is_fp
            row[f"is_fn_{key}"] = is_fn
            n_fp += is_fp
            n_fn += is_fn
        row["n_models_fp"] = n_fp
        row["n_models_fn"] = n_fn
        rows.append(row)

    manifest = pd.DataFrame(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)

    # Summary prints
    n_models = len(frames)
    min_models = args.min_models if args.min_models is not None else -(-n_models // 2)
    threshold_str = ", ".join(f"{k}={v:.3f}" for k, v in thresholds.items())
    print(
        f"\nManifest: {len(manifest)} slides, {n_models} models (seeds averaged), "
        f"thresholds=[{threshold_str}], showing errors in >= {min_models} models"
    )

    pos = manifest[manifest["label"] == 1]
    neg = manifest[manifest["label"] == 0]

    prob_cols = [f"prob_{k}" for k in frames]

    print(f"\nCommon false negatives (missed by >= {min_models}/{n_models} models):")
    pfn = (
        pos[pos["n_models_fn"] >= min_models]
        .sort_values("n_models_fn", ascending=False)
    )
    print(pfn[["slide_id", "cohort", "case_id", "n_models_fn"] + prob_cols].to_string(index=False))

    print(f"\nCommon false positives (flagged by >= {min_models}/{n_models} models):")
    pfp = (
        neg[neg["n_models_fp"] >= min_models]
        .sort_values("n_models_fp", ascending=False)
    )
    print(pfp[["slide_id", "cohort", "case_id", "n_models_fp"] + prob_cols].to_string(index=False))

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
