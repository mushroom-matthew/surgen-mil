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

RUNS = {
    "mean_pool":         "outputs/uni_mean",
    "attention_mil":     "outputs/uni_attention",
    "gated_attention":   "outputs/uni_gated_attention",
    "topk_k4":           "outputs/uni_topk_attention_k4",
}


def _resolve_predictions(base_dir: Path) -> Path | None:
    """Return path to predictions.csv, handling versioned and flat layouts."""
    # flat layout
    flat = base_dir / "predictions.csv"
    if flat.exists():
        return flat
    # versioned: follow latest symlink
    latest = base_dir / "latest"
    if latest.exists():
        p = latest.resolve() / "predictions.csv"
        if p.exists():
            return p
    # versioned: highest-numbered run
    runs = base_dir / "runs"
    if runs.is_dir():
        versions = sorted(d for d in runs.iterdir() if d.is_dir() and d.name.isdigit())
        for d in reversed(versions):
            p = d / "predictions.csv"
            if p.exists():
                return p
    return None


def extract_cohort(slide_id: str) -> str:
    return slide_id.split("_")[0]


def extract_case_id(slide_id: str) -> int:
    m = re.match(r"^SR\d+_40X_HE_T(\d+)_\d+$", slide_id)
    return int(m.group(1)) if m else -1


def load_run(run_dir: Path, split: str) -> pd.DataFrame | None:
    p = _resolve_predictions(run_dir)
    if p is None:
        return None
    df = pd.read_csv(p)
    if split != "all":
        df = df[df["split"] == split].reset_index(drop=True)
    return df


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

    frames: dict[str, pd.DataFrame] = {}
    for key, run_dir in RUNS.items():
        df = load_run(Path(run_dir), args.split)
        if df is None:
            print(f"  SKIP {key} (no predictions.csv)")
            continue
        frames[key] = df.set_index("slide_id")
        print(f"  loaded {key}  ({len(df)} slides)")

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
            pred = int(prob >= args.threshold)
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
    print(
        f"\nManifest: {len(manifest)} slides, {n_models} models, "
        f"threshold={args.threshold}, showing errors in >= {min_models} models"
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
