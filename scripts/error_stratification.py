"""
Error stratification for the best Attention MIL model.

Joins test-set predictions with SurGen metadata and produces:
  - FN / FP cases by cohort, site, mutation subgroup
  - Mean predicted score by subgroup
  - Per-subgroup AUROC / AUPRC where sample size permits

Usage:
    python scripts/error_stratification.py \
        --root /mnt/data-surgen \
        --predictions outputs/uni_attention/predictions.csv \
        --split test \
        --out outputs/error_stratification/
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

SITE_COLS = ["site", "tumour_site", "tumor_site", "primary_site", "location"]
META_COLS = ["primary_metastatic", "primary_metastasis", "metastatic", "tissue_type"]
MUTATION_COLS = ["KRAS", "NRAS", "BRAF", "kras", "nras", "braf"]
ASSAY_COLS = ["msi_method", "assay", "mmr_method", "test_method"]

MIN_AUPRC_N = 2  # minimum positives to attempt AUPRC


def first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_auroc(y_true, y_score) -> float | None:
    u = np.unique(y_true)
    if len(u) < 2:
        return None
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return None


def safe_auprc(y_true, y_score) -> float | None:
    if y_true.sum() < MIN_AUPRC_N:
        return None
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
    	return None


def subgroup_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for val, g in df.groupby(group_col):
        y_true = g["label"].values
        y_score = g["prob"].values
        auroc = safe_auroc(y_true, y_score)
        auprc = safe_auprc(y_true, y_score)
        rows.append({
            group_col: val,
            "n_slides": len(g),
            "n_pos": int(y_true.sum()),
            "n_neg": int((1 - y_true).sum()),
            "mean_prob_pos": round(float(y_score[y_true == 1].mean()), 3) if y_true.sum() > 0 else None,
            "mean_prob_neg": round(float(y_score[y_true == 0].mean()), 3) if (1 - y_true).sum() > 0 else None,
            "auroc": round(auroc, 3) if auroc is not None else None,
            "auprc": round(auprc, 3) if auprc is not None else None,
        })
    return pd.DataFrame(rows)


def error_cases(df: pd.DataFrame, error_type: str, meta_cols: list[str]) -> pd.DataFrame:
    assert error_type in ("fn", "fp")
    if error_type == "fn":
        sub = df[(df["label"] == 1) & (df["pred"] == 0)].copy()
    else:
        sub = df[(df["label"] == 0) & (df["pred"] == 1)].copy()
    cols = ["slide_id", "cohort", "case_id", "label", "prob"] + [c for c in meta_cols if c in df.columns]
    return sub[cols].sort_values("prob", ascending=(error_type == "fn"))


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _resolve_predictions(path: str) -> str:
    """Accept either a direct predictions.csv path or a model base/run dir."""
    p = Path(path)
    if p.is_file():
        return str(p)
    # versioned base dir
    latest = p / "latest"
    if latest.exists():
        cand = latest.resolve() / "predictions.csv"
        if cand.exists():
            return str(cand)
    runs = p / "runs"
    if runs.is_dir():
        versions = sorted(d for d in runs.iterdir() if d.is_dir() and d.name.isdigit())
        for d in reversed(versions):
            cand = d / "predictions.csv"
            if cand.exists():
                return str(cand)
    # flat layout
    cand = p / "predictions.csv"
    if cand.exists():
        return str(cand)
    return path   # fall through; will raise naturally on read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/data-surgen")
    parser.add_argument("--predictions", default="outputs/uni_attention",
                        help="Path to predictions.csv or a model output dir (versioned layout supported)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Operating threshold (use val-tuned threshold if available)")
    parser.add_argument("--out", default="outputs/error_stratification")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)

    # ------------------------------------------------------------------ #
    # Load predictions
    # ------------------------------------------------------------------ #
    preds_path = _resolve_predictions(args.predictions)
    preds = pd.read_csv(preds_path)
    if args.split != "all":
        preds = preds[preds["split"] == args.split].reset_index(drop=True)
    print(f"Predictions loaded: {len(preds)} slides (split={args.split})")

    # Parse cohort and case_id from slide_id
    pat = re.compile(r"^(SR\d+)_40X_HE_T(\d+)_\d+$")
    preds["cohort"] = preds["slide_id"].apply(lambda s: pat.match(s).group(1) if pat.match(s) else None)
    preds["case_id"] = preds["slide_id"].apply(
        lambda s: int(pat.match(s).group(2)) if pat.match(s) else None
    )
    preds["pred"] = (preds["prob"] >= args.threshold).astype(int)

    # ------------------------------------------------------------------ #
    # Load metadata
    # ------------------------------------------------------------------ #
    sr1482 = pd.read_csv(root / "SR1482_labels.csv")
    sr386 = pd.read_csv(root / "SR386_labels.csv")

    # Normalise mutation column names to uppercase
    for df in [sr1482, sr386]:
        for col in list(df.columns):
            if col.upper() in ["KRAS", "NRAS", "BRAF"] and col != col.upper():
                df.rename(columns={col: col.upper()}, inplace=True)

    # Find which metadata columns are available
    available_meta: dict[str, list[str]] = {}
    for df, cohort in [(sr1482, "SR1482"), (sr386, "SR386")]:
        cols = []
        for cands in [SITE_COLS, META_COLS, ASSAY_COLS]:
            c = first_present(df, cands)
            if c:
                cols.append(c)
        for m in ["KRAS", "NRAS", "BRAF"]:
            if m in df.columns:
                cols.append(m)
        available_meta[cohort] = cols
        print(f"{cohort} metadata columns to merge: {cols}")

    # Join per cohort
    merged_parts = []
    for df, cohort in [(sr1482, "SR1482"), (sr386, "SR386")]:
        sub = preds[preds["cohort"] == cohort].copy()
        if sub.empty:
            continue
        extra_cols = ["case_id"] + available_meta.get(cohort, [])
        meta = df[[c for c in extra_cols if c in df.columns]].drop_duplicates("case_id")
        merged = sub.merge(meta, on="case_id", how="left")
        merged_parts.append(merged)

    if not merged_parts:
        print("No predictions after merge — check cohort names / split filter.")
        return

    df = pd.concat(merged_parts, ignore_index=True)
    all_meta_cols = list({c for cols in available_meta.values() for c in cols})

    # ------------------------------------------------------------------ #
    # Overall summary
    # ------------------------------------------------------------------ #
    section("Overall summary")
    y_true = df["label"].values
    y_score = df["prob"].values
    auroc = safe_auroc(y_true, y_score)
    auprc = safe_auprc(y_true, y_score)
    print(f"N slides: {len(df)}, N pos: {int(y_true.sum())}, N neg: {int((1-y_true).sum())}")
    print(f"AUROC: {auroc:.3f}" if auroc else "AUROC: n/a")
    print(f"AUPRC: {auprc:.3f}" if auprc else "AUPRC: n/a")
    tp = int(((df['pred'] == 1) & (df['label'] == 1)).sum())
    fp = int(((df['pred'] == 1) & (df['label'] == 0)).sum())
    fn = int(((df['pred'] == 0) & (df['label'] == 1)).sum())
    tn = int(((df['pred'] == 0) & (df['label'] == 0)).sum())
    print(f"At threshold={args.threshold}: TP={tp} FP={fp} FN={fn} TN={tn}")

    # ------------------------------------------------------------------ #
    # Per-cohort metrics
    # ------------------------------------------------------------------ #
    section("Metrics by cohort")
    ct = subgroup_metrics(df, "cohort")
    print(ct.to_string(index=False))
    ct.to_csv(out_dir / "metrics_by_cohort.csv", index=False)

    # ------------------------------------------------------------------ #
    # Per-stratum metrics for each available metadata column
    # ------------------------------------------------------------------ #
    for col in all_meta_cols:
        if col not in df.columns:
            continue
        n_vals = df[col].nunique(dropna=True)
        if n_vals == 0:
            continue
        section(f"Metrics by {col}")
        mt = subgroup_metrics(df, col)
        print(mt.to_string(index=False))
        mt.to_csv(out_dir / f"metrics_by_{col}.csv", index=False)

    # ------------------------------------------------------------------ #
    # False negatives
    # ------------------------------------------------------------------ #
    section(f"False Negatives (threshold={args.threshold})")
    fn_df = error_cases(df, "fn", all_meta_cols)
    if fn_df.empty:
        print("  None")
    else:
        print(fn_df.to_string(index=False))
    fn_df.to_csv(out_dir / "false_negatives.csv", index=False)

    # ------------------------------------------------------------------ #
    # False positives
    # ------------------------------------------------------------------ #
    section(f"False Positives (threshold={args.threshold})")
    fp_df = error_cases(df, "fp", all_meta_cols)
    if fp_df.empty:
        print("  None")
    else:
        print(fp_df.to_string(index=False))
    fp_df.to_csv(out_dir / "false_positives.csv", index=False)

    # ------------------------------------------------------------------ #
    # Score distribution summary per subgroup
    # ------------------------------------------------------------------ #
    section("Score distribution by cohort × label")
    for (cohort, label), g in df.groupby(["cohort", "label"]):
        desc = g["prob"].describe().round(3)
        tag = "POS" if label == 1 else "NEG"
        print(f"\n  {cohort} {tag} (n={len(g)}):")
        print(f"    mean={desc['mean']:.3f}  std={desc['std']:.3f}  "
              f"min={desc['min']:.3f}  median={desc['50%']:.3f}  max={desc['max']:.3f}")

    # ------------------------------------------------------------------ #
    # Save full merged table
    # ------------------------------------------------------------------ #
    df.to_csv(out_dir / "predictions_with_metadata.csv", index=False)
    print(f"\nOutputs written to {out_dir}/")


if __name__ == "__main__":
    main()
