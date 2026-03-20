"""
Print a formatted failure manifest cross-referenced against slide metadata.

Reads the failure manifest CSV produced by export_failure_manifest.py and joins
it with SR1482_labels.csv to show per-error-category breakdowns by tumour site,
stage, mutation status, sex, and age.

Usage:
    python scripts/failures/failure_report.py \
        --manifest outputs/failure_manifest.csv \
        --labels /mnt/data-surgen/SR1482_labels.csv \
        [--threshold 0.5] [--min_models 1]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_case_id(slide_id: str) -> int:
    m = re.match(r"^SR\d+_40X_HE_T(\d+)_\d+$", slide_id)
    return int(m.group(1)) if m else -1


def _fmt_val(v) -> str:
    if pd.isna(v) or v == "" or v is None:
        return "—"
    return str(v).strip()


# Columns to pull from the labels CSV
META_COLS = ["tumour_site", "dukes", "pT", "pN", "sex", "age", "KRAS", "NRAS", "BRAF"]


def _load_metadata(labels_path: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path)
    # Normalise case_id to int
    df["case_id"] = pd.to_numeric(df["case_id"], errors="coerce").astype("Int64")
    keep = ["case_id"] + [c for c in META_COLS if c in df.columns]
    return df[keep].drop_duplicates("case_id")


def _join_metadata(manifest: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.copy()
    manifest["case_id_int"] = manifest["slide_id"].apply(_extract_case_id)
    manifest = manifest.merge(
        meta.rename(columns={"case_id": "case_id_int"}),
        on="case_id_int",
        how="left",
    )
    return manifest


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

DISPLAY_COLS = ["slide_id", "label", "n_models_fp", "n_models_fn",
                "tumour_site", "dukes", "pT", "pN", "sex", "age",
                "KRAS", "NRAS", "BRAF"]

PROB_COL_ORDER = None  # filled from manifest columns


def _prob_cols(manifest: pd.DataFrame) -> list[str]:
    return sorted(c for c in manifest.columns if c.startswith("prob_"))


def _print_table(df: pd.DataFrame, prob_cols: list[str], title: str, max_rows: int = 40) -> None:
    if df.empty:
        print(f"\n{title}\n  (none)\n")
        return

    # Select display columns present in df
    show = [c for c in DISPLAY_COLS if c in df.columns] + prob_cols
    out = df[show].copy()

    # Round probabilities
    for c in prob_cols:
        out[c] = out[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

    print(f"\n{'=' * 80}")
    print(f"  {title}  (n={len(df)})")
    print(f"{'=' * 80}")
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 30)
    print(out.fillna("—").to_string(index=False))


def _print_breakdown(df: pd.DataFrame, col: str, label: str) -> None:
    if col not in df.columns or df.empty:
        return
    counts = df[col].fillna("unknown").value_counts()
    print(f"\n  By {label}:")
    for val, cnt in counts.items():
        print(f"    {val:<40s}  n={cnt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Print failure manifest cross-referenced with slide metadata."
    )
    parser.add_argument("--manifest", default="outputs/failure_manifest.csv",
                        help="Path to failure_manifest.csv from export_failure_manifest.py")
    parser.add_argument("--labels", default="/mnt/data-surgen/SR1482_labels.csv",
                        help="Path to SR1482_labels.csv for metadata join")
    parser.add_argument("--min_models", type=int, default=1,
                        help="Minimum number of models that must agree on the error (default: 1 = any)")
    parser.add_argument("--sort", default="n_models",
                        choices=["n_models", "prob", "tumour_site", "dukes", "age"],
                        help="Sort order within each error table (default: n_models)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}")
        print("Run first:  python scripts/failures/export_failure_manifest.py --out outputs/failure_manifest.csv")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    prob_cols = _prob_cols(manifest)

    # Join metadata if available
    labels_path = Path(args.labels)
    if labels_path.exists():
        meta = _load_metadata(labels_path)
        manifest = _join_metadata(manifest, meta)
        has_meta = True
    else:
        print(f"Warning: labels file not found at {labels_path} — metadata columns omitted")
        has_meta = False

    # Sort helpers
    def _sort(df: pd.DataFrame, err_col: str) -> pd.DataFrame:
        if args.sort == "n_models":
            return df.sort_values(err_col, ascending=False)
        elif args.sort == "prob" and prob_cols:
            mean_prob = df[prob_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            return df.assign(_mp=mean_prob).sort_values("_mp", ascending=False).drop(columns="_mp")
        elif args.sort in df.columns:
            return df.sort_values(args.sort)
        return df

    # ---- Summary header ----
    n_models = len(prob_cols)
    min_m = args.min_models
    print(f"\nFailure Manifest Report")
    print(f"  manifest: {manifest_path}  ({len(manifest)} slides, {n_models} models)")
    print(f"  min_models threshold: {min_m}")
    if has_meta:
        print(f"  metadata: {labels_path}")

    # ---- False Negatives ----
    fn = manifest[(manifest["label"] == 1) & (manifest["n_models_fn"] >= min_m)].copy()
    fn = _sort(fn, "n_models_fn")
    _print_table(fn, prob_cols,
                 f"FALSE NEGATIVES  (missed positives, n_models_fn >= {min_m})")
    if has_meta:
        _print_breakdown(fn, "tumour_site", "tumour site")
        _print_breakdown(fn, "dukes", "Dukes stage")
        _print_breakdown(fn, "KRAS", "KRAS mutation")
        _print_breakdown(fn, "BRAF", "BRAF mutation")

    # ---- False Positives ----
    fp = manifest[(manifest["label"] == 0) & (manifest["n_models_fp"] >= min_m)].copy()
    fp = _sort(fp, "n_models_fp")
    _print_table(fp, prob_cols,
                 f"FALSE POSITIVES  (flagged negatives, n_models_fp >= {min_m})")
    if has_meta:
        _print_breakdown(fp, "tumour_site", "tumour site")
        _print_breakdown(fp, "dukes", "Dukes stage")
        _print_breakdown(fp, "KRAS", "KRAS mutation")
        _print_breakdown(fp, "BRAF", "BRAF mutation")

    # ---- Consistent TN (most confidently correct negatives — for context) ----
    tn = manifest[(manifest["label"] == 0) & (manifest["n_models_fp"] == 0)].copy()
    if prob_cols:
        mean_prob = tn[prob_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        tn = tn.assign(_mp=mean_prob).sort_values("_mp").drop(columns="_mp")
    _print_table(tn.head(10), prob_cols,
                 "CONSISTENT TRUE NEGATIVES  (all models agree, lowest mean prob, top 10)")

    # ---- Consistent TP ----
    tp = manifest[(manifest["label"] == 1) & (manifest["n_models_fn"] == 0)].copy()
    if prob_cols:
        mean_prob = tp[prob_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        tp = tp.assign(_mp=mean_prob).sort_values("_mp", ascending=False).drop(columns="_mp")
    _print_table(tp.head(10), prob_cols,
                 "CONSISTENT TRUE POSITIVES  (all models agree, highest mean prob, top 10)")

    # ---- Summary stats ----
    print(f"\n{'=' * 80}")
    print("  Summary statistics")
    print(f"{'=' * 80}")
    print(f"  Total slides:           {len(manifest)}")
    print(f"  Positives (label=1):    {(manifest['label']==1).sum()}")
    print(f"  Negatives (label=0):    {(manifest['label']==0).sum()}")
    print(f"  FN (any model):         {(manifest['n_models_fn'] >= 1).sum()}")
    print(f"  FP (any model):         {(manifest['n_models_fp'] >= 1).sum()}")
    print(f"  FN (all {n_models} models):       {(manifest['n_models_fn'] == n_models).sum()}")
    print(f"  FP (all {n_models} models):       {(manifest['n_models_fp'] == n_models).sum()}")

    if has_meta and "age" in manifest.columns:
        fn_ages = pd.to_numeric(fn["age"], errors="coerce").dropna()
        fp_ages = pd.to_numeric(fp["age"], errors="coerce").dropna()
        all_ages = pd.to_numeric(manifest["age"], errors="coerce").dropna()
        if len(fn_ages) and len(all_ages):
            print(f"\n  Mean age — all: {all_ages.mean():.1f}, FN: {fn_ages.mean():.1f}, FP: {fp_ages.mean():.1f}")


if __name__ == "__main__":
    main()
