"""
Metadata audit for SurGen label tables.

Produces:
  - Label distribution by cohort
  - Label distribution by site / primary vs metastatic (if available)
  - Mutation co-occurrence (KRAS, NRAS, BRAF × MSI/MMR)
  - Missingness table for all columns
  - Multi-slide case counts and label consistency check
  - Per-cohort slide counts

Usage:
    python scripts/metadata_audit.py --root /mnt/data-surgen
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.feature_provider import UniFeatureProvider


SITE_COLS = ["site", "tumour_site", "tumor_site", "primary_site", "location"]
META_COLS = ["primary_metastatic", "primary_metastasis", "metastatic", "tissue_type"]
MUTATION_COLS = ["KRAS", "NRAS", "BRAF", "kras", "nras", "braf"]
ASSAY_COLS = ["msi_method", "assay", "mmr_method", "test_method"]


def first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def missingness_table(df: pd.DataFrame, label: str) -> None:
    section(f"Missingness — {label}")
    total = len(df)
    rows = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct = 100 * n_missing / total if total else 0
        rows.append({"column": col, "n_missing": n_missing, "pct_missing": f"{pct:.1f}%", "dtype": str(df[col].dtype)})
    print(pd.DataFrame(rows).to_string(index=False))


def cross_tab(df: pd.DataFrame, row_col: str, col_col: str, label: str) -> None:
    if row_col not in df.columns or col_col not in df.columns:
        return
    sub = df[[row_col, col_col]].dropna()
    if sub.empty:
        print(f"\n[{label}] — no data after dropping NaN")
        return
    ct = pd.crosstab(sub[row_col], sub[col_col], margins=True, margins_name="Total")
    print(f"\n{label}")
    print(ct.to_string())


def parse_slide_ids(zarr_root: Path) -> pd.DataFrame:
    """Build a slide-level DataFrame from available zarr files."""
    pattern = re.compile(r"^(SR\d+)_40X_HE_T(\d+)_(\d+)$")
    rows = []
    for p in sorted(zarr_root.glob("*.zarr")):
        m = pattern.match(p.stem)
        if not m:
            continue
        rows.append({
            "slide_id": p.stem,
            "cohort": m.group(1),
            "case_id": int(m.group(2)),
            "slide_num": int(m.group(3)),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/data-surgen")
    args = parser.parse_args()
    root = Path(args.root)

    # ------------------------------------------------------------------ #
    # Load label tables
    # ------------------------------------------------------------------ #
    sr1482 = pd.read_csv(root / "SR1482_labels.csv")
    sr386 = pd.read_csv(root / "SR386_labels.csv")
    sr1482["cohort"] = "SR1482"
    sr386["cohort"] = "SR386"

    # Normalise label columns to a shared explicit state + binary label.
    sr1482_states = UniFeatureProvider._build_sr1482_label_states(sr1482)
    sr1482 = sr1482.merge(sr1482_states, on="case_id", how="left")
    sr1482["msi_label"] = sr1482["state"].map({"positive": 1, "negative": 0})
    sr1482.rename(columns={"state": "label_state", "basis": "label_basis"}, inplace=True)
    if "mmr_loss_binary" in sr386.columns:
        sr386["msi_label"] = pd.to_numeric(sr386["mmr_loss_binary"], errors="coerce")
        sr386["label_state"] = sr386["msi_label"].map({1: "positive", 0: "negative"})
        sr386["label_basis"] = "mmr_loss_binary"

    # ------------------------------------------------------------------ #
    # Slide manifest from filesystem
    # ------------------------------------------------------------------ #
    zarr_root = root / "embeddings"
    slides_df = parse_slide_ids(zarr_root)
    section("Slide Manifest")
    print(f"Total zarr files found: {len(slides_df)}")
    print(slides_df.groupby("cohort").size().rename("n_slides").to_string())

    # ------------------------------------------------------------------ #
    # Per-cohort label distribution
    # ------------------------------------------------------------------ #
    for df, name, label_col in [
        (sr1482, "SR1482", "msi_label"),
        (sr386, "SR386", "msi_label"),
    ]:
        section(f"Label distribution — {name}")
        if label_col not in df.columns:
            print("  label column not found")
            continue
        counts = df[label_col].value_counts(dropna=False).rename("n_cases")
        counts.index = counts.index.map(
            lambda x: "positive (MSI/MMR-loss)" if x == 1 else ("negative (MSS/MMR-intact)" if x == 0 else "missing")
        )
        print(counts.to_string())
        pct = counts / counts.sum() * 100
        print(pct.rename("pct_cases").map(lambda x: f"{x:.1f}%").to_string())
        if "label_state" in df.columns:
            print("\nlabel_state counts:")
            print(df["label_state"].value_counts(dropna=False).to_string())

    # ------------------------------------------------------------------ #
    # Multi-slide cases
    # ------------------------------------------------------------------ #
    section("Multi-slide cases")
    if not slides_df.empty:
        slide_counts = slides_df.groupby(["cohort", "case_id"]).size().rename("n_slides")
        multi = slide_counts[slide_counts > 1]
        print(f"Cases with >1 slide: {len(multi)}")
        if not multi.empty:
            print(multi.to_string())

    # ------------------------------------------------------------------ #
    # Label consistency for multi-slide cases
    # ------------------------------------------------------------------ #
    section("Label consistency across slides")
    for df, name, id_col, label_col in [
        (sr1482, "SR1482", "case_id", "msi_label"),
        (sr386, "SR386", "case_id", "msi_label"),
    ]:
        if id_col not in df.columns or label_col not in df.columns:
            continue
        dup = df[df.duplicated(id_col, keep=False)]
        if dup.empty:
            print(f"  {name}: no duplicate case_ids in label table")
            continue
        inconsistent = dup.groupby(id_col)[label_col].nunique()
        bad = inconsistent[inconsistent > 1]
        print(f"  {name}: {len(dup[id_col].unique())} case_ids appear more than once; "
              f"{len(bad)} have inconsistent labels")
        if not bad.empty:
            print(dup[dup[id_col].isin(bad.index)][[id_col, label_col]].to_string(index=False))

    # ------------------------------------------------------------------ #
    # Site / primary vs metastatic
    # ------------------------------------------------------------------ #
    for df, name in [(sr1482, "SR1482"), (sr386, "SR386")]:
        site_col = first_present(df, SITE_COLS)
        meta_col = first_present(df, META_COLS)
        if site_col:
            cross_tab(df, site_col, "msi_label", f"{name}: label × {site_col}")
        if meta_col:
            cross_tab(df, meta_col, "msi_label", f"{name}: label × {meta_col}")
        if not site_col and not meta_col:
            print(f"\n{name}: no site/metastatic columns found")

    # ------------------------------------------------------------------ #
    # Mutation co-occurrence
    # ------------------------------------------------------------------ #
    for df, name in [(sr1482, "SR1482"), (sr386, "SR386")]:
        found = [c for c in MUTATION_COLS if c in df.columns]
        if not found:
            print(f"\n{name}: no mutation columns found (checked {MUTATION_COLS})")
            continue
        for mut_col in found:
            cross_tab(df, mut_col, "msi_label", f"{name}: MSI/MMR × {mut_col}")

    # ------------------------------------------------------------------ #
    # Assay / label source
    # ------------------------------------------------------------------ #
    for df, name in [(sr1482, "SR1482"), (sr386, "SR386")]:
        assay_col = first_present(df, ASSAY_COLS)
        if assay_col:
            cross_tab(df, assay_col, "msi_label", f"{name}: label × {assay_col}")
        else:
            print(f"\n{name}: no assay/method column found")

    # ------------------------------------------------------------------ #
    # Missingness
    # ------------------------------------------------------------------ #
    missingness_table(sr1482, "SR1482")
    missingness_table(sr386, "SR386")

    # ------------------------------------------------------------------ #
    # Column inventory
    # ------------------------------------------------------------------ #
    section("Column inventory")
    print(f"\nSR1482 columns ({len(sr1482.columns)}): {list(sr1482.columns)}")
    print(f"SR386  columns ({len(sr386.columns)}): {list(sr386.columns)}")


if __name__ == "__main__":
    main()
