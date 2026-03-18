"""
Inspect slide-per-case distribution and label consistency.

Usage:
    python scripts/inspect_cases.py
"""
from collections import defaultdict

import pandas as pd

from src.data.feature_provider import UniFeatureProvider

provider = UniFeatureProvider("/mnt/data-surgen")

# Group slides by (cohort, case_id)
groups = defaultdict(list)
for idx, rec in enumerate(provider.records):
    groups[(rec.cohort, rec.case_id)].append(rec)

# Build a per-case summary
rows = []
for (cohort, case_id), recs in sorted(groups.items()):
    labels = [r.label for r in recs]
    rows.append({
        "cohort": cohort,
        "case_id": case_id,
        "n_slides": len(recs),
        "label": labels[0],
        "label_consistent": len(set(labels)) == 1,
        "slide_ids": ", ".join(r.slide_id for r in recs),
    })

df = pd.DataFrame(rows)

print("=== Overall ===")
print(f"  Total cases:  {len(df)}")
print(f"  Total slides: {df['n_slides'].sum()}")
print(f"  Positive cases: {df['label'].sum()}")
print(f"  Negative cases: {(df['label'] == 0).sum()}")

print("\n=== Label consistency ===")
inconsistent = df[~df["label_consistent"]]
if inconsistent.empty:
    print("  All cases have consistent labels across slides (expected — labels are case-level).")
else:
    print(f"  WARNING: {len(inconsistent)} cases have inconsistent slide labels:")
    print(inconsistent[["cohort", "case_id", "n_slides", "slide_ids"]].to_string(index=False))

print("\n=== Slides per case ===")
print(df["n_slides"].value_counts().sort_index().rename("n_cases").to_frame().to_string())

print("\n=== Per-cohort breakdown ===")
for cohort, cdf in df.groupby("cohort"):
    print(f"\n  {cohort}")
    print(f"    cases:          {len(cdf)}")
    print(f"    slides:         {cdf['n_slides'].sum()}")
    print(f"    positive cases: {cdf['label'].sum()}")
    print(f"    negative cases: {(cdf['label'] == 0).sum()}")
    print(f"    slides/case:    min={cdf['n_slides'].min()}  max={cdf['n_slides'].max()}  mean={cdf['n_slides'].mean():.2f}")

print("\n=== Multi-slide cases ===")
multi = df[df["n_slides"] > 1].sort_values(["cohort", "case_id"])
if multi.empty:
    print("  No multi-slide cases.")
else:
    print(f"  {len(multi)} cases have more than one slide:")
    print(multi[["cohort", "case_id", "n_slides", "label", "slide_ids"]].to_string(index=False))
