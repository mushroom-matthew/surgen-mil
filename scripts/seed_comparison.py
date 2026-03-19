"""
Aggregate metrics across seeded runs for a set of models.

Each model's output dir is expected to have the versioned layout:
    outputs/uni_mean/runs/001/   (seed 42)
    outputs/uni_mean/runs/002/   (seed 123)
    outputs/uni_mean/runs/003/   (seed 456)

Each run dir must contain metrics.json, predictions.csv, and config.yaml.

Usage:
    python scripts/seed_comparison.py
    python scripts/seed_comparison.py --out outputs/analysis/seed_comparison
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Models to compare
# ---------------------------------------------------------------------------

MODELS = {
    "Mean Pool":         "outputs/uni_mean",
    "Attention MIL":     "outputs/uni_attention",
    "Gated Attention":   "outputs/uni_gated_attention",
    "Top-k (k=4)":       "outputs/uni_topk_attention_k4",
}

AGG_METHODS = ["max", "mean", "noisy_or"]


# ---------------------------------------------------------------------------
# Case-level helpers (mirrors analyse.py)
# ---------------------------------------------------------------------------

def parse_case_id(slide_id: str) -> str:
    m = re.match(r"^(SR\d+)_40X_HE_T(\d+)_\d+$", slide_id)
    return f"{m.group(1)}_T{m.group(2)}" if m else slide_id


def case_level_metrics(df: pd.DataFrame) -> dict[str, dict]:
    """
    Given a slide-level DataFrame with columns [slide_id, label, prob],
    return {agg_method: {auroc, auprc, n_cases, n_pos}} for max/mean/noisy_or.
    Returns empty dict if fewer than 2 classes present.
    """
    df = df.copy()
    df["case_id"] = df["slide_id"].apply(parse_case_id)

    out: dict[str, dict] = {}
    for case_id, group in df.groupby("case_id"):
        pass  # just to validate groupby works

    records: dict[str, list] = {m: [] for m in AGG_METHODS}
    for case_id, group in df.groupby("case_id"):
        label = int(group["label"].max())
        probs = group["prob"]
        records["max"].append({"label": label, "prob": probs.max()})
        records["mean"].append({"label": label, "prob": probs.mean()})
        records["noisy_or"].append({"label": label,
                                    "prob": 1.0 - (1.0 - probs).prod()})

    for agg, rows in records.items():
        cdf = pd.DataFrame(rows)
        if cdf["label"].nunique() < 2:
            continue
        y_true = cdf["label"].values
        y_score = cdf["prob"].values
        out[agg] = {
            "auroc": float(roc_auc_score(y_true, y_score)),
            "auprc": float(average_precision_score(y_true, y_score)),
            "n_cases": len(cdf),
            "n_pos": int(cdf["label"].sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Run loader
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    preds_path   = run_dir / "predictions.csv"
    config_path  = run_dir / "config.yaml"

    if not metrics_path.exists():
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)

    seed = None
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        seed = cfg.get("training", {}).get("seed")

    slide = metrics.get("test", metrics)

    # Case-level metrics from predictions.csv (test split only)
    case: dict[str, dict] = {}
    if preds_path.exists():
        preds = pd.read_csv(preds_path)
        test_preds = preds[preds["split"] == "test"]
        if not test_preds.empty:
            case = case_level_metrics(test_preds)

    return {
        "run": run_dir.name,
        "seed": seed,
        "slide_auroc": slide.get("auroc"),
        "slide_auprc": slide.get("auprc"),
        "temperature": metrics.get("temperature"),
        "case": case,   # {agg: {auroc, auprc, n_cases, n_pos}}
    }


def collect_runs(base_dir: Path) -> list[dict]:
    runs_dir = base_dir / "runs"
    if not runs_dir.is_dir():
        return []
    results = []
    for d in sorted(runs_dir.iterdir()):
        if d.is_dir() and d.name.isdigit():
            rec = load_run(d)
            if rec is not None:
                results.append(rec)
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(vals: list[float]) -> str:
    if not vals:
        return "—"
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if len(arr) == 0:
        return "—"
    if len(arr) == 1:
        return f"{arr[0]:.4f}"
    return f"{arr.mean():.4f} ± {arr.std():.4f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="Save summary CSVs here")
    args = parser.parse_args()

    all_rows: list[dict] = []

    # ------------------------------------------------------------------
    # Per-run slide-level table
    # ------------------------------------------------------------------
    print("\n=== Slide-level results (test) ===")
    hdr = f"{'Model':<22} {'Run':>4} {'Seed':>6} {'AUROC':>8} {'AUPRC':>8} {'Temp':>6}"
    print(hdr)
    print("-" * len(hdr))

    for name, base in MODELS.items():
        runs = collect_runs(Path(base))
        if not runs:
            print(f"{name:<22}   no versioned runs found at {base}")
            continue
        for r in runs:
            auroc = f"{r['slide_auroc']:.4f}" if r["slide_auroc"] is not None else "—"
            auprc = f"{r['slide_auprc']:.4f}" if r["slide_auprc"] is not None else "—"
            temp  = f"{r['temperature']:.3f}" if r["temperature"] is not None else "—"
            seed  = str(r["seed"]) if r["seed"] is not None else "?"
            print(f"{name:<22} {r['run']:>4} {seed:>6} {auroc:>8} {auprc:>8} {temp:>6}")
            all_rows.append({"model": name, **r})

    # ------------------------------------------------------------------
    # Per-run case-level table
    # ------------------------------------------------------------------
    print("\n=== Case-level results (test) ===")
    chdr = f"{'Model':<22} {'Run':>4} {'Seed':>6} {'Agg':>9} {'N_cases':>8} {'N_pos':>6} {'AUROC':>8} {'AUPRC':>8}"
    print(chdr)
    print("-" * len(chdr))

    for r in all_rows:
        name = r["model"]
        seed = str(r["seed"]) if r["seed"] is not None else "?"
        for agg in AGG_METHODS:
            cm = r["case"].get(agg)
            if cm is None:
                print(f"{name:<22} {r['run']:>4} {seed:>6} {agg:>9}  (insufficient classes)")
                continue
            print(f"{name:<22} {r['run']:>4} {seed:>6} {agg:>9} "
                  f"{cm['n_cases']:>8} {cm['n_pos']:>6} "
                  f"{cm['auroc']:>8.4f} {cm['auprc']:>8.4f}")

    # ------------------------------------------------------------------
    # Aggregate summary: mean ± std across seeds
    # ------------------------------------------------------------------
    print("\n=== Aggregate summary (mean ± std, N seeds) ===")

    print(f"\n--- Slide level ---")
    shdr = f"{'Model':<22} {'N':>3} {'AUROC':>20} {'AUPRC':>20}"
    print(shdr)
    print("-" * len(shdr))

    summary_rows = []
    for name in MODELS:
        runs = [r for r in all_rows if r["model"] == name]
        auroc_vals = [r["slide_auroc"] for r in runs if r["slide_auroc"] is not None]
        auprc_vals = [r["slide_auprc"] for r in runs if r["slide_auprc"] is not None]
        print(f"{name:<22} {len(runs):>3} {fmt(auroc_vals):>20} {fmt(auprc_vals):>20}")
        summary_rows.append({
            "model": name, "level": "slide", "agg": "—",
            "n_runs": len(runs),
            "auroc_mean": float(np.mean(auroc_vals)) if auroc_vals else None,
            "auroc_std":  float(np.std(auroc_vals))  if len(auroc_vals) > 1 else None,
            "auprc_mean": float(np.mean(auprc_vals)) if auprc_vals else None,
            "auprc_std":  float(np.std(auprc_vals))  if len(auprc_vals) > 1 else None,
        })

    print(f"\n--- Case level ---")
    cahdr = f"{'Model':<22} {'Agg':>9} {'N':>3} {'AUROC':>20} {'AUPRC':>20}"
    print(cahdr)
    print("-" * len(cahdr))

    for name in MODELS:
        runs = [r for r in all_rows if r["model"] == name]
        for agg in AGG_METHODS:
            auroc_vals = [r["case"][agg]["auroc"] for r in runs if agg in r["case"]]
            auprc_vals = [r["case"][agg]["auprc"] for r in runs if agg in r["case"]]
            print(f"{name:<22} {agg:>9} {len(auroc_vals):>3} {fmt(auroc_vals):>20} {fmt(auprc_vals):>20}")
            summary_rows.append({
                "model": name, "level": "case", "agg": agg,
                "n_runs": len(auroc_vals),
                "auroc_mean": float(np.mean(auroc_vals)) if auroc_vals else None,
                "auroc_std":  float(np.std(auroc_vals))  if len(auroc_vals) > 1 else None,
                "auprc_mean": float(np.mean(auprc_vals)) if auprc_vals else None,
                "auprc_std":  float(np.std(auprc_vals))  if len(auprc_vals) > 1 else None,
            })

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        # flatten all_rows for CSV (drop nested case dict)
        flat_rows = []
        for r in all_rows:
            base = {k: v for k, v in r.items() if k != "case"}
            flat_rows.append(base)
            for agg in AGG_METHODS:
                cm = r["case"].get(agg)
                if cm:
                    flat_rows[-1][f"case_{agg}_auroc"] = cm["auroc"]
                    flat_rows[-1][f"case_{agg}_auprc"] = cm["auprc"]
        pd.DataFrame(flat_rows).to_csv(out / "seed_runs.csv", index=False)
        pd.DataFrame(summary_rows).round(4).to_csv(out / "seed_summary.csv", index=False)
        print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
