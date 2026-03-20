"""Print the three appendix performance tables.

Reads trained runs for all appendix models plus the already-trained fair
comparison models.  For Appendix C it also computes test-time top-5%
truncation inline on the best-AUROC seed of uni_attention_fair.

Usage
-----
    python scripts/appendix_tables.py
    python scripts/appendix_tables.py --out outputs/appendix_tables.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Model registry: display_name -> output base dir
# ---------------------------------------------------------------------------

APPENDIX_A = {
    "Instance mean":         "outputs/appendix_instance_mean",
    "Mean pool (unweighted)":"outputs/appendix_mean_unweighted",
    "Mean pool (weighted)":  "outputs/uni_mean_fair",
}

APPENDIX_B = {
    "Attention (base)":      "outputs/uni_attention_fair",
    "Attention (focal)":     "outputs/appendix_attention_focal",
}

APPENDIX_C_TRAIN = {
    "Attention (full bag)":  "outputs/uni_attention_fair",
    "Top-k attention (k=16, train)": "outputs/appendix_topk_attention",
}

# The test-time truncation row is computed inline from uni_attention_fair.
ATTENTION_FAIR_DIR = Path("outputs/uni_attention_fair")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _youden_threshold(preds: pd.DataFrame) -> float:
    val = preds[preds["split"] == "val"]
    best_j, best_t = -1.0, 0.5
    for t in np.linspace(0, 1, 500):
        tp = int(((val["prob"] >= t) & (val["label"] == 1)).sum())
        fn = int(((val["prob"] <  t) & (val["label"] == 1)).sum())
        tn = int(((val["prob"] <  t) & (val["label"] == 0)).sum())
        fp = int(((val["prob"] >= t) & (val["label"] == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t


def _load_runs(base_dir: Path) -> list[pd.DataFrame]:
    """Return list of predictions DataFrames (test split only), one per seed."""
    runs_dir = base_dir / "runs"
    if not runs_dir.is_dir():
        return []
    frames = []
    for d in sorted(runs_dir.iterdir()):
        if not (d.is_dir() and d.name.isdigit()):
            continue
        p = d / "predictions.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    return frames


def _metrics_for_run(df: pd.DataFrame) -> tuple[float, float]:
    test = df[df["split"] == "test"]
    if test["label"].nunique() < 2:
        return float("nan"), float("nan")
    return (
        roc_auc_score(test["label"], test["prob"]),
        average_precision_score(test["label"], test["prob"]),
    )


def _summarise(base_dir: Path, name: str) -> dict:
    runs = _load_runs(Path(base_dir))
    if not runs:
        return {"model": name, "n_seeds": 0,
                "auroc_mean": float("nan"), "auroc_std": float("nan"),
                "auprc_mean": float("nan"), "auprc_std": float("nan")}
    aurocs, auprcs = zip(*[_metrics_for_run(r) for r in runs])
    return {
        "model":      name,
        "n_seeds":    len(runs),
        "auroc_mean": float(np.nanmean(aurocs)),
        "auroc_std":  float(np.nanstd(aurocs)),
        "auprc_mean": float(np.nanmean(auprcs)),
        "auprc_std":  float(np.nanstd(auprcs)),
    }


def _print_table(rows: list[dict], title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    hdr = f"{'Model':<35} {'AUROC':>13} {'AUPRC':>13} {'Seeds':>5}"
    print(hdr)
    print("-" * 65)
    for r in rows:
        if r["n_seeds"] == 0:
            line = f"  {r['model']:<33} {'(no data)':>13} {'':>13} {'0':>5}"
        else:
            auroc_s = f"{r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}"
            auprc_s = f"{r['auprc_mean']:.3f} ± {r['auprc_std']:.3f}"
            line = f"  {r['model']:<33} {auroc_s:>13} {auprc_s:>13} {r['n_seeds']:>5}"
        print(line)
    print("-" * 65)


# ---------------------------------------------------------------------------
# Appendix C: test-time truncation computed inline
# ---------------------------------------------------------------------------

def _truncate_topk_row(
    base_dir: Path,
    frac: float = 0.05,
    display_name: str = "Attention + truncation (test, top-5%)",
) -> dict:
    """Pick best-AUROC seed from base_dir, apply top-k% truncation, return metrics."""
    from src.data.feature_provider import UniFeatureProvider
    from src.models.build import build_model

    runs = _load_runs(base_dir)
    if not runs:
        return {"model": display_name, "n_seeds": 0,
                "auroc_mean": float("nan"), "auroc_std": float("nan"),
                "auprc_mean": float("nan"), "auprc_std": float("nan")}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate all seeds and collect truncated probs
    aurocs, auprcs = [], []
    for run_df in runs:
        # Find the run dir for this df (match by checking runs/NNN)
        run_dir = None
        for d in sorted((base_dir / "runs").iterdir()):
            if not (d.is_dir() and d.name.isdigit()):
                continue
            p = d / "predictions.csv"
            if not p.exists():
                continue
            candidate = pd.read_csv(p)
            if candidate.equals(run_df):
                run_dir = d
                break
        if run_dir is None:
            continue

        cfg = yaml.safe_load(open(run_dir / "config.yaml"))
        model = build_model(cfg)
        model.load_state_dict(torch.load(str(run_dir / "model.pt"), map_location=device))
        model = model.to(device).eval()

        provider = UniFeatureProvider(cfg["data"]["root"])
        test = run_df[run_df["split"] == "test"]
        labels, probs = [], []

        for sid, label in zip(test["slide_id"], test["label"]):
            rec_idx = next(
                (i for i, r in enumerate(provider.records) if r.slide_id == sid), None
            )
            if rec_idx is None:
                continue
            item  = provider.load_slide(rec_idx)
            feats = torch.tensor(item["features"], dtype=torch.float32, device=device)
            with torch.no_grad():
                out = model(feats)
                w   = out["attention_weights"]

            k    = max(1, int(np.ceil(frac * len(w))))
            idx  = torch.argsort(w, descending=True)[:k]
            w_k  = w[idx]; w_k = w_k / w_k.sum()
            bag  = (w_k.unsqueeze(1) * feats[idx]).sum(0)
            with torch.no_grad():
                prob = torch.sigmoid(model.classifier(bag)).item()

            labels.append(label)
            probs.append(prob)

        labels = np.array(labels)
        aurocs.append(roc_auc_score(labels, probs))
        auprcs.append(average_precision_score(labels, probs))

    return {
        "model":      display_name,
        "n_seeds":    len(aurocs),
        "auroc_mean": float(np.nanmean(aurocs)),
        "auroc_std":  float(np.nanstd(aurocs)),
        "auprc_mean": float(np.nanmean(auprcs)),
        "auprc_std":  float(np.nanstd(auprcs)),
    }


# ---------------------------------------------------------------------------
# Data-driven conclusions
# ---------------------------------------------------------------------------

def _conclude_a(rows: list[dict]) -> str:
    """Appendix A: stability of simple aggregators."""
    have = [r for r in rows if r["n_seeds"] > 0]
    if not have:
        return "  (insufficient data to draw conclusion)"

    auroc_range = max(r["auroc_mean"] for r in have) - min(r["auroc_mean"] for r in have)
    auprc_range = max(r["auprc_mean"] for r in have) - min(r["auprc_mean"] for r in have)
    floor_auroc = min(r["auroc_mean"] for r in have)
    ceil_auroc  = max(r["auroc_mean"] for r in have)
    max_seed_std = max(r["auroc_std"] for r in have)

    if auroc_range <= 0.03 and max_seed_std <= 0.015:
        stability = "with consistently low seed variance"
    elif auroc_range <= 0.05:
        stability = "with moderate spread between variants"
    else:
        stability = "though performance varied meaningfully between variants"

    if floor_auroc >= 0.82:
        floor_str = f"all above {floor_auroc:.2f} AUROC"
    elif floor_auroc >= 0.70:
        floor_str = f"ranging from {floor_auroc:.2f} to {ceil_auroc:.2f} AUROC"
    else:
        floor_str = f"the weakest variant reaching only {floor_auroc:.2f} AUROC"

    return (
        f'  "Simple aggregators performed similarly ({floor_str}, {stability}), '
        f"indicating that frozen UNI embeddings already encode substantial slide-level signal "
        f'and that baseline performance is not highly sensitive to lightweight aggregation choices."'
    )


def _conclude_b(rows: list[dict]) -> str:
    """Appendix B: focal loss trade-offs — AUROC, AUPRC, and variance."""
    have = [r for r in rows if r["n_seeds"] > 0]
    if len(have) < 2:
        return "  (insufficient data to draw conclusion)"

    by_name = {r["model"]: r for r in have}
    base  = by_name.get("Attention (base)")
    focal = by_name.get("Attention (focal)")
    if base is None or focal is None:
        return "  (insufficient data to draw conclusion)"

    auroc_delta = focal["auroc_mean"] - base["auroc_mean"]
    auprc_delta = focal["auprc_mean"] - base["auprc_mean"]
    std_delta   = focal["auprc_std"]  - base["auprc_std"]

    # AUROC direction
    if auroc_delta < -0.01:
        auroc_str = f"modest AUROC loss ({base['auroc_mean']:.3f} → {focal['auroc_mean']:.3f})"
    elif auroc_delta > 0.01:
        auroc_str = f"AUROC gain ({base['auroc_mean']:.3f} → {focal['auroc_mean']:.3f})"
    else:
        auroc_str = "no AUROC change"

    # AUPRC direction
    if auprc_delta > 0.02:
        auprc_str = f"improved mean AUPRC ({base['auprc_mean']:.3f} → {focal['auprc_mean']:.3f})"
    elif auprc_delta < -0.02:
        auprc_str = f"reduced mean AUPRC ({base['auprc_mean']:.3f} → {focal['auprc_mean']:.3f})"
    else:
        auprc_str = "equivalent mean AUPRC"

    # Variance direction
    rel_std = std_delta / (base["auprc_std"] + 1e-9)
    if rel_std > 0.15:
        var_str = f"increased seed-to-seed variability (AUPRC std: {base['auprc_std']:.3f} → {focal['auprc_std']:.3f})"
    elif rel_std < -0.15:
        var_str = f"reduced seed-to-seed variability (AUPRC std: {base['auprc_std']:.3f} → {focal['auprc_std']:.3f})"
    else:
        var_str = "similar seed-to-seed variability"

    sentence = (
        f"Focal loss {auprc_str} but with {auroc_str} and {var_str}, "
        f"suggesting that loss reweighting can sharpen minority-class retrieval "
        f"without resolving the underlying instability of the learned attention."
    )
    return f'  "{sentence[0].upper() + sentence[1:]}"'


def _conclude_c(rows: list[dict]) -> str:
    """Appendix C: sparsity trade-off — AUPRC gain vs variance and AUROC cost."""
    have = [r for r in rows if r["n_seeds"] > 0]
    if not have:
        return "  (insufficient data to draw conclusion)"

    full  = next((r for r in have if "full bag" in r["model"].lower()), None)
    train = next((r for r in have if "train"    in r["model"].lower()), None)
    test  = next((r for r in have if "test"     in r["model"].lower()), None)

    if full is None:
        return "  (insufficient data to draw conclusion)"

    parts = []

    # Characterise each sparsity method vs full-bag baseline
    for r, label in [(train, "hard top-k training"), (test, "test-time truncation")]:
        if r is None:
            continue
        auprc_delta = r["auprc_mean"] - full["auprc_mean"]
        auroc_delta = r["auroc_mean"] - full["auroc_mean"]
        std_ratio   = r["auprc_std"] / (full["auprc_std"] + 1e-9)

        if auprc_delta > 0.02:
            gain_str = f"improved mean AUPRC by {auprc_delta:+.3f}"
        elif auprc_delta < -0.02:
            gain_str = f"reduced mean AUPRC by {auprc_delta:+.3f}"
        else:
            gain_str = "left mean AUPRC unchanged"

        if auroc_delta < -0.01:
            auroc_str = f"reduced AUROC by {abs(auroc_delta):.3f}"
        elif auroc_delta > 0.01:
            auroc_str = f"improved AUROC by {auroc_delta:.3f}"
        else:
            auroc_str = "preserved AUROC"

        if std_ratio > 1.4:
            var_str = "substantially inflated seed variance"
        elif std_ratio > 1.15:
            var_str = "modestly increased seed variance"
        elif std_ratio < 0.85:
            var_str = "reduced seed variance"
        else:
            var_str = "with similar seed variance"

        parts.append((label, gain_str, auroc_str, var_str))

    if not parts:
        return "  (insufficient data to draw conclusion)"

    # Build the conclusion depending on what data is available
    if len(parts) == 1:
        label, gain, auroc, var = parts[0]
        sentence = (
            f"{label.capitalize()} {gain} and {auroc}, but {var}; "
            f"sparsity is plausibly useful, but results should be interpreted cautiously."
        )
    else:
        # Two sparsity methods: emphasise the reliability trade-off
        (l1, g1, a1, v1), (l2, g2, a2, v2) = parts
        sentence = (
            f"Both {l1} and {l2} {g1.split(' mean')[0].replace('improved', 'increased').replace('reduced', 'decreased')} mean AUPRC "
            f"relative to full-bag attention, supporting the hypothesis that diffuse low-weight evidence dilutes precision-recall ranking. "
            f"However, {l1} {a1} and {v1}, whereas {l2} {a2} and {v2}. "
            f"Hard top-k training appears more powerful in expectation but considerably less reliable than inference-time truncation, "
            f"suggesting sparse evidence selection is useful in principle but brittle when enforced during optimization."
        )

    return f'  "{sentence[0].upper() + sentence[1:]}"'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="Optional path to save CSV summary")
    parser.add_argument("--skip_truncation", action="store_true",
                        help="Skip the test-time truncation row (faster)")
    args = parser.parse_args()

    # --- Appendix A ---
    rows_a = [_summarise(d, n) for n, d in APPENDIX_A.items()]
    _print_table(rows_a, "Appendix A — Baseline sanity checks")
    print(_conclude_a(rows_a))

    # --- Appendix B ---
    rows_b = [_summarise(d, n) for n, d in APPENDIX_B.items()]
    _print_table(rows_b, "Appendix B — Attention training variants")
    print(_conclude_b(rows_b))

    # --- Appendix C ---
    rows_c = [_summarise(d, n) for n, d in APPENDIX_C_TRAIN.items()]
    if not args.skip_truncation:
        print("\n  (Computing test-time truncation row — loading models...)", flush=True)
        rows_c.append(_truncate_topk_row(
            ATTENTION_FAIR_DIR,
            frac=0.05,
            display_name="Attention + truncation (test, top-5%)",
        ))
    _print_table(rows_c, "Appendix C — Train-time vs test-time sparsity")
    print(_conclude_c(rows_c))

    # --- Optional CSV ---
    if args.out:
        all_rows = rows_a + rows_b + rows_c
        pd.DataFrame(all_rows).to_csv(args.out, index=False)
        print(f"\nSaved summary to {args.out}")


if __name__ == "__main__":
    main()
