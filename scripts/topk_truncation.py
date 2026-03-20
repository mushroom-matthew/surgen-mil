"""Test-time top-k truncation and attention concentration diagnostics.

Loads a trained Attention MIL checkpoint, then for each test slide:
  - sweeps top-k% truncation (keep top fraction of patches by weight)
  - sweeps minimum-mass truncation (keep fewest patches reaching cumulative mass)
  - computes concentration metrics on the original weights

Outputs three figures:
  - truncation_sweep.png   : AUROC + AUPRC vs truncation level (both modes)
  - concentration_by_outcome.png : violin plots of metrics per TP/FP/FN/TN
  - concentration_scatter.png    : N_eff / entropy vs prob, coloured by outcome

Usage
-----
    python scripts/topk_truncation.py
    python scripts/topk_truncation.py --base_dir outputs/uni_attention --run 002
    python scripts/topk_truncation.py --out outputs/topk_analysis --threshold 0.45
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.feature_provider import UniFeatureProvider
from src.models.build import build_model


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------

def _best_run(base_dir: Path) -> tuple[Path, Path, Path]:
    """Return (run_dir, config_path, ckpt_path) for the run with highest test AUROC."""
    runs_dir = base_dir / "runs"
    best = None
    best_auroc = -1.0
    for d in sorted(runs_dir.iterdir()):
        if not (d.is_dir() and d.name.isdigit()):
            continue
        preds = d / "predictions.csv"
        cfg   = d / "config.yaml"
        ckpt  = d / "model.pt"
        if not (preds.exists() and cfg.exists() and ckpt.exists()):
            continue
        df = pd.read_csv(preds)
        test = df[df["split"] == "test"]
        if test["label"].nunique() < 2:
            continue
        auroc = roc_auc_score(test["label"], test["prob"])
        if auroc > best_auroc:
            best_auroc = auroc
            best = (d, cfg, ckpt)
    if best is None:
        raise FileNotFoundError(f"No valid runs found in {runs_dir}")
    print(f"Using run {best[0].name}  (test AUROC={best_auroc:.4f})")
    return best


# ---------------------------------------------------------------------------
# Truncation helpers — operate on [N] weight tensors and [N, D] feature tensors
# ---------------------------------------------------------------------------

def _bag_from_weights(
    x: torch.Tensor,          # [N, D]
    w: torch.Tensor,          # [N]
    idx: torch.Tensor,        # indices to keep
    classifier: torch.nn.Module,
) -> float:
    w_kept = w[idx]
    w_kept = w_kept / w_kept.sum()
    bag = (w_kept.unsqueeze(1) * x[idx]).sum(0)
    with torch.no_grad():
        prob = torch.sigmoid(classifier(bag)).item()
    return prob


def truncate_topk(
    x: torch.Tensor,
    w: torch.Tensor,
    frac: float,
    classifier: torch.nn.Module,
) -> float:
    """Keep the top-frac fraction of patches by attention weight."""
    k = max(1, int(np.ceil(frac * len(w))))
    idx = torch.argsort(w, descending=True)[:k]
    return _bag_from_weights(x, w, idx, classifier)


def truncate_minmass(
    x: torch.Tensor,
    w: torch.Tensor,
    mass: float,
    classifier: torch.nn.Module,
) -> float:
    """Keep the fewest highest-weight patches whose cumulative mass >= `mass`."""
    sorted_idx = torch.argsort(w, descending=True)
    cumsum = torch.cumsum(w[sorted_idx], dim=0)
    n_keep = int((cumsum < mass).sum().item()) + 1
    n_keep = max(1, min(n_keep, len(w)))
    return _bag_from_weights(x, w, sorted_idx[:n_keep], classifier)


# ---------------------------------------------------------------------------
# Concentration metrics
# ---------------------------------------------------------------------------

def concentration_metrics(w: np.ndarray) -> dict[str, float]:
    """Compute four scalar concentration metrics from normalised weights."""
    w = w / w.sum()                          # ensure sum=1
    eps = 1e-12

    entropy = float(-np.sum(w * np.log(w + eps)))          # nats; high = diffuse
    n_eff   = float(1.0 / np.sum(w ** 2))                  # effective patch count

    # Gini coefficient (1 = perfectly concentrated, 0 = uniform)
    w_sorted = np.sort(w)
    n = len(w_sorted)
    cum = np.cumsum(w_sorted)
    gini = float(1 - 2 * np.sum(cum) / (n * cum[-1]))

    # Fraction of total mass carried by top-1% / top-5% / top-10% of patches
    k1  = max(1, int(np.ceil(0.01 * n)))
    k5  = max(1, int(np.ceil(0.05 * n)))
    k10 = max(1, int(np.ceil(0.10 * n)))
    top1_mass  = float(np.sort(w)[-k1:].sum())
    top5_mass  = float(np.sort(w)[-k5:].sum())
    top10_mass = float(np.sort(w)[-k10:].sum())

    return dict(
        entropy=entropy,
        n_eff=n_eff,
        gini=gini,
        top1_mass=top1_mass,
        top5_mass=top5_mass,
        top10_mass=top10_mass,
    )


def outcome_label(prob: float, true_label: int, threshold: float) -> str:
    pred = int(prob >= threshold)
    if   true_label == 1 and pred == 1: return "TP"
    elif true_label == 0 and pred == 1: return "FP"
    elif true_label == 1 and pred == 0: return "FN"
    else:                               return "TN"


# ---------------------------------------------------------------------------
# Core: collect per-slide data
# ---------------------------------------------------------------------------

def collect(
    model,
    provider: UniFeatureProvider,
    test_df: pd.DataFrame,
    device: torch.device,
    topk_fracs: list[float],
    mass_levels: list[float],
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per test slide containing:
      - slide_id, label, prob_full
      - prob_topk_{frac} for each frac in topk_fracs
      - prob_mass_{mass} for each mass in mass_levels
      - concentration metrics
    """
    classifier = model.classifier

    rows = []
    slide_ids = test_df["slide_id"].tolist()
    labels    = dict(zip(test_df["slide_id"], test_df["label"]))

    for sid in slide_ids:
        rec_idx = next(
            (i for i, r in enumerate(provider.records) if r.slide_id == sid), None
        )
        if rec_idx is None:
            print(f"  SKIP {sid}: not in provider")
            continue

        item  = provider.load_slide(rec_idx)
        feats = torch.tensor(item["features"], dtype=torch.float32, device=device)
        label = labels[sid]

        with torch.no_grad():
            out = model(feats, coords=None)
            w   = out["attention_weights"]   # [N]

        row: dict = {"slide_id": sid, "label": label}

        # Full-bag probability (baseline)
        row["prob_full"] = torch.sigmoid(out["logit"].view(())).item()

        # Top-k% sweep
        for frac in topk_fracs:
            row[f"prob_topk_{frac:.2f}"] = truncate_topk(feats, w, frac, classifier)

        # Min-mass sweep
        for mass in mass_levels:
            row[f"prob_mass_{mass:.2f}"] = truncate_minmass(feats, w, mass, classifier)

        # Concentration metrics from original weights
        row.update(concentration_metrics(w.cpu().numpy()))

        rows.append(row)
        print(f"  {sid}  N={len(feats)}  n_eff={row['n_eff']:.1f}  prob={row['prob_full']:.3f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_truncation_sweep(
    df: pd.DataFrame,
    topk_fracs: list[float],
    mass_levels: list[float],
    threshold: float,
    out: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Test-time truncation sweep — Attention MIL", fontsize=13)

    def _sweep_metrics(prob_cols):
        aurocs, auprcs = [], []
        for col in prob_cols:
            probs = df[col].values
            aurocs.append(roc_auc_score(df["label"], probs))
            auprcs.append(average_precision_score(df["label"], probs))
        return aurocs, auprcs

    topk_cols  = [f"prob_topk_{f:.2f}" for f in topk_fracs]
    mass_cols  = [f"prob_mass_{m:.2f}" for m in mass_levels]
    auroc_tk, auprc_tk = _sweep_metrics(topk_cols)
    auroc_mm, auprc_mm = _sweep_metrics(mass_cols)

    full_auroc = roc_auc_score(df["label"], df["prob_full"])
    full_auprc = average_precision_score(df["label"], df["prob_full"])

    # --- top-k% panels ---
    for ax, vals, full_val, metric in [
        (axes[0, 0], auroc_tk, full_auroc, "AUROC"),
        (axes[1, 0], auprc_tk, full_auprc, "AUPRC"),
    ]:
        ax.plot([f * 100 for f in topk_fracs], vals,
                "o-", color="#1976d2", linewidth=2, markersize=7, label="top-k%")
        ax.axhline(full_val, color="grey", linestyle="--", linewidth=1.2, label="100% (baseline)")
        ax.set_xlabel("Fraction of patches kept (%)")
        ax.set_ylabel(metric)
        ax.set_title(f"Top-k% truncation — {metric}")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 105)
        ax.grid(True, alpha=0.3)

    # --- min-mass panels ---
    for ax, vals, full_val, metric in [
        (axes[0, 1], auroc_mm, full_auroc, "AUROC"),
        (axes[1, 1], auprc_mm, full_auprc, "AUPRC"),
    ]:
        ax.plot([m * 100 for m in mass_levels], vals,
                "s-", color="#c62828", linewidth=2, markersize=7, label="min-mass")
        ax.axhline(full_val, color="grey", linestyle="--", linewidth=1.2, label="100% (baseline)")
        ax.set_xlabel("Cumulative attention mass retained (%)")
        ax.set_ylabel(metric)
        ax.set_title(f"Min-mass truncation — {metric}")
        ax.legend(fontsize=8)
        ax.set_xlim(45, 105)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out / "truncation_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_concentration_by_outcome(
    df: pd.DataFrame,
    threshold: float,
    out: Path,
) -> None:
    df = df.copy()
    df["outcome"] = [
        outcome_label(p, l, threshold)
        for p, l in zip(df["prob_full"], df["label"])
    ]

    order  = ["TP", "FP", "FN", "TN"]
    colors = {"TP": "#2e7d32", "FP": "#e65100", "FN": "#6a1b9a", "TN": "#1565c0"}
    metrics = [
        ("entropy",  "Attention entropy (nats)\nhigh = diffuse"),
        ("n_eff",    "Effective patch count (N_eff)\nhigh = diffuse"),
        ("gini",     "Gini coefficient\nhigh = concentrated"),
        ("top5_mass","Cumulative mass in top-5% patches\nhigh = concentrated"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        f"Attention concentration by prediction outcome  (threshold={threshold:.2f})",
        fontsize=12,
    )

    for ax, (metric, ylabel) in zip(axes, metrics):
        data_by_outcome = [df[df["outcome"] == o][metric].dropna().values for o in order]
        positions = range(len(order))
        vp = ax.violinplot(
            [d for d in data_by_outcome if len(d) > 0],
            positions=[i for i, d in enumerate(data_by_outcome) if len(d) > 0],
            showmedians=True, showextrema=True,
        )
        for i, (o, body) in enumerate(
            zip([o for o, d in zip(order, data_by_outcome) if len(d) > 0],
                vp["bodies"])
        ):
            body.set_facecolor(colors[o])
            body.set_alpha(0.55)

        # Overlay jittered points
        for i, (o, d) in enumerate(zip(order, data_by_outcome)):
            if len(d) == 0:
                continue
            jitter = np.random.default_rng(0).uniform(-0.1, 0.1, len(d))
            ax.scatter(np.full(len(d), i) + jitter, d,
                       s=18, alpha=0.7, color=colors[o], zorder=3)

        visible = [o for o, d in zip(order, data_by_outcome) if len(d) > 0]
        ax.set_xticks([i for i, d in enumerate(data_by_outcome) if len(d) > 0])
        ax.set_xticklabels(visible)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(metric, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        # Add n= labels
        for i, (o, d) in enumerate(zip(order, data_by_outcome)):
            if len(d) > 0:
                ax.text(i, ax.get_ylim()[0], f"n={len(d)}",
                        ha="center", va="bottom", fontsize=7, color=colors[o])

    fig.tight_layout()
    path = out / "concentration_by_outcome.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_concentration_scatter(
    df: pd.DataFrame,
    threshold: float,
    out: Path,
) -> None:
    df = df.copy()
    df["outcome"] = [
        outcome_label(p, l, threshold)
        for p, l in zip(df["prob_full"], df["label"])
    ]

    colors = {"TP": "#2e7d32", "FP": "#e65100", "FN": "#6a1b9a", "TN": "#1565c0"}
    order  = ["TP", "FP", "FN", "TN"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Concentration vs probability  (threshold={threshold:.2f})",
        fontsize=12,
    )

    for ax, (metric, xlabel) in zip(axes, [
        ("entropy", "Attention entropy (nats)"),
        ("n_eff",   "Effective patch count (N_eff)"),
    ]):
        for o in order:
            sub = df[df["outcome"] == o]
            if sub.empty:
                continue
            ax.scatter(
                sub[metric], sub["prob_full"],
                label=f"{o} (n={len(sub)})",
                color=colors[o], s=35, alpha=0.75, edgecolors="none",
            )
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1,
                   label=f"threshold={threshold:.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Predicted probability (full bag)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out / "concentration_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TOPK_FRACS  = [0.05, 0.10, 0.20, 0.40, 0.60, 1.00]
MASS_LEVELS = [0.50, 0.70, 0.85, 0.95, 1.00]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",  default="outputs/uni_attention_fair")
    parser.add_argument("--run",       default=None,
                        help="Specific run number (e.g. 003). Default: best by AUROC.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold. Default: Youden's J from val set.")
    parser.add_argument("--out",       default="outputs/topk_analysis")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out      = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    if args.run:
        run_dir  = base_dir / "runs" / args.run
        cfg_path = run_dir / "config.yaml"
        ckpt_path = run_dir / "model.pt"
        preds_path = run_dir / "predictions.csv"
    else:
        run_dir, cfg_path, ckpt_path = _best_run(base_dir)
        preds_path = run_dir / "predictions.csv"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model = model.to(device).eval()

    # --- Threshold ---
    preds = pd.read_csv(preds_path)
    if args.threshold is not None:
        threshold = args.threshold
    else:
        val = preds[preds["split"] == "val"]
        thresholds = np.linspace(0, 1, 500)
        best_j, threshold = -1.0, 0.5
        for t in thresholds:
            pred = (val["prob"] >= t).astype(int)
            tp = ((pred == 1) & (val["label"] == 1)).sum()
            tn = ((pred == 0) & (val["label"] == 0)).sum()
            fp = ((pred == 1) & (val["label"] == 0)).sum()
            fn = ((pred == 0) & (val["label"] == 1)).sum()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            j = sens + spec - 1
            if j > best_j:
                best_j, threshold = j, t
        print(f"Optimal val threshold (Youden's J): {threshold:.3f}")

    test_df = preds[preds["split"] == "test"][["slide_id", "label"]].copy()
    print(f"Test slides: {len(test_df)}  (MSI: {test_df['label'].sum()})")

    data_root = cfg["data"]["root"]
    provider  = UniFeatureProvider(data_root)

    # --- Collect ---
    print("\nCollecting per-slide data...")
    df = collect(model, provider, test_df, device, TOPK_FRACS, MASS_LEVELS)
    df.to_csv(out / "slide_data.csv", index=False)
    print(f"\nCollected {len(df)} slides.")

    # --- Report baseline metrics ---
    print(f"\nBaseline (full bag):")
    print(f"  AUROC = {roc_auc_score(df['label'], df['prob_full']):.4f}")
    print(f"  AUPRC = {average_precision_score(df['label'], df['prob_full']):.4f}")

    # --- Figures ---
    print("\nGenerating figures...")
    plot_truncation_sweep(df, TOPK_FRACS, MASS_LEVELS, threshold, out)
    plot_concentration_by_outcome(df, threshold, out)
    plot_concentration_scatter(df, threshold, out)
    print(f"\nAll outputs in {out}/")


if __name__ == "__main__":
    main()
