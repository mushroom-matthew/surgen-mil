"""Generate Appendix D figures: train-time bag sampler ablation.

Produces two figures mirroring the Appendix C style:
  docs/figures/appendix_d_roc_pr_curves.png
  docs/figures/appendix_d_confusion_matrices.png

Usage
-----
    python scripts/plot_appendix_d.py
    python scripts/plot_appendix_d.py --out docs/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MEANPOOL_MODELS = {
    "MeanPool + random":          "outputs/appendix/phase1_mean_random",
    "MeanPool + spatial":         "outputs/appendix/phase1_mean_spatial",
    "MeanPool + feature diverse": "outputs/appendix/phase1_mean_feature_diverse",
}

ATTN_MODELS = {
    "AttentionMIL + random":          "outputs/appendix/phase1_attention_random",
    "AttentionMIL + spatial":         "outputs/appendix/phase1_attention_spatial",
    "AttentionMIL + feature diverse": "outputs/appendix/phase1_attention_feature_diverse",
}

# Sampler display names (short, for subplot titles)
SAMPLER_LABELS = {
    "random":          "Random",
    "spatial":         "Spatial balanced",
    "feature diverse": "Feature diverse",
}

# Colors: one per sampler, shared across model families
SAMPLER_COLORS = {
    "random":          "#4C72B0",
    "spatial":         "#DD8452",
    "feature diverse": "#55A868",
}

MEANPOOL_LINESTYLE = "-"
ATTN_LINESTYLE     = "--"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_runs(base_dir: Path) -> list[dict]:
    """Return list of {test_preds, val_preds} dicts, one per seed."""
    runs_dir = base_dir / "runs"
    if not runs_dir.is_dir():
        return []
    results = []
    for d in sorted(runs_dir.iterdir()):
        if not (d.is_dir() and d.name.isdigit()):
            continue
        p = d / "predictions.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        results.append({
            "test_preds": df[df["split"] == "test"].copy(),
            "val_preds":  df[df["split"] == "val"].copy(),
        })
    return results


def _average_preds(runs: list[dict], key: str) -> pd.DataFrame:
    """Average predicted probabilities across seeds (assumes shared slide splits)."""
    frames = [r[key] for r in runs if not r[key].empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    return (
        combined.groupby("slide_id", sort=False)
        .agg(label=("label", "first"), prob=("prob", "mean"))
        .reset_index()
    )


def _youden_threshold(df: pd.DataFrame, n: int = 500) -> float:
    """Threshold that maximises Youden's J on df."""
    y_true  = df["label"].values
    y_score = df["prob"].values
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_j, best_t = -1.0, 0.5
    for t in np.linspace(0, 1, n):
        y_pred = (y_score >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        j  = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t


def _sampler_key(model_name: str) -> str:
    """Extract the sampler identifier from a model display name."""
    for k in ("feature diverse", "spatial", "random"):
        if k in model_name.lower():
            return k
    return "random"


# ---------------------------------------------------------------------------
# Figure 1: ROC and PR curves
# ---------------------------------------------------------------------------

def _plot_family(axes: tuple, family_name: str, model_dict: dict, base_dir: Path) -> None:
    """Plot ROC and PR curves for one model family onto (ax_roc, ax_pr)."""
    from matplotlib.lines import Line2D

    ax_roc, ax_pr = axes
    prevalence = None

    for model_name, rel_dir in model_dict.items():
        runs     = _load_runs(base_dir / rel_dir)
        avg_test = _average_preds(runs, "test_preds")
        avg_val  = _average_preds(runs, "val_preds")
        if avg_test.empty or avg_test["label"].nunique() < 2:
            continue

        y_true  = avg_test["label"].values
        y_score = avg_test["prob"].values
        auroc   = roc_auc_score(y_true, y_score)
        auprc   = average_precision_score(y_true, y_score)

        sampler   = _sampler_key(model_name)
        color     = SAMPLER_COLORS[sampler]
        lw        = 1.8

        # Youden threshold operating point
        threshold = _youden_threshold(avg_val) if not avg_val.empty else 0.5
        y_thr  = (y_score >= threshold).astype(int)
        tp = ((y_thr == 1) & (y_true == 1)).sum()
        fp = ((y_thr == 1) & (y_true == 0)).sum()
        tn = ((y_thr == 0) & (y_true == 0)).sum()
        fn = ((y_thr == 0) & (y_true == 1)).sum()
        tpr_op  = tp / max(tp + fn, 1)
        fpr_op  = fp / max(fp + tn, 1)
        prec_op = tp / max(tp + fp, 1)
        rec_op  = tp / max(tp + fn, 1)

        # Strip family prefix for shorter legend labels
        short_name = model_name.split(" + ", 1)[-1].capitalize()

        fpr_c, tpr_c, _ = roc_curve(y_true, y_score)
        ax_roc.plot(fpr_c, tpr_c, color=color, linewidth=lw,
                    label=f"{short_name}  (AUROC={auroc:.3f})")
        ax_roc.plot(fpr_op, tpr_op, marker="D", color=color, markersize=7, zorder=5)

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax_pr.plot(recall, precision, color=color, linewidth=lw,
                   label=f"{short_name}  (AUPRC={auprc:.3f})")
        ax_pr.plot(rec_op, prec_op, marker="D", color=color, markersize=7, zorder=5)

        if prevalence is None:
            prevalence = float(y_true.mean())

    # Reference lines
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax_roc.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=9)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=9)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.set_title(f"{family_name} — ROC curve", fontsize=10)

    if prevalence is not None:
        ax_pr.axhline(prevalence, color="k", linestyle="--", lw=0.8,
                      label=f"Chance (prevalence={prevalence:.2f})")
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=9)
    ax_pr.set_ylabel("Precision (PPV)", fontsize=9)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.set_title(f"{family_name} — Precision-Recall curve", fontsize=10)

    youden_handle = Line2D([0], [0], color="gray", linestyle="none", marker="D",
                           markersize=7, label="Youden J threshold")
    for ax, loc in ((ax_roc, "lower right"), (ax_pr, "upper right")):
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [youden_handle], labels + ["Youden J threshold"],
                  fontsize=7.5, loc=loc, framealpha=0.85)


def plot_roc_pr(out_path: Path, base_dir: Path) -> None:
    """2×2 grid: rows = MeanPool / AttentionMIL, cols = ROC / PR."""

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    _plot_family((axes[0, 0], axes[0, 1]), "MeanPool",     MEANPOOL_MODELS, base_dir)
    _plot_family((axes[1, 0], axes[1, 1]), "AttentionMIL", ATTN_MODELS,     base_dir)

    fig.suptitle(
        "Appendix D — Train-time Bag Sampler Ablation: ROC and PR Curves\n"
        "Seed-averaged predictions, Youden J threshold marked",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Confusion matrices
# ---------------------------------------------------------------------------

def _draw_cm(ax: plt.Axes, df: pd.DataFrame, threshold: float, title: str) -> None:
    """Draw a confusion matrix (raw counts + row-normalised colours) onto ax."""
    y_true = df["label"].values
    y_pred = (df["prob"].values >= threshold).astype(int)

    # [row=actual, col=pred]: [[TN, FP], [FN, TP]]
    cm = np.zeros((2, 2), dtype=int)
    cm[0, 0] = ((y_pred == 0) & (y_true == 0)).sum()  # TN
    cm[0, 1] = ((y_pred == 1) & (y_true == 0)).sum()  # FP
    cm[1, 0] = ((y_pred == 0) & (y_true == 1)).sum()  # FN
    cm[1, 1] = ((y_pred == 1) & (y_true == 1)).sum()  # TP

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    class_labels = ["MSS", "MSI"]
    for r in range(2):
        for c in range(2):
            count   = cm[r, c]
            frac    = cm_norm[r, c]
            text_c  = "white" if frac > 0.55 else "#1a1a2e"
            ax.text(c, r, f"{count}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_c)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(class_labels, fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(class_labels, fontsize=8)
    ax.set_title(title, fontsize=8, pad=4)


def plot_confusion_matrices(out_path: Path, base_dir: Path) -> None:
    """2×3 grid of confusion matrices: rows = MeanPool / AttentionMIL, cols = sampler."""

    samplers      = ["random", "spatial", "feature diverse"]
    families      = [
        ("MeanPool",      MEANPOOL_MODELS),
        ("AttentionMIL",  ATTN_MODELS),
    ]

    n_rows = len(families)
    n_cols = len(samplers)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.8 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )

    for row_idx, (family_name, model_dict) in enumerate(families):
        for col_idx, sampler in enumerate(samplers):
            ax = axes[row_idx, col_idx]

            # Match this cell to the correct model entry
            model_name = next(
                (k for k in model_dict if _sampler_key(k) == sampler), None
            )
            if model_name is None:
                ax.axis("off")
                continue

            runs     = _load_runs(base_dir / model_dict[model_name])
            avg_test = _average_preds(runs, "test_preds")
            avg_val  = _average_preds(runs, "val_preds")

            if avg_test.empty:
                ax.axis("off")
                continue

            threshold = _youden_threshold(avg_val) if not avg_val.empty else 0.5

            n_seeds = len(runs)
            title   = (
                f"{family_name}\n{SAMPLER_LABELS[sampler]}\n"
                f"(n={n_seeds} avg, threshold={threshold:.2f})"
            )
            _draw_cm(ax, avg_test, threshold, title)

            if col_idx == 0:
                ax.set_ylabel("True label", fontsize=8)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Predicted label", fontsize=8)

    fig.suptitle(
        "Appendix D — Train-time Bag Sampler Ablation\n"
        "Seed-averaged predictions, Youden J threshold (val set)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Appendix D figures.")
    parser.add_argument("--out", default="docs/figures",
                        help="Output directory for figures (default: docs/figures)")
    parser.add_argument("--base", default=".",
                        help="Project root (default: .)")
    args = parser.parse_args()

    out_dir  = Path(args.out)
    base_dir = Path(args.base)

    plot_roc_pr(
        out_path=out_dir / "appendix_d_roc_pr_curves.png",
        base_dir=base_dir,
    )
    plot_confusion_matrices(
        out_path=out_dir / "appendix_d_confusion_matrices.png",
        base_dir=base_dir,
    )


if __name__ == "__main__":
    main()
