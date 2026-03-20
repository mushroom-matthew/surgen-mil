"""Compare multiple trained models across seeds: summary table and figures.

Figures produced
----------------
  summary_metrics.png       strip + bar: slide/case AUROC and AUPRC per model
  roc_pr_curves.png         ROC and PR curves (runs averaged per model)
  seed_variance.png         AUROC std across seeds
  confusion_matrices.png    normalised confusion matrices, slide + case level
  calibration.png           reliability diagram
  score_distributions.png   predicted probability histograms by class
  training_curves.png       val_auroc, val_auprc, train_loss per seed per model
  summary.csv               mean ± std table
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Paper baseline AUROC for reference (Myles et al., GigaScience 2025, doi:10.1093/gigascience/giaf086)
PAPER_AUROC = 0.8273

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

AGG_METHODS = ["max", "mean", "noisy_or"]
_SLIDE_ID_RE = re.compile(r"^(SR\d+)_40X_HE_T(\d+)_\d+$")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_case_id(slide_id: str) -> str:
    m = _SLIDE_ID_RE.match(slide_id)
    return f"{m.group(1)}_T{m.group(2)}" if m else slide_id


def _to_case_df(df: pd.DataFrame, agg: str = "max") -> pd.DataFrame:
    df = df.copy()
    df["case_id"] = df["slide_id"].apply(_parse_case_id)
    records = []
    for case_id, group in df.groupby("case_id"):
        label = int(group["label"].max())
        probs = group["prob"]
        if agg == "max":
            prob = probs.max()
        elif agg == "mean":
            prob = probs.mean()
        else:
            prob = 1.0 - (1.0 - probs).prod()
        records.append({"case_id": case_id, "label": label, "prob": prob})
    return pd.DataFrame(records)


def _case_level_metrics(df: pd.DataFrame) -> dict[str, dict]:
    from sklearn.metrics import roc_auc_score, average_precision_score
    out: dict[str, dict] = {}
    for agg in AGG_METHODS:
        cdf = _to_case_df(df, agg)
        if cdf["label"].nunique() < 2:
            continue
        y_true = cdf["label"].values
        y_score = cdf["prob"].values
        out[agg] = {
            "auroc": float(roc_auc_score(y_true, y_score)),
            "auprc": float(average_precision_score(y_true, y_score)),
        }
    return out


def load_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        metrics = json.load(f)

    seed = None
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        seed = cfg.get("training", {}).get("seed")

    test_preds = pd.DataFrame()
    val_preds = pd.DataFrame()
    preds_path = run_dir / "predictions.csv"
    if preds_path.exists():
        preds = pd.read_csv(preds_path)
        test_preds = preds[preds["split"] == "test"].copy()
        val_preds  = preds[preds["split"] == "val"].copy()

    history = None
    history_path = run_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    slide = metrics.get("test", metrics)
    case = _case_level_metrics(test_preds) if not test_preds.empty else {}

    return {
        "run": run_dir.name,
        "seed": seed,
        "slide_auroc": slide.get("auroc"),
        "slide_auprc": slide.get("auprc"),
        "temperature": metrics.get("temperature"),
        "case": case,
        "test_preds": test_preds,
        "val_preds": val_preds,
        "history": history,
    }


def collect_runs(output_dir: Path) -> list[dict]:
    runs_dir = output_dir / "runs"
    if not runs_dir.is_dir():
        return []
    results = []
    for d in sorted(runs_dir.iterdir()):
        if d.is_dir() and d.name.isdigit():
            rec = load_run(d)
            if rec is not None:
                results.append(rec)
    return results


def summarise(model_name: str, runs: list[dict]) -> dict:
    aurocs = [r["slide_auroc"] for r in runs if r["slide_auroc"] is not None]
    auprcs = [r["slide_auprc"] for r in runs if r["slide_auprc"] is not None]
    return {
        "model": model_name,
        "n_runs": len(runs),
        "auroc_mean": float(np.mean(aurocs)) if aurocs else None,
        "auroc_std":  float(np.std(aurocs))  if len(aurocs) > 1 else None,
        "auprc_mean": float(np.mean(auprcs)) if auprcs else None,
        "auprc_std":  float(np.std(auprcs))  if len(auprcs) > 1 else None,
    }


# ---------------------------------------------------------------------------
# Prediction averaging helpers
# ---------------------------------------------------------------------------

def _splits_consistent(runs: list[dict], key: str) -> bool:
    slide_sets = [
        frozenset(r[key]["slide_id"]) for r in runs
        if not r.get(key, pd.DataFrame()).empty
    ]
    return len(slide_sets) > 0 and len(set(slide_sets)) == 1


def _average_preds(runs: list[dict], key: str) -> tuple[pd.DataFrame, str]:
    frames = [r[key] for r in runs if not r.get(key, pd.DataFrame()).empty]
    if not frames:
        return pd.DataFrame(), ""
    if _splits_consistent(runs, key):
        combined = pd.concat(frames)
        df = (
            combined.groupby("slide_id", sort=False)
            .agg(label=("label", "first"), prob=("prob", "mean"))
            .reset_index()
        )
        n = len(frames)
        return df, f"n={n} avg"
    else:
        best = max(
            (r for r in runs if not r.get(key, pd.DataFrame()).empty),
            key=lambda r: r.get("slide_auroc") or 0.0,
        )
        return best[key].copy(), f"seed {best['seed']} (splits differ)"


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def _youden_threshold(df: pd.DataFrame) -> float:
    if df.empty or df["label"].nunique() < 2:
        return 0.5
    y_true  = df["label"].values
    y_score = df["prob"].values
    best_j, best_t = -1.0, 0.5
    for t in np.linspace(0, 1, 500):
        y_pred = (y_score >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        j = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t


# ---------------------------------------------------------------------------
# Figure 1: strip + bar metric summary
# ---------------------------------------------------------------------------

def plot_metric_summary(
    model_runs: dict[str, list[dict]],
    out_path: Path,
    case_agg: str = "mean",
) -> None:
    """Bar + individual-seed strip plot for slide/case AUROC and AUPRC."""
    metric_keys = [
        ("slide_auroc", "Slide AUROC"),
        ("slide_auprc", "Slide AUPRC"),
        (f"case_{case_agg}_auroc", f"Case AUROC ({case_agg})"),
        (f"case_{case_agg}_auprc", f"Case AUPRC ({case_agg})"),
    ]
    model_names = list(model_runs.keys())
    x_pos = np.arange(len(model_names))

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(4 * len(metric_keys), 5), squeeze=False)
    axes = axes[0]
    rng = np.random.default_rng(0)

    for ax, (key, label) in zip(axes, metric_keys):
        means, stds, all_vals = [], [], []
        for name in model_names:
            vals = []
            for run in model_runs[name]:
                if key.startswith("case_"):
                    agg  = key.split("_", 1)[1].rsplit("_", 1)[0]
                    stat = key.rsplit("_", 1)[-1]
                    v = run.get("case", {}).get(agg, {}).get(stat)
                else:
                    v = run.get(key)
                if v is not None:
                    vals.append(v)
            all_vals.append(vals)
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)

        colors = [PALETTE[i % len(PALETTE)] for i in range(len(model_names))]
        ax.bar(x_pos, means, yerr=stds, capsize=4, color=colors, alpha=0.6, width=0.5)

        for xi, vals in zip(x_pos, all_vals):
            jitter = rng.uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(xi + jitter, vals, color="black", s=30, zorder=3, alpha=0.8)

        # Paper baseline reference line (AUROC panels only)
        if "auroc" in key:
            ax.axhline(PAPER_AUROC, color="red", linestyle="--", linewidth=1.0,
                       alpha=0.7, label=f"Paper ({PAPER_AUROC:.4f})")
            ax.legend(fontsize=7)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: ROC + PR curves
# ---------------------------------------------------------------------------

def plot_roc_pr(model_runs: dict[str, list[dict]], out_path: Path) -> None:
    """ROC and PR curves, one curve per model (runs averaged)."""
    from sklearn.metrics import (
        PrecisionRecallDisplay, RocCurveDisplay,
        roc_auc_score, average_precision_score,
    )
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    for (name, runs), color in zip(model_runs.items(), PALETTE):
        avg_test, note = _average_preds(runs, "test_preds")
        if avg_test.empty or avg_test["label"].nunique() < 2:
            continue
        y_true  = avg_test["label"].values
        y_score = avg_test["prob"].values
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        RocCurveDisplay.from_predictions(
            y_true, y_score, name=f"{name} ({note})  AUROC={auroc:.3f}",
            ax=ax_roc, color=color)
        PrecisionRecallDisplay.from_predictions(
            y_true, y_score, name=f"{name} ({note})  AUPRC={auprc:.3f}",
            ax=ax_pr, color=color)

    ax_roc.axvline(0, color="none")  # ensure axis exists
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax_roc.set_title("ROC Curve (test)", fontsize=12)
    ax_roc.legend(fontsize=8)
    ax_roc.grid(True, alpha=0.3)

    # Chance line on PR at dataset prevalence
    for runs in model_runs.values():
        avg, _ = _average_preds(runs, "test_preds")
        if not avg.empty:
            prevalence = avg["label"].mean()
            ax_pr.axhline(prevalence, color="k", linestyle="--", lw=0.8,
                          label=f"Chance ({prevalence:.2f})")
            break
    ax_pr.set_title("Precision-Recall Curve (test)", fontsize=12)
    ax_pr.legend(fontsize=8)
    ax_pr.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: seed variance bar
# ---------------------------------------------------------------------------

def plot_seed_variance(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    stds = summary_df["auroc_std"].fillna(0)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(summary_df))]
    ax.bar(summary_df["model"], stds, color=colors)
    ax.set_ylabel("AUROC std across seeds")
    ax.set_title("Seed Variance by Model")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: confusion matrices
# ---------------------------------------------------------------------------

def _draw_cm(ax, df: pd.DataFrame, threshold: float, title: str) -> None:
    df = df.copy()
    df["pred"] = (df["prob"] >= threshold).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for outcome, r, c in [("TN", 0, 0), ("FP", 0, 1), ("FN", 1, 0), ("TP", 1, 1)]:
        mask = (df["label"] == (1 if outcome in ("TP", "FN") else 0)) & \
               (df["pred"]  == (1 if outcome in ("TP", "FP") else 0))
        cm[r, c] = mask.sum()
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for r in range(2):
        for c in range(2):
            lbl = ["TN", "FP", "FN", "TP"][r * 2 + c]
            ax.text(c, r, f"{lbl}\n{cm[r, c]}\n({cm_norm[r, c]:.1%})",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[r, c] > 0.6 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Neg", "Pred Pos"], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Neg", "Actual Pos"], fontsize=8)
    ax.set_title(title, fontsize=8)


def plot_confusion_matrices(
    model_runs: dict[str, list[dict]],
    out_path: Path,
    case_agg: str = "max",
) -> None:
    """Rows: slide / case level. Cols: models. Threshold = Youden J on val."""
    n = len(model_runs)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7), squeeze=False)

    for col, (name, runs) in enumerate(model_runs.items()):
        avg_test, note = _average_preds(runs, "test_preds")
        avg_val,  _    = _average_preds(runs, "val_preds")
        threshold = _youden_threshold(avg_val)

        title = f"{name}\n({note}, τ={threshold:.2f})"
        if col == 0:
            axes[0, col].set_ylabel("Slide", fontsize=9, labelpad=8)
            axes[1, col].set_ylabel(f"Case ({case_agg})", fontsize=9, labelpad=8)

        if avg_test.empty:
            axes[0, col].axis("off"); axes[1, col].axis("off")
            continue

        _draw_cm(axes[0, col], avg_test, threshold, title)
        _draw_cm(axes[1, col], _to_case_df(avg_test, case_agg), threshold, title="")

    fig.suptitle("Confusion matrices — Youden's J threshold (from val)", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: calibration
# ---------------------------------------------------------------------------

def plot_calibration(model_runs: dict[str, list[dict]], out_path: Path, n_bins: int = 10) -> None:
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")

    for (name, runs), color in zip(model_runs.items(), PALETTE):
        avg_test, note = _average_preds(runs, "test_preds")
        if avg_test.empty or avg_test["label"].nunique() < 2:
            continue
        frac_pos, mean_pred = calibration_curve(
            avg_test["label"], avg_test["prob"], n_bins=n_bins, strategy="uniform"
        )
        ax.plot(mean_pred, frac_pos, marker="o", color=color, linewidth=1.5, markersize=5,
                label=f"{name} ({note})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration (test)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 6: score distributions
# ---------------------------------------------------------------------------

_LABEL_STYLE = {
    0: {"color": "#4C72B0", "label": "Negative"},
    1: {"color": "#DD8452", "label": "Positive"},
}


def _draw_dist(ax, df: pd.DataFrame, threshold: float, title: str) -> None:
    bins = np.linspace(0, 1, 21)
    for lbl in [0, 1]:
        vals = df.loc[df["label"] == lbl, "prob"].values
        if len(vals) == 0:
            continue
        style = _LABEL_STYLE[lbl]
        ax.hist(vals, bins=bins, alpha=0.6, density=True,
                label=f"{style['label']} (n={len(vals)})",
                color=style["color"], edgecolor="white")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)


def plot_score_distributions(
    model_runs: dict[str, list[dict]],
    out_path: Path,
    case_agg: str = "max",
) -> None:
    """Rows: slide / case level. Cols: models. Threshold = Youden J on val."""
    n = len(model_runs)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 7), squeeze=False)

    for col, (name, runs) in enumerate(model_runs.items()):
        avg_test, note = _average_preds(runs, "test_preds")
        avg_val,  _    = _average_preds(runs, "val_preds")
        threshold = _youden_threshold(avg_val)

        title = f"{name}\n({note}, τ={threshold:.2f})"
        if col == 0:
            axes[0, col].set_ylabel("Slide", fontsize=9, labelpad=8)
            axes[1, col].set_ylabel(f"Case ({case_agg})", fontsize=9, labelpad=8)

        if avg_test.empty:
            axes[0, col].axis("off"); axes[1, col].axis("off")
            continue

        _draw_dist(axes[0, col], avg_test, threshold, title)
        _draw_dist(axes[1, col], _to_case_df(avg_test, case_agg), threshold, title="")

    fig.suptitle("Score distributions — Youden's J threshold (from val)", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 7: training curves
# ---------------------------------------------------------------------------

def plot_training_curves(model_runs: dict[str, list[dict]], out_path: Path) -> None:
    """Val AUROC, val AUPRC, and train loss per seed, one row per model."""
    n = len(model_runs)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(17, 4 * n), squeeze=False)

    for row, (name, runs) in enumerate(model_runs.items()):
        ax_auroc, ax_auprc, ax_loss = axes[row]
        for i, run in enumerate(runs):
            hist = run.get("history")
            if not hist:
                continue
            epochs     = [h["epoch"] for h in hist]
            val_auroc  = [h.get("val_auroc")  for h in hist]
            val_auprc  = [h.get("val_auprc")  for h in hist]
            train_loss = [h.get("train_loss") for h in hist]
            label = f"seed {run['seed']} (run {run['run']})"
            color = PALETTE[i % len(PALETTE)]
            kw = dict(color=color, linewidth=1.5, marker="o", markersize=3, label=label)
            ax_auroc.plot(epochs, val_auroc,  **kw)
            ax_auprc.plot(epochs, val_auprc,  **kw)
            ax_loss.plot( epochs, train_loss, **kw)

        for ax, title, ylabel, ylim in [
            (ax_auroc, f"{name} — Val AUROC",  "AUROC", (0, 1)),
            (ax_auprc, f"{name} — Val AUPRC",  "AUPRC", (0, 1)),
            (ax_loss,  f"{name} — Train Loss", "Loss",  None),
        ]:
            ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            if ylim:
                ax.set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare trained models across seeds.")
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Config YAML paths for each model to compare")
    parser.add_argument("--out", default="outputs/comparison",
                        help="Output directory for summary and plots")
    parser.add_argument("--case-agg", default="mean", choices=AGG_METHODS,
                        help="Case aggregation method for case-level panels (default: mean)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_runs: dict[str, list[dict]] = {}
    for config_path in args.configs:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        output_dir = Path(cfg["output"]["dir"])
        model_name = output_dir.name
        runs = collect_runs(output_dir)
        if not runs:
            print(f"Warning: no completed runs found in {output_dir}")
        model_runs[model_name] = runs

    # Summary table + CSV
    summaries = [summarise(name, runs) for name, runs in model_runs.items()]
    summary_df = pd.DataFrame(summaries)
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== Model Comparison Summary ===")
    for _, row in summary_df.iterrows():
        auroc = f"{row['auroc_mean']:.3f} ± {row['auroc_std']:.3f}" if row['auroc_mean'] is not None else "N/A"
        auprc = f"{row['auprc_mean']:.3f} ± {row['auprc_std']:.3f}" if row['auprc_mean'] is not None else "N/A"
        print(f"  {row['model']:40s}  AUROC: {auroc}  AUPRC: {auprc}  ({row['n_runs']} runs)")
    print(f"  {'[paper baseline]':40s}  AUROC: {PAPER_AUROC:.3f}         AUPRC: N/A             (reference)")
    print(f"\nSummary saved to {summary_csv}")

    # Figures
    plot_metric_summary(model_runs, out_dir / "summary_metrics.png", case_agg=args.case_agg)
    plot_roc_pr(model_runs, out_dir / "roc_pr_curves.png")
    plot_seed_variance(summary_df, out_dir / "seed_variance.png")
    plot_confusion_matrices(model_runs, out_dir / "confusion_matrices.png", case_agg=args.case_agg)
    plot_calibration(model_runs, out_dir / "calibration.png")
    plot_score_distributions(model_runs, out_dir / "score_distributions.png", case_agg=args.case_agg)
    plot_training_curves(model_runs, out_dir / "training_curves.png")


if __name__ == "__main__":
    main()
