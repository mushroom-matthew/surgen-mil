"""
Full experiment analysis: ROC, PR, training curves, calibration,
confusion matrix, per-cohort breakdown, and summary table.

Usage:
    python scripts/analyse.py --out outputs/analysis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Config: which runs to include
# ---------------------------------------------------------------------------

RUNS = {
    "Mean Pool (weighted BCE)": "outputs/uni_mean",
    "Attention MIL (weighted BCE)": "outputs/uni_attention",
    "Mean Pool (unweighted BCE)": "outputs/uni_mean_unweighted",
    "Mean Pool (focal)": "outputs/uni_mean_focal",
    "Mean Pool (BCE + focal)": "outputs/uni_mean_bce_focal",
}

# Short keys used in the combined predictions export
RUN_KEYS = {
    "Mean Pool (weighted BCE)": "mean_weighted",
    "Attention MIL (weighted BCE)": "attention_weighted",
    "Mean Pool (unweighted BCE)": "mean_unweighted",
    "Mean Pool (focal)": "mean_focal",
    "Mean Pool (BCE + focal)": "mean_bce_focal",
}

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
MARKERS = ["o", "s", "^", "D", "v"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_predictions(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "predictions.csv")


def load_history(run_dir: Path) -> pd.DataFrame:
    with open(run_dir / "history.json") as f:
        return pd.DataFrame(json.load(f))


def load_metrics(run_dir: Path) -> dict:
    with open(run_dir / "metrics.json") as f:
        return json.load(f)


def split_df(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"] == split].reset_index(drop=True)


def extract_cohort(slide_id: str) -> str:
    return slide_id.split("_")[0]


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 1: ROC + PR curves (test split)
# ---------------------------------------------------------------------------

def plot_roc_pr(
    run_preds: dict[str, pd.DataFrame],
    split: str,
    out: Path,
) -> None:
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))

    for (name, df), color in zip(run_preds.items(), COLORS):
        sub = split_df(df, split)
        if sub.empty:
            continue
        y_true, y_score = sub["label"].values, sub["prob"].values
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        RocCurveDisplay.from_predictions(
            y_true, y_score,
            name=f"{name}\n(AUROC={auroc:.3f})",
            ax=ax_roc, color=color,
        )
        PrecisionRecallDisplay.from_predictions(
            y_true, y_score,
            name=f"{name}\n(AUPRC={auprc:.3f})",
            ax=ax_pr, color=color,
        )

    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax_roc.set_title(f"ROC Curve ({split})", fontsize=13)
    ax_roc.legend(fontsize=8)

    # chance line for PR
    all_pos = 0
    all_n = 0
    for df in run_preds.values():
        sub = split_df(df, split)
        all_pos += sub["label"].sum()
        all_n += len(sub)
        break  # same splits across runs
    prevalence = all_pos / max(all_n, 1)
    ax_pr.axhline(prevalence, color="k", linestyle="--", lw=0.8, label=f"Chance ({prevalence:.2f})")
    ax_pr.set_title(f"Precision-Recall Curve ({split})", fontsize=13)
    ax_pr.legend(fontsize=8)

    fig.suptitle("Model Comparison", fontsize=14, y=1.01)
    _save(fig, out)


# ---------------------------------------------------------------------------
# Plot 2: Training curves (loss, val AUROC, val AUPRC)
# ---------------------------------------------------------------------------

def plot_training_curves(
    run_histories: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("train_loss", "Train Loss"),
        ("val_auroc", "Val AUROC"),
        ("val_auprc", "Val AUPRC"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        for (name, hist), color, marker in zip(run_histories.items(), COLORS, MARKERS):
            ax.plot(
                hist["epoch"], hist[col],
                label=name, color=color, marker=marker,
                markersize=4, linewidth=1.5,
            )
        ax.set_xlabel("Epoch")
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# Plot 3: Calibration curves (test split)
# ---------------------------------------------------------------------------

def plot_calibration(
    run_preds: dict[str, pd.DataFrame],
    split: str,
    out: Path,
    n_bins: int = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")

    for (name, df), color, marker in zip(run_preds.items(), COLORS, MARKERS):
        sub = split_df(df, split)
        if sub.empty:
            continue
        frac_pos, mean_pred = calibration_curve(
            sub["label"], sub["prob"], n_bins=n_bins, strategy="uniform"
        )
        ax.plot(mean_pred, frac_pos, marker=marker, color=color,
                label=name, linewidth=1.5, markersize=5)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration ({split})", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out)


# ---------------------------------------------------------------------------
# Plot 4: Score distributions by class (test split)
# ---------------------------------------------------------------------------

def plot_score_distributions(
    run_preds: dict[str, pd.DataFrame],
    split: str,
    out: Path,
) -> None:
    n = len(run_preds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 25)
    for ax, (name, df), color in zip(axes, run_preds.items(), COLORS):
        sub = split_df(df, split)
        pos = sub[sub["label"] == 1]["prob"]
        neg = sub[sub["label"] == 0]["prob"]
        ax.hist(neg, bins=bins, alpha=0.6, color="#90A4AE", label="Negative (MSS)")
        ax.hist(pos, bins=bins, alpha=0.7, color=color, label="Positive (MSI)")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Score distributions ({split})", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# Plot 5: Confusion matrices at 0.5 threshold (test split)
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    run_preds: dict[str, pd.DataFrame],
    split: str,
    out: Path,
    threshold: float = 0.5,
) -> None:
    n = len(run_preds)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, df), color in zip(axes, run_preds.items(), COLORS):
        sub = split_df(df, split)
        y_true = sub["label"].values
        y_pred = (sub["prob"].values >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{name}\n(threshold={threshold})", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")

    fig.suptitle(f"Confusion matrices ({split})", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# Plot 6: Per-cohort AUROC / AUPRC bar chart (test split)
# ---------------------------------------------------------------------------

def plot_cohort_breakdown(
    run_preds: dict[str, pd.DataFrame],
    split: str,
    out: Path,
) -> None:
    cohorts = ["SR1482", "SR386"]
    metric_names = ["AUROC", "AUPRC"]
    fns = [roc_auc_score, average_precision_score]

    records = []
    for run_name, df in run_preds.items():
        sub = split_df(df, split)
        sub = sub.copy()
        sub["cohort"] = sub["slide_id"].apply(extract_cohort)
        for cohort in cohorts:
            cdf = sub[sub["cohort"] == cohort]
            if cdf["label"].nunique() < 2:
                continue
            for metric, fn in zip(metric_names, fns):
                records.append({
                    "run": run_name,
                    "cohort": cohort,
                    "metric": metric,
                    "value": fn(cdf["label"], cdf["prob"]),
                })

    if not records:
        print("  no per-cohort data — skipping")
        return

    result = pd.DataFrame(records)
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(len(cohorts))
    width = 0.25

    for ax, metric in zip(axes, metric_names):
        for i, (run_name, color) in enumerate(zip(run_preds, COLORS)):
            vals = []
            for cohort in cohorts:
                row = result[
                    (result["run"] == run_name) &
                    (result["cohort"] == cohort) &
                    (result["metric"] == metric)
                ]
                vals.append(row["value"].values[0] if not row.empty else 0.0)
            bars = ax.bar(x + i * width, vals, width, label=run_name, color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x + width)
        ax.set_xticklabels(cohorts)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by cohort ({split})", fontsize=12)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# Export: combined predictions (all models, all splits)
# ---------------------------------------------------------------------------

def export_combined_predictions(
    run_preds: dict[str, pd.DataFrame],
    out: Path,
) -> pd.DataFrame:
    frames = []
    for name, df in run_preds.items():
        tagged = df.copy()
        tagged["model"] = RUN_KEYS[name]
        frames.append(tagged[["slide_id", "label", "prob", "model", "split"]])
    combined = pd.concat(frames, ignore_index=True)

    combined.to_csv(out / "predictions_all.csv", index=False)

    # also write per-model JSON for test split (the requested format)
    for name, df in run_preds.items():
        sub = split_df(df, "test")
        records = [
            {"slide_id": r["slide_id"], "label": r["label"],
             "prob": round(r["prob"], 6), "model": RUN_KEYS[name]}
            for _, r in sub.iterrows()
        ]
        key = RUN_KEYS[name]
        with open(out / f"predictions_test_{key}.json", "w") as f:
            json.dump(records, f, indent=2)
        print(f"  exported predictions_test_{key}.json ({len(records)} slides)")

    print(f"  exported predictions_all.csv ({len(combined)} rows)")
    return combined


# ---------------------------------------------------------------------------
# Plot 7: Threshold sweep — sensitivity / specificity / precision / F1
# ---------------------------------------------------------------------------

def plot_threshold_sweep(
    run_preds: dict[str, pd.DataFrame],
    split: str,
    out: Path,
    n_thresholds: int = 200,
) -> None:
    thresholds = np.linspace(0, 1, n_thresholds)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_sens_spec, ax_pr_f1 = axes

    for (name, df), color in zip(run_preds.items(), COLORS):
        sub = split_df(df, split)
        if sub.empty or sub["label"].nunique() < 2:
            continue
        y_true = sub["label"].values
        y_score = sub["prob"].values

        sens, spec, prec, f1 = [], [], [], []
        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            sens.append(tp / max(tp + fn, 1))
            spec.append(tn / max(tn + fp, 1))
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            prec.append(p)
            f1.append(2 * p * r / max(p + r, 1e-9))

        ax_sens_spec.plot(thresholds, sens, color=color, lw=1.5, label=f"{name} sensitivity")
        ax_sens_spec.plot(thresholds, spec, color=color, lw=1.5, linestyle="--")

        ax_pr_f1.plot(thresholds, prec, color=color, lw=1.5, label=f"{name} precision")
        ax_pr_f1.plot(thresholds, f1, color=color, lw=1.5, linestyle=":", label=f"{name} F1")

    # dummy legend entries for line styles
    from matplotlib.lines import Line2D
    style_legend = [
        Line2D([0], [0], color="k", lw=1.5, label="Sensitivity / Precision"),
        Line2D([0], [0], color="k", lw=1.5, linestyle="--", label="Specificity"),
        Line2D([0], [0], color="k", lw=1.5, linestyle=":", label="F1"),
    ]

    ax_sens_spec.set_xlabel("Threshold")
    ax_sens_spec.set_ylabel("Rate")
    ax_sens_spec.set_title(f"Sensitivity & Specificity vs Threshold ({split})", fontsize=11)
    ax_sens_spec.legend(fontsize=7)
    ax_sens_spec.grid(True, alpha=0.3)

    ax_pr_f1.set_xlabel("Threshold")
    ax_pr_f1.set_ylabel("Score")
    ax_pr_f1.set_title(f"Precision & F1 vs Threshold ({split})", fontsize=11)
    handles, labels = ax_pr_f1.get_legend_handles_labels()
    ax_pr_f1.legend(handles + style_legend, labels + [l.get_label() for l in style_legend], fontsize=6)
    ax_pr_f1.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# Table: top-k confident errors (test split)
# ---------------------------------------------------------------------------

def print_top_errors(
    run_preds: dict[str, pd.DataFrame],
    split: str,
    k: int = 10,
) -> None:
    print(f"\n=== Top-{k} Confident Errors ({split}) ===")
    for name, df in run_preds.items():
        sub = split_df(df, split).copy()
        sub["cohort"] = sub["slide_id"].apply(extract_cohort)

        fp = sub[(sub["label"] == 0)].nlargest(k, "prob")[["slide_id", "cohort", "prob", "label"]]
        fn = sub[(sub["label"] == 1)].nsmallest(k, "prob")[["slide_id", "cohort", "prob", "label"]]

        print(f"\n--- {name} ---")
        print(f"  Top-{k} False Positives (MSS predicted as MSI):")
        for _, r in fp.iterrows():
            print(f"    {r['slide_id']}  cohort={r['cohort']}  prob={r['prob']:.3f}")
        print(f"  Top-{k} False Negatives (MSI predicted as MSS):")
        for _, r in fn.iterrows():
            print(f"    {r['slide_id']}  cohort={r['cohort']}  prob={r['prob']:.3f}")


# ---------------------------------------------------------------------------
# Table: summary
# ---------------------------------------------------------------------------

def print_summary_table(
    run_preds: dict[str, pd.DataFrame],
    run_histories: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for run_name, df in run_preds.items():
        hist = run_histories[run_name]
        best_epoch = int(hist.loc[hist["val_auprc"].idxmax(), "epoch"])

        for split in ["val", "test"]:
            sub = split_df(df, split)
            if sub["label"].nunique() < 2:
                continue
            y_true, y_score = sub["label"].values, sub["prob"].values
            rows.append({
                "Model": run_name,
                "Split": split,
                "N": len(sub),
                "N_pos": int(sub["label"].sum()),
                "AUROC": round(roc_auc_score(y_true, y_score), 4),
                "AUPRC": round(average_precision_score(y_true, y_score), 4),
                "Best epoch (val AUPRC)": best_epoch,
            })

    summary = pd.DataFrame(rows)
    print("\n=== Summary Table ===")
    print(summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/analysis")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading runs...")
    run_preds: dict[str, pd.DataFrame] = {}
    run_histories: dict[str, pd.DataFrame] = {}

    for name, run_dir in RUNS.items():
        p = Path(run_dir)
        if not p.exists():
            print(f"  SKIP {run_dir} (not found)")
            continue
        run_preds[name] = load_predictions(p)
        run_histories[name] = load_history(p)
        print(f"  loaded {run_dir}")

    if not run_preds:
        print("No runs found. Exiting.")
        return

    split = args.split
    print(f"\nGenerating figures for split={split}...")

    plot_roc_pr(run_preds, split, out / "roc_pr.png")
    plot_training_curves(run_histories, out / "training_curves.png")
    plot_calibration(run_preds, split, out / "calibration.png")
    plot_score_distributions(run_preds, split, out / "score_distributions.png")
    plot_confusion_matrices(run_preds, split, out / "confusion_matrices.png")
    plot_cohort_breakdown(run_preds, split, out / "cohort_breakdown.png")
    plot_threshold_sweep(run_preds, split, out / "threshold_sweep.png")

    print("\nExporting combined predictions...")
    export_combined_predictions(run_preds, out)

    print_top_errors(run_preds, split, k=10)

    summary = print_summary_table(run_preds, run_histories)
    summary.to_csv(out / "summary.csv", index=False)
    print(f"\nSummary saved to {out / 'summary.csv'}")
    print(f"All figures in {out}/")


if __name__ == "__main__":
    main()
