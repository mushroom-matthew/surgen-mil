"""
Full experiment analysis: ROC, PR, training curves, calibration,
confusion matrix, per-cohort breakdown, and summary table.

Usage:
    python scripts/analyse.py --out outputs/analysis
"""
from __future__ import annotations

import argparse
import json
import re
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

from compare_models import adaptive_unit_ylim

# ---------------------------------------------------------------------------
# Config: which runs to include
# ---------------------------------------------------------------------------

RUNS = {
    # Baselines
    "Mean Pool":                      "outputs/uni_mean",
    "Attention MIL":                  "outputs/uni_attention",
    "Gated Attention MIL":            "outputs/uni_gated_attention",
    # Top-k sparse attention sweep
    "Top-k (k=4)":                    "outputs/uni_topk_attention_k4",
    "Top-k (k=8)":                    "outputs/uni_topk_attention_k8",
    "Top-k (k=16)":                   "outputs/uni_topk_attention_k16",
    "Top-k (k=32)":                   "outputs/uni_topk_attention_k32",
    # Previous architecture comparisons (Phase A / B)
    
    # "Region Attention (8×8)":       "outputs/uni_region_attention_8",
    # "Region Attention (16×16)":     "outputs/uni_region_attention_16",
    # Previous aggregation experiments
    # "Mean+Var Pool":                "outputs/uni_mean_var",
    # "Instance MLP + Mean":          "outputs/uni_instance_mean",
    # "LSE Pool (τ=2)":               "outputs/uni_lse_tau2",
    # Loss ablations
    # "Attention MIL (focal)":        "outputs/uni_attention_focal",
    # "Attention MIL (norm. hybrid)": "outputs/uni_attention_bce_focal_normalized",
    # "Attention MIL (curr. hybrid)": "outputs/uni_attention_bce_focal_curriculum",
}

# Short keys used in the combined predictions export
RUN_KEYS = {
    "Mean Pool":                      "mean_weighted",
    "Attention MIL":                  "attention_weighted",
    "Gated Attention MIL":            "gated_attention",
    "Top-k (k=4)":                    "topk_k4",
    "Top-k (k=8)":                    "topk_k8",
    "Top-k (k=16)":                   "topk_k16",
    "Top-k (k=32)":                   "topk_k32",
    # Previous runs
    # "Region Attention (8×8)":       "region_attention_8",
    # "Region Attention (16×16)":     "region_attention_16",
    # "Mean+Var Pool":                "mean_var",
    # "Instance MLP + Mean":          "instance_mean",
    # "LSE Pool (τ=2)":               "lse_tau2",
    # "LSE Pool (τ=5)":               "lse_tau5",
    # "LSE Pool (τ=10)":              "lse_tau10",
    # "Attention MIL (focal)":        "attention_focal",
    # "Attention MIL (norm. hybrid)": "attention_normalized",
    # "Attention MIL (curr. hybrid)": "attention_curriculum",
}

COLORS  = ["#2196F3", "#F44336", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4", "#E91E63", "#795548"]
MARKERS = ["o",       "s",       "^",       "D",       "v",       "P",       "X",       "*"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def resolve_run_dir(path: Path) -> Path:
    """Resolve versioned run layout (runs/NNN + latest symlink) or return path as-is."""
    if (path / "predictions.csv").exists():
        return path                              # old flat layout — backwards compatible
    latest = path / "latest"
    if latest.exists():
        return latest.resolve()                  # follow symlink
    runs = path / "runs"
    if runs.is_dir():
        versions = sorted(d for d in runs.iterdir() if d.is_dir() and d.name.isdigit())
        if versions:
            return versions[-1]                  # highest-numbered run
    return path                                  # fallback — loaders will raise naturally


def load_predictions(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(resolve_run_dir(run_dir) / "predictions.csv")


def load_history(run_dir: Path) -> pd.DataFrame:
    with open(resolve_run_dir(run_dir) / "history.json") as f:
        return pd.DataFrame(json.load(f))


def load_metrics(run_dir: Path) -> dict:
    with open(resolve_run_dir(run_dir) / "metrics.json") as f:
        return json.load(f)


def split_df(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"] == split].reset_index(drop=True)


def extract_cohort(slide_id: str) -> str:
    return slide_id.split("_")[0]


def parse_case_id(slide_id: str) -> str:
    m = re.match(r"^(SR\d+)_40X_HE_T(\d+)_\d+$", slide_id)
    return f"{m.group(1)}_T{m.group(2)}" if m else slide_id


def aggregate_to_case_level(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Given slide-level predictions, return a dict of case-level DataFrames,
    one per aggregation method: 'max', 'mean', 'noisy_or'.
    Each output df has columns: case_id, label, prob.
    """
    df = df.copy()
    df["case_id"] = df["slide_id"].apply(parse_case_id)

    records: dict[str, list] = {"max": [], "mean": [], "noisy_or": []}
    for case_id, group in df.groupby("case_id"):
        label = int(group["label"].max())
        probs = group["prob"]
        records["max"].append({"case_id": case_id, "label": label, "prob": probs.max()})
        records["mean"].append({"case_id": case_id, "label": label, "prob": probs.mean()})
        records["noisy_or"].append({"case_id": case_id, "label": label,
                                    "prob": 1.0 - (1.0 - probs).prod()})

    return {agg: pd.DataFrame(rows) for agg, rows in records.items()}


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

        # Normalise by row (true class) so color encodes within-class rate
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / np.where(row_sums == 0, 1, row_sums)

        ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{name}\n(threshold={threshold})", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black",
                        fontsize=11, fontweight="bold")

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
    n_runs = len(run_preds)
    width = 0.7 / n_runs
    offsets = np.arange(n_runs) * width - (n_runs - 1) * width / 2

    for ax, metric in zip(axes, metric_names):
        metric_vals = []
        for i, (run_name, color) in enumerate(zip(run_preds, COLORS)):
            vals = []
            for cohort in cohorts:
                row = result[
                    (result["run"] == run_name) &
                    (result["cohort"] == cohort) &
                    (result["metric"] == metric)
                ]
                vals.append(row["value"].values[0] if not row.empty else 0.0)
            metric_vals.extend(vals)
            bars = ax.bar(x + offsets[i], vals, width, label=run_name, color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(cohorts)
        ax.set_ylim(*adaptive_unit_ylim(metric_vals))
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by cohort ({split})", fontsize=12)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# Case-level evaluation
# ---------------------------------------------------------------------------

def print_case_level_metrics(
    run_preds: dict[str, pd.DataFrame],
    split: str,
) -> pd.DataFrame:
    agg_methods = ["max", "mean", "noisy_or"]
    rows = []
    print(f"\n=== Case-level metrics ({split}, case aggregation) ===")
    header = f"{'Model':<30} {'Aggregation':<12} {'N_cases':>8} {'N_pos':>6} {'AUROC':>7} {'AUPRC':>7}"
    print(header)
    print("-" * len(header))

    for name, df in run_preds.items():
        sub = split_df(df, split)
        if sub.empty:
            continue
        case_dfs = aggregate_to_case_level(sub)
        for agg in agg_methods:
            cdf = case_dfs[agg]
            if cdf["label"].nunique() < 2:
                continue
            auroc = roc_auc_score(cdf["label"], cdf["prob"])
            auprc = average_precision_score(cdf["label"], cdf["prob"])
            n_cases = len(cdf)
            n_pos = int(cdf["label"].sum())
            print(f"{name:<30} {agg:<12} {n_cases:>8} {n_pos:>6} {auroc:>7.3f} {auprc:>7.3f}")
            rows.append({
                "Model": name,
                "Aggregation": agg,
                "N_cases": n_cases,
                "N_pos": n_pos,
                "AUROC": round(auroc, 4),
                "AUPRC": round(auprc, 4),
            })

    return pd.DataFrame(rows)


def plot_case_level_roc_pr(
    run_preds: dict[str, pd.DataFrame],
    split: str,
    out: Path,
) -> None:
    agg_methods = ["max", "mean", "noisy_or"]
    linestyles = {"max": "-", "mean": "--", "noisy_or": ":"}

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5))

    # prevalence for chance line (use first run as reference)
    first_sub = None
    for df in run_preds.values():
        first_sub = split_df(df, split)
        if not first_sub.empty:
            break

    for (name, df), color in zip(run_preds.items(), COLORS):
        sub = split_df(df, split)
        if sub.empty:
            continue
        case_dfs = aggregate_to_case_level(sub)
        for agg in agg_methods:
            cdf = case_dfs[agg]
            if cdf["label"].nunique() < 2:
                continue
            y_true = cdf["label"].values
            y_score = cdf["prob"].values
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
            ls = linestyles[agg]
            RocCurveDisplay.from_predictions(
                y_true, y_score,
                name=f"{name} ({agg}, AUROC={auroc:.3f})",
                ax=ax_roc, color=color, linestyle=ls,
            )
            PrecisionRecallDisplay.from_predictions(
                y_true, y_score,
                name=f"{name} ({agg}, AUPRC={auprc:.3f})",
                ax=ax_pr, color=color, linestyle=ls,
            )

    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax_roc.set_title(f"Case-level ROC ({split})", fontsize=13)
    ax_roc.legend(fontsize=7)

    if first_sub is not None and not first_sub.empty:
        case_ref = aggregate_to_case_level(first_sub)["max"]
        prevalence = case_ref["label"].mean()
        ax_pr.axhline(prevalence, color="k", linestyle="--", lw=0.8,
                      label=f"Chance ({prevalence:.2f})")
    ax_pr.set_title(f"Case-level PR ({split})", fontsize=13)
    ax_pr.legend(fontsize=7)

    fig.suptitle("Case-level Model Comparison (solid=max, dashed=mean, dotted=noisy_or)",
                 fontsize=12, y=1.01)
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
# Plot 8: Operating-point analysis — annotated threshold at 90% sens + best F1
# ---------------------------------------------------------------------------

def _threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, n: int = 500):
    """Return arrays of (threshold, sens, spec, prec, f1) over n steps."""
    thresholds = np.linspace(0, 1, n)
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * sens / max(prec + sens, 1e-9)
        rows.append((t, sens, spec, prec, f1))
    return np.array(rows)  # [N, 5]


def find_operating_points(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Return threshold, sens, spec, prec, F1 at two operating points."""
    m = _threshold_metrics(y_true, y_score)
    # ~90% sensitivity: find threshold where sens is closest to 0.90
    idx_sens90 = int(np.argmin(np.abs(m[:, 1] - 0.90)))
    # best F1
    idx_f1 = int(np.argmax(m[:, 4]))
    def row(i):
        return {
            "threshold": float(m[i, 0]),
            "sensitivity": float(m[i, 1]),
            "specificity": float(m[i, 2]),
            "precision": float(m[i, 3]),
            "f1": float(m[i, 4]),
        }
    return {"sens90": row(idx_sens90), "best_f1": row(idx_f1)}


def plot_threshold_analysis(
    run_preds: dict[str, pd.DataFrame],
    split: str,
    out: Path,
) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))

    op_records = []
    for (name, df), color, marker in zip(run_preds.items(), COLORS, MARKERS):
        sub = split_df(df, split)
        if sub.empty or sub["label"].nunique() < 2:
            continue
        y_true = sub["label"].values
        y_score = sub["prob"].values
        m = _threshold_metrics(y_true, y_score)
        ops = find_operating_points(y_true, y_score)

        # sensitivity + specificity curves
        ax_left.plot(m[:, 0], m[:, 1], color=color, lw=1.5, label=f"{name} sens")
        ax_left.plot(m[:, 0], m[:, 2], color=color, lw=1.5, linestyle="--")

        # F1 curve
        ax_right.plot(m[:, 0], m[:, 4], color=color, lw=1.5, label=name)

        # annotate operating points
        for op_name, op, ls in [("90% sens", ops["sens90"], "v"), ("best F1", ops["best_f1"], "*")]:
            t, s = op["threshold"], op["sensitivity"]
            ax_left.axvline(t, color=color, alpha=0.35, linestyle=":")
            ax_left.plot(t, s, marker=ls, color=color, ms=9, zorder=5)
            ax_right.plot(t, op["f1"], marker=ls, color=color, ms=9, zorder=5,
                          label=f"{name} {op_name} (t={t:.2f}, F1={op['f1']:.2f})")
            op_records.append({
                "model": name, "operating_point": op_name, **op
            })

    from matplotlib.lines import Line2D
    ax_left.add_artist(ax_left.legend(
        handles=[
            Line2D([0], [0], color="k", lw=1.5, label="Sensitivity"),
            Line2D([0], [0], color="k", lw=1.5, linestyle="--", label="Specificity"),
        ],
        loc="lower left", fontsize=8,
    ))
    ax_left.legend(
        [Line2D([0], [0], color=c, lw=1.5) for c in COLORS[:len(run_preds)]],
        list(run_preds.keys()), fontsize=7, loc="center left",
    )
    ax_left.set_xlabel("Threshold")
    ax_left.set_ylabel("Rate")
    ax_left.set_title(f"Sensitivity & Specificity ({split})", fontsize=11)
    ax_left.grid(True, alpha=0.3)

    ax_right.set_xlabel("Threshold")
    ax_right.set_ylabel("F1")
    ax_right.set_title(f"F1 vs Threshold — annotated operating points ({split})", fontsize=11)
    ax_right.legend(fontsize=7)
    ax_right.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, out)

    # print operating-point table
    print(f"\n=== Operating Points ({split}) ===")
    op_df = pd.DataFrame(op_records).round(3)
    print(op_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Table: systematic errors — slides that fail across both models
# ---------------------------------------------------------------------------

def print_systematic_errors(
    run_preds: dict[str, pd.DataFrame],
    split: str,
    threshold: float = 0.5,
) -> None:
    model_names = list(run_preds.keys())
    if len(model_names) < 2:
        return

    dfs = {
        name: split_df(df, split).set_index("slide_id")
        for name, df in run_preds.items()
    }

    # union of all slides
    all_slides = set(dfs[model_names[0]].index)
    for df in dfs.values():
        all_slides &= set(df.index)

    fp_sets = {}
    fn_sets = {}
    for name, df in dfs.items():
        sub = df.loc[list(all_slides)]
        fp_sets[name] = set(sub[(sub["label"] == 0) & (sub["prob"] >= threshold)].index)
        fn_sets[name] = set(sub[(sub["label"] == 1) & (sub["prob"] < threshold)].index)

    common_fps = set.intersection(*fp_sets.values())
    common_fns = set.intersection(*fn_sets.values())

    print(f"\n=== Systematic Errors — consistent across all models ({split}, t={threshold}) ===")

    ref_df = dfs[model_names[0]]
    print(f"\nFalse Positives in ALL models ({len(common_fps)} slides):")
    for sid in sorted(common_fps):
        cohort = extract_cohort(sid)
        probs = "  ".join(f"{n}: {dfs[n].loc[sid, 'prob']:.3f}" for n in model_names)
        print(f"  {sid}  cohort={cohort}  [{probs}]")

    print(f"\nFalse Negatives in ALL models ({len(common_fns)} slides):")
    for sid in sorted(common_fns):
        cohort = extract_cohort(sid)
        probs = "  ".join(f"{n}: {dfs[n].loc[sid, 'prob']:.3f}" for n in model_names)
        print(f"  {sid}  cohort={cohort}  [{probs}]")


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
    parser.add_argument("--out", default="outputs/analysis/topk_sweep")
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
        resolved = resolve_run_dir(p)
        runs_dir = p / "runs"
        if runs_dir.is_dir():
            n_versions = sum(1 for d in runs_dir.iterdir() if d.is_dir() and d.name.isdigit())
            if n_versions > 1:
                print(f"  loaded {run_dir}  [{resolved.name} of {n_versions} versions]")
            else:
                print(f"  loaded {run_dir}  [{resolved.name}]")
        else:
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

    print("\nComputing case-level metrics...")
    case_metrics = print_case_level_metrics(run_preds, split)
    case_metrics.to_csv(out / "case_level_metrics.csv", index=False)
    print(f"  saved {out / 'case_level_metrics.csv'}")
    plot_case_level_roc_pr(run_preds, split, out / "case_level_roc_pr.png")

    print("\nExporting combined predictions...")
    export_combined_predictions(run_preds, out)

    plot_threshold_analysis(run_preds, split, out / "threshold_analysis.png")

    print_top_errors(run_preds, split, k=10)
    print_systematic_errors(run_preds, split, threshold=0.5)

    summary = print_summary_table(run_preds, run_histories)
    summary.to_csv(out / "summary.csv", index=False)
    print(f"\nSummary saved to {out / 'summary.csv'}")
    print(f"All figures in {out}/")


if __name__ == "__main__":
    main()
