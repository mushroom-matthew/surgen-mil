"""
Aggregate metrics across seeded runs for a set of models.

Each model's output dir is expected to have the versioned layout:
    outputs/uni_mean/runs/001/   (seed 42)
    outputs/uni_mean/runs/002/   (seed 123)
    outputs/uni_mean/runs/003/   (seed 456)

Each run dir must contain metrics.json, predictions.csv, and config.yaml.

Usage:
    python scripts/seed_comparison.py
    python scripts/seed_comparison.py --models uni_mean --out outputs/analysis/seed_comparison
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Models to compare
# ---------------------------------------------------------------------------

ALL_MODELS = {
    "uni_mean":              ("Mean Pool",          "outputs/uni_mean"),
    "uni_attention":         ("Attention MIL",      "outputs/uni_attention"),
    "uni_gated_attention":   ("Gated Attention",    "outputs/uni_gated_attention"),
    "uni_topk_attention_k4": ("Top-k (k=4)",        "outputs/uni_topk_attention_k4"),
    "paper_reproduction":    ("Paper Reproduction", "outputs/paper_reproduction"),
}

# Default when --models is not specified
_DEFAULT_MODELS = list(ALL_MODELS.keys())

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

def load_history(run_dir: Path) -> list[dict] | None:
    p = run_dir / "history.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


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
    test_preds: pd.DataFrame = pd.DataFrame()
    val_preds: pd.DataFrame = pd.DataFrame()
    if preds_path.exists():
        preds = pd.read_csv(preds_path)
        test_preds = preds[preds["split"] == "test"].copy()
        val_preds  = preds[preds["split"] == "val"].copy()
        if not test_preds.empty:
            case = case_level_metrics(test_preds)

    return {
        "run": run_dir.name,
        "seed": seed,
        "slide_auroc": slide.get("auroc"),
        "slide_auprc": slide.get("auprc"),
        "temperature": metrics.get("temperature"),
        "case": case,   # {agg: {auroc, auprc, n_cases, n_pos}}
        "history": load_history(run_dir),
        "test_preds": test_preds,
        "val_preds":  val_preds,
    }


def collect_runs(base_dir: Path, runs_subdir: str = "runs") -> list[dict]:
    runs_dir = base_dir / runs_subdir
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

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_training_curves(model_runs: dict[str, list[dict]], out_path: Path) -> None:
    """One figure per model: val_auroc and train_loss overlaid across seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(model_runs)
    if n_models == 0:
        return

    fig, axes = plt.subplots(n_models, 3, figsize=(17, 4 * n_models), squeeze=False)

    for row_idx, (model_name, runs) in enumerate(model_runs.items()):
        ax_auroc = axes[row_idx, 0]
        ax_auprc = axes[row_idx, 1]
        ax_loss  = axes[row_idx, 2]

        for run_idx, run in enumerate(runs):
            hist = run.get("history")
            if not hist:
                continue
            epochs     = [h["epoch"] for h in hist]
            val_auroc  = [h.get("val_auroc") for h in hist]
            val_auprc  = [h.get("val_auprc") for h in hist]
            train_loss = [h.get("train_loss") for h in hist]
            label = f"seed {run['seed']} (run {run['run']})"
            color = PALETTE[run_idx % len(PALETTE)]

            ax_auroc.plot(epochs, val_auroc, color=color, label=label, linewidth=1.5, marker="o", markersize=3)
            ax_auprc.plot(epochs, val_auprc, color=color, label=label, linewidth=1.5, marker="o", markersize=3)
            ax_loss.plot( epochs, train_loss, color=color, label=label, linewidth=1.5, marker="o", markersize=3)

        ax_auroc.set_title(f"{model_name} — Val AUROC")
        ax_auroc.set_xlabel("Epoch")
        ax_auroc.set_ylabel("AUROC")
        ax_auroc.set_ylim(0, 1)
        ax_auroc.legend(fontsize=8)
        ax_auroc.grid(True, alpha=0.3)

        ax_auprc.set_title(f"{model_name} — Val AUPRC")
        ax_auprc.set_xlabel("Epoch")
        ax_auprc.set_ylabel("AUPRC")
        ax_auprc.set_ylim(0, 1)
        ax_auprc.legend(fontsize=8)
        ax_auprc.grid(True, alpha=0.3)

        ax_loss.set_title(f"{model_name} — Train Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=8)
        ax_loss.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_metric_summary(
    model_runs: dict[str, list[dict]],
    out_path: Path,
    agg_method: str = "mean",
) -> None:
    """Strip + mean bar for slide-level and case-level AUROC/AUPRC per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("slide_auroc", "Slide AUROC"),
        ("slide_auprc", "Slide AUPRC"),
        (f"case_{agg_method}_auroc", f"Case AUROC ({agg_method})"),
        (f"case_{agg_method}_auprc", f"Case AUPRC ({agg_method})"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5), squeeze=False)
    axes = axes[0]

    model_names = list(model_runs.keys())
    x_positions = np.arange(len(model_names))

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        means, stds, all_vals = [], [], []
        for model_name in model_names:
            vals = []
            for run in model_runs[model_name]:
                if metric_key.startswith("case_"):
                    agg = metric_key.split("_", 1)[1].rsplit("_", 1)[0]  # e.g. "mean"
                    stat = metric_key.rsplit("_", 1)[-1]                 # "auroc" or "auprc"
                    v = run.get("case", {}).get(agg, {}).get(stat)
                else:
                    v = run.get(metric_key)
                if v is not None:
                    vals.append(v)
            all_vals.append(vals)
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)

        ax.bar(x_positions, means, yerr=stds, capsize=4,
               color=[PALETTE[i % len(PALETTE)] for i in range(len(model_names))],
               alpha=0.6, width=0.5)

        for xi, vals in zip(x_positions, all_vals):
            jitter = np.random.default_rng(0).uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(xi + jitter, vals, color="black", s=30, zorder=3, alpha=0.8)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _to_case_df(df: pd.DataFrame, agg: str = "max") -> pd.DataFrame:
    """Aggregate slide-level predictions to case level."""
    df = df.copy()
    df["case_id"] = df["slide_id"].apply(parse_case_id)
    records = []
    for case_id, group in df.groupby("case_id"):
        label = int(group["label"].max())
        probs = group["prob"]
        if agg == "max":
            prob = probs.max()
        elif agg == "mean":
            prob = probs.mean()
        else:  # noisy_or
            prob = 1.0 - (1.0 - probs).prod()
        records.append({"case_id": case_id, "label": label, "prob": prob})
    return pd.DataFrame(records)


def _classify_outcomes(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Add 'pred' and 'outcome' (TP/FP/TN/FN) columns to a predictions DataFrame."""
    df = df.copy()
    df["pred"] = (df["prob"] >= threshold).astype(int)
    conditions = [
        (df["label"] == 1) & (df["pred"] == 1),
        (df["label"] == 0) & (df["pred"] == 1),
        (df["label"] == 1) & (df["pred"] == 0),
        (df["label"] == 0) & (df["pred"] == 0),
    ]
    df["outcome"] = np.select(conditions, ["TP", "FP", "FN", "TN"], default="?")
    return df


def find_optimal_threshold_youden(df: pd.DataFrame, n: int = 500) -> float:
    """Threshold that maximises Youden's J (sensitivity + specificity - 1) on df."""
    y_true = df["label"].values
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
        j = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t


def _average_preds(runs: list[dict], key: str) -> pd.DataFrame:
    """Average predicted probabilities across runs; label taken from the first run."""
    frames = [r[key] for r in runs if not r.get(key, pd.DataFrame()).empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    return (
        combined.groupby("slide_id", sort=False)
        .agg(label=("label", "first"), prob=("prob", "mean"))
        .reset_index()
    )


def _draw_cm(ax, df: pd.DataFrame, threshold: float, title: str) -> None:
    """Draw a single normalised confusion matrix onto ax."""
    df = _classify_outcomes(df, threshold)
    cm = np.zeros((2, 2), dtype=int)
    for outcome, r, c in [("TN", 0, 0), ("FP", 0, 1), ("FN", 1, 0), ("TP", 1, 1)]:
        cm[r, c] = (df["outcome"] == outcome).sum()
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for r in range(2):
        for c in range(2):
            lbl = ["TN", "FP", "FN", "TP"][r * 2 + c]
            ax.text(c, r, f"{lbl}\n{cm[r, c]}\n({cm_norm[r, c]:.1%})",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[r, c] > 0.6 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred Neg", "Pred Pos"], fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual Neg", "Actual Pos"], fontsize=8)
    ax.set_title(title, fontsize=8)


def plot_roc_pr(
    model_runs: dict[str, list[dict]],
    out_path: Path,
) -> None:
    """ROC and PR curves, one curve per model (runs averaged)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    for (model_name, runs), color in zip(model_runs.items(), PALETTE):
        avg_test = _average_preds(runs, "test_preds")
        if avg_test.empty or avg_test["label"].nunique() < 2:
            continue
        y_true  = avg_test["label"].values
        y_score = avg_test["prob"].values
        n_seeds = len(runs)
        seed_note = f"n={n_seeds} avg" if n_seeds > 1 else f"seed {runs[0]['seed']}"
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        label_roc = f"{model_name} ({seed_note})  AUROC={auroc:.3f}"
        label_pr  = f"{model_name} ({seed_note})  AUPRC={auprc:.3f}"
        RocCurveDisplay.from_predictions(y_true, y_score, name=label_roc, ax=ax_roc, color=color)
        PrecisionRecallDisplay.from_predictions(y_true, y_score, name=label_pr, ax=ax_pr, color=color)

    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax_roc.set_title("ROC Curve (test)", fontsize=12)
    ax_roc.legend(fontsize=8)
    ax_roc.grid(True, alpha=0.3)

    # chance line: prevalence from the first non-empty model
    for runs in model_runs.values():
        avg = _average_preds(runs, "test_preds")
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


def plot_confusion_matrices(
    model_runs: dict[str, list[dict]],
    out_path: Path,
    case_agg: str = "max",
) -> None:
    """Confusion matrices: cols = models (runs averaged), rows = slide / case.
    Threshold is the optimal Youden's J from the averaged validation predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(model_runs)
    if n_models == 0:
        return

    fig, axes = plt.subplots(2, n_models, figsize=(3.5 * n_models, 7), squeeze=False)

    for col, (model_name, runs) in enumerate(model_runs.items()):
        avg_test = _average_preds(runs, "test_preds")
        avg_val  = _average_preds(runs, "val_preds")

        threshold = find_optimal_threshold_youden(avg_val) if not avg_val.empty else 0.5
        n_seeds = len(runs)
        seed_note = f"n={n_seeds} seeds avg" if n_seeds > 1 else f"seed {runs[0]['seed']}"

        col_title = f"{model_name}\n({seed_note}, τ={threshold:.2f})"
        if col == 0:
            axes[0, col].set_ylabel("Slide", fontsize=9, labelpad=8)
            axes[1, col].set_ylabel(f"Case ({case_agg})", fontsize=9, labelpad=8)

        if avg_test.empty:
            axes[0, col].axis("off")
            axes[1, col].axis("off")
            continue

        _draw_cm(axes[0, col], avg_test, threshold, title=col_title)
        _draw_cm(axes[1, col], _to_case_df(avg_test, case_agg), threshold, title="")

    fig.suptitle("Confusion matrices — optimal val threshold (Youden's J)", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_calibration(
    model_runs: dict[str, list[dict]],
    out_path: Path,
    n_bins: int = 10,
) -> None:
    """Reliability diagram for averaged test predictions per model."""
    from sklearn.calibration import calibration_curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")

    for (model_name, runs), color in zip(model_runs.items(), PALETTE):
        avg_test = _average_preds(runs, "test_preds")
        if avg_test.empty or avg_test["label"].nunique() < 2:
            continue
        n_seeds = len(runs)
        seed_note = f"n={n_seeds} avg" if n_seeds > 1 else f"seed {runs[0]['seed']}"
        frac_pos, mean_pred = calibration_curve(
            avg_test["label"], avg_test["prob"], n_bins=n_bins, strategy="uniform"
        )
        ax.plot(mean_pred, frac_pos, marker="o", color=color, linewidth=1.5, markersize=5,
                label=f"{model_name} ({seed_note})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration (test)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


LABEL_STYLE = {
    0: {"color": "#4C72B0", "label": "Negative"},
    1: {"color": "#DD8452", "label": "Positive"},
}


def _draw_dist(ax, df: pd.DataFrame, threshold: float, title: str) -> None:
    """Draw score distribution by label onto ax."""
    bins = np.linspace(0, 1, 21)
    for lbl in [0, 1]:
        vals = df.loc[df["label"] == lbl, "prob"].values
        if len(vals) == 0:
            continue
        style = LABEL_STYLE[lbl]
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


def plot_error_distributions(
    model_runs: dict[str, list[dict]],
    out_path: Path,
    case_agg: str = "max",
) -> None:
    """Score distributions: cols = models (runs averaged), rows = slide / case.
    Threshold line is the optimal Youden's J from the averaged validation predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(model_runs)
    if n_models == 0:
        return

    fig, axes = plt.subplots(2, n_models, figsize=(4.5 * n_models, 7), squeeze=False)

    for col, (model_name, runs) in enumerate(model_runs.items()):
        avg_test = _average_preds(runs, "test_preds")
        avg_val  = _average_preds(runs, "val_preds")

        threshold = find_optimal_threshold_youden(avg_val) if not avg_val.empty else 0.5
        n_seeds = len(runs)
        seed_note = f"n={n_seeds} seeds avg" if n_seeds > 1 else f"seed {runs[0]['seed']}"

        col_title = f"{model_name}\n({seed_note}, τ={threshold:.2f})"
        if col == 0:
            axes[0, col].set_ylabel("Slide", fontsize=9, labelpad=8)
            axes[1, col].set_ylabel(f"Case ({case_agg})", fontsize=9, labelpad=8)

        if avg_test.empty:
            axes[0, col].axis("off")
            axes[1, col].axis("off")
            continue

        _draw_dist(axes[0, col], avg_test, threshold, title=col_title)
        _draw_dist(axes[1, col], _to_case_df(avg_test, case_agg), threshold, title="")

    fig.suptitle("Score distributions — optimal val threshold (Youden's J)", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


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
    parser.add_argument("--out", default=None, help="Save summary CSVs and figures here")
    parser.add_argument(
        "--models", nargs="+", default=_DEFAULT_MODELS,
        help=f"Model keys to include. Choices: {list(ALL_MODELS)}"
    )
    parser.add_argument(
        "--case-agg", default="mean", choices=AGG_METHODS,
        help="Case aggregation method shown in summary figure (default: mean)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for confusion matrices and error distributions (default: 0.5)",
    )
    parser.add_argument(
        "--runs-subdir", default="runs",
        help="Subdirectory within each model output dir containing numbered run folders (default: runs)",
    )
    args = parser.parse_args()

    MODELS = {ALL_MODELS[k][0]: ALL_MODELS[k][1] for k in args.models if k in ALL_MODELS}
    if not MODELS:
        parser.error(f"No valid model keys. Choose from: {list(ALL_MODELS)}")

    all_rows: list[dict] = []

    # ------------------------------------------------------------------
    # Per-run slide-level table
    # ------------------------------------------------------------------
    print("\n=== Slide-level results (test) ===")
    hdr = f"{'Model':<22} {'Run':>4} {'Seed':>6} {'AUROC':>8} {'AUPRC':>8} {'Temp':>6}"
    print(hdr)
    print("-" * len(hdr))

    for name, base in MODELS.items():
        runs = collect_runs(Path(base), runs_subdir=args.runs_subdir)
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

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    # Build model_runs dict: model_name -> list of run records
    model_runs: dict[str, list[dict]] = {}
    for name in MODELS:
        model_runs[name] = [r for r in all_rows if r["model"] == name]

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)

        # CSVs
        flat_rows = []
        for r in all_rows:
            base = {k: v for k, v in r.items() if k not in ("case", "history", "test_preds")}
            flat_rows.append(base)
            for agg in AGG_METHODS:
                cm = r["case"].get(agg)
                if cm:
                    flat_rows[-1][f"case_{agg}_auroc"] = cm["auroc"]
                    flat_rows[-1][f"case_{agg}_auprc"] = cm["auprc"]
        pd.DataFrame(flat_rows).to_csv(out / "seed_runs.csv", index=False)
        pd.DataFrame(summary_rows).round(4).to_csv(out / "seed_summary.csv", index=False)

        # Figures
        plot_training_curves(model_runs, out / "training_curves.png")
        plot_roc_pr(model_runs, out / "roc_pr.png")
        plot_metric_summary(model_runs, out / "metric_summary.png", agg_method=args.case_agg)
        plot_confusion_matrices(model_runs, out / "confusion_matrices.png", case_agg=args.case_agg)
        plot_error_distributions(model_runs, out / "error_distributions.png", case_agg=args.case_agg)
        plot_calibration(model_runs, out / "calibration.png")

        print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
