"""Analyse the outputs/multisplit experiment layout.

This script is intentionally layered on top of scripts/compare_models.py:
  - overall multisplit summaries treat each (split, seed) run as one draw
  - per-split subdirectories reuse the existing single-split visualisations,
    where averaging predictions across seeds is valid

Usage
-----
    python scripts/compare_multisplit.py
    python scripts/compare_multisplit.py --base outputs/multisplit --out outputs/multisplit/analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import compare_models as cm

PREFERRED_MODEL_ORDER = [
    "uni_mean_fair",
    "uni_attention_fair",
    "paper_reproduction_fair",
    "uni_gated_attention",
    "uni_mean_var",
    "uni_hybrid_attention_mean2",
    "uni_attention_spatial_fair",
    "uni_hybrid_attention_spatial_mean2",
]


def ordered_model_names(names: list[str] | np.ndarray) -> list[str]:
    names = list(names)
    preferred = [name for name in PREFERRED_MODEL_ORDER if name in names]
    remaining = sorted(name for name in names if name not in PREFERRED_MODEL_ORDER)
    return preferred + remaining


def _split_seed_from_dir(path: Path) -> int | None:
    name = path.name
    if not name.startswith("split_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def discover_multisplit_runs(base_dir: Path) -> tuple[dict[str, list[dict]], dict[int, dict[str, list[dict]]]]:
    by_model: dict[str, list[dict]] = {}
    by_split: dict[int, dict[str, list[dict]]] = {}

    for model_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        model_name = model_dir.name
        model_runs: list[dict] = []

        for split_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() or p.is_symlink()):
            split_seed = _split_seed_from_dir(split_dir)
            if split_seed is None:
                continue

            runs = cm.collect_runs(split_dir.resolve())
            if not runs:
                continue

            for run in runs:
                run["split_seed"] = split_seed
                run["model"] = model_name

            model_runs.extend(runs)
            by_split.setdefault(split_seed, {})[model_name] = runs

        if model_runs:
            by_model[model_name] = model_runs

    return by_model, by_split


def runs_to_frame(model_runs: dict[str, list[dict]], case_agg: str = "mean") -> pd.DataFrame:
    rows = []
    for model_name, runs in model_runs.items():
        for run in runs:
            case = run.get("case", {}).get(case_agg, {})
            rows.append({
                "model": model_name,
                "split_seed": run.get("split_seed"),
                "seed": run.get("seed"),
                "run": run.get("run"),
                "slide_auroc": run.get("slide_auroc"),
                "slide_auprc": run.get("slide_auprc"),
                f"case_{case_agg}_auroc": case.get("auroc"),
                f"case_{case_agg}_auprc": case.get("auprc"),
            })
    return pd.DataFrame(rows)


def cohort_runs_to_frame(model_runs: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for model_name, runs in model_runs.items():
        for run in runs:
            for cohort_row in run.get("cohort", []):
                rows.append({
                    "model": model_name,
                    "split_seed": run.get("split_seed"),
                    "seed": run.get("seed"),
                    "run": run.get("run"),
                    "cohort": cohort_row.get("cohort"),
                    "cohort_auroc": cohort_row.get("auroc"),
                    "cohort_auprc": cohort_row.get("auprc"),
                    "n": cohort_row.get("n"),
                })
    return pd.DataFrame(rows)


def summarise_multisplit(df: pd.DataFrame, group_cols: list[str], case_agg: str = "mean") -> pd.DataFrame:
    metric_cols = [
        "slide_auroc",
        "slide_auprc",
        f"case_{case_agg}_auroc",
        f"case_{case_agg}_auprc",
    ]
    grouped = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_")
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


def plot_multisplit_lines(df: pd.DataFrame, out_path: Path, case_agg: str = "mean") -> None:
    metrics = [
        ("slide_auroc", "Slide AUROC"),
        ("slide_auprc", "Slide AUPRC"),
        (f"case_{case_agg}_auroc", f"Case AUROC ({case_agg})"),
        (f"case_{case_agg}_auprc", f"Case AUPRC ({case_agg})"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5), squeeze=False)
    axes = axes[0]

    model_names = ordered_model_names(df["model"].dropna().unique())
    split_seeds = sorted(int(s) for s in df["split_seed"].dropna().unique())

    for ax, (metric, label) in zip(axes, metrics):
        metric_all_vals = []
        for color, model_name in zip(cm.PALETTE, model_names):
            sub = df[df["model"] == model_name]
            means, stds = [], []
            for split_seed in split_seeds:
                vals = sub.loc[sub["split_seed"] == split_seed, metric].dropna().values
                metric_all_vals.extend(vals.tolist())
                means.append(np.mean(vals) if len(vals) else np.nan)
                stds.append(np.std(vals) if len(vals) > 1 else 0.0)
            ax.errorbar(
                split_seeds,
                means,
                yerr=stds,
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=color,
                label=model_name,
            )
        if "auroc" in metric:
            ax.axhline(cm.PAPER_AUROC, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(label)
        ax.set_xlabel("Split seed")
        ax.set_ylabel(label)
        ax.set_xticks(split_seeds)
        ax.set_ylim(*cm.adaptive_unit_ylim(metric_all_vals))
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_multisplit_strips(df: pd.DataFrame, out_path: Path, case_agg: str = "mean") -> None:
    metrics = [
        ("slide_auroc", "Slide AUROC"),
        ("slide_auprc", "Slide AUPRC"),
        (f"case_{case_agg}_auroc", f"Case AUROC ({case_agg})"),
        (f"case_{case_agg}_auprc", f"Case AUPRC ({case_agg})"),
    ]
    model_names = ordered_model_names(df["model"].dropna().unique())
    x_pos = np.arange(len(model_names))
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5), squeeze=False)
    axes = axes[0]

    for ax, (metric, label) in zip(axes, metrics):
        means, stds, all_vals = [], [], []
        for model_name in model_names:
            vals = df.loc[df["model"] == model_name, metric].dropna().values
            all_vals.append(vals)
            means.append(np.mean(vals) if len(vals) else np.nan)
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)

        colors = [cm.PALETTE[i % len(cm.PALETTE)] for i in range(len(model_names))]
        ax.bar(x_pos, means, yerr=stds, capsize=4, color=colors, alpha=0.6, width=0.5)
        for xi, vals in zip(x_pos, all_vals):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(xi + jitter, vals, color="black", s=25, alpha=0.7, zorder=3)

        if "auroc" in metric:
            ax.axhline(cm.PAPER_AUROC, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(*cm.adaptive_unit_ylim([v for vals in all_vals for v in vals]))
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_multisplit_cohort_lines(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    cohorts = sorted(df["cohort"].dropna().unique())
    metrics = [("cohort_auroc", "Cohort AUROC"), ("cohort_auprc", "Cohort AUPRC")]
    split_seeds = sorted(int(s) for s in df["split_seed"].dropna().unique())
    model_names = ordered_model_names(df["model"].dropna().unique())

    fig, axes = plt.subplots(
        len(metrics),
        len(cohorts),
        figsize=(4.8 * len(cohorts), 4.2 * len(metrics)),
        squeeze=False,
    )

    for row_idx, (metric, label) in enumerate(metrics):
        metric_all_vals = []
        for col_idx, cohort in enumerate(cohorts):
            cohort_df = df[df["cohort"] == cohort]
            ax = axes[row_idx, col_idx]
            for color, model_name in zip(cm.PALETTE, model_names):
                sub = cohort_df[cohort_df["model"] == model_name]
                means, stds = [], []
                for split_seed in split_seeds:
                    vals = sub.loc[sub["split_seed"] == split_seed, metric].dropna().values
                    metric_all_vals.extend(vals.tolist())
                    means.append(np.mean(vals) if len(vals) else np.nan)
                    stds.append(np.std(vals) if len(vals) > 1 else 0.0)
                ax.errorbar(
                    split_seeds,
                    means,
                    yerr=stds,
                    marker="o",
                    linewidth=1.8,
                    capsize=3,
                    color=color,
                    label=model_name,
                )
            if metric == "cohort_auroc":
                ax.axhline(cm.PAPER_AUROC, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_title(cohort)
            ax.set_xlabel("Split seed")
            if col_idx == 0:
                ax.set_ylabel(label)
            ax.set_xticks(split_seeds)
            ax.grid(True, alpha=0.3)
        ylim = cm.adaptive_unit_ylim(metric_all_vals)
        for col_idx in range(len(cohorts)):
            axes[row_idx, col_idx].set_ylim(*ylim)

    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_multisplit_cohort_strips(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    cohorts = sorted(df["cohort"].dropna().unique())
    metrics = [("cohort_auroc", "Cohort AUROC"), ("cohort_auprc", "Cohort AUPRC")]
    model_names = ordered_model_names(df["model"].dropna().unique())
    x_pos = np.arange(len(model_names))
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(
        len(metrics),
        len(cohorts),
        figsize=(4.8 * len(cohorts), 4.5 * len(metrics)),
        squeeze=False,
    )

    for row_idx, (metric, label) in enumerate(metrics):
        metric_all_vals = []
        for col_idx, cohort in enumerate(cohorts):
            cohort_df = df[df["cohort"] == cohort]
            ax = axes[row_idx, col_idx]
            means, stds, all_vals = [], [], []
            for model_name in model_names:
                vals = cohort_df.loc[cohort_df["model"] == model_name, metric].dropna().values
                all_vals.append(vals)
                metric_all_vals.extend(vals.tolist())
                means.append(np.mean(vals) if len(vals) else np.nan)
                stds.append(np.std(vals) if len(vals) > 1 else 0.0)
            colors = [cm.PALETTE[i % len(cm.PALETTE)] for i in range(len(model_names))]
            ax.bar(x_pos, means, yerr=stds, capsize=4, color=colors, alpha=0.6, width=0.5)
            for xi, vals in zip(x_pos, all_vals):
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(xi + jitter, vals, color="black", s=22, alpha=0.75, zorder=3)
            if metric == "cohort_auroc":
                ax.axhline(cm.PAPER_AUROC, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
            ax.set_title(cohort)
            if col_idx == 0:
                ax.set_ylabel(label)
            ax.grid(True, axis="y", alpha=0.3)
        ylim = cm.adaptive_unit_ylim(metric_all_vals)
        for col_idx in range(len(cohorts)):
            axes[row_idx, col_idx].set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def extended_split_summary(split_model_runs: dict[int, dict[str, list[dict]]]) -> pd.DataFrame:
    rows = []
    for split_seed, model_runs in sorted(split_model_runs.items()):
        for model_name, runs in model_runs.items():
            avg_test, note = cm._average_preds(runs, "test_preds")
            if avg_test.empty or avg_test["label"].nunique() < 2:
                continue
            y_true = avg_test["label"].values
            y_score = avg_test["prob"].values
            lo, hi = cm.bootstrap_auroc_ci(y_true, y_score)
            ece = cm.expected_calibration_error(y_true, y_score)
            per_cohort = cm.cohort_metrics(avg_test)
            rows.append({
                "split_seed": split_seed,
                "model": model_name,
                "note": note,
                "auroc_ci_lo": lo,
                "auroc_ci_hi": hi,
                "ece": ece,
                **{f"{c}_auroc": m["auroc"] for c, m in per_cohort.items()},
                **{f"{c}_auprc": m["auprc"] for c, m in per_cohort.items()},
                **{f"{c}_n": m["n"] for c, m in per_cohort.items()},
            })
    return pd.DataFrame(rows)


def write_split_reports(split_model_runs: dict[int, dict[str, list[dict]]], out_dir: Path, case_agg: str = "mean") -> None:
    for split_seed, model_runs in sorted(split_model_runs.items()):
        split_out = out_dir / f"split_{split_seed}"
        split_out.mkdir(parents=True, exist_ok=True)
        model_runs = {
            name: model_runs[name]
            for name in ordered_model_names(list(model_runs.keys()))
        }

        summaries = [cm.summarise(name, runs) for name, runs in model_runs.items()]
        pd.DataFrame(summaries).to_csv(split_out / "summary.csv", index=False)

        cm.plot_metric_summary(model_runs, split_out / "summary_metrics.png", case_agg=case_agg)
        cm.plot_roc_pr(model_runs, split_out / "roc_pr_curves.png")
        cm.plot_confusion_matrices(model_runs, split_out / "confusion_matrices.png", case_agg=case_agg)
        cm.plot_calibration(model_runs, split_out / "calibration.png")
        cm.plot_score_distributions(model_runs, split_out / "score_distributions.png", case_agg=case_agg)
        cm.plot_training_curves(model_runs, split_out / "training_curves.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse multi-split experiment outputs.")
    parser.add_argument("--base", default="outputs/multisplit", help="Base directory of the multisplit layout")
    parser.add_argument("--out", default="outputs/multisplit/analysis", help="Output directory for reports")
    parser.add_argument("--case-agg", default="mean", choices=cm.AGG_METHODS, help="Case aggregation method")
    args = parser.parse_args()

    base_dir = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_runs, split_model_runs = discover_multisplit_runs(base_dir)
    if not model_runs:
        raise SystemExit(f"No multisplit runs found under {base_dir}")

    raw_df = runs_to_frame(model_runs, case_agg=args.case_agg)
    raw_df.to_csv(out_dir / "raw_runs.csv", index=False)
    cohort_df = cohort_runs_to_frame(model_runs)
    if not cohort_df.empty:
        cohort_df.to_csv(out_dir / "raw_cohort_runs.csv", index=False)

    overall_summary = summarise_multisplit(raw_df, ["model"], case_agg=args.case_agg)
    overall_summary.to_csv(out_dir / "summary_overall.csv", index=False)

    split_summary = summarise_multisplit(raw_df, ["split_seed", "model"], case_agg=args.case_agg)
    split_summary.to_csv(out_dir / "summary_by_split.csv", index=False)

    ext_df = extended_split_summary(split_model_runs)
    if not ext_df.empty:
        ext_df.to_csv(out_dir / "summary_extended_by_split.csv", index=False)

    plot_multisplit_lines(raw_df, out_dir / "multisplit_lines.png", case_agg=args.case_agg)
    plot_multisplit_strips(raw_df, out_dir / "multisplit_strips.png", case_agg=args.case_agg)
    plot_multisplit_cohort_lines(cohort_df, out_dir / "multisplit_cohort_lines.png")
    plot_multisplit_cohort_strips(cohort_df, out_dir / "multisplit_cohort_strips.png")

    write_split_reports(split_model_runs, out_dir / "per_split", case_agg=args.case_agg)

    print(f"Saved multisplit analysis to {out_dir}")


if __name__ == "__main__":
    main()
