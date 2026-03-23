"""
Compare attention maps across models and seeds.

Modes
-----
Single slide, latest run per model:
    python scripts/failures/compare_attention.py \\
        --slide_id SR1482_40X_HE_T176_01 --topk 100 --out outputs/attention_viz

Auto-select TP/FP/FN, latest run per model:
    python scripts/failures/compare_attention.py \\
        --auto --n_examples 3 --topk 100 --out outputs/attention_viz

Seed-grid mode (rows = models, cols = seeds):
    python scripts/failures/compare_attention.py \\
        --auto --seed_grid --n_examples 2 --out outputs/attention_viz/seed_grid

Figure layout
-------------
Default:  one row, columns = [CZI thumbnail?] + [model panels]
Seed grid: grid of (n_models rows × n_seeds cols), + optional CZI column
           Each cell shows the attention heatmap for that (model, seed) pair.
           Scores are normalised per-cell to [0,1] so spatial distribution is comparable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.feature_provider import UniFeatureProvider

# ---------------------------------------------------------------------------
# Registry: display_name -> base output dir
# ---------------------------------------------------------------------------

MODELS: dict[str, str] = {
    "Attention MIL":        "outputs/uni_attention",
    "Gated Attention":      "outputs/uni_gated_attention",
    "Top-k (k=4)":          "outputs/uni_topk_attention_k4",
    # Fair comparison runs
    "MeanPool (fair)":      "outputs/uni_mean_fair",
    "Attn MIL (fair)":      "outputs/uni_attention_fair",
    "Attn MIL + coords":    "outputs/uni_attention_spatial_fair",
    "Hybrid Attn + Mean":   "outputs/uni_hybrid_attention_mean2",
    "Hybrid + coords":      "outputs/uni_hybrid_attention_spatial_mean2",
    "Paper Repro (fair)":   "outputs/paper_reproduction_fair",
}


# ---------------------------------------------------------------------------
# Versioned layout helpers
# ---------------------------------------------------------------------------

def enumerate_seeds(base_dir: Path) -> list[tuple[str, Path, Path]]:
    """
    Return [(run_label, config_path, checkpoint_path), ...] sorted by run number.
    Falls back to flat layout (prehistory) if no versioned runs exist yet.
    """
    runs_dir = base_dir / "runs"
    entries = []
    if runs_dir.is_dir():
        for d in sorted(runs_dir.iterdir()):
            if not (d.is_dir() and d.name.isdigit()):
                continue
            cfg  = d / "config.yaml"
            ckpt = d / "model.pt"
            if cfg.exists() and ckpt.exists():
                entries.append((d.name, cfg, ckpt))
    if not entries:
        # flat / prehistory fallback
        cfg  = base_dir / "config.yaml"
        ckpt = base_dir / "model.pt"
        if not cfg.exists():
            # try the config from the project root (use model name heuristic)
            pass
        if ckpt.exists():
            entries.append(("flat", cfg if cfg.exists() else None, ckpt))
    return entries


def resolve_latest_predictions(base_dir: Path) -> Path | None:
    latest = base_dir / "latest"
    if latest.exists():
        p = latest.resolve() / "predictions.csv"
        if p.exists():
            return p
    runs = base_dir / "runs"
    if runs.is_dir():
        for d in sorted(runs.iterdir(), reverse=True):
            if d.is_dir() and d.name.isdigit():
                p = d / "predictions.csv"
                if p.exists():
                    return p
    flat = base_dir / "predictions.csv"
    return flat if flat.exists() else None


# ---------------------------------------------------------------------------
# Model loading & score extraction
# ---------------------------------------------------------------------------

def _load_model(cfg_path: Path, ckpt_path: Path, device: torch.device):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    from src.models.build import build_model
    model = build_model(cfg)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def _integrated_gradients(
    model,
    features: np.ndarray,
    coords: np.ndarray,
    device: torch.device,
    n_steps: int = 50,
) -> tuple[float, np.ndarray]:
    """Integrated Gradients: attributions = (x - baseline) * avg_grad along path.
    Baseline is the zero embedding. Per-patch importance is the L2 norm of the
    attribution vector, which is more spatially smooth than vanilla gradients."""
    model.eval()
    x        = torch.tensor(features, dtype=torch.float32, device=device)
    c        = torch.tensor(coords,   dtype=torch.float32, device=device)
    baseline = torch.zeros_like(x)

    integrated_grads = torch.zeros_like(x)
    for k in range(1, n_steps + 1):
        x_step = (baseline + (k / n_steps) * (x - baseline)).detach().requires_grad_(True)
        out    = model(x_step, coords=c)
        out["logit"].view(()).backward()
        integrated_grads += x_step.grad

    integrated_grads /= n_steps
    attributions = (x - baseline) * integrated_grads   # [N, D]
    saliency = attributions.norm(dim=-1).detach().cpu().numpy()  # [N]

    with torch.no_grad():
        prob = torch.sigmoid(model(x, coords=c)["logit"].view(())).item()
    return prob, saliency


def extract_scores(
    model,
    features: np.ndarray,
    coords: np.ndarray,
    device: torch.device,
    multihead_mode: str = "mean",
) -> tuple[float, np.ndarray | None, np.ndarray | None, str]:
    """Returns (prob, scores, coords, score_type) where score_type is one of
    'attention', 'instance', 'region', 'grad-saliency', or 'none'."""
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).to(device)
        c = torch.tensor(coords,   dtype=torch.float32).to(device)
        out = model(x, coords=c)
        prob = torch.sigmoid(out["logit"].view(())).item()

        if "attention_weights_multi" in out:
            attn_multi = out["attention_weights_multi"].cpu().numpy()  # [H, N]
            if multihead_mode == "mean":
                scores = attn_multi.mean(axis=0)
            elif multihead_mode == "max":
                scores = attn_multi.max(axis=0)
            elif multihead_mode == "head0":
                scores = attn_multi[0]
            else:
                raise ValueError(f"Unknown multihead_mode: {multihead_mode}")
            return prob, scores, coords, f"attention-multi({multihead_mode},H={attn_multi.shape[0]})"
        if "attention_weights" in out:
            return prob, out["attention_weights"].cpu().numpy(), coords, "attention"
        if "instance_scores" in out:
            return prob, out["instance_scores"].cpu().numpy(), coords, "instance"
        if "region_attention_weights" in out:
            region_ids     = model._bin_coords(c).cpu().numpy()
            region_weights = out["region_attention_weights"].cpu().numpy()
            unique_ids     = np.unique(region_ids)
            id_to_w        = {rid: region_weights[i] for i, rid in enumerate(unique_ids)}
            patch_scores   = np.array([id_to_w[rid] for rid in region_ids])
            return prob, patch_scores, coords, "region"

    # Fallback: integrated gradients (requires grad, so outside no_grad block)
    prob, saliency = _integrated_gradients(model, features, coords, device)
    return prob, saliency, coords, "integ-grad"


def extract_score_panels(
    model,
    features: np.ndarray,
    coords: np.ndarray,
    device: torch.device,
    multihead_mode: str = "mean",
    show_multihead_panels: bool = True,
) -> list[tuple[str, float, np.ndarray | None, np.ndarray | None]]:
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).to(device)
        c = torch.tensor(coords, dtype=torch.float32).to(device)
        out = model(x, coords=c)
        prob = torch.sigmoid(out["logit"].view(())).item()

        if "attention_weights_multi" in out:
            attn_multi = out["attention_weights_multi"].cpu().numpy()  # [H, N]
            if multihead_mode == "mean":
                agg_scores = attn_multi.mean(axis=0)
            elif multihead_mode == "max":
                agg_scores = attn_multi.max(axis=0)
            elif multihead_mode == "head0":
                agg_scores = attn_multi[0]
            else:
                raise ValueError(f"Unknown multihead_mode: {multihead_mode}")

            panels = [
                (f"attention-multi({multihead_mode},H={attn_multi.shape[0]})", prob, agg_scores, coords)
            ]
            if show_multihead_panels:
                for head_idx in range(attn_multi.shape[0]):
                    panels.append((f"head {head_idx}", prob, attn_multi[head_idx], coords))
            return panels

    prob, scores, eff_coords, score_type = extract_scores(
        model, features, coords, device, multihead_mode=multihead_mode
    )
    return [(score_type, prob, scores, eff_coords)]


def normalise(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def log_normalise(scores: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    """Linear-normalise to [0,1], then log-stretch and re-normalise.
    Expands the low-score range so dim patches get more colour contrast;
    compresses the bright tail so inter-model intensity is more comparable."""
    lin = normalise(scores)
    stretched = np.log(lin + eps)
    return normalise(stretched)


# ---------------------------------------------------------------------------
# CZI thumbnail
# ---------------------------------------------------------------------------

def _load_czi_thumbnail(slide_id: str, data_root: str, scale_factor: float = 0.02):
    czi_path = Path(data_root) / f"{slide_id}.czi"
    if not czi_path.exists():
        return None
    try:
        import aicspylibczi
    except ImportError:
        return None
    try:
        czi = aicspylibczi.CziFile(str(czi_path))
        bb  = czi.get_mosaic_bounding_box()
        img = czi.read_mosaic(scale_factor=scale_factor, C=0)
    except Exception as exc:
        print(f"WARNING: CZI load failed for {slide_id}: {exc}", file=sys.stderr)
        return None
    return img[0, :, :, :3], bb.w, bb.h


# ---------------------------------------------------------------------------
# Slide selection (auto mode) — uses latest run of each model
# ---------------------------------------------------------------------------

def select_slides(
    threshold: float = 0.5,
    n_examples: int = 3,
    split: str = "test",
) -> dict[str, list[str]]:
    """Select representative slides for each outcome category.

    Selection prefers slides where the outcome is *consistent* across all
    model×seed combinations — i.e., every run agrees it is a TP/FP/FN/TN.
    Ties are broken by mean predicted probability (high for TP/FP, low for TN/FN).

    This is analogous to the cross-model consistency used in
    export_failure_manifest.py, but generalised to all four outcomes.
    """
    rows = []
    for name, base in MODELS.items():
        entries = enumerate_seeds(Path(base))
        if not entries:
            print(f"  SKIP {name}: no checkpoints found")
            continue
        for run_label, cfg_path, ckpt_path in entries:
            p = Path(base) / "runs" / run_label / "predictions.csv" if cfg_path else None
            if p is None or not p.exists():
                # fall back to resolve_latest_predictions for flat layout
                p = resolve_latest_predictions(Path(base))
            if p is None or not p.exists():
                continue
            df = pd.read_csv(p)
            if split != "all":
                df = df[df["split"] == split]
            df = df[["slide_id", "label", "prob"]].copy()
            df["model"] = name
            df["run"] = run_label
            rows.append(df)

    if not rows:
        return {"tp": [], "fp": [], "fn": [], "tn": []}

    combined = pd.concat(rows, ignore_index=True)
    n_runs = combined[["model", "run"]].drop_duplicates().shape[0]

    combined["pred"] = (combined["prob"] >= threshold).astype(int)
    combined["is_tp"] = ((combined["label"] == 1) & (combined["pred"] == 1)).astype(int)
    combined["is_fp"] = ((combined["label"] == 0) & (combined["pred"] == 1)).astype(int)
    combined["is_fn"] = ((combined["label"] == 1) & (combined["pred"] == 0)).astype(int)
    combined["is_tn"] = ((combined["label"] == 0) & (combined["pred"] == 0)).astype(int)

    agg = (
        combined.groupby("slide_id")
        .agg(
            label=("label", "first"),
            mean_prob=("prob", "mean"),
            n_tp=("is_tp", "sum"),
            n_fp=("is_fp", "sum"),
            n_fn=("is_fn", "sum"),
            n_tn=("is_tn", "sum"),
        )
        .reset_index()
    )

    tp = (agg[agg["label"] == 1]
          .sort_values(["n_tp", "mean_prob"], ascending=[False, False])
          .head(n_examples)["slide_id"].tolist())
    fp = (agg[agg["label"] == 0]
          .sort_values(["n_fp", "mean_prob"], ascending=[False, False])
          .head(n_examples)["slide_id"].tolist())
    fn = (agg[agg["label"] == 1]
          .sort_values(["n_fn", "mean_prob"], ascending=[False, True])
          .head(n_examples)["slide_id"].tolist())
    tn = (agg[agg["label"] == 0]
          .sort_values(["n_tn", "mean_prob"], ascending=[False, True])
          .head(n_examples)["slide_id"].tolist())

    print(f"  {n_runs} model×seed combinations surveyed")
    print(f"  Selected {len(tp)} TP, {len(fp)} FP, {len(fn)} FN, {len(tn)} TN slides")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

OUTCOME_COLORS = {"TP": "#2e7d32", "TN": "#1565c0", "FP": "#c62828", "FN": "#6a1b9a"}
ATTENTION_CMAP = "viridis"  # perceptually uniform; no black/white extremes


def _outcome_label(prob: float, true_label: int, threshold: float) -> str:
    pred = int(prob >= threshold)
    if   true_label == 1 and pred == 1: return "TP"
    elif true_label == 0 and pred == 1: return "FP"
    elif true_label == 1 and pred == 0: return "FN"
    else:                               return "TN"


def _scatter_panel(ax, coords, scores_norm, topk_idx, title, prob, outcome: str):
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=scores_norm, cmap=ATTENTION_CMAP, s=14, alpha=0.85,
        vmin=0, vmax=1,
    )
    plt.colorbar(sc, ax=ax, label="Log-norm. score", fraction=0.03)
    outcome_color = OUTCOME_COLORS.get(outcome, "black")
    ax.set_title(f"{title}\nprob={prob:.3f}", fontsize=8)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    # Outcome badge in top-right corner
    ax.text(0.98, 0.98, outcome, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="white",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=outcome_color, alpha=0.85))


# ---------------------------------------------------------------------------
# Figure: default (one row, models as columns)
# ---------------------------------------------------------------------------

def make_comparison_figure(
    slide_id: str,
    label: int,
    category: str,
    features: np.ndarray,
    coords: np.ndarray,
    results: list[tuple[str, float, np.ndarray | None, np.ndarray | None]],
    topk: int,
    data_root: str,
    out: Path,
    threshold: float = 0.5,
) -> None:
    scored = [(n, p, sc, ec) for n, p, sc, ec in results if sc is not None]
    czi_result = _load_czi_thumbnail(slide_id, data_root)

    n_panels = len(scored) + (1 if czi_result is not None else 0)
    if n_panels == 0:
        print(f"  No panels for {slide_id}.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    label_str  = "MSI" if label == 1 else "MSS"
    cat_labels = {"tp": "True Positive", "fp": "False Positive", "fn": "False Negative", "tn": "True Negative"}
    fig.suptitle(
        f"{slide_id}  |  True: {label_str}  |  {cat_labels.get(category, category)}  "
        f"|  threshold={threshold:.2f}",
        fontsize=11, y=1.01,
    )

    ax_idx = 0
    if czi_result is not None:
        img_rgb, slide_w, slide_h = czi_result
        ax = axes[ax_idx]; ax_idx += 1
        ax.imshow(img_rgb, extent=[0, slide_w, slide_h, 0], aspect="equal")
        if scored:
            _, _, s0, c0 = scored[0]
            tk = np.argsort(normalise(s0))[-topk:]
            ax.scatter(c0[tk, 0], c0[tk, 1], s=55,
                       facecolors="none", edgecolors="cyan", linewidths=1.1,
                       label=f"Top-{topk}")
            ax.legend(fontsize=8, loc="lower right")
        ax.set_title("H&E (CZI)", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    for name, prob, scores, eff_coords in scored:
        scores_norm = log_normalise(scores)
        topk_idx    = np.argsort(scores_norm)[-topk:]
        outcome     = _outcome_label(prob, label, threshold)
        _scatter_panel(axes[ax_idx], eff_coords, scores_norm, topk_idx, name, prob, outcome)
        ax_idx += 1

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig_path = out / f"{category}_{slide_id}_compare_attention.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_path}")


# ---------------------------------------------------------------------------
# Figure: seed grid (rows = models, cols = seeds)
# ---------------------------------------------------------------------------

def make_seed_grid_figure(
    slide_id: str,
    label: int,
    category: str,
    features: np.ndarray,
    coords: np.ndarray,
    # grid_data[model_name][run_label] = (prob, scores | None, eff_coords | None)
    grid_data: dict[str, dict[str, tuple[float, np.ndarray | None, np.ndarray | None]]],
    topk: int,
    data_root: str,
    out: Path,
    threshold: float = 0.5,
) -> None:
    model_names = []
    for base_name in MODELS:
        derived = sorted(
            name for name in grid_data
            if name == base_name or name.startswith(f"{base_name} [")
        )
        model_names.extend([name for name in derived if grid_data[name]])
    if not model_names:
        print(f"  No grid data for {slide_id}.")
        return

    # Collect seed labels (union across models, sorted)
    seed_labels = sorted({s for m in model_names for s in grid_data[m]})
    n_models    = len(model_names)
    n_seeds     = len(seed_labels)

    czi_result   = _load_czi_thumbnail(slide_id, data_root)
    n_cols = n_seeds + (1 if czi_result is not None else 0)
    n_rows = n_models

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    label_str  = "MSI" if label == 1 else "MSS"
    cat_labels = {"tp": "True Positive", "fp": "False Positive", "fn": "False Negative", "tn": "True Negative"}
    fig.suptitle(
        f"{slide_id}  |  True: {label_str}  |  {cat_labels.get(category, category)}",
        fontsize=12, y=1.01,
    )

    for row, model_name in enumerate(model_names):
        col_offset = 0

        # Optional CZI column (first col of each row)
        if czi_result is not None:
            ax = axes[row, 0]
            col_offset = 1
            if row == 0:
                # Only render CZI once (first row); blank others
                img_rgb, slide_w, slide_h = czi_result
                ax.imshow(img_rgb, extent=[0, slide_w, slide_h, 0], aspect="equal")
                ax.set_title("H&E", fontsize=9)
                ax.set_xlabel("x"); ax.set_ylabel("y")
            else:
                ax.axis("off")

        for col, seed_label in enumerate(seed_labels):
            ax = axes[row, col + col_offset]
            cell = grid_data[model_name].get(seed_label)

            if cell is None or cell[1] is None:
                ax.axis("off")
                ax.set_title(f"{model_name}\nseed={seed_label}\n(no scores)", fontsize=7)
                continue

            prob, scores, eff_coords = cell
            scores_norm = log_normalise(scores)
            outcome     = _outcome_label(prob, label, threshold)

            sc = ax.scatter(
                eff_coords[:, 0], eff_coords[:, 1],
                c=scores_norm, cmap=ATTENTION_CMAP, s=12, alpha=0.85, vmin=0, vmax=1,
            )
            plt.colorbar(sc, ax=ax, fraction=0.03, label="Log-norm.")
            title = f"{model_name}\nseed {seed_label}  prob={prob:.3f}"
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.invert_yaxis()
            ax.set_aspect("equal")
            outcome_color = OUTCOME_COLORS.get(outcome, "black")
            ax.text(0.98, 0.98, outcome, transform=ax.transAxes,
                    fontsize=10, fontweight="bold", color="white",
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=outcome_color, alpha=0.85))

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig_path = out / f"{category}_{slide_id}_seed_grid.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_path}")


# ---------------------------------------------------------------------------
# Core: run one slide
# ---------------------------------------------------------------------------

def run_slide_default(
    slide_id: str,
    category: str,
    provider: UniFeatureProvider,
    device: torch.device,
    topk: int,
    data_root: str,
    out: Path,
    threshold: float = 0.5,
    multihead_mode: str = "mean",
    show_multihead_panels: bool = True,
) -> None:
    rec_idx = next(
        (i for i, r in enumerate(provider.records) if r.slide_id == slide_id), None
    )
    if rec_idx is None:
        print(f"  SKIP {slide_id}: not in provider.")
        return

    item     = provider.load_slide(rec_idx)
    features = item["features"]
    coords   = item["coords"]
    label    = item["label"]
    print(f"\n[{category.upper()}] {slide_id}  label={label}  patches={len(features)}")

    results = []
    for name, base in MODELS.items():
        seeds = enumerate_seeds(Path(base))
        if not seeds:
            print(f"  SKIP {name}: no checkpoint found")
            continue
        # use latest seed only in default mode
        run_label, cfg_path, ckpt_path = seeds[-1]
        if cfg_path is None:
            print(f"  SKIP {name}: no config for run {run_label}")
            continue
        model = _load_model(cfg_path, ckpt_path, device)
        panels = extract_score_panels(
            model, features, coords, device,
            multihead_mode=multihead_mode,
            show_multihead_panels=show_multihead_panels,
        )
        for panel_label, prob, scores, eff_coords in panels:
            tag = panel_label if scores is not None else "no patch scores"
            print(f"    {name:25s}  run={run_label}  prob={prob:.4f}  [{tag}]")
            display_name = f"{name}\n[{tag}]"
            results.append((display_name, prob, scores, eff_coords))

    make_comparison_figure(
        slide_id, label, category, features, coords, results,
        topk=min(topk, len(features)), data_root=data_root, out=out,
        threshold=threshold,
    )


def run_slide_seed_grid(
    slide_id: str,
    category: str,
    provider: UniFeatureProvider,
    device: torch.device,
    topk: int,
    data_root: str,
    out: Path,
    threshold: float = 0.5,
    multihead_mode: str = "mean",
    show_multihead_panels: bool = True,
) -> None:
    rec_idx = next(
        (i for i, r in enumerate(provider.records) if r.slide_id == slide_id), None
    )
    if rec_idx is None:
        print(f"  SKIP {slide_id}: not in provider.")
        return

    item     = provider.load_slide(rec_idx)
    features = item["features"]
    coords   = item["coords"]
    label    = item["label"]
    print(f"\n[{category.upper()}] {slide_id}  label={label}  patches={len(features)}")

    grid_data: dict[str, dict[str, tuple]] = {}
    for name, base in MODELS.items():
        seeds = enumerate_seeds(Path(base))
        if not seeds:
            print(f"  SKIP {name}: no checkpoints found")
            continue
        grid_data[name] = {}
        for run_label, cfg_path, ckpt_path in seeds:
            if cfg_path is None:
                print(f"    SKIP {name} run={run_label}: no config")
                continue
            model = _load_model(cfg_path, ckpt_path, device)
            panels = extract_score_panels(
                model, features, coords, device,
                multihead_mode=multihead_mode,
                show_multihead_panels=show_multihead_panels,
            )
            for panel_label, prob, scores, eff_coords in panels:
                row_name = name if panel_label.startswith("attention") or panel_label in {"instance", "region", "integ-grad"} else f"{name} [{panel_label}]"
                tag = panel_label if scores is not None else "no scores"
                print(f"    {row_name:25s}  run={run_label}  prob={prob:.4f}  [{tag}]")
                grid_data.setdefault(row_name, {})[run_label] = (prob, scores, eff_coords)

    make_seed_grid_figure(
        slide_id, label, category, features, coords, grid_data,
        topk=min(topk, len(features)), data_root=data_root, out=out,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--slide_id", help="Single slide to visualise")
    group.add_argument("--auto", action="store_true",
                       help="Auto-select TP/FP/FN slides from test predictions")

    parser.add_argument("--seed_grid", action="store_true",
                        help="Show all seeds as columns (rows = models)")
    parser.add_argument("--topk",       type=int,   default=100)
    parser.add_argument("--n_examples", type=int,   default=3,
                        help="Slides per category in --auto mode")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--split",      default="test")
    parser.add_argument("--multihead_mode", choices=["mean", "max", "head0"], default="mean",
                        help="How to collapse multi-head attention for visualisation")
    parser.add_argument("--no_multihead_panels", action="store_true",
                        help="Disable per-head panels for multi-head attention models")
    parser.add_argument("--out",        default="outputs/attention_viz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve data root from first available config
    data_root = None
    for base in MODELS.values():
        seeds = enumerate_seeds(Path(base))
        if seeds:
            _, cfg_path, _ = seeds[0]
            if cfg_path and cfg_path.exists():
                with open(cfg_path) as f:
                    data_root = yaml.safe_load(f)["data"]["root"]
                break
    if data_root is None:
        print("ERROR: no config files found in MODELS.")
        return

    provider = UniFeatureProvider(data_root)
    out = Path(args.out)
    run_fn = run_slide_seed_grid if args.seed_grid else run_slide_default

    if args.slide_id:
        run_fn(args.slide_id, "single", provider, device, args.topk, data_root, out,
               threshold=args.threshold, multihead_mode=args.multihead_mode,
               show_multihead_panels=not args.no_multihead_panels)
    else:
        categories = select_slides(args.threshold, args.n_examples, args.split)
        for category in ("tp", "fp", "fn", "tn"):
            slide_ids = categories.get(category, [])
            if not slide_ids:
                print(f"  No slides for category: {category}")
                continue
            for slide_id in slide_ids:
                run_fn(slide_id, category, provider, device, args.topk, data_root, out,
                       threshold=args.threshold, multihead_mode=args.multihead_mode,
                       show_multihead_panels=not args.no_multihead_panels)


if __name__ == "__main__":
    main()
