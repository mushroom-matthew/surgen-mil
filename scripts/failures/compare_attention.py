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
    "Attention MIL":   "outputs/uni_attention",
    "Gated Attention": "outputs/uni_gated_attention",
    "Top-k (k=4)":     "outputs/uni_topk_attention_k4",
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


@torch.no_grad()
def extract_scores(
    model,
    features: np.ndarray,
    coords: np.ndarray,
    device: torch.device,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    x = torch.tensor(features, dtype=torch.float32).to(device)
    c = torch.tensor(coords,   dtype=torch.float32).to(device)
    out = model(x, coords=c)
    prob = torch.sigmoid(out["logit"].view(())).item()

    if "attention_weights" in out:
        return prob, out["attention_weights"].cpu().numpy(), coords
    if "instance_scores" in out:
        return prob, out["instance_scores"].cpu().numpy(), coords
    if "region_attention_weights" in out:
        region_ids     = model._bin_coords(c).cpu().numpy()
        region_weights = out["region_attention_weights"].cpu().numpy()
        unique_ids     = np.unique(region_ids)
        id_to_w        = {rid: region_weights[i] for i, rid in enumerate(unique_ids)}
        patch_scores   = np.array([id_to_w[rid] for rid in region_ids])
        return prob, patch_scores, coords
    return prob, None, None


def normalise(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


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
    frames = []
    for name, base in MODELS.items():
        p = resolve_latest_predictions(Path(base))
        if p is None:
            print(f"  SKIP {name}: no predictions found")
            continue
        df = pd.read_csv(p)
        if split != "all":
            df = df[df["split"] == split]
        df = df[["slide_id", "label", "prob"]].copy()
        df["model"] = name
        frames.append(df)

    if not frames:
        return {"tp": [], "fp": [], "fn": []}

    combined = pd.concat(frames, ignore_index=True)
    agg = (
        combined.groupby("slide_id")
        .agg(label=("label", "first"), mean_prob=("prob", "mean"))
        .reset_index()
    )

    tp = (agg[(agg["label"] == 1) & (agg["mean_prob"] >= threshold)]
          .sort_values("mean_prob", ascending=False)
          .head(n_examples)["slide_id"].tolist())
    fp = (agg[(agg["label"] == 0) & (agg["mean_prob"] >= threshold)]
          .sort_values("mean_prob", ascending=False)
          .head(n_examples)["slide_id"].tolist())
    fn = (agg[(agg["label"] == 1) & (agg["mean_prob"] < threshold)]
          .sort_values("mean_prob", ascending=True)
          .head(n_examples)["slide_id"].tolist())

    print(f"  Selected {len(tp)} TP, {len(fp)} FP, {len(fn)} FN slides")
    return {"tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _scatter_panel(ax, coords, scores_norm, topk_idx, title, prob):
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=scores_norm, cmap="hot", s=14, alpha=0.85,
        vmin=0, vmax=1,
    )
    ax.scatter(
        coords[topk_idx, 0], coords[topk_idx, 1],
        s=55, facecolors="none", edgecolors="cyan", linewidths=1.1,
    )
    plt.colorbar(sc, ax=ax, label="Norm. score", fraction=0.03)
    ax.set_title(f"{title}\nprob={prob:.3f}", fontsize=8)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.invert_yaxis()
    ax.set_aspect("equal")


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
    cat_labels = {"tp": "True Positive", "fp": "False Positive", "fn": "False Negative"}
    fig.suptitle(
        f"{slide_id}  |  True: {label_str}  |  {cat_labels.get(category, category)}",
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
        scores_norm = normalise(scores)
        topk_idx    = np.argsort(scores_norm)[-topk:]
        _scatter_panel(axes[ax_idx], eff_coords, scores_norm, topk_idx, name, prob)
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
) -> None:
    model_names = [n for n in MODELS if n in grid_data and grid_data[n]]
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
    cat_labels = {"tp": "True Positive", "fp": "False Positive", "fn": "False Negative"}
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
            scores_norm = normalise(scores)
            topk_idx    = np.argsort(scores_norm)[-min(topk, len(scores)):]

            sc = ax.scatter(
                eff_coords[:, 0], eff_coords[:, 1],
                c=scores_norm, cmap="hot", s=12, alpha=0.85, vmin=0, vmax=1,
            )
            ax.scatter(
                eff_coords[topk_idx, 0], eff_coords[topk_idx, 1],
                s=50, facecolors="none", edgecolors="cyan", linewidths=1.0,
            )
            plt.colorbar(sc, ax=ax, fraction=0.03, label="Norm.")
            title = f"{model_name}\nseed {seed_label}  prob={prob:.3f}"
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.invert_yaxis()
            ax.set_aspect("equal")

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
        prob, scores, eff_coords = extract_scores(model, features, coords, device)
        tag = "scores OK" if scores is not None else "no patch scores"
        print(f"    {name:25s}  run={run_label}  prob={prob:.4f}  {tag}")
        results.append((name, prob, scores, eff_coords))

    make_comparison_figure(
        slide_id, label, category, features, coords, results,
        topk=min(topk, len(features)), data_root=data_root, out=out,
    )


def run_slide_seed_grid(
    slide_id: str,
    category: str,
    provider: UniFeatureProvider,
    device: torch.device,
    topk: int,
    data_root: str,
    out: Path,
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
            prob, scores, eff_coords = extract_scores(model, features, coords, device)
            tag = "OK" if scores is not None else "no scores"
            print(f"    {name:25s}  run={run_label}  prob={prob:.4f}  {tag}")
            grid_data[name][run_label] = (prob, scores, eff_coords)

    make_seed_grid_figure(
        slide_id, label, category, features, coords, grid_data,
        topk=min(topk, len(features)), data_root=data_root, out=out,
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
        run_fn(args.slide_id, "single", provider, device, args.topk, data_root, out)
    else:
        categories = select_slides(args.threshold, args.n_examples, args.split)
        for category, slide_ids in categories.items():
            if not slide_ids:
                print(f"  No slides for category: {category}")
                continue
            for slide_id in slide_ids:
                run_fn(slide_id, category, provider, device, args.topk, data_root, out)


if __name__ == "__main__":
    main()
