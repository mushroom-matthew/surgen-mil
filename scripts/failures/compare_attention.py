"""
Compare attention maps across multiple models.

Modes
-----
Single slide:
    python scripts/failures/compare_attention.py \\
        --slide_id SR1482_40X_HE_T176_01 --topk 100 --out outputs/attention_viz

Auto (selects representative TP / FP / FN slides from test predictions):
    python scripts/failures/compare_attention.py \\
        --auto --n_examples 3 --topk 100 --out outputs/attention_viz

For each slide a figure is saved with:
  - CZI thumbnail panel (H&E) with top-k circles
  - One attention-score panel per model that produces per-patch scores

Normalisation: each model's scores are independently rescaled to [0, 1]
so the spatial distribution is directly comparable across models.

RUNS registry maps display name -> (config_path, checkpoint_path).
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
# Registry: display_name -> (config_path, checkpoint_path)
# ---------------------------------------------------------------------------

RUNS: dict[str, tuple[str, str]] = {
    "Attention MIL": (
        "configs/uni_attention.yaml",
        "outputs/uni_attention/model.pt",
    ),
    "Gated Attention": (
        "configs/uni_gated_attention.yaml",
        "outputs/uni_gated_attention/model.pt",
    ),
    "Region Attn (8×8)": (
        "configs/uni_region_attention_8.yaml",
        "outputs/uni_region_attention_8/model.pt",
    ),
    "Region Attn (16×16)": (
        "configs/uni_region_attention_16.yaml",
        "outputs/uni_region_attention_16/model.pt",
    ),
}


# ---------------------------------------------------------------------------
# Model loading & score extraction
# ---------------------------------------------------------------------------

def _load_model(cfg: dict, checkpoint: Path, device: torch.device):
    from src.models.build import build_model
    model = build_model(cfg)
    state = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def extract_scores(
    model,
    features: np.ndarray,
    coords: np.ndarray,
    device: torch.device,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """
    Returns (prob, patch_scores, effective_coords).
    patch_scores are in raw model units; normalise separately for display.
    Returns (prob, None, None) for models with no per-patch output.
    """
    x = torch.tensor(features, dtype=torch.float32).to(device)
    c = torch.tensor(coords,   dtype=torch.float32).to(device)
    out = model(x, coords=c)

    prob = torch.sigmoid(out["logit"].view(())).item()

    if "attention_weights" in out:
        return prob, out["attention_weights"].cpu().numpy(), coords

    if "instance_scores" in out:
        return prob, out["instance_scores"].cpu().numpy(), coords

    if "region_attention_weights" in out:
        region_ids    = model._bin_coords(c).cpu().numpy()
        region_weights = out["region_attention_weights"].cpu().numpy()
        unique_ids    = np.unique(region_ids)
        id_to_w = {rid: region_weights[i] for i, rid in enumerate(unique_ids)}
        patch_scores  = np.array([id_to_w[rid] for rid in region_ids])
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
        bb = czi.get_mosaic_bounding_box()
        img = czi.read_mosaic(scale_factor=scale_factor, C=0)
    except Exception as exc:
        print(
            f"WARNING: failed to load CZI thumbnail for {slide_id}: {exc}",
            file=sys.stderr,
        )
        return None

    return img[0, :, :, :3], bb.w, bb.h


# ---------------------------------------------------------------------------
# Slide selection (auto mode)
# ---------------------------------------------------------------------------

def _load_run_predictions(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "predictions.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def select_slides(
    threshold: float = 0.5,
    n_examples: int = 3,
    split: str = "test",
) -> dict[str, list[str]]:
    """
    Return {"tp": [...], "fp": [...], "fn": [...]} slide_ids.

    Selection criteria (applied to the test split):
      TP — label=1, mean_prob >= threshold, ranked by mean_prob desc  (most confident correct)
      FP — label=0, mean_prob >= threshold, ranked by mean_prob desc  (most confidently wrong)
      FN — label=1, mean_prob <  threshold, ranked by mean_prob asc   (most confidently missed)
    """
    frames = []
    for name, (cfg_path, _) in RUNS.items():
        if not Path(cfg_path).exists():
            continue
        with open(cfg_path) as f:
            run_dir = Path(yaml.safe_load(f)["output"]["dir"])
        df = _load_run_predictions(run_dir)
        if df is None:
            continue
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
# Plotting
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
    ax.set_title(f"{title}\nprob={prob:.3f}", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.invert_yaxis()
    ax.set_aspect("equal")


def make_comparison_figure(
    slide_id: str,
    label: int,
    category: str,          # "tp" / "fp" / "fn"
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
        print(f"  No panels to render for {slide_id}.")
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

    # CZI panel
    if czi_result is not None:
        img_rgb, slide_w, slide_h = czi_result
        ax = axes[ax_idx]; ax_idx += 1
        ax.imshow(img_rgb, extent=[0, slide_w, slide_h, 0], aspect="equal")
        if scored:
            _, _, first_scores, first_coords = scored[0]
            first_norm = normalise(first_scores)
            first_topk = np.argsort(first_norm)[-topk:]
            ax.scatter(
                first_coords[first_topk, 0], first_coords[first_topk, 1],
                s=55, facecolors="none", edgecolors="cyan", linewidths=1.1,
                label=f"Top-{topk}",
            )
            ax.legend(fontsize=8, loc="lower right")
        ax.set_title("H&E (CZI)", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    # One panel per scored model
    for name, prob, scores, eff_coords in scored:
        scores_norm = normalise(scores)
        topk_idx    = np.argsort(scores_norm)[-topk:]
        _scatter_panel(axes[ax_idx], eff_coords, scores_norm, topk_idx, name, prob)
        ax_idx += 1

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fname = f"{category}_{slide_id}_compare_attention.png"
    fig_path = out / fname
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_path}")


# ---------------------------------------------------------------------------
# Core: run comparison for one slide
# ---------------------------------------------------------------------------

def run_slide(
    slide_id: str,
    category: str,
    provider: UniFeatureProvider,
    selected_models: list[str],
    device: torch.device,
    topk: int,
    data_root: str,
    out: Path,
) -> None:
    rec_idx = next(
        (i for i, r in enumerate(provider.records) if r.slide_id == slide_id),
        None,
    )
    if rec_idx is None:
        print(f"  SKIP {slide_id}: not found in provider.")
        return

    item     = provider.load_slide(rec_idx)
    features = item["features"]
    coords   = item["coords"]
    label    = item["label"]
    print(f"\n[{category.upper()}] {slide_id}  label={label}  patches={len(features)}")

    results = []
    for name in selected_models:
        cfg_path, ckpt_path = RUNS[name]
        if not Path(ckpt_path).exists():
            print(f"  SKIP {name}: checkpoint not found")
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model = _load_model(cfg, Path(ckpt_path), device)
        prob, scores, eff_coords = extract_scores(model, features, coords, device)
        tag = "scores OK" if scores is not None else "no patch scores"
        print(f"    {name:25s}  prob={prob:.4f}  {tag}")
        results.append((name, prob, scores, eff_coords))

    actual_topk = min(topk, len(features))
    make_comparison_figure(
        slide_id=slide_id,
        label=label,
        category=category,
        features=features,
        coords=coords,
        results=results,
        topk=actual_topk,
        data_root=data_root,
        out=out,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--slide_id", help="Single slide to visualise")
    group.add_argument("--auto", action="store_true",
                       help="Auto-select TP / FP / FN slides from test predictions")

    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of RUNS keys to include (default: all available)")
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
    for cfg_path, _ in RUNS.values():
        if Path(cfg_path).exists():
            with open(cfg_path) as f:
                data_root = yaml.safe_load(f)["data"]["root"]
            break
    if data_root is None:
        print("ERROR: no config files found in RUNS.")
        return

    provider = UniFeatureProvider(data_root)

    selected = args.models if args.models else [
        n for n, (c, k) in RUNS.items() if Path(k).exists()
    ]
    if not selected:
        print("ERROR: no available checkpoints found.")
        return

    out = Path(args.out)

    if args.slide_id:
        run_slide(args.slide_id, "single", provider, selected, device,
                  args.topk, data_root, out)
    else:
        categories = select_slides(args.threshold, args.n_examples, args.split)
        for category, slide_ids in categories.items():
            if not slide_ids:
                print(f"  No slides found for category: {category}")
                continue
            for slide_id in slide_ids:
                run_slide(slide_id, category, provider, selected, device,
                          args.topk, data_root, out)


if __name__ == "__main__":
    main()
