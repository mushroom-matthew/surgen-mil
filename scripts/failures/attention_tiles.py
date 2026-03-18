"""
Visualise attention weights for a single slide from an AttentionMIL model.

Produces:
  - side-by-side figure: left = CZI thumbnail (if available), right = attention scatter
  - list of top-k attended patch coordinates (printed + saved)

Usage:
    python scripts/failures/attention_tiles.py \
        --slide_id SR1482_40X_HE_T176_01 \
        --config configs/uni_attention.yaml \
        --checkpoint outputs/uni_attention/model.pt \
        --topk 20 \
        --out outputs/attention_viz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.feature_provider import UniFeatureProvider
from src.models.aggregators.attention_mil import AttentionMIL


def load_model(cfg: dict, checkpoint: Path, device: torch.device) -> AttentionMIL:
    model = AttentionMIL(
        input_dim=cfg["model"]["input_dim"],
        attention_dim=cfg["model"]["attention_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )
    state = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def run_attention(model: AttentionMIL, features: np.ndarray, device: torch.device):
    x = torch.tensor(features, dtype=torch.float32).to(device)
    out = model(x)
    logit = out["logit"].item()
    prob = torch.sigmoid(torch.tensor(logit)).item()
    attn = out["attention_weights"].cpu().numpy()  # [N]
    return prob, attn


def _load_czi_thumbnail(slide_id: str, data_root: str, scale_factor: float = 0.02):
    """
    Load a low-resolution thumbnail from the CZI file.

    Returns (img_rgb, slide_w, slide_h) where img_rgb is HxWx3 uint8 and
    slide_w/slide_h are the full slide dimensions in the same coordinate space
    as the zarr patch coords (bounding-box-relative pixels).

    Returns None if the CZI file is not found or aicspylibczi is unavailable.
    """
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
        # read_mosaic returns (S, H_px, W_px, A) uint8
        img = czi.read_mosaic(scale_factor=scale_factor, C=0)
    except Exception as exc:
        print(
            f"WARNING: failed to load CZI thumbnail for {slide_id}: {exc}",
            file=sys.stderr,
        )
        return None

    img_rgb = img[0, :, :, :3]  # drop alpha if present, keep RGB
    return img_rgb, bb.w, bb.h


def plot_attention_map(
    coords: np.ndarray,
    attn: np.ndarray,
    slide_id: str,
    label: int,
    prob: float,
    topk: int,
    out: Path,
    data_root: str,
) -> None:
    topk_idx = np.argsort(attn)[-topk:]

    czi_result = _load_czi_thumbnail(slide_id, data_root)
    has_czi = czi_result is not None

    ncols = 2 if has_czi else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 8))
    if ncols == 1:
        axes = [axes]

    label_str = "MSI (pos)" if label == 1 else "MSS (neg)"
    title = f"{slide_id}\nTrue: {label_str}   Pred prob: {prob:.3f}"

    # ---- left panel: CZI thumbnail (if available) -------------------------
    if has_czi:
        img_rgb, slide_w, slide_h = czi_result
        ax_img = axes[0]
        # extent=[0, W, H, 0] puts y=0 at top — matches patch coordinate convention
        ax_img.imshow(img_rgb, extent=[0, slide_w, slide_h, 0], aspect="equal")
        ax_img.scatter(
            coords[topk_idx, 0], coords[topk_idx, 1],
            s=60, facecolors="none", edgecolors="cyan", linewidths=1.2,
            label=f"Top-{topk} patches",
        )
        ax_img.legend(fontsize=8, loc="lower right")
        ax_img.set_title(title, fontsize=10)
        ax_img.set_xlabel("x coordinate")
        ax_img.set_ylabel("y coordinate")

    # ---- right panel (or only panel): attention scatter -------------------
    ax_att = axes[1] if has_czi else axes[0]
    sc2 = ax_att.scatter(
        coords[:, 0], coords[:, 1],
        c=attn, cmap="hot", s=18, alpha=0.85,
        vmin=0, vmax=attn.max(),
    )
    plt.colorbar(sc2, ax=ax_att, label="Attention weight", fraction=0.03)
    ax_att.scatter(
        coords[topk_idx, 0], coords[topk_idx, 1],
        s=60, facecolors="none", edgecolors="cyan", linewidths=1.2,
        label=f"Top-{topk} patches",
    )
    ax_att.set_title("Attention weights" if has_czi else title, fontsize=10)
    ax_att.set_xlabel("x coordinate")
    ax_att.set_ylabel("y coordinate")
    ax_att.invert_yaxis()
    ax_att.legend(fontsize=9)
    ax_att.set_aspect("equal")

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig_path = out / f"{slide_id}_attention.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--out", default="outputs/attention_viz")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, Path(args.checkpoint), device)

    provider = UniFeatureProvider(cfg["data"]["root"])

    # find the slide
    rec_idx = next(
        (i for i, r in enumerate(provider.records) if r.slide_id == args.slide_id),
        None,
    )
    if rec_idx is None:
        print(f"ERROR: slide_id '{args.slide_id}' not found in provider.")
        return

    item = provider.load_slide(rec_idx)
    features = item["features"]
    coords = item["coords"]
    label = item["label"]

    print(f"Slide: {args.slide_id}  label={label}  patches={len(features)}")

    prob, attn = run_attention(model, features, device)
    print(f"Predicted prob: {prob:.4f}")

    topk = min(args.topk, len(attn))
    topk_idx = np.argsort(attn)[-topk:][::-1]

    print(f"\nTop-{topk} attended patches (rank, attn_weight, x, y):")
    for rank, idx in enumerate(topk_idx, 1):
        print(f"  {rank:3d}  attn={attn[idx]:.5f}  x={coords[idx, 0]}  y={coords[idx, 1]}")

    plot_attention_map(
        coords, attn, args.slide_id, label, prob,
        topk=topk, out=Path(args.out),
        data_root=cfg["data"]["root"],
    )

    # save top-k coords to csv
    import pandas as pd
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    topk_df = pd.DataFrame({
        "rank": range(1, topk + 1),
        "patch_idx": topk_idx,
        "attn_weight": attn[topk_idx],
        "x": coords[topk_idx, 0],
        "y": coords[topk_idx, 1],
    })
    csv_path = out_dir / f"{args.slide_id}_topk.csv"
    topk_df.to_csv(csv_path, index=False)
    print(f"  saved {csv_path}")


if __name__ == "__main__":
    main()
