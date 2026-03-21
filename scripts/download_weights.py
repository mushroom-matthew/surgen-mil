"""Download model weights from HuggingFace Hub into the outputs/ directory.

Model weights (.pt files) are not stored in this git repository.
This script fetches them from HuggingFace Hub and places them alongside
the tracked run artifacts (metrics.json, predictions.csv, etc.).

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --repo <hf-user>/<hf-repo>
    python scripts/download_weights.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Registry: maps each run directory to its weight filename on the Hub.
# Update HF_REPO to the actual HuggingFace repo once weights are uploaded:
#   huggingface-cli upload <HF_REPO> outputs/uni_mean_fair/runs/001/checkpoint.pt \
#       outputs/uni_mean_fair/runs/001/checkpoint.pt
# ---------------------------------------------------------------------------
HF_REPO = "mushroom-matthew/surgen-mil"

WEIGHT_FILES = [
    "outputs/uni_mean_fair/runs/001/checkpoint.pt",
    "outputs/uni_mean_fair/runs/002/checkpoint.pt",
    "outputs/uni_mean_fair/runs/003/checkpoint.pt",
    "outputs/uni_attention_fair/runs/001/checkpoint.pt",
    "outputs/uni_attention_fair/runs/002/checkpoint.pt",
    "outputs/uni_attention_fair/runs/003/checkpoint.pt",
    "outputs/paper_reproduction_fair/runs/001/checkpoint.pt",
    "outputs/paper_reproduction_fair/runs/002/checkpoint.pt",
    "outputs/paper_reproduction_fair/runs/003/checkpoint.pt",
    # Appendix — sampler ablations
    "outputs/appendix/phase1_mean_random/runs/001/checkpoint.pt",
    "outputs/appendix/phase1_mean_random/runs/002/checkpoint.pt",
    "outputs/appendix/phase1_mean_random/runs/003/checkpoint.pt",
    "outputs/appendix/phase1_mean_spatial/runs/001/checkpoint.pt",
    "outputs/appendix/phase1_mean_spatial/runs/002/checkpoint.pt",
    "outputs/appendix/phase1_mean_spatial/runs/003/checkpoint.pt",
    "outputs/appendix/phase1_mean_feature_diverse/runs/001/checkpoint.pt",
    "outputs/appendix/phase1_mean_feature_diverse/runs/002/checkpoint.pt",
    "outputs/appendix/phase1_mean_feature_diverse/runs/003/checkpoint.pt",
    "outputs/appendix/phase1_attention_random/runs/001/checkpoint.pt",
    "outputs/appendix/phase1_attention_random/runs/002/checkpoint.pt",
    "outputs/appendix/phase1_attention_random/runs/003/checkpoint.pt",
    "outputs/appendix/phase1_attention_spatial/runs/001/checkpoint.pt",
    "outputs/appendix/phase1_attention_spatial/runs/002/checkpoint.pt",
    "outputs/appendix/phase1_attention_spatial/runs/003/checkpoint.pt",
    "outputs/appendix/phase1_attention_feature_diverse/runs/001/checkpoint.pt",
    "outputs/appendix/phase1_attention_feature_diverse/runs/002/checkpoint.pt",
    "outputs/appendix/phase1_attention_feature_diverse/runs/003/checkpoint.pt",
    # Appendix — loss / architecture ablations
    "outputs/appendix/attention_focal/runs/001/checkpoint.pt",
    "outputs/appendix/attention_focal/runs/002/checkpoint.pt",
    "outputs/appendix/attention_focal/runs/003/checkpoint.pt",
    "outputs/appendix/instance_mean/runs/001/checkpoint.pt",
    "outputs/appendix/instance_mean/runs/002/checkpoint.pt",
    "outputs/appendix/instance_mean/runs/003/checkpoint.pt",
    "outputs/appendix/mean_unweighted/runs/001/checkpoint.pt",
    "outputs/appendix/mean_unweighted/runs/002/checkpoint.pt",
    "outputs/appendix/mean_unweighted/runs/003/checkpoint.pt",
    "outputs/appendix/topk_attention/runs/001/checkpoint.pt",
    "outputs/appendix/topk_attention/runs/002/checkpoint.pt",
    "outputs/appendix/topk_attention/runs/003/checkpoint.pt",
]


def _ensure_hf_hub():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])


def main():
    parser = argparse.ArgumentParser(description="Download model weights from HuggingFace Hub.")
    parser.add_argument("--repo", default=HF_REPO, help="HuggingFace repo id (user/repo)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be downloaded without fetching")
    args = parser.parse_args()

    # Load token from env / .env file
    token = os.environ.get("HF_TOKEN")
    if token is None:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break

    if "PLACEHOLDER" in args.repo:
        print(
            "ERROR: HF_REPO has not been set.\n"
            "Update HF_REPO in this script or pass --repo <user>/<repo>."
        )
        sys.exit(1)

    _ensure_hf_hub()
    from huggingface_hub import hf_hub_download

    root = Path(__file__).parent.parent
    token = token or None  # hf_hub_download accepts None (unauthenticated)
    missing = [f for f in WEIGHT_FILES if not (root / f).exists()]

    if not missing:
        print(f"All {len(WEIGHT_FILES)} weight files already present.")
        return

    print(f"Downloading {len(missing)} weight file(s) from {args.repo} ...")
    for rel_path in missing:
        dest = root / rel_path
        if args.dry_run:
            print(f"  [dry-run] would download {rel_path}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=args.repo,
            filename=rel_path,
            repo_type="model",
            local_dir=str(root),
            token=token,
        )
        print(f"  ✓ {rel_path}")

    if not args.dry_run:
        print("Done.")


if __name__ == "__main__":
    main()
