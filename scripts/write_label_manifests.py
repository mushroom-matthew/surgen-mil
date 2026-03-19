"""
Write slide-level review manifests for MSI/MMR labels before training.

Outputs:
  - positive.csv
  - negative.csv
  - unknown.csv
  - discordant.csv

Usage:
    python3 scripts/write_label_manifests.py --root /mnt/data-surgen
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.feature_provider import UniFeatureProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/data-surgen")
    parser.add_argument("--out-dir", default="outputs/label_manifests")
    return parser.parse_args()


def build_slide_manifest(root: Path) -> pd.DataFrame:
    provider = UniFeatureProvider(root)
    sr1482_states = provider._build_sr1482_label_states(provider.sr1482_labels).set_index("case_id")
    sr386_lookup = provider.sr386_labels.set_index("case_id")

    rows = []
    for zarr_path in sorted((root / "embeddings").glob("*.zarr")):
        parsed = provider._parse_slide_id(zarr_path.stem)
        if parsed == (None, None):
            continue
        cohort, case_id = parsed
        czi_path = root / f"{zarr_path.stem}.czi"

        if cohort == "SR1482":
            label_row = provider.sr1482_labels.loc[provider.sr1482_labels["case_id"] == case_id].iloc[0]
            state_row = sr1482_states.loc[case_id]
            state = str(state_row["state"])
            basis = str(state_row["basis"])
            binary_label = 1 if state == "positive" else 0 if state == "negative" else None
            rows.append({
                "slide_id": zarr_path.stem,
                "cohort": cohort,
                "case_id": case_id,
                "label_state": state,
                "binary_label": binary_label,
                "label_basis": basis,
                "msi_raw": label_row["MSI"],
                "mmr_raw": label_row["MMR"],
                "mmr_ihc_raw": None,
                "mmr_loss_binary_raw": None,
                "zarr_path": str(zarr_path),
                "czi_path": str(czi_path) if czi_path.exists() else None,
            })
        elif cohort == "SR386":
            label_row = sr386_lookup.loc[case_id]
            binary_label = int(label_row["mmr_loss_binary"])
            rows.append({
                "slide_id": zarr_path.stem,
                "cohort": cohort,
                "case_id": case_id,
                "label_state": "positive" if binary_label == 1 else "negative",
                "binary_label": binary_label,
                "label_basis": "mmr_loss_binary",
                "msi_raw": None,
                "mmr_raw": None,
                "mmr_ihc_raw": label_row["mmr_ihc"],
                "mmr_loss_binary_raw": label_row["mmr_loss_binary"],
                "zarr_path": str(zarr_path),
                "czi_path": str(czi_path) if czi_path.exists() else None,
            })

    return pd.DataFrame(rows).sort_values(["label_state", "cohort", "case_id", "slide_id"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_slide_manifest(root)
    for state in ["positive", "negative", "unknown", "discordant"]:
        sub = df[df["label_state"] == state].copy()
        sub.to_csv(out_dir / f"{state}.csv", index=False)
        print(f"{state}: {len(sub)} slides -> {out_dir / f'{state}.csv'}")


if __name__ == "__main__":
    main()
