from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import zarr


@dataclass
class SlideRecord:
    slide_id: str
    cohort: str
    case_id: int
    label: int
    zarr_path: Path


class UniFeatureProvider:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.emb_root = self.root / "embeddings"
        self.sr1482_labels = pd.read_csv(self.root / "SR1482_labels.csv")
        self.sr386_labels = pd.read_csv(self.root / "SR386_labels.csv")

        self.sr1482_map = self._build_sr1482_label_map(self.sr1482_labels)
        self.sr386_map = self._build_sr386_label_map(self.sr386_labels)

        self.records = self._build_records()

    @staticmethod
    def _parse_slide_id(slide_id: str) -> tuple[str, int]:
        m = re.match(r"^(SR\d+)_40X_HE_T(\d+)_\d+$", slide_id)
        if not m:
            raise ValueError(f"Could not parse slide_id: {slide_id}")
        cohort = m.group(1)
        case_id = int(m.group(2))
        return cohort, case_id

    @staticmethod
    def _build_sr1482_label_map(df: pd.DataFrame) -> dict[int, int]:
        out = {}
        for _, row in df.iterrows():
            case_id = int(row["case_id"])
            msi = str(row["MSI"]).strip().lower()
            if msi == "no msi":
                out[case_id] = 0
            elif "msi" in msi:
                out[case_id] = 1
        return out

    @staticmethod
    def _build_sr386_label_map(df: pd.DataFrame) -> dict[int, int]:
        out = {}
        for _, row in df.iterrows():
            case_id = int(row["case_id"])
            v = row["mmr_loss_binary"]
            if pd.isna(v):
                continue
            out[case_id] = int(v)
        return out

    def _build_records(self) -> list[SlideRecord]:
        records: list[SlideRecord] = []
        for zarr_path in sorted(self.emb_root.glob("*.zarr")):
            slide_id = zarr_path.stem
            cohort, case_id = self._parse_slide_id(slide_id)

            if cohort == "SR1482":
                label = self.sr1482_map.get(case_id)
            elif cohort == "SR386":
                label = self.sr386_map.get(case_id)
            else:
                label = None

            if label is None:
                continue

            records.append(
                SlideRecord(
                    slide_id=slide_id,
                    cohort=cohort,
                    case_id=case_id,
                    label=label,
                    zarr_path=zarr_path,
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def get_record(self, idx: int) -> SlideRecord:
        return self.records[idx]

    def load_slide(self, idx: int) -> dict:
        rec = self.records[idx]
        z = zarr.open(str(rec.zarr_path), mode="r")
        features = np.asarray(z["features"])
        coords = np.asarray(z["coords"])
        return {
            "slide_id": rec.slide_id,
            "cohort": rec.cohort,
            "case_id": rec.case_id,
            "label": rec.label,
            "features": features,
            "coords": coords,
        }
