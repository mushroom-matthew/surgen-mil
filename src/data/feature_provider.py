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
    label_state: str
    label_basis: str
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
    def _parse_slide_id(slide_id: str) -> tuple[str, int] | tuple[None, None]:
        m = re.match(r"^(SR\d+)_40X_HE_T(\d+)_\d+$", slide_id)
        if not m:
            return None, None
        cohort = m.group(1)
        case_id = int(m.group(2))
        return cohort, case_id

    @staticmethod
    def _build_sr1482_label_map(df: pd.DataFrame) -> dict[int, int]:
        out: dict[int, int] = {}
        for case_id, state, _ in UniFeatureProvider._build_sr1482_label_states(df).itertuples(index=False):
            if state == "positive":
                out[int(case_id)] = 1
            elif state == "negative":
                out[int(case_id)] = 0
        return out

    @staticmethod
    def _normalize_sr1482_msi(value) -> str:
        s = str(value).strip().lower()
        # Lower-casing already collapses the observed typo "MSI HIgh".
        if s == "msi high":
            return "msi high"
        return s

    @staticmethod
    def _sr1482_msi_state(value) -> tuple[str, str]:
        s = UniFeatureProvider._normalize_sr1482_msi(value)
        if s == "no msi":
            return "negative", "MSI"
        if s == "msi high":
            return "positive", "MSI"
        if s in {"msi low", "not performed", "insufficient", "failed"}:
            return "unknown", "MSI"
        return "unknown", "MSI"

    @staticmethod
    def _sr1482_mmr_state(value) -> tuple[str, str]:
        s = str(value).strip().lower()
        if s in {"no loss"}:
            return "negative", "MMR"
        if "loss" in s:
            return "positive", "MMR"
        if s in {"not performed"}:
            return "unknown", "MMR"
        return "unknown", "MMR"

    @staticmethod
    def _resolve_sr1482_state(msi_value, mmr_value) -> tuple[str, str]:
        msi_state, _ = UniFeatureProvider._sr1482_msi_state(msi_value)
        mmr_state, _ = UniFeatureProvider._sr1482_mmr_state(mmr_value)

        states = {msi_state, mmr_state}
        if "positive" in states and "negative" in states:
            return "discordant", "MSI+MMR"
        if "positive" in states:
            if msi_state == "positive" and mmr_state == "positive":
                return "positive", "MSI+MMR"
            return "positive", "MSI" if msi_state == "positive" else "MMR"
        if "negative" in states:
            if msi_state == "negative" and mmr_state == "negative":
                return "negative", "MSI+MMR"
            return "negative", "MSI" if msi_state == "negative" else "MMR"
        return "unknown", "none"

    @staticmethod
    def _build_sr1482_label_states(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            case_id = int(row["case_id"])
            state, basis = UniFeatureProvider._resolve_sr1482_state(row["MSI"], row["MMR"])
            rows.append({"case_id": case_id, "state": state, "basis": basis})
        return pd.DataFrame(rows)

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
        sr1482_states = self._build_sr1482_label_states(self.sr1482_labels).set_index("case_id")
        for zarr_path in sorted(self.emb_root.glob("*.zarr")):
            slide_id = zarr_path.stem
            cohort, case_id = self._parse_slide_id(slide_id)

            if cohort is None:
                continue

            if cohort == "SR1482":
                state_row = sr1482_states.loc[case_id]
                label_state = str(state_row["state"])
                label_basis = str(state_row["basis"])
                label = self.sr1482_map.get(case_id)
            elif cohort == "SR386":
                label = self.sr386_map.get(case_id)
                label_state = "positive" if label == 1 else "negative"
                label_basis = "mmr_loss_binary"
            else:
                label_state = "unknown"
                label_basis = "none"
                label = None

            if label is None:
                continue

            records.append(
                SlideRecord(
                    slide_id=slide_id,
                    cohort=cohort,
                    case_id=case_id,
                    label=label,
                    label_state=label_state,
                    label_basis=label_basis,
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
            "label_state": rec.label_state,
            "label_basis": rec.label_basis,
            "features": features,
            "coords": coords,
        }
