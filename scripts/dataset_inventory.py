"""
Comprehensive inventory of the SurGen dataset root.

Reports:
  - root-level file inventory
  - raw image inventory (.czi)
  - embedding inventory (.zarr)
  - metadata inventory (.csv label tables)
  - image <-> embedding joins
  - embedding <-> metadata joins
  - provider-compatible labeled slide inventory

Usage:
    python3 scripts/dataset_inventory.py --root /mnt/data-surgen
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys

import pandas as pd
import zarr

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.feature_provider import UniFeatureProvider


SLIDE_RE = re.compile(r"^(SR\d+)_40X_HE_T(\d+)_(\d+)$")


def section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")


def parse_slide_stem(stem: str) -> tuple[str, int, int] | None:
    m = SLIDE_RE.match(stem)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def inventory_root(root: Path) -> None:
    section("Root Inventory")
    entries = sorted(root.iterdir())
    suffix_counts = Counter(p.suffix if p.is_file() else "<dir>" for p in entries)
    print(f"root: {root}")
    print(f"entries: {len(entries)}")
    print(f"suffix_counts: {dict(suffix_counts)}")


def collect_czi_inventory(root: Path) -> dict:
    rows = []
    invalid = []
    for p in sorted(root.glob("*.czi")):
        parsed = parse_slide_stem(p.stem)
        if parsed is None:
            invalid.append(p.name)
            continue
        cohort, case_id, slide_num = parsed
        rows.append({
            "slide_id": p.stem,
            "file_name": p.name,
            "cohort": cohort,
            "case_id": case_id,
            "slide_num": slide_num,
            "size_bytes": p.stat().st_size,
        })
    df = pd.DataFrame(rows)
    return {"df": df, "invalid": invalid}


def report_czi_inventory(czi: dict) -> None:
    section("Raw Images (.czi)")
    df = czi["df"]
    print(f"valid_czi_files: {len(df)}")
    print(f"invalid_czi_names: {len(czi['invalid'])}")
    if czi["invalid"]:
        print("invalid_czi_examples:")
        for name in czi["invalid"][:10]:
            print(f"  {name}")

    if df.empty:
        return

    counts = df.groupby("cohort").size().rename("n_slides")
    print("\nslides_by_cohort:")
    print(counts.to_string())

    case_counts = df.groupby("cohort")["case_id"].nunique().rename("n_cases")
    print("\ncases_by_cohort:")
    print(case_counts.to_string())

    for cohort, g in df.groupby("cohort"):
        per_case = g.groupby("case_id").size()
        dist = dict(sorted(per_case.value_counts().to_dict().items()))
        min_row = g.loc[g["size_bytes"].idxmin()]
        max_row = g.loc[g["size_bytes"].idxmax()]
        print(f"\n{cohort}")
        print(f"  slides: {len(g)}")
        print(f"  cases: {g['case_id'].nunique()}")
        print(f"  slides_per_case_distribution: {dist}")
        print(f"  smallest_file: {min_row['file_name']} ({int(min_row['size_bytes'])} bytes)")
        print(f"  largest_file:  {max_row['file_name']} ({int(max_row['size_bytes'])} bytes)")


def collect_zarr_inventory(root: Path) -> dict:
    emb_root = root / "embeddings"
    rows = []
    invalid_names = []
    empty_layout = []
    for p in sorted(emb_root.glob("*.zarr")):
        parsed = parse_slide_stem(p.stem)
        child_names = sorted(child.name for child in p.iterdir()) if p.is_dir() else []
        has_coords_dir = (p / "coords").is_dir()
        has_features_dir = (p / "features").is_dir()

        if parsed is None:
            invalid_names.append(p.name)
            if not has_coords_dir or not has_features_dir:
                empty_layout.append(p.name)
            rows.append({
                "slide_id": p.stem,
                "dir_name": p.name,
                "cohort": None,
                "case_id": None,
                "slide_num": None,
                "has_coords_dir": has_coords_dir,
                "has_features_dir": has_features_dir,
                "child_names": tuple(child_names),
            })
            continue

        cohort, case_id, slide_num = parsed
        rows.append({
            "slide_id": p.stem,
            "dir_name": p.name,
            "cohort": cohort,
            "case_id": case_id,
            "slide_num": slide_num,
            "has_coords_dir": has_coords_dir,
            "has_features_dir": has_features_dir,
            "child_names": tuple(child_names),
        })
    df = pd.DataFrame(rows)
    return {"df": df, "invalid_names": invalid_names, "empty_layout": empty_layout}


def report_zarr_inventory(zarr_inv: dict) -> None:
    section("Embeddings (.zarr)")
    df = zarr_inv["df"]
    parsed = df[df["cohort"].notna()].copy()
    print(f"zarr_directories_total: {len(df)}")
    print(f"zarr_directories_parsed: {len(parsed)}")
    print(f"zarr_directories_invalid_name: {len(zarr_inv['invalid_names'])}")
    if zarr_inv["invalid_names"]:
        for name in zarr_inv["invalid_names"]:
            print(f"  invalid_name: {name}")

    missing_dirs = df[(~df["has_coords_dir"]) | (~df["has_features_dir"])]
    print(f"zarr_missing_coords_or_features_dir: {len(missing_dirs)}")
    if not missing_dirs.empty:
        for name in missing_dirs["dir_name"].head(10):
            print(f"  incomplete_layout: {name}")

    if parsed.empty:
        return

    print("\nparsed_zarr_by_cohort:")
    print(parsed.groupby("cohort").size().rename("n_slides").to_string())

    for cohort, g in parsed.groupby("cohort"):
        per_case = g.groupby("case_id").size()
        dist = dict(sorted(per_case.value_counts().to_dict().items()))
        print(f"\n{cohort}")
        print(f"  parsed_zarr_slides: {len(g)}")
        print(f"  parsed_zarr_cases: {g['case_id'].nunique()}")
        print(f"  slides_per_case_distribution: {dist}")


def validate_zarr_arrays(root: Path, zarr_inv: dict) -> None:
    section("Embedding Array Validation")
    emb_root = root / "embeddings"
    parsed = zarr_inv["df"][zarr_inv["df"]["cohort"].notna()].copy()

    status = Counter()
    examples: dict[str, str] = {}
    total_patches = 0
    min_patches: tuple[int, str] | None = None
    max_patches: tuple[int, str] | None = None

    for _, row in parsed.iterrows():
        p = emb_root / row["dir_name"]
        z = zarr.open(str(p), mode="r")
        keys = tuple(sorted(z.array_keys()))
        if keys != ("coords", "features"):
            key = f"bad_keys:{keys}"
            status[key] += 1
            examples.setdefault(key, row["dir_name"])
            continue

        coords = z["coords"]
        features = z["features"]

        if coords.shape[0] != features.shape[0]:
            status["row_mismatch"] += 1
            examples.setdefault("row_mismatch", row["dir_name"])
            continue
        if coords.shape[1:] != (2,):
            status["bad_coords_shape"] += 1
            examples.setdefault("bad_coords_shape", row["dir_name"])
            continue
        if features.shape[1:] != (1024,):
            status["bad_features_shape"] += 1
            examples.setdefault("bad_features_shape", row["dir_name"])
            continue
        if str(coords.dtype) != "int64":
            status["bad_coords_dtype"] += 1
            examples.setdefault("bad_coords_dtype", row["dir_name"])
            continue
        if str(features.dtype) != "float32":
            status["bad_features_dtype"] += 1
            examples.setdefault("bad_features_dtype", row["dir_name"])
            continue

        n_patches = int(features.shape[0])
        total_patches += n_patches
        status["valid"] += 1
        if min_patches is None or n_patches < min_patches[0]:
            min_patches = (n_patches, row["dir_name"])
        if max_patches is None or n_patches > max_patches[0]:
            max_patches = (n_patches, row["dir_name"])

    print(f"parsed_zarr_checked: {len(parsed)}")
    print(f"validation_status_counts: {dict(status)}")
    if examples:
        print("validation_examples:")
        for key, name in examples.items():
            print(f"  {key}: {name}")
    print(f"total_patch_rows_across_valid_zarr: {total_patches}")
    if min_patches is not None:
        print(f"min_patch_count: {min_patches[0]} ({min_patches[1]})")
    if max_patches is not None:
        print(f"max_patch_count: {max_patches[0]} ({max_patches[1]})")


def load_metadata_tables(root: Path) -> dict[str, pd.DataFrame]:
    return {
        "SR1482": pd.read_csv(root / "SR1482_labels.csv"),
        "SR386": pd.read_csv(root / "SR386_labels.csv"),
    }


def report_metadata(metadata: dict[str, pd.DataFrame]) -> None:
    section("Metadata Tables")
    for cohort, df in metadata.items():
        print(f"{cohort}")
        print(f"  rows: {len(df)}")
        print(f"  columns: {len(df.columns)}")
        print(f"  duplicate_case_ids: {int(df['case_id'].duplicated().sum())}")
        print(f"  columns_list: {list(df.columns)}")
        if cohort == "SR1482":
            print("  MSI_value_counts:")
            print(df["MSI"].astype(str).str.strip().value_counts(dropna=False).to_string())
        elif cohort == "SR386":
            print("  mmr_loss_binary_value_counts:")
            print(df["mmr_loss_binary"].value_counts(dropna=False).to_string())
        print()


def report_image_embedding_joins(czi: dict, zarr_inv: dict) -> None:
    section("Image <-> Embedding Joins")
    czi_ids = set(czi["df"]["slide_id"])
    parsed_zarr = zarr_inv["df"][zarr_inv["df"]["cohort"].notna()].copy()
    zarr_ids = set(parsed_zarr["slide_id"])

    czi_without_zarr = sorted(czi_ids - zarr_ids)
    zarr_without_czi = sorted(zarr_ids - czi_ids)

    print(f"raw_images_with_matching_parsed_embedding: {len(czi_ids & zarr_ids)}")
    print(f"raw_images_without_parsed_embedding: {len(czi_without_zarr)}")
    print(f"parsed_embeddings_without_raw_image: {len(zarr_without_czi)}")
    if czi_without_zarr:
        print("raw_images_without_embedding_examples:")
        for slide_id in czi_without_zarr[:10]:
            print(f"  {slide_id}")
    if zarr_without_czi:
        print("parsed_embeddings_without_raw_image_examples:")
        for slide_id in zarr_without_czi[:10]:
            print(f"  {slide_id}")

    invalid_only = sorted(zarr_inv["invalid_names"])
    print(f"embedding_dirs_not_joinable_by_slide_id_naming: {len(invalid_only)}")
    for name in invalid_only[:10]:
        print(f"  non_joinable_embedding_dir: {name}")


def build_provider_label_maps(metadata: dict[str, pd.DataFrame]) -> tuple[dict[int, int], dict[int, int]]:
    sr1482_map = UniFeatureProvider._build_sr1482_label_map(metadata["SR1482"])

    sr386_map: dict[int, int] = {}
    for _, row in metadata["SR386"].iterrows():
        case_id = int(row["case_id"])
        v = row["mmr_loss_binary"]
        if pd.isna(v):
            continue
        sr386_map[case_id] = int(v)

    return sr1482_map, sr386_map


def report_embedding_metadata_joins(zarr_inv: dict, metadata: dict[str, pd.DataFrame]) -> None:
    section("Embedding <-> Metadata Joins")
    parsed = zarr_inv["df"][zarr_inv["df"]["cohort"].notna()].copy()
    sr1482_labels, sr386_labels = build_provider_label_maps(metadata)
    sr1482_states = UniFeatureProvider._build_sr1482_label_states(metadata["SR1482"]).set_index("case_id")
    raw_meta_cases = {
        "SR1482": set(metadata["SR1482"]["case_id"].astype(int)),
        "SR386": set(metadata["SR386"]["case_id"].astype(int)),
    }

    provider_status_rows = []
    for _, row in parsed.iterrows():
        cohort = row["cohort"]
        case_id = int(row["case_id"])
        raw_present = case_id in raw_meta_cases[cohort]
        if cohort == "SR1482":
            provider_label = sr1482_labels.get(case_id)
            label_state = str(sr1482_states.loc[case_id, "state"])
            label_basis = str(sr1482_states.loc[case_id, "basis"])
        else:
            provider_label = sr386_labels.get(case_id)
            label_state = "positive" if provider_label == 1 else "negative"
            label_basis = "mmr_loss_binary"
        provider_status_rows.append({
            "cohort": cohort,
            "case_id": case_id,
            "slide_id": row["slide_id"],
            "raw_metadata_row_exists": raw_present,
            "provider_label_exists": provider_label is not None,
            "provider_label": provider_label,
            "label_state": label_state,
            "label_basis": label_basis,
        })
    provider_df = pd.DataFrame(provider_status_rows)

    print(f"parsed_embedding_slides_total: {len(provider_df)}")
    print(f"slides_with_raw_metadata_row: {int(provider_df['raw_metadata_row_exists'].sum())}")
    print(f"slides_without_raw_metadata_row: {int((~provider_df['raw_metadata_row_exists']).sum())}")
    print(f"slides_with_provider_label: {int(provider_df['provider_label_exists'].sum())}")
    print(f"slides_without_provider_label: {int((~provider_df['provider_label_exists']).sum())}")

    for cohort, g in provider_df.groupby("cohort"):
        kept = g[g["provider_label_exists"]]
        dropped = g[~g["provider_label_exists"]]
        print(f"\n{cohort}")
        print(f"  parsed_embedding_slides: {len(g)}")
        print(f"  parsed_embedding_cases: {g['case_id'].nunique()}")
        print(f"  slides_with_raw_metadata_row: {int(g['raw_metadata_row_exists'].sum())}")
        print(f"  slides_with_provider_label: {len(kept)}")
        print(f"  slides_without_provider_label: {len(dropped)}")
        print(f"  cases_with_provider_label: {kept['case_id'].nunique()}")
        print(f"  cases_without_provider_label: {dropped['case_id'].nunique()}")
        print(f"  label_state_counts: {g['label_state'].value_counts().to_dict()}")
        print(f"  label_basis_counts: {g['label_basis'].value_counts().to_dict()}")
        if not dropped.empty:
            dropped_cases = sorted(dropped["case_id"].unique().tolist())
            print(f"  first_dropped_case_ids: {dropped_cases[:25]}")
        if not kept.empty:
            label_counts = kept["provider_label"].value_counts().sort_index().to_dict()
            print(f"  provider_label_counts: {label_counts}")


def report_code_entry_points() -> None:
    section("Code Entry Points In This Repo")
    print("Primary code paths that currently explore or consume the dataset:")
    print("  src/data/feature_provider.py: UniFeatureProvider.__init__")
    print("  src/data/feature_provider.py: UniFeatureProvider._build_records")
    print("  src/data/feature_provider.py: UniFeatureProvider.load_slide")
    print("  src/data/dataset.py: SurgenBagDataset.__getitem__")
    print("  src/data/splits.py: case_grouped_stratified_split")
    print("  scripts/inspect_surgen.py")
    print("  scripts/inspect_cases.py")
    print("  scripts/metadata_audit.py")
    print("  scripts/error_stratification.py")
    print("  scripts/dataset_inventory.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/data-surgen")
    parser.add_argument(
        "--skip-zarr-validation",
        action="store_true",
        help="Skip opening each parsed zarr store to validate shapes and dtypes.",
    )
    args = parser.parse_args()

    root = Path(args.root)

    inventory_root(root)
    czi = collect_czi_inventory(root)
    report_czi_inventory(czi)

    zarr_inv = collect_zarr_inventory(root)
    report_zarr_inventory(zarr_inv)
    if not args.skip_zarr_validation:
        validate_zarr_arrays(root, zarr_inv)

    metadata = load_metadata_tables(root)
    report_metadata(metadata)
    report_image_embedding_joins(czi, zarr_inv)
    report_embedding_metadata_joins(zarr_inv, metadata)
    report_code_entry_points()


if __name__ == "__main__":
    main()
