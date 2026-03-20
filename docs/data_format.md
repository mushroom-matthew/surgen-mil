# Data Format

## Directory Structure

```
<data_root>/
  embeddings/
    SR1482_40X_HE_T1_0.zarr      # slide embeddings
    SR1482_40X_HE_T1_1.zarr      # second slide from same case
    SR1482_40X_HE_T2_0.zarr
    ...
  SR1482_labels.csv
  SR386_labels.csv
```

## Zarr Embedding Format

Each `.zarr` store contains two arrays:

| Array | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `features` | `[N, 1024]` | float32 | UNI patch embeddings (one row per patch) |
| `coords` | `[N, 2]` | float32 | Patch coordinates in the slide (x, y) |

`N` varies by slide (typically 500–5000 patches at 40× magnification).

Read example:
```python
import zarr
z = zarr.open("SR1482_40X_HE_T1_0.zarr", mode="r")
features = z["features"][:]  # np.ndarray [N, 1024]
coords = z["coords"][:]      # np.ndarray [N, 2]
```

## Label Files

### SR1482_labels.csv

| Column | Type | Values |
|--------|------|--------|
| `case_id` | int | Patient identifier |
| `MSI` | str | `"MSI HIGH"` or `"NO MSI"` |
| `MMR` | str | `"MMR loss"` or `"No loss"` |

### SR386_labels.csv

| Column | Type | Values |
|--------|------|--------|
| `case_id` | int | Patient identifier |
| `mmr_loss_binary` | int | `1` = MMR loss, `0` = no loss |

## Slide ID Naming Convention

```
{COHORT}_40X_HE_T{CASE_ID}_{SLIDE_NUM}
```

Examples:
- `SR1482_40X_HE_T1_0` — cohort SR1482, case 1, first slide
- `SR1482_40X_HE_T1_1` — cohort SR1482, case 1, second slide
- `SR386_40X_HE_T5_0`  — cohort SR386, case 5, first slide

Slides sharing a `CASE_ID` are always assigned to the same train/val/test split (case-grouped split).

## Split Handling

The `case_grouped_stratified_split` function (in `src/data/splits.py`) partitions cases (not slides)
into train/val/test, preserving class balance. The `split_seed` config parameter controls
randomization; `split_seed: 0` is used for all fair-comparison configs.

## Minimum Inference Manifest

To evaluate a checkpoint on new data, you need:
1. A directory of `.zarr` embedding files following the naming convention above.
2. A label CSV with `case_id` and either `MSI`/`MMR` (SR1482) or `mmr_loss_binary` (SR386).
3. A config YAML with `data.root` pointing to your directory.

See `examples/sample_manifest.csv` for a minimal example.
