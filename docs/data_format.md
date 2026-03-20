# Data Format

Dataset: [github.com/CraigMyles/SurGen-Dataset](https://github.com/CraigMyles/SurGen-Dataset)
Dataset DOI: https://doi.org/10.6019/S-BIAD1285
Paper: https://doi.org/10.1093/gigascience/giaf086

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

| Column | Type | Observed values |
|--------|------|-----------------|
| `case_id` | int | Patient identifier |
| `MSI` | str | `"MSI HIGH"`, `"NO MSI"`, `"MSI LOW"`, `"Not performed"`, `"Insufficient"`, `"Failed"` |
| `MMR` | str | `"MMR loss"`, `"No loss"`, `"Not performed"` |

**Binary label derivation** (`src/data/feature_provider.py: _resolve_sr1482_state`):

Each column is first mapped to an intermediate state:

| `MSI` value | State |
|-------------|-------|
| `"MSI HIGH"` | positive |
| `"NO MSI"` | negative |
| `"MSI LOW"`, `"Not performed"`, `"Insufficient"`, `"Failed"` | unknown |

| `MMR` value | State |
|-------------|-------|
| any string containing `"loss"` | positive |
| `"No loss"` | negative |
| `"Not performed"` | unknown |

The two states are then reconciled into a final binary label:

| MSI state | MMR state | Final label | Note |
|-----------|-----------|-------------|------|
| positive | positive | **1** | basis: MSI+MMR |
| positive | unknown | **1** | basis: MSI |
| unknown | positive | **1** | basis: MMR |
| negative | negative | **0** | basis: MSI+MMR |
| negative | unknown | **0** | basis: MSI |
| unknown | negative | **0** | basis: MMR |
| positive | negative | excluded | discordant — case dropped |
| unknown | unknown | excluded | insufficient evidence — case dropped |

Discordant and unknown cases are silently excluded from the dataset;
their slides never appear in any split.

### SR386_labels.csv

| Column | Type | Values |
|--------|------|--------|
| `case_id` | int | Patient identifier |
| `mmr_loss_binary` | int | `1` = positive (MMR loss), `0` = negative, NaN = excluded |

**Binary label derivation**: `mmr_loss_binary` is used directly as the label.
Cases with a NaN value are excluded.

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
