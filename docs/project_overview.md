# Project Overview

## Task

Predict slide-level microsatellite instability (MSI) and mismatch repair (MMR) status from whole-slide
histology images of colorectal cancer. This is a weakly-supervised binary classification problem: each
training label is assigned at the patient level, with no patch-level annotation.

## Dataset

The SurGen dataset contains two cohorts:

- **SR1482**: Primary colorectal tumours with MSI/MMR labels derived from IHC/PCR testing.
- **SR386**: A separate cohort with `mmr_loss_binary` labels.

Slides are available as precomputed patch embeddings from the **UNI** foundation model
(Chen et al., 2024), stored as Zarr archives. No raw WSI preprocessing is required to run this
codebase.

## Modeling Approach

We adopt a multiple-instance learning (MIL) framework:

- A slide is treated as a *bag* of patch embeddings.
- A bag-level label (MSI / no-MSI) supervises the model.
- Patch-level supervision is not used.

Three aggregation strategies are evaluated under fair, controlled conditions:

| Model | Key idea |
|-------|---------|
| **MeanPool** | Average all patch embeddings, classify with MLP |
| **AttentionMIL** | Learn a softmax attention distribution over patches, then weighted sum |
| **TransformerMIL** | Self-attention across patches before mean pooling (paper-style) |

## Core Result

Frozen UNI embeddings contain strong discriminative signal for MSI/MMR prediction.
Mean pooling is the most stable baseline. AttentionMIL is competitive but seed-sensitive.
TransformerMIL is not justified in this data regime (sample size, label quality).

See `docs/results_summary.md` for the full performance table and interpretation.

## Design Choices

- **Precomputed embeddings**: UNI features are fixed; only the aggregator and classifier are trained.
- **Case-grouped splits**: Slides from the same patient are always in the same split to prevent leakage.
- **Stratified**: Positive/negative class balance is maintained across train/val/test.
- **Fixed split seed**: `split_seed: 0` across all fair-comparison configs ensures identical data splits.
- **Temperature scaling**: A post-hoc calibration scalar is fit on the validation set and applied at test time.
- **EMA smoothing**: Validation metric is smoothed with exponential moving average before checkpoint selection.
