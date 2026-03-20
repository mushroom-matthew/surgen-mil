# Packaging and Deployment

## What "Deploy" Means Here

This codebase does inference from **precomputed UNI patch embeddings**. It does not include:
- Raw WSI ingestion or tile extraction
- UNI feature extraction (requires the UNI model weights and a CUDA GPU)

To deploy, you need Zarr embedding files already computed by an upstream preprocessing pipeline.

## Installation

### Option 1: venv (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-core.txt
```

### Option 2: conda

```bash
conda create -n surgen python=3.10
conda activate surgen
pip install -e .
pip install -r requirements-core.txt
```

For GPU support, install the CUDA-compatible PyTorch first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Inference-Only Usage

To apply a trained checkpoint to new data without retraining:

```bash
python scripts/export_predictions.py \
  --config configs/uni_mean_fair.yaml \
  --checkpoint outputs/uni_mean_fair/runs/001/model.pt \
  --split all \
  --out my_predictions.csv
```

The script will:
1. Load the config and build data loaders from `data.root`.
2. Load the model checkpoint.
3. Apply temperature scaling (loaded from `metrics.json` next to the checkpoint).
4. Export per-slide probabilities to CSV.

## Threshold Configuration

Two thresholds are available:

| Threshold | Value | Recommendation |
|-----------|-------|----------------|
| Default | 0.5 | Use when calibration is unknown |
| Youden J | Fit on val | Use for clinical decision support |

> REVIEW: "Use for clinical decision support" is too strong for this repository. There is no clinical validation, external validation cohort, or deployment study here.

The Youden J threshold is computed in `scripts/appendix_tables.py`:
```python
from scripts.appendix_tables import _youden_threshold
threshold = _youden_threshold(preds_df)
```

> REVIEW: The previous example signature was incorrect. `_youden_threshold` takes a predictions DataFrame, not separate label/prob arrays.

## GPU Requirements

Training and inference were developed and tested on an NVIDIA Tesla T4.
A CUDA GPU is strongly recommended for training; the codebase will fall back to CPU
automatically but training will be very slow without one.
Inference on CPU is practical for small slide counts.

> REVIEW: Keep this only if you can personally vouch for the T4 testing statement. "strongly recommended" and "practical" are reasonable operational judgments, but they are still informal unless backed by benchmark numbers.

The codebase automatically falls back to CPU if CUDA is unavailable:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## What This Repo Does NOT Include

- Raw WSI files (requires institutional data access)
- UNI model weights (available separately from the UNI authors)
- Preprocessing scripts for tile extraction and embedding computation
- TCGA or external validation cohort data
