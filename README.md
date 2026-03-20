# surgen-mil

Modular multiple-instance learning (MIL) framework for slide-level MSI/MMR prediction from
precomputed UNI patch embeddings on the SurGen colorectal cancer dataset.

This repository accompanies a computational pathology evaluation comparing three aggregation
strategies — mean pooling, attention MIL, and transformer MIL — under identical experimental
conditions. The core finding is that frozen UNI embeddings already contain strong discriminative
signal, and simple pooled baselines are competitive with more expressive architectures at this
data scale.

---

## Main Findings

- **Frozen UNI embeddings are strongly discriminative** for MSI/MMR status without any fine-tuning.
- **Mean pooling is the most stable baseline**: consistent performance across seeds, low variance.
- **AttentionMIL is competitive but seed-dependent**: higher peak performance than mean pooling
  in some runs, but higher variance — suggesting training instability at this sample size.
- **TransformerMIL (6.8M params) is not justified** in this data regime: no improvement over
  simpler aggregators, higher compute cost.
- **Sparse evidence selection (top-k attention)** is conditionally useful but not robustly superior
  to full-bag attention.

---

## Repository Layout

```
surgen-mil/
  train.py                     # main training script
  configs/
    uni_mean_fair.yaml         # MeanPool (main comparison)
    uni_attention_fair.yaml    # AttentionMIL (main comparison)
    paper_reproduction_fair.yaml  # TransformerMIL (main comparison)
    appendix/                  # ablation configs
    ...                        # additional exploratory configs
  src/
    data/                      # feature provider, sampler, splits, dataset
    models/                    # aggregators: mean, attention, transformer, gated, lse, topk, region
    losses.py                  # BCE, focal loss factories
  scripts/
    smoke_test.py              # synthetic end-to-end test
    evaluate.py                # standalone evaluation from checkpoint
    compare_models.py          # multi-seed summary table + plots
    make_figures.py            # generate ROC/PR figures
    export_predictions.py      # export per-slide predictions from checkpoint
    inspect_attention.py       # attention weight diagnostics
    run_fair_comparison.sh     # train all 3 models x 3 seeds
    run_appendix.sh            # train appendix models x 3 seeds
    analyse.py                 # cohort-level analysis
    seed_comparison.py         # detailed multi-seed aggregation
    appendix_tables.py         # generate appendix A/B/C tables
    failures/                  # exploratory error analysis
  docs/
    project_overview.md
    data_format.md
    reproducibility.md
    results_summary.md
    appendix.md
    future_work.md
    packaging_and_deployment.md
    architecture_configs.md
  tests/                       # unit tests (no real data required)
  examples/                    # sample manifest, config overrides, expected outputs
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-core.txt
```

For GPU (CUDA 11.8):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Quickstart

```bash
# 1. Verify installation (no real data needed)
make smoke

# 2. Train a model
python train.py --config configs/uni_mean_fair.yaml --seed 42

# 3. Evaluate a checkpoint
python scripts/evaluate.py \
  --config configs/uni_mean_fair.yaml \
  --checkpoint outputs/uni_mean_fair/runs/001/model.pt
```

---

## Expected Data Format

```
<data_root>/
  embeddings/
    SR1482_40X_HE_T1_0.zarr   # features: [N, 1024], coords: [N, 2]
    SR1482_40X_HE_T2_0.zarr
    ...
  SR1482_labels.csv            # columns: case_id, MSI, MMR
  SR386_labels.csv             # columns: case_id, mmr_loss_binary
```

Set `data.root` in your config to point to this directory. See `docs/data_format.md` for full details.

---

## Reproducing the Main Comparison

Train all three models with three seeds each (parallel streams, fixed split):

```bash
bash scripts/run_fair_comparison.sh
```

Summarise results:
```bash
python scripts/compare_models.py \
  --configs configs/uni_mean_fair.yaml \
            configs/uni_attention_fair.yaml \
            configs/paper_reproduction_fair.yaml \
  --out outputs/comparison
```

---

## Reproducing Appendix Analyses

```bash
# Train appendix models
bash scripts/run_appendix.sh

# Generate appendix tables (A, B, C)
python scripts/appendix_tables.py --out outputs/appendix_tables.csv
```

See `docs/appendix.md` for interpretation of each section.

---

## Outputs

Each training run writes to `outputs/<model_name>/runs/NNN/`:

| File | Description |
|------|-------------|
| `config.yaml` | Config used for this run |
| `model.pt` | Best model weights (by val AUPRC) |
| `metrics.json` | Test AUROC, AUPRC, temperature, case-level metrics |
| `predictions.csv` | Per-slide probabilities for all splits |
| `history.json` | Per-epoch training metrics |
| `training_curve.png` | Loss and AUROC training curves |

See `examples/expected_outputs.md` for the full schema.

---

## Deployment / Inference

To export predictions from a trained checkpoint without retraining:

```bash
python scripts/export_predictions.py \
  --config configs/uni_mean_fair.yaml \
  --checkpoint outputs/uni_mean_fair/runs/001/model.pt \
  --split all \
  --out my_predictions.csv
```

See `docs/packaging_and_deployment.md` for GPU requirements, threshold configuration, and notes
on what this repo does NOT include (raw WSI preprocessing, UNI feature extraction).

---

## Limitations

- Small sample size: both cohorts have limited cases relative to expressive model complexity.
- Weak supervision: slide-level labels only; no patch-level annotation available.
- Single-site data: generalisability to other scanners/staining protocols is not validated.
- Label noise: IHC- and PCR-derived labels may have different error rates.

---

## Future Directions

- Mutation status prediction (KRAS, NRAS, BRAF) from the same embeddings
- Cohort-aware modelling (SR1482 vs SR386 distribution shift)
- Uncertainty-aware attention for reliability estimation
- Case-level aggregation across multiple slides per patient

See `docs/future_work.md` for a detailed roadmap.

---

## Citation / Contact

This repository is a take-home research artifact. If you use this codebase, please cite the SurGen
dataset and the UNI foundation model paper.

**SurGen dataset**
- GitHub: https://github.com/CraigMyles/SurGen-Dataset
- Dataset DOI: https://doi.org/10.6019/S-BIAD1285
- Paper: https://doi.org/10.1093/gigascience/giaf086

**UNI foundation model**
- Chen, R.J. et al. (2024). Towards a general-purpose foundation model for computational pathology. *Nature Medicine*.
