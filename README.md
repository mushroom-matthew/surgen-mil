# surgen-mil

Modular multiple-instance learning (MIL) framework for slide-level MSI/MMR prediction from
precomputed UNI patch embeddings on the SurGen colorectal cancer dataset.

This repository accompanies a computational pathology evaluation comparing three aggregation
strategies — mean pooling, attention MIL, and transformer MIL — under comparable experimental
conditions. The core finding is that frozen UNI embeddings already contain strong discriminative
signal, and simple pooled baselines are competitive with more expressive architectures at this
data scale (see Main Findings table below).

---

## Main Findings

| Model | AUROC (mean ± std) | AUPRC (mean ± std) |
|-------|--------------------|--------------------|
| MeanPool (weighted BCE) | **0.860 ± 0.005** | 0.447 ± 0.019 |
| AttentionMIL (weighted BCE) | 0.869 ± 0.020 | 0.381 ± 0.052 |
| TransformerMIL (weighted BCE) | 0.806 ± 0.057 | 0.391 ± 0.116 |
| *Paper baseline (Myles et al.)* | *0.827* | *—* |

*3 seeds × fixed split (split_seed=0), temperature scaling. See `docs/results_summary.md`.*
*Paper baseline: Myles et al. (2025), GigaScience — doi:10.1093/gigascience/giaf086.*

![Confusion matrices](docs/figures/fair_comparison_confusion_matrices.png)

- **Frozen UNI embeddings are strongly discriminative** for MSI/MMR status without any fine-tuning:
  MeanPool and AttentionMIL exceed the paper reference AUROC of 0.827 on this split, while the
  fair TransformerMIL reproduction does not.
- **Mean pooling is the most stable baseline**: lowest cross-seed AUROC variance (±0.005),
  consistent performance across all three seeds.
- **AttentionMIL is competitive but seed-dependent**: matched or exceeded MeanPool in some runs,
  but with 4× higher AUROC variance (±0.020 vs ±0.005) — an observation from this experiment,
  not a general claim. See `docs/results_summary.md` for per-seed breakdown.
- **TransformerMIL (6.8M params)** produced the lowest mean AUROC (0.806), highest variance
  (±0.057), and highest compute cost across this three-seed, fixed-split comparison.
- **Sparse evidence selection (top-k attention)** improves AUPRC in this experiment (0.455 vs
  0.381) but reduces AUROC (0.853 vs 0.869); not robustly superior overall. See Appendix C
  in `docs/appendix.md`.
- **Multisplit evaluation is the stronger summary**: across 3 data splits × 3 seeds, the best
  model is `HybridAttentionMIL` at 0.903 ± 0.033 AUROC and 0.591 ± 0.054 AUPRC.

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
    sampler_diagnostics.py     # quantify sampler coverage/diversity on real slides
    run_fair_comparison.sh     # train all 3 models x 3 seeds
    run_main_multisplit_updates.sh  # mainline updates across split seeds 0/1/2
    run_phase1_sampler_ablation.sh  # train mean/attention x sampler ablation
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
    attention_visualization.md
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

# 2. Run unit tests
make test

# 3. Train a fair-comparison baseline
make train-mean

# 4. Evaluate the latest checkpoint for a config
make evaluate CONFIG=configs/uni_mean_fair.yaml
```

Training note:
- When `data.max_patches` is set, train-time patch bags are sampled on fetch, so the same slide can be seen with different patch subsets across epochs.
- Validation and test remain full-bag by default.

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
make fair-comparison
```

Summarise results:
```bash
make compare
```

## Multi-Split Mainline Updates

The expanded mainline sweep includes the original fair-comparison trio plus:

- `configs/uni_gated_attention.yaml`
- `configs/uni_mean_var.yaml`
- `configs/uni_hybrid_attention_mean2.yaml`
- `configs/uni_attention_spatial_fair.yaml`
- `configs/uni_hybrid_attention_spatial_mean2.yaml`
- `configs/uni_transformer_spatial_fair.yaml` via the dedicated transformer launcher below

Run the full suite across split seeds `{0,1,2}` and training seeds `{42,123,456}` with:

```bash
make multisplit-updates
```

or:

```bash
MAX_PARALLEL=2 bash scripts/run_main_multisplit_updates.sh
```

Outputs are written to `outputs/multisplit/<config_name>/split_<seed>/`.
If a config already has canonical runs in its original output directory for the requested split
(for example the existing split-0 fair-comparison runs), the launcher skips retraining and
creates a symlink into the `outputs/multisplit/` tree instead.

Generate the multisplit summary tables and plots with:

```bash
make multisplit-analyse
```

Generate multisplit attention visualisations with:

```bash
make multisplit-attn
```

Run the spatial TransformerMIL multisplit sweep with:

```bash
bash scripts/run_transformer_spatial_multisplit.sh
```

or with reduced parallelism:

```bash
MAX_PARALLEL=2 bash scripts/run_transformer_spatial_multisplit.sh
```

Representative multisplit analysis plots:

![Multisplit lines](outputs/multisplit/analysis/multisplit_lines.png)

![Multisplit cohort lines](outputs/multisplit/analysis/multisplit_cohort_lines.png)

---

## Reproducing Appendix Analyses

```bash
# Train appendix models
make appendix

# Train Phase 1 sampler ablation
bash scripts/run_phase1_sampler_ablation.sh

# Validate sampler behaviour before training:
python scripts/sampler_diagnostics.py \
  --configs configs/appendix/phase1_mean_random.yaml \
            configs/appendix/phase1_mean_spatial.yaml \
            configs/appendix/phase1_mean_feature_diverse.yaml \
  --split train --repeats 3 --out outputs/sampler_diagnostics

# Generate appendix tables (A, B, C)
make appendix-tables
```

See `docs/appendix.md` for interpretation of each section.

---

## Attention Visualization

Attention-MIL models produce per-patch attention weights that can be projected back onto slide
coordinates to identify which tissue regions received high weight from the model. These weights
are learned end-to-end from slide-level BCE supervision — they are not validated pathology
annotations and should not be interpreted as direct indicators of biological ground truth. Attention
patterns vary across seeds and model checkpoints, reflecting the stochasticity of training at this
sample size; any individual attention map is one representative draw from an uncertain distribution
over which patches the model found useful during that training run.

The script auto-selects TP/FP/FN/TN examples ranked by classification count and mean predicted
probability across model×seed combinations:

```bash
# Auto-select representative TP/FP/FN/TN slides (3 per category)
make attn-auto

# Seed-variance grid
make attn-seed-grid

# Single-slide view
make attn-slide SLIDE_ID=SR1482_40X_HE_T1_0

# Attention statistics from the latest AttentionMIL checkpoint
make attn-stats
```

Slides are selected by ranking: for each category (TP/FP/FN/TN), slides are ranked by how
frequently they fall into that category across model×seed combinations and by their mean predicted
probability. The examples shown are the highest-ranked slides under that criterion — not slides
that are unanimously classified the same way by all models.

False positive (true MSS, all models predict MSI — systematic failure):

![FP attention example](docs/figures/attn_fp_SR1482_T061.png)

True positive (true MSI, models consistently recover the signal):

![TP attention example](docs/figures/attn_tp_SR1482_T297.png)

True negative (true MSS, all models correctly suppress):

![TN attention example](docs/figures/attn_tn_SR386_T129.png)

False negative (true MSI, model evidence remains weak or diffuse):

![FN attention example](docs/figures/attn_fn_SR386_T436.png)

See `docs/attention_visualization.md` for full usage, figure layout, colormap details,
and interpretation guidance.

## Make Targets

The `Makefile` is the fastest way to discover the current command surface:

```bash
make help
```

Common targets:

- `make smoke`
- `make test`
- `make train-mean`
- `make train-attention`
- `make train-transformer`
- `make fair-comparison`
- `make multisplit-updates`
- `make multisplit-analyse`
- `make multisplit-attn`
- `make appendix`
- `make appendix-tables`
- `make compare`
- `make evaluate CONFIG=...`
- `make attn-auto`
- `make attn-seed-grid`
- `make attn-slide SLIDE_ID=...`
- `make attn-stats`
- `make errors`
- `make error-report`

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
