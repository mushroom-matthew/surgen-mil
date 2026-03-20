# Reproducibility

## Seeds

| Purpose | Values |
|---------|--------|
| Training seeds | 42, 123, 456 |
| Split seed (data partition) | 0 (fixed for all fair-comparison and appendix configs) |

Three training seeds produce three independent runs per model. Results are reported as mean ± std.

## Config Names

### Core fair-comparison configs

| Config file | Model | Appendix ref |
|-------------|-------|-------------|
| `configs/uni_mean_fair.yaml` | MeanPoolMIL | Main comparison |
| `configs/uni_attention_fair.yaml` | AttentionMIL | Main comparison |
| `configs/paper_reproduction_fair.yaml` | TransformerMIL | Main comparison |

### Appendix configs

| Config file | Model | Appendix section |
|-------------|-------|-----------------|
| `configs/appendix/uni_instance_mean.yaml` | InstanceMeanMIL | Appendix A |
| `configs/appendix/uni_mean_unweighted.yaml` | MeanPoolMIL (unweighted loss) | Appendix A |
| `configs/appendix/uni_attention_focal.yaml` | AttentionMIL (focal loss) | Appendix B |
| `configs/appendix/uni_topk_attention_k16.yaml` | TopKAttentionMIL (k=16) | Appendix C |

## Reproducing Runs

```bash
# Main three-way comparison (3 seeds each, parallel streams)
bash scripts/run_fair_comparison.sh

# Appendix ablations
bash scripts/run_appendix.sh

# Or train a single config manually
python train.py --config configs/uni_mean_fair.yaml --seed 42
```

## Checkpoint Selection Logic

1. After each epoch, validate on `val` split using `val_auprc`.
2. Apply EMA smoothing: `ema = alpha * ema + (1 - alpha) * raw` with `alpha = 0.7`.
3. Save best checkpoint whenever EMA metric improves.
4. After training, restore best checkpoint (unless `use_best_checkpoint: false`).

## Temperature Scaling

Temperature scaling is applied post hoc on the validation set:

```python
temperature = find_temperature(model, val_loader, device)
# Minimises NLL: T* = argmin_T BCE(sigmoid(logit / T), label)
prob = sigmoid(logit / T)
```

The fitted temperature is saved to `metrics.json` and is used by `evaluate.py` and
`export_predictions.py` automatically.

## Metric Definitions

- **AUROC**: Area under the ROC curve (slide-level).
- **AUPRC**: Area under the precision-recall curve (slide-level).
- **Case-level metrics**: Slides from the same case are aggregated using `max`, `mean`, or `noisy_or`
  before computing AUROC/AUPRC.

## Threshold Selection

Two threshold conventions are used:

1. **Default threshold: 0.5** on temperature-scaled probabilities. Used in most analysis scripts.
2. **Youden J threshold**: Optimised on the validation set to maximise `sensitivity + specificity - 1`.
   Computed in `scripts/appendix_tables.py` (`_youden_threshold` function).

For deployment, we recommend the Youden J threshold fit on validation data.
