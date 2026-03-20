# Appendix

## Appendix A: Aggregator Ablations

**Configs**: `configs/appendix/uni_instance_mean.yaml`, `configs/appendix/uni_mean_unweighted.yaml`
(with `configs/uni_mean_fair.yaml` as the weighted-BCE reference)

**Question**: Does weighted BCE help? Does instance-level classification (classify-then-pool) differ
from bag-level mean pooling?

**How to regenerate**:
```bash
bash scripts/run_appendix.sh   # trains all appendix models
python scripts/appendix_tables.py --out outputs/appendix_tables.csv
```

**Interpretation**: Simple mean-pool with weighted BCE is generally preferred. Unweighted BCE
degrades on imbalanced datasets. Instance-level mean is similar to bag-level but with noisier gradients.

## Appendix B: Loss Function Ablation

**Configs**: `configs/appendix/uni_attention_focal.yaml`
(with `configs/uni_attention_fair.yaml` as the weighted-BCE reference)

**Question**: Does focal loss improve performance on hard negatives vs weighted BCE for AttentionMIL?

**Interpretation**: Focal loss reduces variance in some seeds but does not systematically outperform
weighted BCE. Weighted BCE is recommended as the default.

## Appendix C: Sparse Evidence Selection

**Configs**: `configs/appendix/uni_topk_attention_k16.yaml`
(with test-time truncation computed inline in `appendix_tables.py`)

**Question**: Is a model trained with top-k attention (k=16, ~3% of patches) more or less robust than
test-time truncation of a full-bag AttentionMIL?

**How to regenerate**:
```bash
python scripts/appendix_tables.py --out outputs/appendix_tables.csv
# Appendix C uses topk_truncation.py logic internally
```

**Interpretation**: Train-time sparsity (top-k) is conditionally useful when attention is poorly
calibrated, but is not robustly superior. Full-bag AttentionMIL with careful training is generally
sufficient.
