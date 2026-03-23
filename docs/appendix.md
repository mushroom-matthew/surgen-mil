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

**Interpretation**: In this experiment, unweighted BCE produced marginally higher mean AUROC
(0.862 ± 0.003) and AUPRC (0.465 ± 0.004) than weighted BCE (0.860 ± 0.005 / 0.447 ± 0.019).
The differences are small and within the range of seed variation. Instance-level mean pooling
performs nearly identically to bag-level mean pooling (AUROC 0.859 vs 0.860), with slightly
higher AUPRC variance (±0.027 vs ±0.019), consistent with noisier gradients from classifying
before pooling. See Appendix A table in `docs/results_summary.md`.

## Appendix B: Loss Function Ablation

**Configs**: `configs/appendix/uni_attention_focal.yaml`
(with `configs/uni_attention_fair.yaml` as the weighted-BCE reference)

**Question**: Does focal loss improve performance on hard negatives vs weighted BCE for AttentionMIL?

**Interpretation**: Focal loss did not clearly outperform weighted BCE in this experiment.
AttentionMIL with focal loss achieved slightly lower mean AUROC (0.863 vs 0.869) but higher
mean AUPRC (0.420 vs 0.381) and lower AUROC variance (±0.014 vs ±0.020). The tradeoff is:
focal loss down-weights easy negatives and concentrates gradient on hard examples, which may
improve precision at moderate recall (reflected in higher AUPRC) but can reduce overall ranking
performance (AUROC). Weighted BCE applies a fixed class-frequency correction and is simpler to
tune. Neither dominates the other across metrics in this experiment.

## Appendix C: Sparse Evidence Selection

**Configs**: `configs/appendix/uni_topk_attention_k16.yaml`
(with test-time truncation computed inline in `appendix_tables.py`)

**Question**: Is a model trained with top-k attention (k=16, ≈3% of the 512-patch training bag) more or less robust than
test-time truncation of a full-bag AttentionMIL?

**How to regenerate**:
```bash
python scripts/appendix_tables.py --out outputs/appendix_tables.csv
# Appendix C uses topk_truncation.py logic internally
```

**Interpretation**: In this experiment, TopK-16 training improved AUPRC (0.455 ± 0.139 vs
0.381 ± 0.052) but reduced AUROC (0.853 ± 0.032 vs 0.869 ± 0.020) relative to full-bag
AttentionMIL. Train-time sparsity forces the model to rely on a very small subset of patches
(k=16 ≈ 3% of the 512-patch bag), which may sharpen the learned weighting distribution at the
cost of reduced ranking across the full bag. The AUPRC gain and AUROC loss are complementary
rather than contradictory. Neither configuration is unambiguously superior; the appropriate
choice depends on which metric matters more for the downstream use case.

![Appendix C ROC and PR curves](figures/appendix_c_roc_pr_curves.png)

**Reading the PR curve**: AUPRC is the area under the precision-recall curve — it measures how
well the model ranks true positives above true negatives across all possible thresholds.
TopK-16 (red) sits above AttentionMIL (blue) across most of the PR curve, meaning at
intermediate recall levels it maintains higher precision for the same sensitivity. However, the
Youden J threshold (diamond markers) selects the operating point that maximises
sensitivity + specificity − 1 — which for AttentionMIL lands at high recall (0.92) with moderate
precision (0.28), while TopK-16's Youden point is worse on both dimensions (rec=0.83, prec=0.19).

A model can have higher AUPRC and a worse Youden operating point simultaneously: AUPRC rewards
good ranking across the full curve, while Youden selects the single best threshold under that
metric. The two are complementary, not redundant.

![Appendix C confusion matrices](figures/appendix_c_confusion_matrices.png)

The confusion matrices make the operating-point shift concrete: TopK-16's threshold drops from
0.24 to 0.12, FPs increase from 28 to 42, and sensitivity drops from 11/12 to 10/12. Whether
the AUPRC improvement is worth the worse Youden operating point depends on the relative cost of
false positives versus false negatives.

## Appendix D: Train-Time Bag Sampling Protocol

This repository distinguishes between:

- **train-time bag construction**: when `max_patches` is set, a subset of the full slide bag is drawn before the model sees it
- **evaluation-time bag construction**: validation and test use full bags unless a config explicitly disables that

The important implementation detail is that train-time sampling is **dynamic**, not fixed once per run.

- Sampling happens inside dataset `__getitem__`
- therefore the selected patch subset can change each time a slide is fetched
- in practice, the model is exposed to different sampled views of the same slide across epochs

This is the intended default. Whether dynamic re-sampling produces better generalisation than a
fixed-per-run subset has not been directly ablated in this repository. To test the hypothesis,
one could compare training with dynamic re-sampling vs a pre-fixed bag (shuffled once at the
start of training, held constant across epochs), keeping all other settings identical.

For sampler ablations, the intended clean comparison is:

- keep train/val/test split fixed
- keep training seeds fixed
- keep `max_patches` fixed
- keep evaluation on full bags
- change only the train-time bag sampler

Under that protocol, differences should be interpreted as differences in how the model is exposed to slide evidence during training, not differences in evaluation-time evidence usage.

### Appendix D Figures

![Appendix D ROC and PR curves](figures/appendix_d_roc_pr_curves.png)

![Appendix D confusion matrices](figures/appendix_d_confusion_matrices.png)
