# Results Summary

## Primary Comparison

All models trained with seeds {42, 123, 456}, fixed split (`split_seed=0`), temperature scaling.
Confusion matrices use seed-averaged predictions with Youden J threshold fit on the validation set.

| Model | AUROC (mean ± std) | AUPRC (mean ± std) |
|-------|--------------------|--------------------|
| MeanPool (weighted BCE) | 0.860 ± 0.005 | 0.447 ± 0.019 |
| AttentionMIL (weighted BCE) | 0.869 ± 0.020 | 0.381 ± 0.052 |
| TransformerMIL (weighted BCE) | 0.806 ± 0.057 | 0.391 ± 0.116 |
| *Paper baseline (Myles et al.)* | *0.827* | *—* |

*Paper baseline: Myles et al. (2025), GigaScience — "SurGen: a multimodal, multi-centre surgical and genomics colorectal cancer dataset", doi:10.1093/gigascience/giaf086.*

![Confusion matrices](figures/fair_comparison_confusion_matrices.png)

## Qualitative Interpretation

- **MeanPool is the strongest stable baseline.** Simple averaging of frozen UNI features achieves
  competitive AUROC (0.860) with the lowest cross-seed variance (±0.005), consistent with UNI
  embeddings carrying strong discriminative signal that does not require learned aggregation
  to surface.

- **AttentionMIL is promising but seed-dependent.** Mean AUROC (0.869) is marginally higher than
  MeanPool, but cross-seed variance is 4× larger (±0.020 vs ±0.005). This observation is from
  three seeds on one fixed split; whether the variance reflects inherent training sensitivity or
  an artefact of this dataset size is not conclusively established here.

- **TransformerMIL has the highest parameter count (6.8M) and lowest mean AUROC (0.806) in this
  experiment**, with the highest cross-seed variance (±0.057). Whether this reflects overfitting,
  optimisation difficulty under weak supervision, or data-scale mismatch is not directly
  established by this comparison alone; it is included here alongside the smaller architectures
  to match the scope of prior work.

## Observations

1. MeanPool and AttentionMIL exceed the paper reference AUROC of 0.827 in this three-seed, fixed-split comparison, while the fair TransformerMIL reproduction does not. The broader conclusion is still that frozen UNI embeddings provide useful discriminative signal for MSI/MMR status.
2. MeanPool achieves competitive AUROC with the lowest cross-seed variance, making it the most reproducible aggregator in this experiment.
3. AttentionMIL shows higher cross-seed variance (±0.020) than MeanPool (±0.005). Whether this reflects training instability, signal absence in some seeds, or both is not directly resolved by this experiment. A direct test would compare held-out performance as a function of training set size or fix random initialisation while varying bag sampling.
4. In this experiment, TopK-16 training improves AUPRC but reduces AUROC relative to full-bag AttentionMIL; neither configuration dominates across metrics.

### Why A Better Confusion Matrix Can Coexist With Worse PRC

A confusion matrix is tied to one specific threshold. Precision-recall curves and AUPRC summarize
performance across *all* thresholds. That means an architecture can look better at the chosen
operating point, for example after fitting a Youden J threshold on validation, while still ranking
positives below negatives less consistently over the full score range.

In practice, this usually means:

- the model has a stronger operating point near one threshold
- but its overall probability ordering is less clean across the rest of the curve

So a model can have a better confusion matrix at the selected threshold and still have worse AUPRC.
These are not contradictory findings; they answer different questions.

## Multi-Split Comparison

To reduce dependence on any single case-grouped partition, the mainline models were also evaluated
across `split_seed ∈ {0,1,2}` and training seeds `{42,123,456}`, giving 9 runs per model. This is
the same framing used in the current presentation deck: multisplit performance is the more reliable
summary of expected behaviour, while the fixed-split fair comparison remains useful for controlled
apples-to-apples plots.

| Model | AUROC (mean ± std) | AUPRC (mean ± std) | Interpretation |
|-------|--------------------|--------------------|----------------|
| HybridAttentionMIL (mean + 2 heads) | 0.903 ± 0.033 | 0.591 ± 0.054 | best overall performer |
| AttentionMIL | 0.900 ± 0.030 | 0.532 ± 0.120 | strong mean AUROC, high AUPRC variance |
| HybridAttentionMIL + coords | 0.897 ± 0.038 | 0.541 ± 0.069 | spatial encoding is effectively neutral here |
| Gated AttentionMIL | 0.896 ± 0.037 | 0.516 ± 0.130 | competitive AUROC, unstable AUPRC |
| MeanVar Pool | 0.895 ± 0.026 | 0.497 ± 0.079 | strong non-attention baseline |
| AttentionMIL + coords | 0.882 ± 0.038 | 0.485 ± 0.138 | spatial coordinates hurt plain attention |
| MeanPool | 0.877 ± 0.031 | 0.495 ± 0.047 | stable baseline, still competitive |
| Spatial TransformerMIL | 0.859 ± 0.053 | 0.465 ± 0.108 | slight AUROC gain vs no-coords transformer, still poor/high-variance |
| TransformerMIL (paper repro) | 0.850 ± 0.062 | 0.508 ± 0.116 | worst AUROC, highest variance |

### Multi-Split Interpretation

- **Hybrid attention is the clearest winner.** Combining mean pooling with two learned attention
  heads gives the best overall AUROC and AUPRC, with variance that stays in the same range as plain
  AttentionMIL despite higher capacity.
- **Frozen UNI embeddings remain the main story.** MeanPool, MeanVar, AttentionMIL, and the hybrid
  models all perform well enough that the limiting factor is reliable aggregation, not lack of
  discriminative patch features.
- **Spatial coordinates are model-dependent.** They are harmful for plain AttentionMIL, essentially
  neutral for the hybrid, and only slightly improve transformer AUROC while leaving transformer
  variance high.
- **The transformer result is now more concrete.** Adding the small MLP coordinate branch does not
  rescue TransformerMIL. The correct framing is not "transformers need coordinates and will likely
  recover", but "this simple coordinate injection is insufficient at the current data scale."

The current presentation deck uses this multisplit framing as the primary summary because it is
less sensitive to one lucky or unlucky split than the original fixed-split table.

## Appendix Results

### Appendix A — Aggregator and loss ablations

Comparison against the MeanPool weighted-BCE baseline (`uni_mean_fair`).

| Model | AUROC | AUPRC | Note |
|-------|-------|-------|------|
| MeanPool (weighted BCE) | 0.860 ± 0.005 | 0.447 ± 0.019 | baseline |
| InstanceMean (weighted BCE) | 0.859 ± 0.008 | 0.443 ± 0.027 | classify-then-pool |
| MeanPool (unweighted BCE) | 0.862 ± 0.003 | 0.465 ± 0.004 | no class reweighting |

### Appendix B — Loss function ablation (AttentionMIL)

| Model | AUROC | AUPRC | Note |
|-------|-------|-------|------|
| AttentionMIL (weighted BCE) | 0.869 ± 0.020 | 0.381 ± 0.052 | baseline |
| AttentionMIL (focal, α=0.5, γ=2) | 0.863 ± 0.014 | 0.420 ± 0.066 | focal loss |

### Appendix C — Sparse evidence selection

| Model | AUROC | AUPRC | Note |
|-------|-------|-------|------|
| AttentionMIL (weighted BCE) | 0.869 ± 0.020 | 0.381 ± 0.052 | full bag |
| TopK-16 AttentionMIL (weighted BCE) | 0.853 ± 0.032 | 0.455 ± 0.139 | k=16 (≈3% of 512-patch training bag) |

![Appendix C ROC and PR curves](figures/appendix_c_roc_pr_curves.png)

**Reading the PR curve**: AUPRC is the area under the precision-recall curve — it measures how
well the model ranks true positives above true negatives across *all* possible thresholds.
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
the AUPRC improvement is worth the worse Youden operating point depends on the clinical cost of
false positives versus false negatives.

### Appendix D — Train-time sampler ablation

This section is reserved for the Phase 1 bag-sampler comparison. The intended protocol is:

- same split (`split_seed=0`)
- same training seeds `{42, 123, 456}`
- same `max_patches=512`
- same full-bag evaluation
- same optimizer / LR / early stopping within each model family
- different **re-sampled** train-time bag construction rules only

The train-time sampler is applied on fetch, so sampled bags can differ across epochs. Results here
should therefore be interpreted as differences in training-time evidence exposure, not differences
in evaluation-time evidence usage.

#### MeanPoolMIL

| Model | AUROC | AUPRC | Note |
|-------|-------|-------|------|
| MeanPool + random | 0.860 ± 0.005 | 0.447 ± 0.019 | baseline train-time sampler |
| MeanPool + spatial balanced | 0.859 ± 0.007 | 0.447 ± 0.014 | grid-based coverage sampler |
| MeanPool + feature diverse | 0.852 ± 0.005 | 0.358 ± 0.035 | feature-space coverage sampler |

#### AttentionMIL

| Model | AUROC | AUPRC | Note |
|-------|-------|-------|------|
| AttentionMIL + random | 0.869 ± 0.020 | 0.381 ± 0.052 | baseline train-time sampler |
| AttentionMIL + spatial balanced | 0.861 ± 0.031 | 0.404 ± 0.058 | grid-based coverage sampler |
| AttentionMIL + feature diverse | 0.879 ± 0.006 | 0.407 ± 0.058 | feature-space coverage sampler |

#### Figures

![ROC and PR curves](figures/appendix_d_roc_pr_curves.png)

![Confusion matrices](figures/appendix_d_confusion_matrices.png)

#### Interpretation

**MeanPoolMIL is sampler-invariant — and that is a positive result.** Random and spatial samplers
match closely (AUROC 0.860/0.859, AUPRC 0.447/0.447). This is not a null result: mean pooling
behaves as a stable estimator of the slide-level expectation, and random subsampling is a
sufficient Monte Carlo approximation of it. The implication is that discriminative information is
broadly distributed across patches rather than concentrated in rare regions. Spatial balancing
provides a marginally lower-variance estimate of the same expectation; the gains are negligible
because the signal density is already high enough for random sampling to capture it.

Feature-diverse sampling hurts MeanPool AUPRC substantially (0.358 ± 0.035 vs 0.447). This is
not simply about selecting atypical patches — it reflects a distribution shift in the training bag.
Feature-diverse selection downweights high-density embedding modes, which correspond to dominant
tissue patterns. In weakly-supervised MIL those dominant regions carry most of the statistical
signal. Feature diversity in embedding space is not the same as discriminative diversity: optimising
for coverage of representation space can actively degrade mean-based aggregation by
underrepresenting the most label-relevant evidence.

**AttentionMIL: the result is not "no signal" — it is "variance dominates."** Mean performance
differences are small (AUPRC 0.381–0.407), but the variance story is more informative. Random
sampling yields AUROC std of ±0.020–0.031 across seeds; feature-diverse sampling reduces this to
±0.006. That is structural stabilisation, not noise. Attention MIL is sensitive to which patches
are seen early and often: random bag construction introduces stochastic exposure and
seed-dependent convergence paths, while diversity-constrained sampling provides more consistent
coverage of the embedding space and more stable gradients. Sampling does not strongly shift mean
performance, but it materially affects optimisation stability.

**The bottleneck is not evidence availability — it is evidence utilisation.** Evaluation always
uses the full bag, so any benefit from structured sampling must appear as improved learning from
limited exposure. The fact that it largely does not suggests the model already sees enough
informative signal under random sampling; the limitation is how it aggregates that signal.
The lack of improvement from structured sampling indicates that performance is not constrained by
missing informative patches, but by the model's ability to assign stable and generalisable
importance weights to the patches it already observes.

**Where the stack stands:**
- *Representation:* strong — discriminative signal is broadly distributed and well-captured by UNI embeddings
- *Sampling:* sufficient — random subsampling is an adequate estimator at this patch density
- *Aggregation:* the primary bottleneck — attention instability is an optimisation problem, not an information problem

*All appendix metrics: mean ± std across seeds {42, 123, 456}, same split as main comparison.
Regenerate with `python scripts/appendix_tables.py --out outputs/appendix_tables.csv`.*

## Limitations

- Sample size: both cohorts are small relative to the parameter counts of more expressive models.
- Label quality: MSI/MMR labels are assigned at the patient level; slide-level ground truth is not
  available.
- Single site: results may not generalise to slides from different scanners or staining protocols.
- No external validation: all evaluation is on held-out cases from the same cohort distribution.
- No significance testing: results are summarised as mean ± std across a small number of seeds and
  splits; confidence intervals and formal paired significance tests are not yet reported.
- Limited split count: multisplit evaluation across `{0,1,2}` is stronger than a single split, but
  it is not full repeated nested cross-validation.
- No dynamic-vs-fixed bag-sampling ablation: the repository uses dynamic train-time re-sampling by
  default, but does not yet directly compare that against a fixed sampled bag per run.
