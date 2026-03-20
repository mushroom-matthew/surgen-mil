# Future Work

## A: Label and Task Refinement

- **False positive morphology**: Slides with high model confidence but negative labels may reveal
  morphological correlates worth investigating. See `scripts/failures/` for exploratory error analysis.
- **Calibration**: Temperature scaling is applied globally. Per-cohort or per-slide calibration may
  improve reliability estimates.
- **Attention statistics as reliability signal**: Slides with high attention entropy (diffuse attention)
  may be harder cases. Using diagnostics from `inspect_attention.py` as confidence proxies is worth
  exploring.

## B: Data-Centric Improvements

- **Cohort effects**: SR1482 and SR386 have different label conventions and patient populations.
  Cohort-aware training or evaluation would clarify generalisability.
- **Primary vs metastatic**: The dataset may mix primary tumours and metastases, which have different
  morphology and MSI rates.
- **Label provenance**: IHC-derived vs PCR-derived MMR/MSI labels may have different error rates.
  Stratifying by label source could reduce noise.
- **Sample size sensitivity**: Repeated cross-validation with different train fractions would
  characterise how performance degrades with fewer cases.

## C: Additional Tasks

- **Mutation status**: KRAS, NRAS, BRAF status prediction from the same embeddings.
- **Multi-task learning**: Joint prediction of MSI, MMR, and mutation status may regularise feature use.
- **Survival prediction**: UNI embeddings may encode prognostic features beyond biomarker status.
- **Case-vs-slide aggregation**: A dedicated case-level model (aggregating across multiple slides per
  patient) could improve performance when multiple slides are available.

## D: Method Questions

- **Robust sparse aggregation**: Top-k attention is sensitive to k choice. Learned sparsity (e.g.,
  attention with regularisation) may be more principled.
- **Uncertainty-aware attention**: Modelling uncertainty over which patches are relevant, rather than
  point-estimate attention weights.
- **Repeated cross-validation**: Single train/val/test split is high-variance at small N. Nested CV
  would give better uncertainty estimates on performance.
- **Spatial attention**: The `coords` field is loaded but ignored by all three main models. Region-based
  or spatial attention (see `uni_region_attention_*.yaml`) is a natural extension.
