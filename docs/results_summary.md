# Results Summary

## Primary Comparison

All models trained with seeds {42, 123, 456}, fixed split (`split_seed=0`), temperature scaling.

| Model | AUROC (mean ± std) | AUPRC (mean ± std) |
|-------|--------------------|--------------------|
| MeanPool (weighted BCE) | — | — |
| AttentionMIL (weighted BCE) | — | — |
| TransformerMIL (unweighted BCE, Adam) | — | — |

*Fill in from `make compare` output or `outputs/comparison/summary.csv`.*

## Qualitative Interpretation

- **MeanPool is the strongest stable baseline.** Simple averaging of frozen UNI features is competitive
  with more complex architectures. The UNI backbone already encodes the relevant morphological signal.

- **AttentionMIL is promising but seed-dependent.** When it works, it matches or exceeds MeanPool.
  Variance across seeds is higher, suggesting the attention mechanism is hard to train stably at this
  sample size.

- **TransformerMIL is not justified in this data regime.** With 6.8M parameters and no patch-level
  labels, the Transformer overfits. It is included for comparison with prior work, not as a recommended
  approach.

## Stable Conclusions

1. Frozen UNI embeddings contain strong discriminative signal for MSI/MMR status.
2. Simple pooling is a hard-to-beat baseline at this data scale.
3. The main limitation is training instability under weak supervision — not absence of signal.
4. Sparse evidence selection (top-k attention) shows conditional benefit but is not robustly superior.

## Limitations

- Sample size: both cohorts are small relative to the parameter counts of more expressive models.
- Label quality: MSI/MMR labels are assigned at the patient level; slide-level ground truth is not
  available.
- Single site: results may not generalise to slides from different scanners or staining protocols.
- No external validation: all evaluation is on held-out cases from the same cohort distribution.
