# Future Work

## A: Label and Task Refinement

- **False positive morphology**: Slides with high model confidence but negative labels may reveal
  morphological correlates worth investigating. See `scripts/failures/` for exploratory error analysis.
- **Calibration**: Temperature scaling is applied globally. Per-cohort or per-slide calibration may
  improve reliability estimates.
- **Attention statistics as reliability signal**: Slides with high attention entropy (diffuse attention)
  may be harder cases. Using diagnostics from `inspect_attention.py` as confidence proxies is worth
  exploring.

## B: Data Augmentation

Data augmentation is an open area of interest for this project. The current pipeline applies no
augmentation at any stage — UNI embeddings are precomputed and fixed, and bag construction is
the only source of stochasticity during training. Whether augmentation could meaningfully improve
performance or robustness at this sample size is an open question.

Directions worth exploring:

- **Bag-level augmentation**: randomly drop patches, shuffle bag order, or duplicate patches with
  additive noise before aggregation — analogous to dropout at the evidence level.
- **Embedding-space perturbation**: add small Gaussian noise to patch embeddings during training.
  This is a rough proxy for the variability that would arise from re-extracting features under
  slightly different preprocessing conditions.
- **Stain normalisation effects**: if raw tiles are available, training on slides normalised to
  different reference stains (Macenko, Vahadane) may simulate scanner/protocol variation.
- **Mixup at the slide level**: interpolate two slide embeddings and their labels as a
  regularisation strategy — this is established in standard MIL literature but unexplored here.
- **Synthetic minority oversampling**: generate synthetic positive-class bag embeddings by
  interpolating between known positive slides in embedding space, to address class imbalance
  beyond loss reweighting.

No strong prior exists for which of these will help at this scale; empirical ablation is needed.

## C: Data-Centric Improvements

- **Cohort effects**: SR1482 and SR386 have different label conventions and patient populations.
  Cohort-aware training or evaluation would clarify generalisability.
- **Primary vs metastatic**: The dataset may mix primary tumours and metastases, which have different
  morphology and MSI rates.
- **Label provenance**: IHC-derived vs PCR-derived MMR/MSI labels may have different error rates.
  Stratifying by label source could reduce noise.
- **Sample size sensitivity**: Repeated cross-validation with different train fractions would
  characterise how performance degrades with fewer cases.

## D: Additional Tasks

- **Mutation status**: KRAS, NRAS, BRAF status prediction from the same embeddings.
- **Multi-task learning**: Joint prediction of MSI, MMR, and mutation status may regularise feature use.
- **Survival prediction**: UNI embeddings may encode prognostic features beyond biomarker status.
- **Case-vs-slide aggregation**: A dedicated case-level model (aggregating across multiple slides per
  patient) could improve performance when multiple slides are available.

## E: Method Questions

- **Robust sparse aggregation**: Top-k attention is sensitive to k choice. Learned sparsity (e.g.,
  attention with regularisation) may be more principled.
- **Uncertainty-aware attention**: Modelling uncertainty over which patches are relevant, rather than
  point-estimate attention weights.
- **Repeated cross-validation**: Multi-split evaluation across `split_seed ∈ {0,1,2}` is now
  supported via `--override data.split_seed=N` and `scripts/run_main_multisplit_updates.sh`.
  This gives a first estimate of split-sensitivity. True nested CV remains a longer-term goal.
- **Spatial attention**: A rough coordinate extension is now implemented — per-slide min-max
  normalised `(x,y)` encoded via a small MLP and concatenated into the attention scorer input.
  In the current multisplit results this hurts plain `AttentionMIL` and is effectively neutral for
  the hybrid model, so the lesson is not that spatial information is useless, but that a weak
  learned absolute-position branch is not enough. It does not encode inter-patch distances or
  neighbourhood structure. Region-based pooling (`uni_region_attention_*.yaml`) and stronger priors
  such as sinusoidal or rotary encodings are more defensible next steps.

## F: Clinical Interpretability via Multi-Head Attention and Tissue Phenotype Discovery

### Current status

`HybridAttentionMIL` (mean pooling + two independent attention heads + optional diversity penalty)
has now been evaluated across the full 3-split × 3-seed multisplit grid. The current summary is:

- `HybridAttentionMIL`: AUROC `0.903 ± 0.033`, AUPRC `0.591 ± 0.054` — best overall model
- `HybridAttentionMIL + coords`: AUROC `0.897 ± 0.038`, AUPRC `0.541 ± 0.069` — essentially neutral
  relative to the non-spatial hybrid
- `AttentionMIL`: AUROC `0.900 ± 0.030`, AUPRC `0.532 ± 0.120`

So the framing should now be decisive rather than tentative: the hybrid architecture is the
current multisplit winner, and mean-pool anchoring improves robustness enough to justify treating
it as the mainline architecture for follow-up work.

### Immediate priorities

The next round of work should focus on understanding what makes multi-head attention training
stable or unstable before adding further architectural complexity:

1. **Head count**: Two heads is a starting point, not a principled choice. Run ablations over
   `n_heads ∈ {1, 2, 4, 6}` at fixed penalty weight to separate the effect of head count from
   the effect of regularisation.

2. **Diversity penalty weight**: The aux_loss coefficient in `HybridAttentionMIL` controls how
   strongly heads are pushed apart. Too weak and heads collapse onto the same patches; too strong
   and heads are forced to attend to uninformative regions. Grid-search or schedule the penalty
   weight (`lam ∈ {0, 0.01, 0.05, 0.1, 0.5}`) and inspect per-head attention entropy as a proxy
   for whether heads have genuinely specialised.

3. **Initialisation**: Attention weight initialisation affects early-epoch gradient flow. Worth
   testing orthogonal initialisation of the two attention scorer heads as a diversity-promoting
   prior, versus the default PyTorch initialisation.

4. **Evaluation criterion**: Use cross-seed AUROC variance (not just mean) as the primary
   stability metric. In practice this should now be extended to the multisplit setting: a head
   configuration that preserves AUROC while tightening both seed-level and split-level variance is
   more valuable than a fragile mean-only gain.

### Motivation

The current `AttentionMIL` assigns a single scalar weight to each patch. A high-attention patch is
"important" — but important *how*? The model cannot distinguish between, for example, a patch
containing a tumour-infiltrating lymphocyte cluster (which may signal MSI-H) and a patch of
necrotic debris (which may be a confound). A clinician reviewing an attention heatmap has no way to
decompose what each region is contributing or why.

Multi-head attention, already scaffolded in `HybridAttentionMIL`, provides a path toward this. If
each head specialises on a distinct tissue phenotype, the resulting attention maps become
interpretable in morphological terms: head 1 highlights stroma, head 2 highlights tumour
epithelium, head 3 highlights immune infiltrate. The classifier then learns which *combination* of
tissue compartment signals predicts MSI/MMR status.

### Tissue phenotypes expected in colorectal cancer slides

A histopathologist reviewing H&E sections of colorectal cancer would expect to find the following
recurring patch-level structures, which likely occupy distinct regions of the UNI embedding space:

| Phenotype | Relevance to MSI/MMR |
|---|---|
| Tumour epithelium (glandular) | Core signal; poorly differentiated glands are MSI-H associated |
| Tumour-infiltrating lymphocytes (TILs) | Strong MSI-H marker; Crohn's-like reaction |
| Stroma (desmoplastic) | Background; may dilute signal in high-stroma slides |
| Mucin pools / mucinous differentiation | Associated with MSI-H in some subtypes |
| Necrosis | Common in high-grade tumours; potentially confounding |
| Normal mucosa / smooth muscle | Off-tumour tissue; ideally low attention |
| Immune aggregates (peritumoral) | Peritumoral lymphocytic infiltration, MSI-H association |

This suggests somewhere between **4 and 7** meaningful phenotype clusters in a typical CRC dataset.
The crude clustering analysis in `scripts/patch_embedding_viz.py` (see below) is a first empirical
estimate of how many clusters the UNI embeddings naturally support before any label supervision.

### Relationship to head count in HybridAttentionMIL

The optimal number of attention heads is loosely bounded by the number of *task-relevant* tissue
phenotypes. Too few heads and phenotypically distinct patches (e.g. TILs vs tumour glands) are
pooled into a single summary vector, losing discriminative information. Too many heads and the
diversity penalty is required to prevent collapse, adding a hyperparameter whose tuning is
unjustified without clinical grounding.

A practical strategy:

1. Run `scripts/patch_embedding_viz.py` to estimate the number of natural patch clusters in the
   embedding space (k = 2–8, evaluated by silhouette score).
2. Request a clinical expert to annotate representative tiles per cluster — this validates whether
   the clusters correspond to recognisable tissue types.
3. Use the clinically validated cluster count as the starting point for `n_attention_heads` in
   `HybridAttentionMIL`.
4. After training, inspect per-head attention maps and check whether spatial heatmaps for each head
   co-localise with the morphological regions identified in step 2.

### Crude initial analysis: patch embedding clusters

**Script**: `scripts/patch_embedding_viz.py`

The script samples a small number of patches per slide across the full dataset, reduces
dimensionality with PCA (→ 50 components), projects to 3D with UMAP, and applies k-means for
k ∈ {2..8}. It outputs:

- Silhouette scores per k, to identify the natural cluster count empirically
- An interactive 3D Plotly scatter (HTML) with colour toggles:
  - Cluster assignment (for each k)
  - Slide MSI/MMR label (to see whether clusters are label-correlated or purely morphological)
  - Cohort (SR1482 vs SR386), to detect embedding-space cohort effects

**Interpretation guidance**: if the silhouette curve peaks at k=5 and the 3D UMAP shows 5 visually
separable blobs, that is strong evidence for 5 meaningful tissue phenotypes in the embedding space
and suggests `n_attention_heads = 4` or `5` (leaving one head plus mean pooling for the
classifier). If clusters are label-correlated (MSI-H patches concentrated in one or two clusters),
the multi-head model has a clear mechanistic story to offer a clinician.

### From multi-head attention to transformer architecture

If multi-head attention proves capable of robustly differentiating tissue phenotypes — i.e. heads
reliably specialise and their spatial heatmaps align with clinically recognisable structures — that
result is a strong signal that a full transformer encoder over the patch sequence would be the
natural next step.

The current transformer evidence should narrow that claim. Plain `TransformerMIL` is weak on the
multisplit benchmark (`0.850 ± 0.062` AUROC), and adding the current MLP coordinate encoder only
nudges it to `0.859 ± 0.053` while leaving variance high. So "move to a transformer next" is not
the right operational recommendation. The more defensible path is:

1. Use `HybridAttentionMIL` as the strong baseline.
2. Validate whether heads specialise in clinically legible ways.
3. If richer spatial reasoning is still justified, try transformer variants with stronger spatial
   inductive bias rather than the current weak absolute-coordinate concatenation.

The reasoning is direct: `HybridAttentionMIL` computes attention independently per head and pools
each head's weighted sum into a fixed vector before the classifier. There is no interaction between
patches during aggregation — each patch "votes" in isolation. A transformer, by contrast, allows
patches to contextualise each other through self-attention before any pooling occurs. A TIL patch
adjacent to a tumour gland can attend to that gland and produce a richer representation than it
could in isolation. This is precisely the kind of spatial reasoning a pathologist performs when
reading a slide.

The progression in terms of model maturity and interpretability would be:

```
AttentionMIL (single scalar weight per patch)
    ↓  multi-head attention proves tissue specialisation
HybridAttentionMIL (K independent heads, each attending to a phenotype)
    ↓  heads are stable, clinically validated, and task-relevant
Transformer encoder (patches interact; context-aware representations before pooling)
```

Two practical architectures at the transformer stage are worth considering, but only after a
stronger spatial prior is chosen:

- **TRANSMIL** (Shao et al., 2021): adapts the standard transformer encoder directly to the MIL
  setting with a correlated position encoding scheme; a near drop-in upgrade from attention MIL.
- **HIPT** (Chen et al., 2022): hierarchical image pyramid transformer that processes patches at
  multiple scales, more closely mirroring the multi-scale reasoning of a pathologist.

Critically, the interpretability case for the transformer is only well-founded if the multi-head
attention step has already produced clinically legible heads. Without that validation, and given
the current weak multisplit transformer results, a transformer is just a larger black box with no
clear empirical upside. The multi-head analysis is therefore not just a stepping stone in model
performance; it is the standard a transformer variant now has to beat.

### Limitations and caveats

- UNI is a foundation model trained on a broad pathology corpus; its embedding space reflects
  general morphological structure, not MSI-specific features. Clusters found here are tissue
  phenotype clusters, not necessarily task-relevant ones.
- Patch-level ground truth (which tile contains which tissue type) does not exist in this dataset.
  Clinical annotation is required to validate cluster identity.
- The diversity penalty in `HybridAttentionMIL` is a soft constraint. Without label-supervised
  head specialisation (e.g. auxiliary patch-level classification losses), heads may not align with
  clinically interpretable phenotypes even if the correct head count is chosen.
- Spatial context matters: a TIL cluster at the tumour invasive margin is more informative than
  TILs in the centre. But the current `CoordinateEncoder` result suggests that simply appending a
  learned absolute-position embedding is too weak. Future spatial work should prefer stronger
  inductive bias: structured positional encodings, region-level aggregation, or architectures that
  model neighbourhood structure explicitly.

## G: Remaining Gaps And Scope-Limited Limitations

These are the main open items that were not reconciled because they require additional experiments
or broader project scope rather than quick documentation cleanup:

- **External validation**: all reported results come from held-out SurGen cases. A third-cohort or
  cross-site evaluation is the most important missing robustness check.
- **Patch-level validation**: attention maps and clustering analyses remain hypothesis-generating
  because there is no patch-level annotation or pathologist-reviewed ROI benchmark in the repo.
- **Statistical uncertainty**: mean ± std is reported, but there are no confidence intervals,
  paired bootstrap tests, or formal significance tests between models.
- **Limited CV depth**: the multisplit protocol uses three split seeds and three training seeds.
  This is materially better than a single split, but still short of repeated nested CV.
- **Bag sampling ablation depth**: the code documents dynamic train-time re-sampling, but does not
  yet isolate whether dynamic sampling itself is better than fixing one sampled bag per slide per
  run.
- **Upstream pipeline omission**: this repository begins from precomputed UNI embeddings. It does
  not include raw WSI tiling, stain normalization, or feature extraction, so those stages are held
  fixed rather than evaluated.
- **Deployment scope**: temperature scaling and thresholding are implemented, but no operational
  study exists for threshold selection under a clinical prevalence shift, workflow constraints, or
  downstream reviewer-in-the-loop use.
