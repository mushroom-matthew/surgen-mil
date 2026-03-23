.PHONY: smoke test train-mean train-attention train-transformer fair-comparison multisplit-updates multisplit-analyse appendix appendix-tables compare evaluate attn-auto attn-seed-grid attn-slide attn-stats errors error-report help

smoke:  ## Run smoke test with synthetic data
	python scripts/smoke_test.py

test:  ## Run unit tests
	python -m pytest tests/ -v

train-mean:  ## Train MeanPool model (fair comparison config)
	python train.py --config configs/uni_mean_fair.yaml

train-attention:  ## Train AttentionMIL model (fair comparison config)
	python train.py --config configs/uni_attention_fair.yaml

train-transformer:  ## Train TransformerMIL model (fair comparison config)
	python train.py --config configs/paper_reproduction_fair.yaml

fair-comparison:  ## Train all three models with 3 seeds each (parallel)
	bash scripts/run_fair_comparison.sh

multisplit-updates:  ## Train mainline updated suite across split seeds 0/1/2
	bash scripts/run_main_multisplit_updates.sh

multisplit-analyse:  ## Analyse outputs/multisplit with overall + per-split reports
	python scripts/compare_multisplit.py

appendix:  ## Train all appendix models with 3 seeds each
	bash scripts/run_appendix.sh

appendix-tables:  ## Generate appendix performance tables
	python scripts/appendix_tables.py --out outputs/appendix_tables.csv

compare:  ## Compare models across seeds: summary table + plots
	python scripts/compare_models.py \
		--configs configs/uni_mean_fair.yaml \
		          configs/uni_attention_fair.yaml \
		          configs/paper_reproduction_fair.yaml \
		--out outputs/comparison

evaluate:  ## Evaluate latest checkpoint: make evaluate CONFIG=configs/uni_mean_fair.yaml
	python scripts/evaluate.py \
		--config $(CONFIG) \
		--checkpoint $$(ls -t $$(python -c "import yaml; print(yaml.safe_load(open('$(CONFIG)'))['output']['dir'])")/runs/*/model.pt 2>/dev/null | head -1)

attn-auto:  ## Attention maps: compare models on auto-selected TP/FP/FN test slides
	python scripts/failures/compare_attention.py \
		--auto --n_examples 3 --topk 100 \
		--out outputs/attention_viz

attn-seed-grid:  ## Attention maps: seed variance grid (rows=models, cols=seeds)
	python scripts/failures/compare_attention.py \
		--auto --seed_grid --n_examples 2 \
		--out outputs/attention_viz/seed_grid

attn-slide:  ## Attention map for a single slide: make attn-slide SLIDE_ID=SR1482_40X_HE_T1_0
	python scripts/failures/compare_attention.py \
		--slide_id $(SLIDE_ID) --topk 100 \
		--out outputs/attention_viz

attn-stats:  ## Attention weight statistics for AttentionMIL (latest checkpoint)
	python scripts/inspect_attention.py \
		--config configs/uni_attention_fair.yaml \
		--checkpoint $$(ls -t outputs/uni_attention_fair/runs/*/model.pt 2>/dev/null | head -1) \
		--split test \
		--out outputs/attention_stats.csv

errors:  ## Build failure manifest CSV across all model runs
	python scripts/failures/export_failure_manifest.py \
		--split test \
		--out outputs/failure_manifest.csv

error-report:  ## Print failure manifest cross-referenced with slide metadata
	python scripts/failures/export_failure_manifest.py \
		--split test \
		--out outputs/failure_manifest.csv
	python scripts/failures/failure_report.py \
		--manifest outputs/failure_manifest.csv \
		--labels /mnt/data-surgen/SR1482_labels.csv

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'
