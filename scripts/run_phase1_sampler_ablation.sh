#!/usr/bin/env bash
# Run the Phase 1 sampler ablation:
#   mean_pool      x {random, spatial_balanced, feature_diverse}
#   attention_mil  x {random, spatial_balanced, feature_diverse}
#
# All configs share:
#   - split_seed=0
#   - max_patches=512
#   - full-bag evaluation
#   - the same optimizer / early-stopping settings within each model family
#
# Usage:
#   bash scripts/run_phase1_sampler_ablation.sh

set -euo pipefail

SEEDS=(42 123 456)

CONFIGS=(
  configs/appendix/phase1_mean_random.yaml
  configs/appendix/phase1_mean_spatial.yaml
  configs/appendix/phase1_mean_feature_diverse.yaml
  configs/appendix/phase1_attention_random.yaml
  configs/appendix/phase1_attention_spatial.yaml
  configs/appendix/phase1_attention_feature_diverse.yaml
)

run_model() {
  local config="$1"
  local name
  name=$(basename "$config" .yaml)
  echo "[${name}] starting"
  for seed in "${SEEDS[@]}"; do
    echo "[${name}] seed ${seed} - start $(date '+%H:%M:%S')"
    python train.py --config "$config" --seed "$seed"
    echo "[${name}] seed ${seed} - done  $(date '+%H:%M:%S')"
  done
  echo "[${name}] all seeds complete"
}

export -f run_model

pids=()
for config in "${CONFIGS[@]}"; do
  run_model "$config" &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "Stream PID ${pid} failed" >&2
    failed=1
  fi
done

if [ "$failed" -eq 0 ]; then
  echo "All Phase 1 sampler runs complete."
else
  echo "One or more Phase 1 sampler streams failed - check the log." >&2
  exit 1
fi
