#!/usr/bin/env bash
# Run the fair 3-way comparison: uni_mean_fair, uni_attention_fair, paper_reproduction_fair
# Each config is trained with seeds 42, 123, 456 against a fixed split_seed=0.
#
# The three model streams run in parallel; seeds within each stream are sequential
# to avoid GPU memory contention between runs of the same model.
#
# Usage:
#   bash scripts/run_fair_comparison.sh
#   bash scripts/run_fair_comparison.sh 2>&1 | tee outputs/fair_comparison.log

set -euo pipefail

SEEDS=(42 123 456)

CONFIGS=(
  configs/uni_mean_fair.yaml
  configs/uni_attention_fair.yaml
  configs/paper_reproduction_fair.yaml
)

run_model() {
  local config="$1"
  local name
  name=$(basename "$config" .yaml)
  echo "[${name}] starting"
  for seed in "${SEEDS[@]}"; do
    echo "[${name}] seed ${seed} — start $(date '+%H:%M:%S')"
    python train.py --config "$config" --seed "$seed"
    echo "[${name}] seed ${seed} — done  $(date '+%H:%M:%S')"
  done
  echo "[${name}] all seeds complete"
}

export -f run_model

# Launch one stream per config in parallel
pids=()
for config in "${CONFIGS[@]}"; do
  run_model "$config" &
  pids+=($!)
done

# Wait for all streams and collect exit codes
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "Stream PID ${pid} failed" >&2
    failed=1
  fi
done

if [ "$failed" -eq 0 ]; then
  echo "All runs complete."
else
  echo "One or more runs failed — check the log." >&2
  exit 1
fi
