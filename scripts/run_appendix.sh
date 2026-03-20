#!/usr/bin/env bash
# Train the four appendix models (3 seeds each) for the final writeup.
#
# Appendix A: instance_mean, mean_unweighted  (+ uni_mean_fair already done)
# Appendix B: attention_focal                 (+ uni_attention_fair already done)
# Appendix C: topk_attention (k=16)           (+ uni_attention_fair already done)
#
# Each model stream runs in parallel; seeds within a stream are sequential
# to avoid GPU contention between runs of the same model.
#
# Usage:
#   bash scripts/run_appendix.sh
#   bash scripts/run_appendix.sh 2>&1 | tee outputs/appendix.log

set -euo pipefail

SEEDS=(42 123 456)

CONFIGS=(
  configs/appendix/uni_instance_mean.yaml
  configs/appendix/uni_mean_unweighted.yaml
  configs/appendix/uni_attention_focal.yaml
  configs/appendix/uni_topk_attention_k16.yaml
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
  echo "All appendix runs complete."
else
  echo "One or more runs failed — check the log." >&2
  exit 1
fi
