#!/usr/bin/env bash
# Run the spatial TransformerMIL experiment across multiple split seeds and training seeds.
#
# Config: configs/uni_transformer_spatial_fair.yaml
# Splits: 3 split seeds (0, 1, 2) — one stream per split
# Seeds:  3 training seeds (42, 123, 456) run sequentially within each stream
#
# Usage:
#   bash scripts/run_transformer_spatial_multisplit.sh
#   MAX_PARALLEL=2 bash scripts/run_transformer_spatial_multisplit.sh
#   SPLITS="0 1" SEEDS="42 123" bash scripts/run_transformer_spatial_multisplit.sh
#
# Logs:
#   outputs/multisplit/logs/uni_transformer_spatial_fair__split<split>.log

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

SEEDS_STR=${SEEDS:-"42 123 456"}
SPLITS_STR=${SPLITS:-"0 1 2"}
MAX_PARALLEL=${MAX_PARALLEL:-3}
PYTHON_BIN=${PYTHON_BIN:-python}

read -r -a SEEDS_ARR <<< "$SEEDS_STR"
read -r -a SPLITS_ARR <<< "$SPLITS_STR"

CONFIG="configs/uni_transformer_spatial_fair.yaml"
LOG_DIR="outputs/multisplit/logs"
mkdir -p "$LOG_DIR"

wait_for_slot() {
  while true; do
    local running
    running=$(jobs -rp | wc -l | tr -d ' ')
    if [ "$running" -lt "$MAX_PARALLEL" ]; then
      break
    fi
    sleep 2
  done
}

run_stream() {
  local split_seed="$1"
  local name
  local log_path
  local split_out_dir

  name=$(basename "$CONFIG" .yaml)
  log_path="$LOG_DIR/${name}__split${split_seed}.log"
  split_out_dir=$("$PYTHON_BIN" - "$CONFIG" "$split_seed" <<'PY'
from pathlib import Path
import sys
import yaml

base_config = Path(sys.argv[1])
split_seed = int(sys.argv[2])
with open(base_config) as f:
    cfg = yaml.safe_load(f)
base_out = Path(cfg["output"]["dir"])
print(Path("outputs") / "multisplit" / base_out.name / f"split_{split_seed}")
PY
)

  {
    echo "[${name}][split ${split_seed}] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "[${name}][split ${split_seed}] config ${CONFIG}"
    echo "[${name}][split ${split_seed}] output ${split_out_dir}"
    for seed in "${SEEDS_ARR[@]}"; do
      echo "[${name}][split ${split_seed}] seed ${seed} start $(date -u '+%H:%M:%S')"
      "$PYTHON_BIN" train.py \
        --config "$CONFIG" \
        --seed "$seed" \
        --override "data.split_seed=${split_seed}" "output.dir=${split_out_dir}"
      echo "[${name}][split ${split_seed}] seed ${seed} done  $(date -u '+%H:%M:%S')"
    done
    echo "[${name}][split ${split_seed}] complete $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  } 2>&1 | tee "$log_path"
}

echo "Spatial TransformerMIL multi-split run"
echo "Config       : ${CONFIG}"
echo "Split seeds  : ${SPLITS_STR}"
echo "Train seeds  : ${SEEDS_STR}"
echo "Max parallel : ${MAX_PARALLEL}"
echo "Logs         : ${LOG_DIR}"

pids=()
for split_seed in "${SPLITS_ARR[@]}"; do
  wait_for_slot
  run_stream "$split_seed" &
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
  echo "All spatial transformer multi-split runs complete."
else
  echo "One or more spatial transformer multi-split runs failed." >&2
  exit 1
fi
