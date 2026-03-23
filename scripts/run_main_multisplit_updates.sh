#!/usr/bin/env bash
# Run the mainline experiment suite across multiple split seeds and training seeds.
#
# This script covers:
#   - original fair-comparison models
#   - newly added hybrid model
#   - newly added coordinate-aware attention model
#   - newly added coordinate-aware hybrid model
#   - dormant mainline architectures worth reviving quickly (gated attention, mean+var)
#
# Each stream is one (config, split_seed) pair. Seeds within a stream run
# sequentially; streams are throttled with MAX_PARALLEL to avoid overloading
# a single machine.
#
# Usage:
#   bash scripts/run_main_multisplit_updates.sh
#   MAX_PARALLEL=2 bash scripts/run_main_multisplit_updates.sh
#   SPLITS="0 1 2" SEEDS="42 123 456" bash scripts/run_main_multisplit_updates.sh
#
# Logs:
#   outputs/multisplit/logs/<config_name>__split<split>.log
#
# Existing-run behavior:
#   If a config already has canonical runs in its original output dir for the
#   requested split_seed, the script skips retraining that stream and creates a
#   symlink under outputs/multisplit/.../split_<seed> pointing to the original
#   output directory.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

SEEDS_STR=${SEEDS:-"42 123 456"}
SPLITS_STR=${SPLITS:-"0 1 2"}
MAX_PARALLEL=${MAX_PARALLEL:-4}
PYTHON_BIN=${PYTHON_BIN:-python}

read -r -a SEEDS_ARR <<< "$SEEDS_STR"
read -r -a SPLITS_ARR <<< "$SPLITS_STR"

CONFIGS=(
  "configs/uni_mean_fair.yaml"
  "configs/uni_attention_fair.yaml"
  "configs/paper_reproduction_fair.yaml"
  "configs/uni_gated_attention.yaml"
  "configs/uni_mean_var.yaml"
  "configs/uni_hybrid_attention_mean2.yaml"
  "configs/uni_attention_spatial_fair.yaml"
  "configs/uni_hybrid_attention_spatial_mean2.yaml"
)

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

link_existing_output_if_available() {
  local base_config="$1"
  local split_seed="$2"

  "$PYTHON_BIN" - "$base_config" "$split_seed" <<'PY'
from pathlib import Path
import sys
import yaml

base_config = Path(sys.argv[1])
split_seed = int(sys.argv[2])

with open(base_config) as f:
    cfg = yaml.safe_load(f)

base_out = Path(cfg["output"]["dir"])
base_split_seed = cfg.get("data", {}).get("split_seed")
target = Path("outputs") / "multisplit" / base_out.name / f"split_{split_seed}"

has_runs = (base_out / "runs").is_dir() and any((base_out / "runs").glob("*/config.yaml"))
if not (has_runs and base_split_seed == split_seed):
    print("NO_LINK")
    raise SystemExit(0)

target.parent.mkdir(parents=True, exist_ok=True)

if target.is_symlink():
    if target.resolve() == base_out.resolve():
        print(f"LINKED\t{target}\t{base_out}")
        raise SystemExit(0)
    target.unlink()
elif target.exists():
    print(f"EXISTS\t{target}\t{base_out}")
    raise SystemExit(0)

target.symlink_to(base_out.resolve())
print(f"LINKED\t{target}\t{base_out}")
PY
}

run_stream() {
  local base_config="$1"
  local split_seed="$2"
  local name
  local log_path
  local split_out_dir

  name=$(basename "$base_config" .yaml)
  log_path="$LOG_DIR/${name}__split${split_seed}.log"
  split_out_dir=$("$PYTHON_BIN" - "$base_config" "$split_seed" <<'PY'
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
    echo "[${name}][split ${split_seed}] config ${base_config}"
    echo "[${name}][split ${split_seed}] output ${split_out_dir}"
    for seed in "${SEEDS_ARR[@]}"; do
      echo "[${name}][split ${split_seed}] seed ${seed} start $(date -u '+%H:%M:%S')"
      "$PYTHON_BIN" train.py \
        --config "$base_config" \
        --seed "$seed" \
        --override "data.split_seed=${split_seed}" "output.dir=${split_out_dir}"
      echo "[${name}][split ${split_seed}] seed ${seed} done  $(date -u '+%H:%M:%S')"
    done
    echo "[${name}][split ${split_seed}] complete $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  } 2>&1 | tee "$log_path"
}

echo "Running mainline multi-split suite"
echo "Configs      : ${#CONFIGS[@]}"
echo "Split seeds  : ${SPLITS_STR}"
echo "Train seeds  : ${SEEDS_STR}"
echo "Max parallel : ${MAX_PARALLEL}"
echo "Logs         : ${LOG_DIR}"

pids=()
for config in "${CONFIGS[@]}"; do
  for split_seed in "${SPLITS_ARR[@]}"; do
    link_result=$(link_existing_output_if_available "$config" "$split_seed")
    if [ "$link_result" != "NO_LINK" ]; then
      echo "$(basename "$config" .yaml) split ${split_seed} -> ${link_result}"
      continue
    fi
    wait_for_slot
    run_stream "$config" "$split_seed" &
    pids+=($!)
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "Stream PID ${pid} failed" >&2
    failed=1
  fi
done

if [ "$failed" -eq 0 ]; then
  echo "All multi-split update runs complete."
else
  echo "One or more multi-split update runs failed." >&2
  exit 1
fi
