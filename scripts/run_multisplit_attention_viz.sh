#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-outputs/multisplit}"
OUT_DIR="${OUT_DIR:-outputs/multisplit/attention_viz}"
SPLITS="${SPLITS:-0 1 2}"
TOPK="${TOPK:-100}"
N_EXAMPLES="${N_EXAMPLES:-2}"
THRESHOLD="${THRESHOLD:-0.5}"
SPLIT_NAME="${SPLIT_NAME:-test}"
MULTIHEAD_MODE="${MULTIHEAD_MODE:-mean}"
AUTO="${AUTO:-1}"
SEED_GRID="${SEED_GRID:-1}"
NO_MULTIHEAD_PANELS="${NO_MULTIHEAD_PANELS:-0}"
SLIDE_ID="${SLIDE_ID:-}"

MODEL_NAMES=(
  "MeanPool (fair)"
  "Attn MIL (fair)"
  "Paper Repro (fair)"
  "Gated Attention"
  "Mean + Var"
  "Hybrid Attn + Mean"
  "Attn MIL + coords"
  "Hybrid + coords"
)

MODEL_DIRS=(
  "uni_mean_fair"
  "uni_attention_fair"
  "paper_reproduction_fair"
  "uni_gated_attention"
  "uni_mean_var"
  "uni_hybrid_attention_mean2"
  "uni_attention_spatial_fair"
  "uni_hybrid_attention_spatial_mean2"
)

read -r -a SPLIT_LIST <<< "${SPLITS}"

for split_seed in "${SPLIT_LIST[@]}"; do
  split_dir="${BASE_DIR}"
  split_out="${OUT_DIR}/split_${split_seed}"
  models_file="$(mktemp /tmp/multisplit_attention_models_${split_seed}_XXXX.yaml)"
  existing_count=0

  {
    for idx in "${!MODEL_NAMES[@]}"; do
      name="${MODEL_NAMES[$idx]}"
      model_dir="${MODEL_DIRS[$idx]}"
      target="${split_dir}/${model_dir}/split_${split_seed}"
      if [[ -e "${target}" ]]; then
        existing_count=$((existing_count + 1))
      fi
      printf '"%s": "%s"\n' "${name}" "${target}"
    done
  } > "${models_file}"

  if [[ "${existing_count}" -eq 0 ]]; then
    echo "Skipping split ${split_seed}: no multisplit model directories found under ${split_dir}"
    rm -f "${models_file}"
    continue
  fi

  mkdir -p "${split_out}"
  echo "Running attention visualisation for split ${split_seed} -> ${split_out}"

  cmd=(
    python scripts/failures/compare_attention.py
    --models_file "${models_file}"
    --topk "${TOPK}"
    --threshold "${THRESHOLD}"
    --split "${SPLIT_NAME}"
    --multihead_mode "${MULTIHEAD_MODE}"
    --out "${split_out}"
  )

  if [[ "${AUTO}" == "1" ]]; then
    cmd+=(--auto --n_examples "${N_EXAMPLES}")
  else
    if [[ -z "${SLIDE_ID}" ]]; then
      echo "SLIDE_ID must be set when AUTO=0" >&2
      rm -f "${models_file}"
      exit 1
    fi
    cmd+=(--slide_id "${SLIDE_ID}")
  fi

  if [[ "${SEED_GRID}" == "1" ]]; then
    cmd+=(--seed_grid)
  fi

  if [[ "${NO_MULTIHEAD_PANELS}" == "1" ]]; then
    cmd+=(--no_multihead_panels)
  fi

  "${cmd[@]}"
  rm -f "${models_file}"
done
