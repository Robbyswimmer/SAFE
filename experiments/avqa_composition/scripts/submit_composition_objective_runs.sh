#!/bin/bash
# Submit compact sbatch jobs for composition interleaved experiments
# with/without the new compatibility regularizer objective.
#
# Usage:
#   bash experiments/avqa_composition/scripts/submit_composition_objective_runs.sh compreg_disjoint
#   bash experiments/avqa_composition/scripts/submit_composition_objective_runs.sh baseline_disjoint
#   bash experiments/avqa_composition/scripts/submit_composition_objective_runs.sh compreg_samelayer
#   bash experiments/avqa_composition/scripts/submit_composition_objective_runs.sh lambda_sweep
#   bash experiments/avqa_composition/scripts/submit_composition_objective_runs.sh all
#
# Optional env overrides:
#   SBATCH_RESOURCES="--gres=gpu:1 --exclude=gpu03"
#   EPOCHS=10 MAX_SAMPLES=0 FUSION_GATE=0.2 SEED=42
#   COMPAT_REG_LAMBDA=0.05 COMPAT_REG_RANK=8

set -euo pipefail

MODE="${1:-compreg_disjoint}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFE_ROOT="${SAFE_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TRAIN_SCRIPT="$SAFE_ROOT/experiments/avqa_composition/scripts/train_composition_interleaved.sh"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "[error] train script not found: $TRAIN_SCRIPT" >&2
  exit 1
fi

# sbatch resource flags can be overridden by caller
SBATCH_RESOURCES="${SBATCH_RESOURCES:---gres=gpu:1}"

# Shared defaults
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-SAFE-Composition}"
OUTPUT_BASE="${OUTPUT_BASE:-checkpoints/composition_runs}"
RUN_PREFIX="${RUN_PREFIX:-composition}"
EPOCHS="${EPOCHS:-10}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_AUDIO_TOKENS="${NUM_AUDIO_TOKENS:-8}"
FUSION_GATE="${FUSION_GATE:-0.2}"
SEED="${SEED:-42}"
LAYER_ADDITIVITY_PROBE="${LAYER_ADDITIVITY_PROBE:-0}"
LAYER_PROBE_SAMPLES="${LAYER_PROBE_SAMPLES:-256}"
LAYER_PROBE_EVERY="${LAYER_PROBE_EVERY:-1}"

# Compatibility objective defaults
COMPAT_REG_LAMBDA="${COMPAT_REG_LAMBDA:-0.05}"
COMPAT_REG_RANK="${COMPAT_REG_RANK:-8}"
COMPAT_REG_AUDIO_SAMPLES="${COMPAT_REG_AUDIO_SAMPLES:-256}"
COMPAT_REG_MIN_SAMPLES="${COMPAT_REG_MIN_SAMPLES:-64}"
COMPAT_REG_REFRESH_EVERY="${COMPAT_REG_REFRESH_EVERY:-1}"
COMPAT_REG_LAYERS="${COMPAT_REG_LAYERS:-}"
COMPAT_REG_WEIGHT_BY_SHIFT_NORM="${COMPAT_REG_WEIGHT_BY_SHIFT_NORM:-1}"

COMPAT_ADD_REG_ENABLE="${COMPAT_ADD_REG_ENABLE:-0}"
COMPAT_ADD_REG_LAMBDA="${COMPAT_ADD_REG_LAMBDA:-0.01}"
COMPAT_ADD_REG_EVERY="${COMPAT_ADD_REG_EVERY:-200}"
COMPAT_ADD_REG_LAYERS="${COMPAT_ADD_REG_LAYERS:-}"
COMPAT_ADD_REG_NORMALIZE="${COMPAT_ADD_REG_NORMALIZE:-1}"
COMPAT_ADD_BANK_SIZE="${COMPAT_ADD_BANK_SIZE:-64}"

COMPAT_NOHARM_ENABLE="${COMPAT_NOHARM_ENABLE:-0}"
COMPAT_NOHARM_LAMBDA="${COMPAT_NOHARM_LAMBDA:-0.02}"
COMPAT_NOHARM_MARGIN="${COMPAT_NOHARM_MARGIN:-0.0}"
COMPAT_NOHARM_USE_BEST_SINGLE="${COMPAT_NOHARM_USE_BEST_SINGLE:-1}"

COMPAT_LOGIT_FUSION_ENABLE="${COMPAT_LOGIT_FUSION_ENABLE:-0}"
COMPAT_LOGIT_FUSION_LAMBDA="${COMPAT_LOGIT_FUSION_LAMBDA:-0.02}"
COMPAT_LOGIT_FUSION_CONF_TEMP="${COMPAT_LOGIT_FUSION_CONF_TEMP:-0.5}"

COMPAT_POE_ENABLE="${COMPAT_POE_ENABLE:-0}"
COMPAT_POE_LAMBDA="${COMPAT_POE_LAMBDA:-0.02}"
COMPAT_POE_WEIGHT_TEMP="${COMPAT_POE_WEIGHT_TEMP:-0.5}"
COMPAT_POE_LOSS_TYPE="${COMPAT_POE_LOSS_TYPE:-kl}"
COMPAT_POE_LOGIT_TEMP="${COMPAT_POE_LOGIT_TEMP:-1.0}"

COMPAT_ROUTING_ENABLE="${COMPAT_ROUTING_ENABLE:-0}"
COMPAT_ROUTING_MIN_SCALE="${COMPAT_ROUTING_MIN_SCALE:-0.25}"
COMPAT_ROUTING_MAX_SCALE="${COMPAT_ROUTING_MAX_SCALE:-1.0}"

LAMBDA_LIST="${LAMBDA_LIST:-0.02 0.05 0.10}"
STAMP="$(date +%m%d_%H%M%S)"

mkdir -p "$SAFE_ROOT/$OUTPUT_BASE"

usage() {
  cat <<USAGE
Usage: $0 [mode]

Modes:
  baseline_disjoint   Train interleaved with composition_independent, compat reg OFF
  compreg_disjoint    Train interleaved with composition_independent, compat reg ON
  compreg_samelayer   Train interleaved with composition_study, compat reg ON
  lambda_sweep        Submit disjoint compat runs over LAMBDA_LIST
  all                 Submit baseline_disjoint + compreg_disjoint + compreg_samelayer

Examples:
  $0 compreg_disjoint
  SBATCH_RESOURCES="--gres=gpu:1 --exclude=gpu03" $0 lambda_sweep
USAGE
}

submit_one() {
  local run_key="$1"
  local model_config="$2"
  local compat_enable="$3"
  local compat_lambda="$4"
  local tag="$5"

  local output_dir="$OUTPUT_BASE/${run_key}_${STAMP}"
  local run_name="${RUN_PREFIX}_${run_key}_${STAMP}"

  local exports=(
    "ALL"
    "SAFE_ROOT=$SAFE_ROOT"
    "MODEL_CONFIG=$model_config"
    "OUTPUT_DIR=$output_dir"
    "WANDB=$WANDB"
    "WANDB_PROJECT=$WANDB_PROJECT"
    "WANDB_RUN_NAME=$run_name"
    "WANDB_TAGS=$tag"
    "EPOCHS=$EPOCHS"
    "MAX_SAMPLES=$MAX_SAMPLES"
    "LR=$LR"
    "BATCH_SIZE=$BATCH_SIZE"
    "NUM_AUDIO_TOKENS=$NUM_AUDIO_TOKENS"
    "FUSION_GATE=$FUSION_GATE"
    "SEED=$SEED"
    "LAYER_ADDITIVITY_PROBE=$LAYER_ADDITIVITY_PROBE"
    "LAYER_PROBE_SAMPLES=$LAYER_PROBE_SAMPLES"
    "LAYER_PROBE_EVERY=$LAYER_PROBE_EVERY"
    "COMPAT_REG_ENABLE=$compat_enable"
    "COMPAT_REG_LAMBDA=$compat_lambda"
    "COMPAT_REG_RANK=$COMPAT_REG_RANK"
    "COMPAT_REG_AUDIO_SAMPLES=$COMPAT_REG_AUDIO_SAMPLES"
    "COMPAT_REG_MIN_SAMPLES=$COMPAT_REG_MIN_SAMPLES"
    "COMPAT_REG_REFRESH_EVERY=$COMPAT_REG_REFRESH_EVERY"
    "COMPAT_REG_WEIGHT_BY_SHIFT_NORM=$COMPAT_REG_WEIGHT_BY_SHIFT_NORM"
    "COMPAT_ADD_REG_ENABLE=$COMPAT_ADD_REG_ENABLE"
    "COMPAT_ADD_REG_LAMBDA=$COMPAT_ADD_REG_LAMBDA"
    "COMPAT_ADD_REG_EVERY=$COMPAT_ADD_REG_EVERY"
    "COMPAT_ADD_REG_NORMALIZE=$COMPAT_ADD_REG_NORMALIZE"
    "COMPAT_ADD_BANK_SIZE=$COMPAT_ADD_BANK_SIZE"
    "COMPAT_NOHARM_ENABLE=$COMPAT_NOHARM_ENABLE"
    "COMPAT_NOHARM_LAMBDA=$COMPAT_NOHARM_LAMBDA"
    "COMPAT_NOHARM_MARGIN=$COMPAT_NOHARM_MARGIN"
    "COMPAT_NOHARM_USE_BEST_SINGLE=$COMPAT_NOHARM_USE_BEST_SINGLE"
    "COMPAT_LOGIT_FUSION_ENABLE=$COMPAT_LOGIT_FUSION_ENABLE"
    "COMPAT_LOGIT_FUSION_LAMBDA=$COMPAT_LOGIT_FUSION_LAMBDA"
    "COMPAT_LOGIT_FUSION_CONF_TEMP=$COMPAT_LOGIT_FUSION_CONF_TEMP"
    "COMPAT_POE_ENABLE=$COMPAT_POE_ENABLE"
    "COMPAT_POE_LAMBDA=$COMPAT_POE_LAMBDA"
    "COMPAT_POE_WEIGHT_TEMP=$COMPAT_POE_WEIGHT_TEMP"
    "COMPAT_POE_LOSS_TYPE=$COMPAT_POE_LOSS_TYPE"
    "COMPAT_POE_LOGIT_TEMP=$COMPAT_POE_LOGIT_TEMP"
    "COMPAT_ROUTING_ENABLE=$COMPAT_ROUTING_ENABLE"
    "COMPAT_ROUTING_MIN_SCALE=$COMPAT_ROUTING_MIN_SCALE"
    "COMPAT_ROUTING_MAX_SCALE=$COMPAT_ROUTING_MAX_SCALE"
  )

  if [[ -n "$COMPAT_REG_LAYERS" ]]; then
    exports+=("COMPAT_REG_LAYERS=$COMPAT_REG_LAYERS")
  fi
  if [[ -n "$COMPAT_ADD_REG_LAYERS" ]]; then
    exports+=("COMPAT_ADD_REG_LAYERS=$COMPAT_ADD_REG_LAYERS")
  fi

  local export_str
  export_str="$(IFS=,; echo "${exports[*]}")"

  echo "[submit] mode=$run_key model=$model_config compat=$compat_enable lambda=$compat_lambda"
  echo "[submit] output_dir=$output_dir"
  # shellcheck disable=SC2086
  sbatch $SBATCH_RESOURCES --export="$export_str" "$TRAIN_SCRIPT"
}

case "$MODE" in
  baseline_disjoint)
    submit_one "baseline_disjoint" "composition_independent" "0" "0.0" "composition_baseline"
    ;;
  compreg_disjoint)
    submit_one "compreg_disjoint" "composition_independent" "1" "$COMPAT_REG_LAMBDA" "composition_compreg_disjoint"
    ;;
  compreg_samelayer)
    submit_one "compreg_samelayer" "composition_study" "1" "$COMPAT_REG_LAMBDA" "composition_compreg_samelayer"
    ;;
  lambda_sweep)
    for L in $LAMBDA_LIST; do
      l_key="l$(echo "$L" | tr '.' 'p')"
      submit_one "compreg_disjoint_${l_key}" "composition_independent" "1" "$L" "composition_compreg_sweep"
    done
    ;;
  all)
    submit_one "baseline_disjoint" "composition_independent" "0" "0.0" "composition_baseline"
    submit_one "compreg_disjoint" "composition_independent" "1" "$COMPAT_REG_LAMBDA" "composition_compreg_disjoint"
    submit_one "compreg_samelayer" "composition_study" "1" "$COMPAT_REG_LAMBDA" "composition_compreg_samelayer"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "[error] unknown mode: $MODE" >&2
    usage
    exit 1
    ;;
esac
