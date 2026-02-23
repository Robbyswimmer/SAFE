#!/bin/bash
# Submit unpaired ICM composition run (frozen LLM, unimodal-safe, no joint data).
#
# Usage:
#   bash experiments/avqa_composition/scripts/submit_icm_unpaired.sh
#
# Optional overrides:
#   SEED=7 EPOCHS=15 OUTPUT_DIR=checkpoints/comp_icm_s7 \
#   bash experiments/avqa_composition/scripts/submit_icm_unpaired.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/submit_level2_full_config.sh"

if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "[error] base submit script not found: $BASE_SCRIPT" >&2
  exit 1
fi

# Core run setup
export MODEL_CONFIG="${MODEL_CONFIG:-composition_independent}"
export EPOCHS="${EPOCHS:-10}"
export LR="${LR:-5e-5}"
export OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/composition_icm_unpaired}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}"

# Keep existing unimodal adapters and train composition sidecar/objectives.
export INIT_AUDIO_CKPT="${INIT_AUDIO_CKPT:-checkpoints/comp_indep_additive_v1/best_model.pt}"
export INIT_VISION_CKPT="${INIT_VISION_CKPT:-checkpoints/comp_indep_additive_v1/best_model.pt}"

# Architecture: enable ICM (set-style mixer), no hard transport constraints.
export ICM_ENABLE="${ICM_ENABLE:-1}"
export ICM_DIM="${ICM_DIM:-512}"
export ICM_HEADS="${ICM_HEADS:-8}"
export ICM_LAYERS="${ICM_LAYERS:-1}"
export ICM_DROPOUT="${ICM_DROPOUT:-0.1}"
export ICM_GATE_INIT="${ICM_GATE_INIT:--2.0}"
export ICM_MIN_MODALITIES="${ICM_MIN_MODALITIES:-2}"
export ICM_UTIL_TARGET="${ICM_UTIL_TARGET:-0.7}"
export DELTA_NORM_CAP_RATIO="${DELTA_NORM_CAP_RATIO:-0.0}"
export AUDIO_GATE_DEPTH_DECAY="${AUDIO_GATE_DEPTH_DECAY:-1.0}"
export VISION_GATE_DEPTH_DECAY="${VISION_GATE_DEPTH_DECAY:-1.0}"

# Unpaired objectives (ICM-centric)
export COMPAT_REG_ENABLE="${COMPAT_REG_ENABLE:-0}"
export COMPAT_ADD_REG_ENABLE="${COMPAT_ADD_REG_ENABLE:-1}"
export COMPAT_ADD_REG_LAMBDA="${COMPAT_ADD_REG_LAMBDA:-0.01}"
export COMPAT_ADD_REG_EVERY="${COMPAT_ADD_REG_EVERY:-100}"
export COMPAT_TRANSPORT_ENABLE="${COMPAT_TRANSPORT_ENABLE:-1}"
export COMPAT_TRANSPORT_LAMBDA="${COMPAT_TRANSPORT_LAMBDA:-0.02}"
export COMPAT_ICM_CANCEL_ENABLE="${COMPAT_ICM_CANCEL_ENABLE:-1}"
export COMPAT_ICM_CANCEL_LAMBDA="${COMPAT_ICM_CANCEL_LAMBDA:-0.02}"
export COMPAT_ICM_CANCEL_LOSS_TYPE="${COMPAT_ICM_CANCEL_LOSS_TYPE:-mse}"
export COMPAT_ICM_CANCEL_START_STEP="${COMPAT_ICM_CANCEL_START_STEP:-500}"
export COMPAT_ICM_IDENTITY_LAMBDA="${COMPAT_ICM_IDENTITY_LAMBDA:-0.01}"
export COMPAT_ICM_SMALL_LAMBDA="${COMPAT_ICM_SMALL_LAMBDA:-0.001}"
export COMPAT_ICM_UTIL_LAMBDA="${COMPAT_ICM_UTIL_LAMBDA:-0.005}"
export COMPAT_ICM_UTIL_START_STEP="${COMPAT_ICM_UTIL_START_STEP:-1000}"
export COMPAT_NOHARM_ENABLE="${COMPAT_NOHARM_ENABLE:-1}"
export COMPAT_NOHARM_LAMBDA="${COMPAT_NOHARM_LAMBDA:-0.02}"
export COMPAT_ICM_NOHARM_START_STEP="${COMPAT_ICM_NOHARM_START_STEP:-1000}"

# Disable unrelated objectives by default for clean ablation.
export COMPAT_GATE_ADD_ENABLE="${COMPAT_GATE_ADD_ENABLE:-0}"
export COMPAT_LOGIT_FUSION_ENABLE="${COMPAT_LOGIT_FUSION_ENABLE:-0}"
export COMPAT_POE_ENABLE="${COMPAT_POE_ENABLE:-0}"
export COMPAT_ROUTING_ENABLE="${COMPAT_ROUTING_ENABLE:-0}"

export WANDB_RUN_NAME="${WANDB_RUN_NAME:-comp_icm_unpaired_s${SEED:-42}_$(date +%m%d_%H%M)}"
export WANDB_TAGS="${WANDB_TAGS:-composition,icm,unpaired,no_joint_data,sidecar_mixer}"

exec "$BASE_SCRIPT"
