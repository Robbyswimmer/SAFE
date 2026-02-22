#!/bin/bash
# Submit a full-data Level-2 run with all available auxiliary objectives enabled.
#
# Usage:
#   bash experiments/avqa_composition/scripts/submit_level2_full_all_objectives.sh
#
# Optional overrides (example):
#   SEED=7 EPOCHS=5 COMPAT_POE_LAMBDA=0.01 \
#   bash experiments/avqa_composition/scripts/submit_level2_full_all_objectives.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/submit_level2_full_config.sh"

if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "[error] base submit script not found: $BASE_SCRIPT" >&2
  exit 1
fi

# Full objective stack defaults (can be overridden by caller env).
export COMPAT_NOHARM_ENABLE="${COMPAT_NOHARM_ENABLE:-1}"
export COMPAT_NOHARM_LAMBDA="${COMPAT_NOHARM_LAMBDA:-0.02}"
export COMPAT_NOHARM_USE_BEST_SINGLE="${COMPAT_NOHARM_USE_BEST_SINGLE:-1}"

export COMPAT_LOGIT_FUSION_ENABLE="${COMPAT_LOGIT_FUSION_ENABLE:-1}"
export COMPAT_LOGIT_FUSION_LAMBDA="${COMPAT_LOGIT_FUSION_LAMBDA:-0.02}"
export COMPAT_LOGIT_FUSION_CONF_TEMP="${COMPAT_LOGIT_FUSION_CONF_TEMP:-0.5}"

export COMPAT_POE_ENABLE="${COMPAT_POE_ENABLE:-1}"
export COMPAT_POE_LAMBDA="${COMPAT_POE_LAMBDA:-0.02}"
export COMPAT_POE_WEIGHT_TEMP="${COMPAT_POE_WEIGHT_TEMP:-0.5}"
export COMPAT_POE_LOSS_TYPE="${COMPAT_POE_LOSS_TYPE:-kl}"
export COMPAT_POE_LOGIT_TEMP="${COMPAT_POE_LOGIT_TEMP:-1.0}"

export COMPAT_ROUTING_ENABLE="${COMPAT_ROUTING_ENABLE:-1}"
export COMPAT_ROUTING_MIN_SCALE="${COMPAT_ROUTING_MIN_SCALE:-0.25}"
export COMPAT_ROUTING_MAX_SCALE="${COMPAT_ROUTING_MAX_SCALE:-1.0}"

export WANDB_RUN_NAME="${WANDB_RUN_NAME:-comp_level2_full32k_all_objectives_s${SEED:-42}_$(date +%m%d_%H%M)}"
export WANDB_TAGS="${WANDB_TAGS:-composition,level2,full32k,all_objectives,poe,noharm,logit_fusion,routing,transport_bound}"

exec "$BASE_SCRIPT"
