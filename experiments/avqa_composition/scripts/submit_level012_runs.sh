#!/bin/bash
# Queue Level-0/1/2 composability runs from a single base checkpoint.
set -euo pipefail

if [[ -z "${SAFE_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    SAFE_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SAFE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
  fi
fi
cd "$SAFE_ROOT"

BASE_CKPT=${BASE_CKPT:-checkpoints/comp_indep_additive_v1/best_model.pt}
MODEL_CONFIG=${MODEL_CONFIG:-composition_independent}
SEED=${SEED:-42}
FUSION_GATE=${FUSION_GATE:-0.2}

# Pilot defaults (quick empirical read). Set to 0 for full dataset.
LEVEL0_MAX_SAMPLES=${LEVEL0_MAX_SAMPLES:-1000}
LEVEL0_PROBE_SAMPLES=${LEVEL0_PROBE_SAMPLES:-512}
LEVEL1_MAX_SAMPLES=${LEVEL1_MAX_SAMPLES:-1000}
LEVEL2_MAX_SAMPLES=${LEVEL2_MAX_SAMPLES:-1000}

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "FATAL: BASE_CKPT not found: $BASE_CKPT" >&2
  exit 2
fi

echo "[submit] SAFE_ROOT=$SAFE_ROOT"
echo "[submit] BASE_CKPT=$BASE_CKPT"
echo "[submit] MODEL_CONFIG=$MODEL_CONFIG"

# -------------------------
# Level 0: zero-training diagnostics
# -------------------------
J0=$(sbatch --parsable --gres=gpu:1 \
  --export=ALL,SAFE_ROOT="$SAFE_ROOT",MODEL_CONFIG="$MODEL_CONFIG",COMPOSE_AUDIO_CKPT="$BASE_CKPT",COMPOSE_VISION_CKPT="$BASE_CKPT",OUTPUT_DIR=checkpoints/composability_diagnostics/level0,OUTPUT_JSON=checkpoints/composability_diagnostics/level0/diag.json,MAX_SAMPLES="$LEVEL0_MAX_SAMPLES",PROBE_SAMPLES="$LEVEL0_PROBE_SAMPLES",FUSION_GATE="$FUSION_GATE",SEED="$SEED",SLIM_PROJECTOR=0 \
  experiments/avqa_composition/scripts/run_composability_diagnostics.sh)

echo "[submit] Level-0 diagnostics job: $J0"

# -------------------------
# Level 1: cheap calibration (gates-only)
# -------------------------
J1=$(sbatch --parsable --gres=gpu:1 \
  --export=ALL,SAFE_ROOT="$SAFE_ROOT",MODEL_CONFIG="$MODEL_CONFIG",OUTPUT_DIR=checkpoints/composition_level1_calibration,SEED="$SEED",FUSION_GATE="$FUSION_GATE",SLIM_PROJECTOR=0,INIT_AUDIO_CKPT="$BASE_CKPT",INIT_VISION_CKPT="$BASE_CKPT",EPOCHS=2,LR=1e-3,MAX_SAMPLES="$LEVEL1_MAX_SAMPLES",LEARNED_GATE=1,LEARNED_GATE_INIT=2.0,TRAIN_GATES_ONLY=1,COMPAT_GATE_ADD_ENABLE=1,COMPAT_GATE_ADD_LAMBDA=0.05,COMPAT_GATE_PAIRING=zip,COMPAT_GATE_TARGET_MODE=inverse_rho,COMPAT_GATE_RHO_BETA=2.0,COMPAT_GATE_PRODUCT_TARGET=-1,COMPAT_GATE_MIN_EFFECTIVE=0.02,COMPAT_GATE_FLOOR_LAMBDA=0.10,COMPAT_NOHARM_ENABLE=1,COMPAT_NOHARM_LAMBDA=0.01,COMPAT_NOHARM_USE_BEST_SINGLE=1,COMPAT_POE_ENABLE=1,COMPAT_POE_LAMBDA=0.01,COMPAT_POE_WEIGHT_TEMP=0.5,COMPAT_POE_LOSS_TYPE=kl,COMPAT_POE_LOGIT_TEMP=1.0,COMPAT_ROUTING_ENABLE=0,WANDB_RUN_NAME=level1_calibration_${SEED},WANDB_TAGS=composition,level1,calibration,gates_only,gate_add \
  experiments/avqa_composition/scripts/train_composition_interleaved.sh)

echo "[submit] Level-1 calibration job: $J1"

# -------------------------
# Level 2: full informed independent training objectives
# -------------------------
J2=$(sbatch --parsable --gres=gpu:1 \
  --export=ALL,SAFE_ROOT="$SAFE_ROOT",MODEL_CONFIG="$MODEL_CONFIG",OUTPUT_DIR=checkpoints/composition_level2_full,SEED="$SEED",FUSION_GATE="$FUSION_GATE",SLIM_PROJECTOR=0,INIT_AUDIO_CKPT="$BASE_CKPT",INIT_VISION_CKPT="$BASE_CKPT",EPOCHS=3,MAX_SAMPLES="$LEVEL2_MAX_SAMPLES",LEARNED_GATE=1,LEARNED_GATE_INIT=2.0,COMPAT_REG_ENABLE=1,COMPAT_REG_LAMBDA=0.05,COMPAT_REG_RANK=8,COMPAT_REG_AUDIO_SAMPLES=256,COMPAT_REG_WEIGHT_BY_SHIFT_NORM=1,COMPAT_ADD_REG_ENABLE=1,COMPAT_ADD_REG_LAMBDA=0.01,COMPAT_ADD_REG_EVERY=200,COMPAT_ADD_REG_NORMALIZE=1,COMPAT_GATE_ADD_ENABLE=1,COMPAT_GATE_ADD_LAMBDA=0.01,COMPAT_GATE_PAIRING=zip,COMPAT_GATE_TARGET_MODE=inverse_rho,COMPAT_GATE_RHO_BETA=2.0,COMPAT_GATE_PRODUCT_TARGET=-1,COMPAT_NOHARM_ENABLE=1,COMPAT_NOHARM_LAMBDA=0.02,COMPAT_NOHARM_USE_BEST_SINGLE=1,COMPAT_POE_ENABLE=1,COMPAT_POE_LAMBDA=0.02,COMPAT_POE_WEIGHT_TEMP=0.5,COMPAT_POE_LOSS_TYPE=kl,COMPAT_POE_LOGIT_TEMP=1.0,COMPAT_ROUTING_ENABLE=1,COMPAT_ROUTING_MIN_SCALE=0.25,COMPAT_ROUTING_MAX_SCALE=1.0,WANDB_RUN_NAME=level2_full_${SEED},WANDB_TAGS=composition,level2,full_objective,gate_add \
  experiments/avqa_composition/scripts/train_composition_interleaved.sh)

echo "[submit] Level-2 full job: $J2"

echo "[submit] queued: level0=$J0 level1=$J1 level2=$J2"
echo "[submit] monitor: squeue -j $J0,$J1,$J2"
