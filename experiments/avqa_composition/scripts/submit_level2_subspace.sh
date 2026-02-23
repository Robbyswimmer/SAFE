#!/bin/bash
# Submit Level-2 subspace avoidance (Eq. 24) + task loss ONLY (clean).
# No PoE, no noharm, no routing — derivation-only.
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
MAX_SAMPLES=${MAX_SAMPLES:-1000}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/level2_clean}

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "FATAL: BASE_CKPT not found: $BASE_CKPT" >&2
  exit 2
fi

echo "[level2] SAFE_ROOT=$SAFE_ROOT"
echo "[level2] BASE_CKPT=$BASE_CKPT"
echo "[level2] MODEL_CONFIG=$MODEL_CONFIG"
echo "[level2] OUTPUT_DIR=$OUTPUT_DIR"

J2=$(sbatch --parsable --gres=gpu:1 \
  --export=ALL,\
SAFE_ROOT="$SAFE_ROOT",\
MODEL_CONFIG="$MODEL_CONFIG",\
OUTPUT_DIR="$OUTPUT_DIR",\
SEED="$SEED",\
FUSION_GATE="$FUSION_GATE",\
SLIM_PROJECTOR=0,\
INIT_AUDIO_CKPT="$BASE_CKPT",\
INIT_VISION_CKPT="$BASE_CKPT",\
EPOCHS=3,\
LR=5e-5,\
MAX_SAMPLES="$MAX_SAMPLES",\
LEARNED_GATE=1,\
LEARNED_GATE_INIT=2.0,\
COMPAT_REG_ENABLE=1,\
COMPAT_REG_LAMBDA=0.05,\
COMPAT_REG_RANK=8,\
COMPAT_REG_AUDIO_SAMPLES=256,\
COMPAT_REG_WEIGHT_BY_SHIFT_NORM=1,\
COMPAT_ADD_REG_ENABLE=0,\
COMPAT_TRANSPORT_ENABLE=0,\
COMPAT_GATE_ADD_ENABLE=0,\
COMPAT_NOHARM_ENABLE=0,\
COMPAT_POE_ENABLE=0,\
COMPAT_ROUTING_ENABLE=0,\
COMPAT_LOGIT_FUSION_ENABLE=0,\
COMPAT_ICM_CANCEL_ENABLE=0,\
EVAL_MODALITIES=text,audio,image,both,\
LAYER_ADDITIVITY_PROBE=1,\
LAYER_PROBE_SAMPLES=256,\
WANDB_RUN_NAME=level2_clean_${SEED},\
WANDB_TAGS=composition,level2,clean,subspace_avoidance,derivation \
  experiments/avqa_composition/scripts/train_composition_interleaved.sh)

echo "[level2] submitted job: $J2"
echo "[level2] monitor: squeue -j $J2"
