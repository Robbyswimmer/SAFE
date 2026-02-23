#!/bin/bash
# Submit Level-1 gates-only training (clean, derivation-only).
# Task loss + learned gates — no auxiliary objectives.
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
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/level1_gates_only}

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "FATAL: BASE_CKPT not found: $BASE_CKPT" >&2
  exit 2
fi

echo "[level1] SAFE_ROOT=$SAFE_ROOT"
echo "[level1] BASE_CKPT=$BASE_CKPT"
echo "[level1] MODEL_CONFIG=$MODEL_CONFIG"
echo "[level1] OUTPUT_DIR=$OUTPUT_DIR"

J1=$(sbatch --parsable --gres=gpu:1 \
  --export=ALL,\
SAFE_ROOT="$SAFE_ROOT",\
MODEL_CONFIG="$MODEL_CONFIG",\
OUTPUT_DIR="$OUTPUT_DIR",\
SEED="$SEED",\
FUSION_GATE="$FUSION_GATE",\
SLIM_PROJECTOR=0,\
INIT_AUDIO_CKPT="$BASE_CKPT",\
INIT_VISION_CKPT="$BASE_CKPT",\
EPOCHS=2,\
LR=1e-3,\
MAX_SAMPLES="$MAX_SAMPLES",\
LEARNED_GATE=1,\
LEARNED_GATE_INIT=2.0,\
TRAIN_GATES_ONLY=1,\
COMPAT_REG_ENABLE=0,\
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
WANDB_RUN_NAME=level1_gates_only_${SEED},\
WANDB_TAGS=composition,level1,clean,gates_only,derivation \
  experiments/avqa_composition/scripts/train_composition_interleaved.sh)

echo "[level1] submitted job: $J1"
echo "[level1] monitor: squeue -j $J1"
