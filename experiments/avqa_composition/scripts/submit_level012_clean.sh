#!/bin/bash
# Queue clean Level-0/1/2 runs — derivation-only objectives, no extras.
# Level 0: diagnostics (m*, rho_l, subspace overlap) — zero training
# Level 1: gates-only + task loss — no auxiliary objectives
# Level 2: subspace avoidance (Eq. 24) + task loss — no PoE, no noharm, no routing
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

LEVEL0_MAX_SAMPLES=${LEVEL0_MAX_SAMPLES:-1000}
LEVEL0_PROBE_SAMPLES=${LEVEL0_PROBE_SAMPLES:-512}
LEVEL1_MAX_SAMPLES=${LEVEL1_MAX_SAMPLES:-1000}
LEVEL2_MAX_SAMPLES=${LEVEL2_MAX_SAMPLES:-1000}

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "FATAL: BASE_CKPT not found: $BASE_CKPT" >&2
  exit 2
fi

echo "[submit-clean] SAFE_ROOT=$SAFE_ROOT"
echo "[submit-clean] BASE_CKPT=$BASE_CKPT"
echo "[submit-clean] MODEL_CONFIG=$MODEL_CONFIG"

# -------------------------
# Level 0: zero-training diagnostics (already clean)
# -------------------------
J0=$(sbatch --parsable --gres=gpu:1 \
  --export=ALL,SAFE_ROOT="$SAFE_ROOT",MODEL_CONFIG="$MODEL_CONFIG",COMPOSE_AUDIO_CKPT="$BASE_CKPT",COMPOSE_VISION_CKPT="$BASE_CKPT",OUTPUT_DIR=checkpoints/level0_clean,OUTPUT_JSON=checkpoints/level0_clean/diag.json,MAX_SAMPLES="$LEVEL0_MAX_SAMPLES",PROBE_SAMPLES="$LEVEL0_PROBE_SAMPLES",FUSION_GATE="$FUSION_GATE",SEED="$SEED",SLIM_PROJECTOR=0 \
  experiments/avqa_composition/scripts/run_composability_diagnostics.sh)

echo "[submit-clean] Level-0 diagnostics job: $J0"

# -------------------------
# Level 1: gates-only + task loss ONLY
# No gate-add, no noharm, no PoE, no routing
# -------------------------
J1=$(sbatch --parsable --gres=gpu:1 \
  --export=ALL,\
SAFE_ROOT="$SAFE_ROOT",\
MODEL_CONFIG="$MODEL_CONFIG",\
OUTPUT_DIR=checkpoints/level1_clean,\
SEED="$SEED",\
FUSION_GATE="$FUSION_GATE",\
SLIM_PROJECTOR=0,\
INIT_AUDIO_CKPT="$BASE_CKPT",\
INIT_VISION_CKPT="$BASE_CKPT",\
EPOCHS=2,\
LR=1e-3,\
MAX_SAMPLES="$LEVEL1_MAX_SAMPLES",\
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
WANDB_RUN_NAME=level1_clean_${SEED},\
WANDB_TAGS=composition,level1,clean,gates_only,derivation \
  experiments/avqa_composition/scripts/train_composition_interleaved.sh)

echo "[submit-clean] Level-1 gates-only (clean) job: $J1"

# -------------------------
# Level 2: subspace avoidance (Eq. 24) + task loss ONLY
# COMPAT_REG = subspace regularizer from derivation
# Everything else OFF
# -------------------------
J2=$(sbatch --parsable --gres=gpu:1 \
  --export=ALL,\
SAFE_ROOT="$SAFE_ROOT",\
MODEL_CONFIG="$MODEL_CONFIG",\
OUTPUT_DIR=checkpoints/level2_clean,\
SEED="$SEED",\
FUSION_GATE="$FUSION_GATE",\
SLIM_PROJECTOR=0,\
INIT_AUDIO_CKPT="$BASE_CKPT",\
INIT_VISION_CKPT="$BASE_CKPT",\
EPOCHS=3,\
LR=5e-5,\
MAX_SAMPLES="$LEVEL2_MAX_SAMPLES",\
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

echo "[submit-clean] Level-2 subspace avoidance (clean) job: $J2"

echo ""
echo "[submit-clean] queued: level0=$J0 level1=$J1 level2=$J2"
echo "[submit-clean] monitor: squeue -j $J0,$J1,$J2"
