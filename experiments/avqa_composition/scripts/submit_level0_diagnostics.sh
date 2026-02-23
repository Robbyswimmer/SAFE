#!/bin/bash
# Submit Level-0 diagnostics only (zero training).
# Measures: m*, rho_l, shift norms, subspace overlap, additivity probe.
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
PROBE_SAMPLES=${PROBE_SAMPLES:-512}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/level0_diagnostics}

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "FATAL: BASE_CKPT not found: $BASE_CKPT" >&2
  exit 2
fi

echo "[level0] SAFE_ROOT=$SAFE_ROOT"
echo "[level0] BASE_CKPT=$BASE_CKPT"
echo "[level0] MODEL_CONFIG=$MODEL_CONFIG"
echo "[level0] OUTPUT_DIR=$OUTPUT_DIR"

J0=$(sbatch --parsable --gres=gpu:1 \
  --export=ALL,SAFE_ROOT="$SAFE_ROOT",MODEL_CONFIG="$MODEL_CONFIG",COMPOSE_AUDIO_CKPT="$BASE_CKPT",COMPOSE_VISION_CKPT="$BASE_CKPT",OUTPUT_DIR="$OUTPUT_DIR",OUTPUT_JSON="$OUTPUT_DIR/diag.json",MAX_SAMPLES="$MAX_SAMPLES",PROBE_SAMPLES="$PROBE_SAMPLES",FUSION_GATE="$FUSION_GATE",SEED="$SEED",SLIM_PROJECTOR=0 \
  experiments/avqa_composition/scripts/run_composability_diagnostics.sh)

echo "[level0] submitted job: $J0"
echo "[level0] monitor: squeue -j $J0"
