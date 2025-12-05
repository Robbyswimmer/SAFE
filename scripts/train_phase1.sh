#!/bin/bash
#
# Phase 1 Training Script - Clean SAFE Training
#
# Usage:
#   bash scripts/train_phase1.sh
#   sbatch scripts/train_phase1.sh  # For SLURM

#SBATCH --job-name=SAFE-Train
#SBATCH --output=logs/train_%j.txt
#SBATCH --error=logs/train_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH -p gpu

set -euo pipefail

# Environment setup - activate conda environment
CONDA_ENV=${CONDA_ENV:-"safe-env"}

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

echo "Activating conda environment '${CONDA_ENV}'"
conda activate "${CONDA_ENV}"

# Verify environment
python --version
which python

# Configuration
MODEL_CONFIG=${MODEL_CONFIG:-"phase1"}
DATA_PATH=${DATA_PATH:-"$PWD/experiments/full_training/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/phase1_clean"}
NUM_EPOCHS=${NUM_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-4}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-32}
LR_PROJECTOR=${LR_PROJECTOR:-1e-3}
LR_ADAPTER=${LR_ADAPTER:-5e-4}
WARMUP_STEPS=${WARMUP_STEPS:-500}
EVAL_FREQUENCY=${EVAL_FREQUENCY:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-20}
NUM_BEAMS=${NUM_BEAMS:-1}
FP16=${FP16:-"--fp16"}
SEED=${SEED:-42}

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# Verify data path exists
if [[ ! -d "${DATA_PATH}" ]]; then
  echo "ERROR: Data path not found: ${DATA_PATH}" >&2
  echo "Please set DATA_PATH environment variable to your data directory" >&2
  exit 1
fi

# Log configuration
echo "========================================"
echo "SAFE Training Configuration"
echo "========================================"
echo "Model config: ${MODEL_CONFIG}"
echo "Data path: ${DATA_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Gradient accumulation: ${GRADIENT_ACCUMULATION}"
echo "Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "LR projector: ${LR_PROJECTOR}"
echo "LR adapter: ${LR_ADAPTER}"
echo "Warmup steps: ${WARMUP_STEPS}"
echo "Mixed precision: ${FP16}"
echo "Seed: ${SEED}"
echo "========================================"
echo ""

# Run training
python train_safe.py \
    --model-config "${MODEL_CONFIG}" \
    --data-path "${DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-epochs "${NUM_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION}" \
    --learning-rate-projector "${LR_PROJECTOR}" \
    --learning-rate-adapter "${LR_ADAPTER}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --eval-frequency "${EVAL_FREQUENCY}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --num-beams "${NUM_BEAMS}" \
    --seed "${SEED}" \
    ${FP16}

echo ""
echo "========================================"
echo "Training complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"
