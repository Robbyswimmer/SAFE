#!/bin/bash -l

#SBATCH --job-name="SAFE-SCST"
#SBATCH --output=logs/scst_%j.out
#SBATCH --error=logs/scst_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH -p gpu

set -euo pipefail

# Force unbuffered output for Python
export PYTHONUNBUFFERED=1

# Activate conda environment
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

CONDA_ENV=${CONDA_ENV:-safe-env}
echo "Activating conda environment '${CONDA_ENV}'"
conda activate "${CONDA_ENV}"

# ============================================================================
# SCST (Self-Critical Sequence Training) for Audio Captioning
# ============================================================================
#
# This script fine-tunes a trained SAFE model using reinforcement learning
# with CIDEr as the reward signal.
#
# Usage:
#   CHECKPOINT=path/to/checkpoint.pt sbatch scripts/run_scst.sh
#
# Or run directly:
#   CHECKPOINT=path/to/checkpoint.pt bash scripts/run_scst.sh
#
# ============================================================================

set -e

# Configuration with defaults
CHECKPOINT=${CHECKPOINT:-"checkpoints/bottleneck_baseline/checkpoint_best.pt"}
FUSION_LAYER_INDICES=${FUSION_LAYER_INDICES:-"12,24,36"}
DATA_PATH=${DATA_PATH:-"experiments/full_training/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/scst"}
CONFIG=${CONFIG:-"phase1"}

# SCST hyperparameters
NUM_EPOCHS=${NUM_EPOCHS:-5}
LR=${LR:-1e-5}
TEMPERATURE=${TEMPERATURE:-0.7}
NUM_SAMPLES=${NUM_SAMPLES:-5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-20}

# Training configuration
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-16}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
FP16=${FP16:-"--fp16"}
SEED=${SEED:-42}

# Evaluation
EVAL_FREQUENCY=${EVAL_FREQUENCY:-1}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-500}

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

echo "========================================"
echo "SCST Training for Audio Captioning"
echo "========================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Fusion layers: ${FUSION_LAYER_INDICES}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Temperature: ${TEMPERATURE}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Learning rate: ${LR}"
echo "Epochs: ${NUM_EPOCHS}"
echo "========================================"

# Verify checkpoint exists
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

# Verify data path exists
if [[ ! -d "${DATA_PATH}" ]]; then
    echo "ERROR: Data path not found: ${DATA_PATH}"
    exit 1
fi

# Build fusion layer args
FUSION_ARGS=""
if [[ -n "${FUSION_LAYER_INDICES}" ]]; then
    FUSION_ARGS="--fusion-layer-indices ${FUSION_LAYER_INDICES}"
fi

# Run training
python scripts/train_scst.py \
    --checkpoint "${CHECKPOINT}" \
    --data-path "${DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --config "${CONFIG}" \
    ${FUSION_ARGS} \
    --num-epochs "${NUM_EPOCHS}" \
    --lr "${LR}" \
    --temperature "${TEMPERATURE}" \
    --num-samples "${NUM_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation "${GRADIENT_ACCUMULATION}" \
    --max-grad-norm "${MAX_GRAD_NORM}" \
    --eval-frequency "${EVAL_FREQUENCY}" \
    --max-eval-samples "${MAX_EVAL_SAMPLES}" \
    --seed "${SEED}" \
    ${FP16}

echo ""
echo "========================================"
echo "SCST Training Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"
