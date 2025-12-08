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
LR_PROJECTOR=${LR_PROJECTOR:-2e-4}
LR_ADAPTER=${LR_ADAPTER:-1e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
EVAL_FREQUENCY=${EVAL_FREQUENCY:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-20}
NUM_BEAMS=${NUM_BEAMS:-1}
FP16=${FP16:-"--fp16"}
SEED=${SEED:-42}
USE_WAVCAPS=${USE_WAVCAPS:-0}
WAVCAPS_RATIO=${WAVCAPS_RATIO:-0.8}
AUDIO_CONTRASTIVE_WEIGHT=${AUDIO_CONTRASTIVE_WEIGHT:-0.0}
AUDIO_CONTRASTIVE_TEMPERATURE=${AUDIO_CONTRASTIVE_TEMPERATURE:-0.07}
AUDIO_CONTRASTIVE_MAX_LENGTH=${AUDIO_CONTRASTIVE_MAX_LENGTH:-48}
GATE_WARMUP_STEPS=${GATE_WARMUP_STEPS:-0}

# Memory optimization - enable by default for 48GB GPUs with large datasets
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
NUM_WORKERS=${NUM_WORKERS:-2}  # Reduced from 4 to save ~2GB memory
MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-100}  # Limit eval batches to prevent memory buildup

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
echo "Use WavCaps: ${USE_WAVCAPS} (ratio=${WAVCAPS_RATIO})"
echo "Audio contrastive weight: ${AUDIO_CONTRASTIVE_WEIGHT}"
echo "Gate warmup steps: ${GATE_WARMUP_STEPS}"
echo "Gradient checkpointing: ${GRADIENT_CHECKPOINTING}"
echo "Num workers: ${NUM_WORKERS}"
echo "Max eval batches: ${MAX_EVAL_BATCHES}"
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
    $( [[ "${USE_WAVCAPS}" != "0" ]] && echo --use-wavcaps ) \
    --wavcaps-ratio "${WAVCAPS_RATIO}" \
    --audio-contrastive-weight "${AUDIO_CONTRASTIVE_WEIGHT}" \
    --audio-contrastive-temperature "${AUDIO_CONTRASTIVE_TEMPERATURE}" \
    --audio-contrastive-max-length "${AUDIO_CONTRASTIVE_MAX_LENGTH}" \
    --gate-warmup-steps "${GATE_WARMUP_STEPS}" \
    --num-workers "${NUM_WORKERS}" \
    --max-eval-batches "${MAX_EVAL_BATCHES}" \
    $( [[ "${GRADIENT_CHECKPOINTING}" != "0" ]] && echo --gradient-checkpointing ) \
    ${FP16}

echo ""
echo "========================================"
echo "Training complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"
