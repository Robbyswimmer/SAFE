#!/bin/bash
#
# Point Cloud Training Script - SAFE Architecture on Point Clouds
#
# Usage:
#   bash scripts/train_pointcloud.sh                    # Single GPU
#   sbatch --gres=gpu:1 scripts/train_pointcloud.sh     # SLURM

#SBATCH --job-name=SAFE-PointCloud
#SBATCH --output=logs/pointcloud_%j.txt
#SBATCH --error=logs/pointcloud_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH -p gpu

set -euo pipefail

# Environment setup
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

# Configuration (override via environment variables)
CONFIG=${CONFIG:-"modelnet40"}
PHASE=${PHASE:-"classification"}
DATA_PATH=${DATA_PATH:-"data"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/pointcloud"}
NUM_EPOCHS=${NUM_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-4}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-8}
WARMUP_STEPS=${WARMUP_STEPS:-100}
EVAL_EVERY=${EVAL_EVERY:-1}
MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-50}
SAVE_EVERY=${SAVE_EVERY:-5}
NUM_WORKERS=${NUM_WORKERS:-4}
FP16=${FP16:-0}
DEBUG=${DEBUG:-0}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-""}
POINTBERT_CHECKPOINT=${POINTBERT_CHECKPOINT:-"checkpoints/pointbert/pointbert_shapenet.pt"}

# Create logs directory
mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "SAFE Point Cloud Training"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "========================================"
echo "Config: ${CONFIG}"
echo "Phase: ${PHASE}"
echo "Data path: ${DATA_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LR}"
echo "Gradient accumulation: ${GRADIENT_ACCUMULATION}"
echo "PointBERT checkpoint: ${POINTBERT_CHECKPOINT}"
echo "========================================"

# Build command
CMD="python train_pointcloud.py"
CMD="$CMD --config ${CONFIG}"
CMD="$CMD --phase ${PHASE}"
CMD="$CMD --data-path ${DATA_PATH}"
CMD="$CMD --output-dir ${OUTPUT_DIR}"
CMD="$CMD --num-epochs ${NUM_EPOCHS}"
CMD="$CMD --batch-size ${BATCH_SIZE}"
CMD="$CMD --lr ${LR}"
CMD="$CMD --gradient-accumulation ${GRADIENT_ACCUMULATION}"
CMD="$CMD --warmup-steps ${WARMUP_STEPS}"
CMD="$CMD --eval-every ${EVAL_EVERY}"
CMD="$CMD --max-eval-batches ${MAX_EVAL_BATCHES}"
CMD="$CMD --save-every ${SAVE_EVERY}"
CMD="$CMD --num-workers ${NUM_WORKERS}"

if [[ "${FP16}" == "1" ]]; then
  CMD="$CMD --fp16"
fi

if [[ "${DEBUG}" == "1" ]]; then
  CMD="$CMD --debug"
fi

if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  CMD="$CMD --max-train-samples ${MAX_TRAIN_SAMPLES}"
fi

if [[ -f "${POINTBERT_CHECKPOINT}" ]]; then
  CMD="$CMD --encoder-checkpoint ${POINTBERT_CHECKPOINT}"
  echo "Using PointBERT checkpoint: ${POINTBERT_CHECKPOINT}"
else
  echo "No PointBERT checkpoint found, training encoder from scratch"
fi

echo "Running: ${CMD}"
echo "========================================"

# Run training
$CMD

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
