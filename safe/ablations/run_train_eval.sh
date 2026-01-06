#!/bin/bash
#
# Evaluate SAFE checkpoint on training data
#
# Usage:
#   CHECKPOINT=/path/to/checkpoint.pt bash safe/ablations/run_train_eval.sh
#   sbatch safe/ablations/run_train_eval.sh

#SBATCH --job-name=SAFE-TrainEval
#SBATCH --output=logs/train_eval_%j.txt
#SBATCH --error=logs/train_eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH -p gpu

set -euo pipefail

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

# Configuration
CHECKPOINT=${CHECKPOINT:-""}
DATA_PATH=${DATA_PATH:-"$PWD/experiments/full_training/data"}
CONFIG=${CONFIG:-"phase1"}
SPLIT=${SPLIT:-"train"}
MAX_SAMPLES=${MAX_SAMPLES:-2000}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-30}
NUM_BEAMS=${NUM_BEAMS:-1}

# Validate checkpoint
if [[ -z "${CHECKPOINT}" ]]; then
    echo "ERROR: CHECKPOINT not set. Usage: CHECKPOINT=/path/to/checkpoint.pt bash $0"
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Log configuration
echo "========================================"
echo "SAFE Training Data Evaluation"
echo "========================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data path: ${DATA_PATH}"
echo "Config: ${CONFIG}"
echo "Split: ${SPLIT}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Num beams: ${NUM_BEAMS}"
echo "========================================"
echo ""

# Run evaluation
python safe/ablations/eval_train_metrics.py \
    --checkpoint "${CHECKPOINT}" \
    --data_path "${DATA_PATH}" \
    --config "${CONFIG}" \
    --split "${SPLIT}" \
    --max_samples "${MAX_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --num_beams "${NUM_BEAMS}"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "========================================"
