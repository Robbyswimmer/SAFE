#!/bin/bash -l

#SBATCH --job-name="SAFE-Eval"
#SBATCH --output=logs/eval_%j.txt
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

# Force unbuffered output for Python
export PYTHONUNBUFFERED=1

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

CONDA_ENV=${CONDA_ENV:-safe-env}
echo "Activating conda environment '${CONDA_ENV}'"
conda activate "${CONDA_ENV}"

# Default arguments
RUN_ID_OR_PATH=${1:-""}
SPLIT=${2:-"val"}
MAX_SAMPLES=${3:-""}
# Model configuration overrides.
MODEL_CONFIG=${MODEL_CONFIG:-phase1}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-32}

if [[ -z "$RUN_ID_OR_PATH" ]]; then
    echo "Usage: sbatch scripts/run_eval.sh <RUN_ID_OR_CHECKPOINT_PATH> [SPLIT] [MAX_SAMPLES]"
    echo "Examples:"
    echo "  sbatch scripts/run_eval.sh 232228 test"
    echo "  sbatch scripts/run_eval.sh checkpoints/ablation_6layer/checkpoint_best.pt train 3000"
    exit 1
fi

# Determine if argument is a path or run ID
if [[ -f "$RUN_ID_OR_PATH" ]]; then
    CHECKPOINT_ARG="--checkpoint-path $RUN_ID_OR_PATH"
    echo "Starting Evaluation for Checkpoint: $RUN_ID_OR_PATH on Split: $SPLIT"
else
    CHECKPOINT_ARG="--run_id $RUN_ID_OR_PATH"
    echo "Starting Evaluation for Run ID: $RUN_ID_OR_PATH on Split: $SPLIT"
fi

MAX_SAMPLES_ARG=""
if [[ -n "$MAX_SAMPLES" ]]; then
    MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"
    echo "Max samples: $MAX_SAMPLES"
fi
echo "Date: $(date)"
echo "Node: $(hostname)"

DATA_ROOT=${DATA_ROOT:-"$PWD/experiments/full_training/data"}

python -u scripts/evaluate_checkpoint.py \
    $CHECKPOINT_ARG \
    --split "$SPLIT" \
    --data_root "$DATA_ROOT" \
    --device cuda \
    --model_config "$MODEL_CONFIG" \
    --num_audio_tokens "$NUM_AUDIO_TOKENS" \
    $MAX_SAMPLES_ARG

echo "Evaluation complete."
