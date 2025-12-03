#!/bin/bash -l

# Heavier Stage-A-style evaluation (uses StageATrainer + CLAP reranking).
# Requests more memory than the lightweight eval script.

#SBATCH --job-name="SAFE-Eval-StageA"
#SBATCH --output=logs/eval_stagea_%j.txt
#SBATCH --error=logs/eval_stagea_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail
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

RUN_ID=${1:-""}
SPLIT=${2:-"test"}

if [[ -z "$RUN_ID" ]]; then
    echo "Usage: sbatch scripts/run_eval_stagea.sh <RUN_ID> [SPLIT]"
    echo "Example: sbatch scripts/run_eval_stagea.sh 232363 test"
    exit 1
fi

echo "Starting Stage-A style Evaluation for Run ID: $RUN_ID on Split: $SPLIT"
echo "Date: $(date)"
echo "Node: $(hostname)"

DATA_ROOT=${DATA_ROOT:-"$PWD/experiments/full_training/data"}
MODEL_CONFIG=${MODEL_CONFIG:-phase1}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-32}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_AUDIO_EVAL_SAMPLES=${MAX_AUDIO_EVAL_SAMPLES:-800}

python -u scripts/evaluate_stagea_checkpoint.py \
    --run_id "$RUN_ID" \
    --split "$SPLIT" \
    --data_root "$DATA_ROOT" \
    --device cuda \
    --model_config "$MODEL_CONFIG" \
    --num_audio_tokens "$NUM_AUDIO_TOKENS" \
    --batch_size "$BATCH_SIZE" \
    --max_audio_eval_samples "$MAX_AUDIO_EVAL_SAMPLES"

echo "Stage-A style evaluation complete."

