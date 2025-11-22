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
RUN_ID=${1:-""}
SPLIT=${2:-"val"}

if [[ -z "$RUN_ID" ]]; then
    echo "Usage: sbatch scripts/run_eval.sh <RUN_ID> [SPLIT]"
    echo "Example: sbatch scripts/run_eval.sh 232228 test"
    exit 1
fi

echo "Starting Evaluation for Run ID: $RUN_ID on Split: $SPLIT"
echo "Date: $(date)"
echo "Node: $(hostname)"

python -u scripts/evaluate_checkpoint.py \
    --run_id "$RUN_ID" \
    --split "$SPLIT" \
    --device cuda

echo "Evaluation complete."
