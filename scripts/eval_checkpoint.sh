#!/bin/bash
#SBATCH --job-name=eval-ckpt
#SBATCH --output=logs/eval_%j.txt
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH -p gpu

set -euo pipefail

CHECKPOINT_PATH=${1:-""}
SPLIT=${2:-"train"}
MAX_SAMPLES=${3:-"3000"}

if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "Usage: sbatch --gres=gpu:1 scripts/eval_checkpoint.sh <checkpoint_path> [split] [max_samples]"
    exit 1
fi

source ~/.bashrc
conda activate safe-env

mkdir -p logs

python scripts/evaluate_checkpoint.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --split "$SPLIT" \
    --model_config phase1 \
    --max_samples "$MAX_SAMPLES"
