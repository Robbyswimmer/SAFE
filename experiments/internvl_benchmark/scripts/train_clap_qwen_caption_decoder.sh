#!/bin/bash
#SBATCH --job-name=clap-qwen-cap
#SBATCH --output=logs/clap_qwen_cap_%j.out
#SBATCH --error=logs/clap_qwen_cap_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SAFE_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-8B}

DATA_PATH=${DATA_PATH:-$SAFE_ROOT/experiments/full_training/data}
OUTPUT_DIR=${OUTPUT_DIR:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/clap_qwen_caption_decoder}
USE_WAVCAPS=${USE_WAVCAPS:-0}
BATCH_SIZE=${BATCH_SIZE:-32}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
NUM_EPOCHS=${NUM_EPOCHS:-10}
NUM_WORKERS=${NUM_WORKERS:-2}
LEARNING_RATE=${LEARNING_RATE:-3e-4}

cd "$SAFE_ROOT"
mkdir -p "$OUTPUT_DIR" logs

EXTRA_ARGS=()
if [[ "$USE_WAVCAPS" == "1" ]]; then
  EXTRA_ARGS+=(--use-wavcaps)
fi

python3 experiments/internvl_benchmark/train_clap_qwen_caption_decoder.py \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --llm-model "$LLM_MODEL_PATH" \
  --batch-size "$BATCH_SIZE" \
  --val-batch-size "$VAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --num-epochs "$NUM_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  "${EXTRA_ARGS[@]}"
