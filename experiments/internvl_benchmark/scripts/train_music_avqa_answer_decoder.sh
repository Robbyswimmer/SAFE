#!/bin/bash
#SBATCH --job-name=clap-avqa-ans
#SBATCH --output=logs/clap_avqa_answer_%j.out
#SBATCH --error=logs/clap_avqa_answer_%j.err
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
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT/data/music_avqa}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$SAFE_ROOT/data/music_avqa/manifests/train.jsonl}
VAL_MANIFEST=${VAL_MANIFEST:-$SAFE_ROOT/data/music_avqa/manifests/validation.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/clap_music_avqa_answer_decoder}
HOLDOUT_MODEL_CONFIG=${HOLDOUT_MODEL_CONFIG:-rkca_joint}
HOLDOUT_CHECKPOINT=${HOLDOUT_CHECKPOINT:-}
BATCH_SIZE=${BATCH_SIZE:-32}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
HOLDOUT_BATCH_SIZE=${HOLDOUT_BATCH_SIZE:-2}
NUM_EPOCHS=${NUM_EPOCHS:-10}
NUM_WORKERS=${NUM_WORKERS:-2}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
MAX_LENGTH=${MAX_LENGTH:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4}

cd "$SAFE_ROOT"
mkdir -p "$OUTPUT_DIR" logs

python3 experiments/internvl_benchmark/train_clap_qwen_caption_decoder.py \
  --dataset-mode music_avqa \
  --data-path "$DATA_PATH" \
  --train-manifest "$TRAIN_MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  --media-root "$MEDIA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --llm-model "$LLM_MODEL_PATH" \
  --batch-size "$BATCH_SIZE" \
  --val-batch-size "$VAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --num-epochs "$NUM_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --max-length "$MAX_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --holdout-model-config "$HOLDOUT_MODEL_CONFIG" \
  --holdout-checkpoint "$HOLDOUT_CHECKPOINT" \
  --holdout-batch-size "$HOLDOUT_BATCH_SIZE"
