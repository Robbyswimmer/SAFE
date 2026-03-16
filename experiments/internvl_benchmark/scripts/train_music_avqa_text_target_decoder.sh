#!/bin/bash
#SBATCH --job-name=clap-avqa-text
#SBATCH --output=logs/clap_avqa_text_%j.out
#SBATCH --error=logs/clap_avqa_text_%j.err
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
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/music_avqa_teacher_captions_train_semantic.jsonl}
VAL_MANIFEST=${VAL_MANIFEST:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/music_avqa_teacher_captions_validation_semantic.jsonl}
TARGET_FIELD=${TARGET_FIELD:-teacher_caption_structured}
OUTPUT_DIR=${OUTPUT_DIR:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/clap_music_avqa_text_target_decoder}
HOLDOUT_MODEL_CONFIG=${HOLDOUT_MODEL_CONFIG:-rkca_joint}
HOLDOUT_CHECKPOINT=${HOLDOUT_CHECKPOINT:-}
HOLDOUT_BACKEND=${HOLDOUT_BACKEND:-raw_internvl}
DEDUP_BY_AUDIO=${DEDUP_BY_AUDIO:-1}
CONSTRAINED_DECODING=${CONSTRAINED_DECODING:-1}

cd "$SAFE_ROOT"
mkdir -p "$OUTPUT_DIR" logs

CMD=(
  python3 experiments/internvl_benchmark/train_clap_qwen_caption_decoder.py
  --dataset-mode music_avqa
  --data-path "$DATA_PATH"
  --train-manifest "$TRAIN_MANIFEST"
  --val-manifest "$VAL_MANIFEST"
  --media-root "$MEDIA_ROOT"
  --target-field "$TARGET_FIELD"
  --output-dir "$OUTPUT_DIR"
  --llm-model "$LLM_MODEL_PATH"
  --batch-size 32
  --val-batch-size 32
  --num-workers 2
  --num-epochs "${NUM_EPOCHS:-10}"
  --learning-rate "${LEARNING_RATE:-1e-4}"
  --max-length "${MAX_LENGTH:-16}"
  --max-new-tokens "${MAX_NEW_TOKENS:-8}"
  --holdout-model-config "$HOLDOUT_MODEL_CONFIG"
  --holdout-checkpoint "$HOLDOUT_CHECKPOINT"
  --holdout-batch-size "${HOLDOUT_BATCH_SIZE:-2}"
  --holdout-backend "$HOLDOUT_BACKEND"
)

if [[ "$DEDUP_BY_AUDIO" == "1" ]]; then
  CMD+=(--dedup-by-audio)
fi

if [[ "$CONSTRAINED_DECODING" == "1" ]]; then
  CMD+=(--constrained-decoding)
fi

"${CMD[@]}"
