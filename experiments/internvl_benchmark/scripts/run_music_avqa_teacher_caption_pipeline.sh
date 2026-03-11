#!/bin/bash
#SBATCH --job-name=avqa-teacher-pipe
#SBATCH --output=logs/avqa_teacher_pipe_%j.out
#SBATCH --error=logs/avqa_teacher_pipe_%j.err
#SBATCH --time=48:00:00
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

export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export FP16=0
export LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-8B}

MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT/data/music_avqa}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$SAFE_ROOT/data/music_avqa/manifests/train.jsonl}
VAL_MANIFEST=${VAL_MANIFEST:-$SAFE_ROOT/data/music_avqa/manifests/validation.jsonl}
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/audiocaps_align_rkca_joint_caption16/audio_aligned_best.pt}
TEACHER_CONFIG=${TEACHER_CONFIG:-rkca_joint_caption16}
TEACHER_BACKEND=${TEACHER_BACKEND:-internal}
TEACHER_MODEL=${TEACHER_MODEL:-$SAFE_ROOT/models/audio_caption_teacher}
TEACHER_TRAIN_MANIFEST=${TEACHER_TRAIN_MANIFEST:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/music_avqa_teacher_captions_train.jsonl}
TEACHER_VAL_MANIFEST=${TEACHER_VAL_MANIFEST:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/music_avqa_teacher_captions_validation.jsonl}
SEMANTIC_TRAIN_MANIFEST=${SEMANTIC_TRAIN_MANIFEST:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/music_avqa_teacher_captions_train_semantic.jsonl}
SEMANTIC_VAL_MANIFEST=${SEMANTIC_VAL_MANIFEST:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/music_avqa_teacher_captions_validation_semantic.jsonl}
DECODER_OUTPUT_DIR=${DECODER_OUTPUT_DIR:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/clap_music_avqa_structured_decoder}
TARGET_FIELD=${TARGET_FIELD:-teacher_caption_structured}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-1e-4}

if [[ -z "${RAW_CAPTION_FIELD:-}" ]]; then
  if [[ "$TEACHER_BACKEND" == "external" ]]; then
    RAW_CAPTION_FIELD=teacher_caption_raw
  else
    RAW_CAPTION_FIELD=rich_audio_caption
  fi
fi

cd "$SAFE_ROOT"
mkdir -p logs "$DECODER_OUTPUT_DIR"

if [[ "$TEACHER_BACKEND" == "external" ]]; then
  python3 experiments/internvl_benchmark/generate_music_avqa_teacher_captions_external.py \
    --teacher-model "$TEACHER_MODEL" \
    --teacher-backend "${EXTERNAL_TEACHER_BACKEND:-auto}" \
    --manifest "$TRAIN_MANIFEST" \
    --media-root "$MEDIA_ROOT" \
    --output-manifest "$TEACHER_TRAIN_MANIFEST" \
    --caption-field teacher_caption_raw \
    ${LOAD_IN_8BIT:+--load-in-8bit} \
    ${LOAD_IN_4BIT:+--load-in-4bit}
else
  python3 experiments/internvl_benchmark/generate_music_avqa_audio_captions.py \
    --model-config "$TEACHER_CONFIG" \
    --checkpoint "$TEACHER_CHECKPOINT" \
    --manifest "$TRAIN_MANIFEST" \
    --media-root "$MEDIA_ROOT" \
    --output-manifest "$TEACHER_TRAIN_MANIFEST"
fi

python3 experiments/internvl_benchmark/build_music_avqa_semantic_targets.py \
  --input-manifest "$TEACHER_TRAIN_MANIFEST" \
  --output-manifest "$SEMANTIC_TRAIN_MANIFEST" \
  --raw-caption-field "$RAW_CAPTION_FIELD"

if [[ "$TEACHER_BACKEND" == "external" ]]; then
  python3 experiments/internvl_benchmark/generate_music_avqa_teacher_captions_external.py \
    --teacher-model "$TEACHER_MODEL" \
    --teacher-backend "${EXTERNAL_TEACHER_BACKEND:-auto}" \
    --manifest "$VAL_MANIFEST" \
    --media-root "$MEDIA_ROOT" \
    --output-manifest "$TEACHER_VAL_MANIFEST" \
    --caption-field teacher_caption_raw \
    ${LOAD_IN_8BIT:+--load-in-8bit} \
    ${LOAD_IN_4BIT:+--load-in-4bit}
else
  python3 experiments/internvl_benchmark/generate_music_avqa_audio_captions.py \
    --model-config "$TEACHER_CONFIG" \
    --checkpoint "$TEACHER_CHECKPOINT" \
    --manifest "$VAL_MANIFEST" \
    --media-root "$MEDIA_ROOT" \
    --output-manifest "$TEACHER_VAL_MANIFEST"
fi

python3 experiments/internvl_benchmark/build_music_avqa_semantic_targets.py \
  --input-manifest "$TEACHER_VAL_MANIFEST" \
  --output-manifest "$SEMANTIC_VAL_MANIFEST" \
  --raw-caption-field "$RAW_CAPTION_FIELD"

python3 experiments/internvl_benchmark/train_clap_qwen_caption_decoder.py \
  --dataset-mode music_avqa \
  --data-path "$SAFE_ROOT/experiments/full_training/data" \
  --train-manifest "$SEMANTIC_TRAIN_MANIFEST" \
  --val-manifest "$SEMANTIC_VAL_MANIFEST" \
  --media-root "$MEDIA_ROOT" \
  --target-field "$TARGET_FIELD" \
  --output-dir "$DECODER_OUTPUT_DIR" \
  --llm-model "$LLM_MODEL_PATH" \
  --batch-size 32 \
  --val-batch-size 32 \
  --num-workers 2 \
  --num-epochs "$NUM_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --max-length 16 \
  --max-new-tokens 8 \
  --holdout-model-config rkca_joint
