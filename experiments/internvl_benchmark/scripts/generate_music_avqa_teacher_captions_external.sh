#!/bin/bash
#SBATCH --job-name=avqa-teacher-ext
#SBATCH --output=logs/avqa_teacher_ext_%j.out
#SBATCH --error=logs/avqa_teacher_ext_%j.err
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

TEACHER_MODEL=${TEACHER_MODEL:-$SAFE_ROOT/models/audio_caption_teacher}
TEACHER_BACKEND=${TEACHER_BACKEND:-auto}
MANIFEST=${MANIFEST:-$SAFE_ROOT/data/music_avqa/manifests/validation.jsonl}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT/data/music_avqa}
OUTPUT_MANIFEST=${OUTPUT_MANIFEST:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/music_avqa_teacher_captions_validation_external.jsonl}
CAPTION_FIELD=${CAPTION_FIELD:-teacher_caption_raw}
MAX_SAMPLES=${MAX_SAMPLES:-0}
QWEN_QUANTIZATION=${QWEN_QUANTIZATION:-4bit}
QWEN_CPU_OFFLOAD=${QWEN_CPU_OFFLOAD:-0}

# Qwen-Omni: process one sample at a time (chat-template model), more tokens for detail
if [[ "$TEACHER_BACKEND" == "qwen_omni" ]] || [[ "$TEACHER_MODEL" == *qwen*omni* ]] || [[ "$TEACHER_MODEL" == *Qwen*Omni* ]]; then
    BATCH_SIZE=${BATCH_SIZE:-1}
    MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
    NUM_BEAMS=${NUM_BEAMS:-1}
    NUM_WORKERS=${NUM_WORKERS:-0}
else
    BATCH_SIZE=${BATCH_SIZE:-8}
    MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
    NUM_BEAMS=${NUM_BEAMS:-4}
    NUM_WORKERS=${NUM_WORKERS:-2}
fi

cd "$SAFE_ROOT"
mkdir -p logs "$(dirname "$OUTPUT_MANIFEST")"

python3 experiments/internvl_benchmark/generate_music_avqa_teacher_captions_external.py \
  --teacher-model "$TEACHER_MODEL" \
  --teacher-backend "$TEACHER_BACKEND" \
  --manifest "$MANIFEST" \
  --media-root "$MEDIA_ROOT" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --caption-field "$CAPTION_FIELD" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-samples "$MAX_SAMPLES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-beams "$NUM_BEAMS" \
  --qwen-quantization "$QWEN_QUANTIZATION" \
  $( [[ "$QWEN_CPU_OFFLOAD" == "1" ]] && printf '%s' '--qwen-cpu-offload' )
