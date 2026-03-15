#!/bin/bash
#SBATCH --job-name=avqa-refine
#SBATCH --output=logs/avqa_refine_%j.out
#SBATCH --error=logs/avqa_refine_%j.err
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

CONDA_ENV=${CONDA_ENV:-safe-internvl}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export SAFE_PREFER_FLASH2=0
export FP16=0
export LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-8B}

CHECKPOINT=${CHECKPOINT:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/clap_music_avqa_text_target_decoder/checkpoint_best.pt}
MANIFEST=${MANIFEST:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/music_avqa_teacher_captions_train_qwen8bit_compose_semantic.jsonl}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT/data/music_avqa}
OUTPUT_MANIFEST=${OUTPUT_MANIFEST:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/music_avqa_refined_decoder_targets_train.jsonl}
CLIP_CACHE_MANIFEST=${CLIP_CACHE_MANIFEST:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/music_avqa_refined_decoder_targets_train_clips.jsonl}
CAPTION_FIELD=${CAPTION_FIELD:-refined_decoder_caption}
NUM_CANDIDATES=${NUM_CANDIDATES:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
TEMPERATURE=${TEMPERATURE:-0.9}
TOP_P=${TOP_P:-0.95}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
MAX_CLIPS=${MAX_CLIPS:-0}
MAX_ROWS_PER_CLIP=${MAX_ROWS_PER_CLIP:-0}

cd "$SAFE_ROOT"
mkdir -p logs "$(dirname "$OUTPUT_MANIFEST")"

python3 experiments/internvl_benchmark/refine_music_avqa_decoder_targets.py \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --media-root "$MEDIA_ROOT" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --clip-cache-manifest "$CLIP_CACHE_MANIFEST" \
  --caption-field "$CAPTION_FIELD" \
  --llm-model "$LLM_MODEL_PATH" \
  --num-candidates "$NUM_CANDIDATES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --max-clips "$MAX_CLIPS" \
  --max-rows-per-clip "$MAX_ROWS_PER_CLIP"
