#!/bin/bash
#SBATCH --job-name=avqa-captions
#SBATCH --output=logs/avqa_captions_%j.out
#SBATCH --error=logs/avqa_captions_%j.err
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

export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export FP16=0
export LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-8B}

MODEL_CONFIG=${MODEL_CONFIG:-rkca_joint_caption16}
CHECKPOINT=${CHECKPOINT:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/audiocaps_align_rkca_joint_caption16/audio_aligned_best.pt}
MANIFEST=${MANIFEST:-$SAFE_ROOT/data/music_avqa/manifests/validation.jsonl}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT/data/music_avqa}
OUTPUT_MANIFEST=${OUTPUT_MANIFEST:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/music_avqa_validation_audio_captions.jsonl}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_SAMPLES=${MAX_SAMPLES:-0}

cd "$SAFE_ROOT"
mkdir -p logs "$(dirname "$OUTPUT_MANIFEST")"

python3 experiments/internvl_benchmark/generate_music_avqa_audio_captions.py \
  --model-config "$MODEL_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --media-root "$MEDIA_ROOT" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-samples "$MAX_SAMPLES"
