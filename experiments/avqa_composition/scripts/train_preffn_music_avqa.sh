#!/bin/bash
#SBATCH --job-name=music-avqa-preffn
#SBATCH --output=logs/music_avqa_preffn_%j.out
#SBATCH --error=logs/music_avqa_preffn_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
# pass --gres=gpu:1 at submit time

set -euo pipefail

# Resolve SAFE root robustly for sbatch launches from arbitrary directories.
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

DATA_ROOT=${DATA_ROOT:-data/music_avqa}
MEDIA_ROOT=${MEDIA_ROOT:-$DATA_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/avqa_composition/music_preffn}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-10}
LR=${LR:-5e-5}
FUSION_LAYERS=${FUSION_LAYERS:-1,5,9,13,17,21}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}
TRAIN_MODALITY=${TRAIN_MODALITY:-both}
EVAL_MODALITIES=${EVAL_MODALITIES:-both,audio,image}

mkdir -p logs "$OUTPUT_DIR"

cd "$SAFE_ROOT"

python3 "$SAFE_ROOT/experiments/avqa_composition/train_avqa_composition.py" \
  --dataset music_avqa \
  --train-manifest "$DATA_ROOT/manifests/train.jsonl" \
  --val-manifest "$DATA_ROOT/manifests/validation.jsonl" \
  --media-root "$MEDIA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs "$EPOCHS" \
  --learning-rate "$LR" \
  --fusion-layers "$FUSION_LAYERS" \
  --num-audio-tokens "$NUM_AUDIO_TOKENS" \
  --train-modality "$TRAIN_MODALITY" \
  --eval-modalities "$EVAL_MODALITIES" \
  --fp16
