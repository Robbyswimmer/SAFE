#!/bin/bash
#SBATCH --job-name=epic-avqa-kv
#SBATCH --output=logs/epic_avqa_kv_%j.out
#SBATCH --error=logs/epic_avqa_kv_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
# pass --gres=gpu:1 at submit time

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

DATA_ROOT=${DATA_ROOT:-experiments/epic_sounds_avqa_composition/data}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/epic_sounds_avqa/kv_augment}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-5}
LR=${LR:-5e-5}
TRAIN_MODALITY=${TRAIN_MODALITY:-both}
EVAL_MODALITIES=${EVAL_MODALITIES:-both,audio,image}

mkdir -p logs "$OUTPUT_DIR"

python3 experiments/epic_sounds_avqa_composition/train_epic_sounds_avqa.py \
  --architecture kv_augment \
  --data-root "$DATA_ROOT" \
  --train-manifest "$DATA_ROOT/manifests/train.jsonl" \
  --val-manifest "$DATA_ROOT/manifests/validation.jsonl" \
  --media-root "$DATA_ROOT/processed" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs "$EPOCHS" \
  --learning-rate "$LR" \
  --train-modality "$TRAIN_MODALITY" \
  --eval-modalities "$EVAL_MODALITIES" \
  --fp16
