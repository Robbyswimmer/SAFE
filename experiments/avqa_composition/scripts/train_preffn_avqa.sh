#!/bin/bash
#SBATCH --job-name=avqa-preffn
#SBATCH --output=logs/avqa_preffn_%j.out
#SBATCH --error=logs/avqa_preffn_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
# pass --gres=gpu:1 at submit time

set -euo pipefail

SAFE_ROOT="/data/SalmanAsif/RobbyMoseley/SAFE/SAFE"

CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

DATA_ROOT=${DATA_ROOT:-data/avqa}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/avqa_composition/preffn}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-10}
LR=${LR:-5e-5}
FUSION_LAYERS=${FUSION_LAYERS:-1,5,9,13,17,21}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}
TRAIN_MODALITY=${TRAIN_MODALITY:-both}
EVAL_MODALITIES=${EVAL_MODALITIES:-both,audio,image}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-SAFE-AVQA-Composition}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-avqa_preffn_${SLURM_JOB_ID:-local}}
WANDB_TAGS=${WANDB_TAGS:-avqa,preffn}

mkdir -p logs "$OUTPUT_DIR"

cd "$SAFE_ROOT"

WANDB_ARGS=()
if [[ "$WANDB" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME" --wandb-tags "$WANDB_TAGS")
fi

python3 "$SAFE_ROOT/experiments/avqa_composition/train_avqa_composition.py" \
  --dataset avqa \
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
  --fp16 \
  "${WANDB_ARGS[@]}"
