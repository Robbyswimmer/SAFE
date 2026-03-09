#!/bin/bash
#SBATCH --job-name=compose-calib
#SBATCH --output=logs/compose_calib_%j.out
#SBATCH --error=logs/compose_calib_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ -z "${SAFE_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    SAFE_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SAFE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
  fi
fi

CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi
export SAFE_GRAD_CKPT=0

MODEL_CONFIG="${MODEL_CONFIG:-composition_ttc}"
DATA_ROOT="${DATA_ROOT:-data/music_avqa}"
MEDIA_ROOT="${MEDIA_ROOT:-$SAFE_ROOT}"
COMPOSE_AUDIO_CKPT="${COMPOSE_AUDIO_CKPT:-checkpoints/composition_ttc_audio/best_model.pt}"
COMPOSE_VISION_CKPT="${COMPOSE_VISION_CKPT:-checkpoints/composition_ttc_vision/best_model.pt}"
CALIBRATION_TRAINABLE="${CALIBRATION_TRAINABLE:-fusion}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-10}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/composition_calib_n${TRAIN_MAX_SAMPLES}_${CALIBRATION_TRAINABLE}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
NUM_AUDIO_TOKENS="${NUM_AUDIO_TOKENS:-8}"

mkdir -p "$SAFE_ROOT/logs" "$OUTPUT_DIR"
cd "$SAFE_ROOT"

python3 "$SAFE_ROOT/experiments/avqa_composition/train_avqa_composition.py" \
  --dataset music_avqa \
  --model-config "$MODEL_CONFIG" \
  --train-manifest "$DATA_ROOT/manifests/train.jsonl" \
  --val-manifest "$DATA_ROOT/manifests/validation.jsonl" \
  --media-root "$MEDIA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs "$EPOCHS" \
  --learning-rate "$LR" \
  --num-audio-tokens "$NUM_AUDIO_TOKENS" \
  --train-modality both \
  --eval-modalities text,audio,image,both \
  --freeze-audio-encoder \
  --compose-audio-ckpt "$COMPOSE_AUDIO_CKPT" \
  --compose-vision-ckpt "$COMPOSE_VISION_CKPT" \
  --compose-calibration-enable \
  --compose-calibration-trainable "$CALIBRATION_TRAINABLE" \
  --train-max-samples "$TRAIN_MAX_SAMPLES" \
  --val-max-samples "$VAL_MAX_SAMPLES"
