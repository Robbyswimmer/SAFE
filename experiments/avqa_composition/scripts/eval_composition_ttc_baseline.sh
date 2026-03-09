#!/bin/bash
#SBATCH --job-name=ttc-eval-base
#SBATCH --output=logs/ttc_eval_baseline_%j.out
#SBATCH --error=logs/ttc_eval_baseline_%j.err
#SBATCH --time=6:00:00
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
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/composition_ttc_eval_baseline}"
COMPOSE_AUDIO_CKPT="${COMPOSE_AUDIO_CKPT:-checkpoints/composition_ttc_audio/best_model.pt}"
COMPOSE_VISION_CKPT="${COMPOSE_VISION_CKPT:-checkpoints/composition_ttc_vision/best_model.pt}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_AUDIO_TOKENS="${NUM_AUDIO_TOKENS:-8}"
FUSION_GATE="${FUSION_GATE:-0.2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

MAX_SAMPLES_ARGS=()
if [[ "$MAX_SAMPLES" != "0" ]]; then
  MAX_SAMPLES_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

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
  --num-epochs 0 \
  --num-audio-tokens "$NUM_AUDIO_TOKENS" \
  --train-modality both \
  --eval-modalities both,text,audio,image \
  --freeze-audio-encoder \
  --fusion-gate "$FUSION_GATE" \
  --compose-audio-ckpt "$COMPOSE_AUDIO_CKPT" \
  --compose-vision-ckpt "$COMPOSE_VISION_CKPT" \
  "${MAX_SAMPLES_ARGS[@]}"
