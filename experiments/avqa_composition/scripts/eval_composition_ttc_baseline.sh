#!/bin/bash
set -euo pipefail

SAFE_ROOT="${SAFE_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
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
  --eval-modalities text,audio,image,both \
  --freeze-audio-encoder \
  --fusion-gate "$FUSION_GATE" \
  --compose-audio-ckpt "$COMPOSE_AUDIO_CKPT" \
  --compose-vision-ckpt "$COMPOSE_VISION_CKPT" \
  "${MAX_SAMPLES_ARGS[@]}"
