#!/bin/bash
set -euo pipefail

SAFE_ROOT="${SAFE_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"

AUDIO_OUT="${AUDIO_OUT:-checkpoints/composition_ttc_audio}"
VISION_OUT="${VISION_OUT:-checkpoints/composition_ttc_vision}"
BASELINE_OUT="${BASELINE_OUT:-checkpoints/composition_ttc_eval_baseline}"
SIMPLE_OUT="${SIMPLE_OUT:-checkpoints/composition_ttc_eval_simple}"
SIMPLE_INTER_OUT="${SIMPLE_INTER_OUT:-checkpoints/composition_ttc_eval_simple_interaction}"
COMPLEX_OUT="${COMPLEX_OUT:-checkpoints/composition_ttc_eval_complex}"
COMPLEX_INTER_OUT="${COMPLEX_INTER_OUT:-checkpoints/composition_ttc_eval_complex_interaction}"

cd "$SAFE_ROOT"

OUTPUT_DIR="$AUDIO_OUT" "$SAFE_ROOT/experiments/avqa_composition/scripts/train_composition_audio_ttc.sh"
OUTPUT_DIR="$VISION_OUT" "$SAFE_ROOT/experiments/avqa_composition/scripts/train_composition_vision_ttc.sh"
COMPOSE_AUDIO_CKPT="$AUDIO_OUT/best_model.pt" \
COMPOSE_VISION_CKPT="$VISION_OUT/best_model.pt" \
OUTPUT_DIR="$BASELINE_OUT" \
  "$SAFE_ROOT/experiments/avqa_composition/scripts/eval_composition_ttc_baseline.sh"
COMPOSE_AUDIO_CKPT="$AUDIO_OUT/best_model.pt" \
COMPOSE_VISION_CKPT="$VISION_OUT/best_model.pt" \
OUTPUT_DIR="$SIMPLE_OUT" \
  "$SAFE_ROOT/experiments/avqa_composition/scripts/eval_composition_ttc_simple.sh"
COMPOSE_AUDIO_CKPT="$AUDIO_OUT/best_model.pt" \
COMPOSE_VISION_CKPT="$VISION_OUT/best_model.pt" \
OUTPUT_DIR="$SIMPLE_INTER_OUT" \
TTC_INTERACTION_ENABLE=1 \
  "$SAFE_ROOT/experiments/avqa_composition/scripts/eval_composition_ttc_simple.sh"
COMPOSE_AUDIO_CKPT="$AUDIO_OUT/best_model.pt" \
COMPOSE_VISION_CKPT="$VISION_OUT/best_model.pt" \
OUTPUT_DIR="$COMPLEX_OUT" \
  "$SAFE_ROOT/experiments/avqa_composition/scripts/eval_composition_ttc_complex.sh"
COMPOSE_AUDIO_CKPT="$AUDIO_OUT/best_model.pt" \
COMPOSE_VISION_CKPT="$VISION_OUT/best_model.pt" \
OUTPUT_DIR="$COMPLEX_INTER_OUT" \
TTC_INTERACTION_ENABLE=1 \
  "$SAFE_ROOT/experiments/avqa_composition/scripts/eval_composition_ttc_complex.sh"
