#!/bin/bash
# Submit the two-stage sequential Qwen experiment:
# 1. Train vision on image+text only.
# 2. Load the vision checkpoint, then train only audio params on audio+vision+text.

set -euo pipefail

VISION_SCRIPT="${VISION_SCRIPT:-Neurips/Scripts/run_qwen_vision_only_music_avqa.sh}"
AUDIO_SCRIPT="${AUDIO_SCRIPT:-Neurips/Scripts/run_qwen_audio_after_vision_music_avqa.sh}"
VISION_OUT="${VISION_OUT:-/data/SalmanAsif/RobbyMoseley/SAFE/SAFE/checkpoints/qwen_vision_only_from_scratch}"
AUDIO_OUT="${AUDIO_OUT:-/data/SalmanAsif/RobbyMoseley/SAFE/SAFE/checkpoints/qwen_audio_after_vision_joint}"
VISION_CKPT="${VISION_CKPT:-$VISION_OUT/best_model.pt}"

submit_output=$(sbatch --gres=gpu:1 --export=ALL,OUTPUT_DIR="$VISION_OUT" "$VISION_SCRIPT")
vision_jobid=$(echo "$submit_output" | awk '{print $4}')
echo "$submit_output"

sbatch --gres=gpu:1 \
  --dependency=afterok:"$vision_jobid" \
  --export=ALL,OUTPUT_DIR="$AUDIO_OUT",VISION_CKPT="$VISION_CKPT",INIT_VISION_CKPT="$VISION_CKPT" \
  "$AUDIO_SCRIPT"
