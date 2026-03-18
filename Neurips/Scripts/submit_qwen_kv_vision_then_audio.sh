#!/bin/bash
set -euo pipefail

if [[ -z "${SAFE_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    SAFE_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SAFE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
  fi
fi
cd "$SAFE_ROOT"

VISION_SCRIPT="${VISION_SCRIPT:-Neurips/Scripts/run_qwen_kv_vision_only_music_avqa.sh}"
AUDIO_SCRIPT="${AUDIO_SCRIPT:-Neurips/Scripts/run_qwen_kv_audio_after_vision_music_avqa.sh}"
VISION_OUT="${VISION_OUT:-$SAFE_ROOT/checkpoints/qwen_kv_vision_only}"
VISION_CKPT="${VISION_CKPT:-$VISION_OUT/best_model.pt}"

VISION_JOB=$(sbatch --parsable --gres=gpu:1 "$VISION_SCRIPT")
echo "[submit] vision job: $VISION_JOB"

AUDIO_JOB=$(VISION_CKPT="$VISION_CKPT" sbatch --parsable --dependency=afterok:${VISION_JOB} --gres=gpu:1 "$AUDIO_SCRIPT")
echo "[submit] audio job: $AUDIO_JOB"
echo "[submit] dependency: audio starts after vision checkpoint job succeeds"
