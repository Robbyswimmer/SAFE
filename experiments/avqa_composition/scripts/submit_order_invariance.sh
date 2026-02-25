#!/bin/bash
# Order invariance experiment (paper analysis section).
# Tests whether modality addition order matters:
#   Run A: audio-first interleaved (default: audio phase → vision phase per epoch)
#   Run B: vision-first interleaved (vision phase → audio phase per epoch)
# With frozen backbone, both should yield identical composition gains.
#
# Uses InternVL 8B (internvl_binding) with bn=756 (best known config).
set -euo pipefail

if [[ -z "${SAFE_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    SAFE_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SAFE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
  fi
fi
cd "$SAFE_ROOT"

MODEL_CONFIG=${MODEL_CONFIG:-internvl_binding}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-5}
LR=${LR:-5e-5}
BOTTLENECK_DIM=${BOTTLENECK_DIM:-756}

echo "[order-inv] SAFE_ROOT=$SAFE_ROOT"
echo "[order-inv] MODEL_CONFIG=$MODEL_CONFIG"
echo "[order-inv] BOTTLENECK_DIM=$BOTTLENECK_DIM"
echo ""

# Run A: audio-first (default interleaved order)
JA=$(sbatch --parsable --gres=gpu:1 \
  --job-name="order-audio-first" \
  --export=ALL,\
SAFE_ROOT="$SAFE_ROOT",\
MODEL_CONFIG="$MODEL_CONFIG",\
OUTPUT_DIR=checkpoints/order_invariance/audio_first,\
SEED="$SEED",\
EPOCHS="$EPOCHS",\
LR="$LR",\
BOTTLENECK_DIM="$BOTTLENECK_DIM",\
SLIM_PROJECTOR=1,\
TRAIN_MODALITY=interleaved,\
EVAL_MODALITIES=text,audio,image,both,\
WANDB_RUN_NAME="order_audio_first_s${SEED}",\
WANDB_TAGS="order_invariance,audio_first,8b,paper_analysis" \
  experiments/avqa_composition/scripts/train_composition_interleaved.sh)

echo "[order-inv] audio-first → job $JA"

# Run B: vision-first
JB=$(sbatch --parsable --gres=gpu:1 \
  --job-name="order-vision-first" \
  --export=ALL,\
SAFE_ROOT="$SAFE_ROOT",\
MODEL_CONFIG="$MODEL_CONFIG",\
OUTPUT_DIR=checkpoints/order_invariance/vision_first,\
SEED="$SEED",\
EPOCHS="$EPOCHS",\
LR="$LR",\
BOTTLENECK_DIM="$BOTTLENECK_DIM",\
SLIM_PROJECTOR=1,\
TRAIN_MODALITY=interleaved_vision_first,\
EVAL_MODALITIES=text,audio,image,both,\
WANDB_RUN_NAME="order_vision_first_s${SEED}",\
WANDB_TAGS="order_invariance,vision_first,8b,paper_analysis" \
  experiments/avqa_composition/scripts/train_composition_interleaved.sh)

echo "[order-inv] vision-first → job $JB"
echo ""
echo "[order-inv] Monitor: squeue -j $JA,$JB"
