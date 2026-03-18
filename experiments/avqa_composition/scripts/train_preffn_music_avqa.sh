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

SAFE_ROOT="/data/SalmanAsif/RobbyMoseley/SAFE/SAFE"

CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

MODEL_CONFIG=${MODEL_CONFIG:-phase1}
DATA_ROOT=${DATA_ROOT:-data/music_avqa}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/avqa_composition/music_preffn}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-10}
LR=${LR:-5e-5}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
FUSION_LAYERS=${FUSION_LAYERS:-}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}
TRAIN_MODALITY=${TRAIN_MODALITY:-both}
EVAL_MODALITIES=${EVAL_MODALITIES:-both,audio,image}
FUSION_GATE=${FUSION_GATE:-0.2}
USE_FP16=${USE_FP16:-1}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-SAFE-AVQA-Composition}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-music_avqa_preffn_${SLURM_JOB_ID:-local}}
WANDB_TAGS=${WANDB_TAGS:-music_avqa,preffn}
MAX_SAMPLES=${MAX_SAMPLES:-0}
BOTTLENECK_DIM=${BOTTLENECK_DIM:-}
INIT_AUDIO_CKPT=${INIT_AUDIO_CKPT:-}
INIT_VISION_CKPT=${INIT_VISION_CKPT:-}

# InternVL defaults: gradient checkpointing currently breaks hook-based adapter grads.
# Keep override-friendly behavior (user can still set these explicitly via --export).
if [[ "${MODEL_CONFIG}" == internvl* ]]; then
  export SAFE_QWEN_QUANT=${SAFE_QWEN_QUANT:-none}
  export SAFE_GRAD_CKPT=${SAFE_GRAD_CKPT:-0}
  if [[ "${USE_FP16}" == "1" && -z "${FORCE_FP16:-}" ]]; then
    USE_FP16=0
    echo "[launcher] InternVL detected: USE_FP16->0 (bf16 path). Set FORCE_FP16=1 to keep fp16."
  fi
fi

mkdir -p logs "$OUTPUT_DIR"

cd "$SAFE_ROOT"

echo "[launcher] model_config=${MODEL_CONFIG} SAFE_GRAD_CKPT=${SAFE_GRAD_CKPT:-unset} SAFE_QWEN_QUANT=${SAFE_QWEN_QUANT:-unset} USE_FP16=${USE_FP16}"

WANDB_ARGS=()
if [[ "$WANDB" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME" --wandb-tags "$WANDB_TAGS")
fi

FUSION_ARGS=()
if [[ -n "$FUSION_LAYERS" ]]; then
  FUSION_ARGS+=(--fusion-layers "$FUSION_LAYERS")
fi

FP16_ARGS=()
if [[ "$USE_FP16" == "1" ]]; then
  FP16_ARGS+=(--fp16)
fi

MAX_SAMPLES_ARGS=()
if [[ "$MAX_SAMPLES" != "0" ]]; then
  MAX_SAMPLES_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

BOTTLENECK_ARGS=()
if [[ -n "$BOTTLENECK_DIM" ]]; then
  BOTTLENECK_ARGS+=(--bottleneck-dim "$BOTTLENECK_DIM")
fi

INIT_ARGS=()
if [[ -n "$INIT_AUDIO_CKPT" ]]; then
  INIT_ARGS+=(--init-audio-ckpt "$INIT_AUDIO_CKPT")
fi
if [[ -n "$INIT_VISION_CKPT" ]]; then
  INIT_ARGS+=(--init-vision-ckpt "$INIT_VISION_CKPT")
fi

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
  --lr-scheduler "$LR_SCHEDULER" \
  --warmup-ratio "$WARMUP_RATIO" \
  --min-lr-ratio "$MIN_LR_RATIO" \
  --num-audio-tokens "$NUM_AUDIO_TOKENS" \
  --train-modality "$TRAIN_MODALITY" \
  --eval-modalities "$EVAL_MODALITIES" \
  --fusion-gate "$FUSION_GATE" \
  --freeze-audio-encoder \
  "${FP16_ARGS[@]}" \
  "${FUSION_ARGS[@]}" \
  "${MAX_SAMPLES_ARGS[@]}" \
  "${BOTTLENECK_ARGS[@]}" \
  "${INIT_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
