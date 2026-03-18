#!/bin/bash
#SBATCH --job-name=qwen_kv_aud_l1s4
#SBATCH --output=logs/qwen_kv_aud_l1s4_%j.out
#SBATCH --error=logs/qwen_kv_aud_l1s4_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu

# Audio-after-vision KV aug: 8 layers starting at 1, stride 4
# Layers: 1, 5, 9, 13, 17, 21, 25, 29

set -euo pipefail

if [[ -z "${SAFE_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    SAFE_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SAFE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
  fi
fi
cd "$SAFE_ROOT"

export VISION_CKPT="${VISION_CKPT:-$SAFE_ROOT/checkpoints/qwen_kv_vision_only_l1s4/best_model.pt}"
export MODEL_CONFIG="${MODEL_CONFIG:-composition_kv_qwen}"
export FUSION_LAYERS="1,5,9,13,17,21,25,29"
export OUTPUT_DIR="${OUTPUT_DIR:-$SAFE_ROOT/checkpoints/qwen_kv_audio_after_vision_l1s4}"
export TRAIN_MODALITY="${TRAIN_MODALITY:-both}"
export TRAINABLE_MODALITIES="${TRAINABLE_MODALITIES:-audio}"
export EVAL_MODALITIES="${EVAL_MODALITIES:-text,audio,image,both}"
export INIT_VISION_CKPT="${INIT_VISION_CKPT:-$VISION_CKPT}"
export FUSION_GATE="${FUSION_GATE:-1.0}"
export GATE_WARMUP_STEPS="${GATE_WARMUP_STEPS:-500}"
export EPOCHS="${EPOCHS:-10}"
export LR="${LR:-5e-5}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export SAFE_QWEN_QUANT="${SAFE_QWEN_QUANT:-none}"
export SAFE_GRAD_CKPT="${SAFE_GRAD_CKPT:-0}"
export USE_FP16="${USE_FP16:-0}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen_kv_aud_l1s4_${SLURM_JOB_ID:-local}}"
export WANDB_TAGS="${WANDB_TAGS:-neurips,qwen,kv,music_avqa,audio_after_vision,l1s4,8layers}"

bash experiments/avqa_composition/scripts/train_preffn_music_avqa.sh
