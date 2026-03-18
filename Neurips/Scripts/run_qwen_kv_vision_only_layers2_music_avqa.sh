#!/bin/bash
#SBATCH --job-name=qwen_kv_vision_l2
#SBATCH --output=logs/qwen_kv_vision_l2_%j.out
#SBATCH --error=logs/qwen_kv_vision_l2_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
# pass --gres=gpu:1 at submit time

# Vision-only KV aug with layers starting at 2 instead of 8.
# Tests whether gradient concentration at the first injection layer
# is a property of layer 8 specifically or of being the first site.

set -euo pipefail

if [[ -z "${SAFE_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    SAFE_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SAFE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
  fi
fi
cd "$SAFE_ROOT"

export MODEL_CONFIG="${MODEL_CONFIG:-composition_kv_qwen}"
export FUSION_LAYERS="2,14,20,26"
export OUTPUT_DIR="${OUTPUT_DIR:-$SAFE_ROOT/checkpoints/qwen_kv_vision_only_l2}"
export TRAIN_MODALITY="${TRAIN_MODALITY:-image}"
export TRAINABLE_MODALITIES="${TRAINABLE_MODALITIES:-vision}"
export EVAL_MODALITIES="${EVAL_MODALITIES:-text,image,both}"
export FUSION_GATE="${FUSION_GATE:-0.1}"
export EPOCHS="${EPOCHS:-10}"
export LR="${LR:-5e-5}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export SAFE_QWEN_QUANT="${SAFE_QWEN_QUANT:-none}"
export SAFE_GRAD_CKPT="${SAFE_GRAD_CKPT:-0}"
export USE_FP16="${USE_FP16:-0}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen_kv_vision_l2_${SLURM_JOB_ID:-local}}"
export WANDB_TAGS="${WANDB_TAGS:-neurips,qwen,kv,music_avqa,vision_only,layer_ablation,l2}"

bash experiments/avqa_composition/scripts/train_preffn_music_avqa.sh
