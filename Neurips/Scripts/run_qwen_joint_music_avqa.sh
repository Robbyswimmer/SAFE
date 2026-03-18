#!/bin/bash
#SBATCH --job-name=run_qwen_joint
#SBATCH --output=logs/run_qwen_joint_%j.out
#SBATCH --error=logs/run_qwen_joint_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
# pass --gres=gpu:1 at submit time

# Submit a Qwen joint-data MUSIC-AVQA run from scratch.
#
# Usage:
#   sbatch --gres=gpu:1 Neurips/Scripts/run_qwen_joint_music_avqa.sh
# Optional overrides:
#   DATA_ROOT=... OUTPUT_DIR=... EPOCHS=10 MAX_SAMPLES=4000 \
#   sbatch --gres=gpu:1 Neurips/Scripts/run_qwen_joint_music_avqa.sh

set -euo pipefail

SAFE_ROOT="${SAFE_ROOT:-/data/SalmanAsif/RobbyMoseley/SAFE/SAFE}"
TRAIN_SCRIPT="$SAFE_ROOT/experiments/avqa_composition/scripts/train_preffn_music_avqa.sh"

export DATA_ROOT="${DATA_ROOT:-/data/SalmanAsif/RobbyMoseley/SAFE/SAFE/experiments/full_training/data/music_avqa}"
export MODEL_CONFIG="${MODEL_CONFIG:-composition_study}"
export TRAIN_MODALITY="${TRAIN_MODALITY:-both}"
export EVAL_MODALITIES="${EVAL_MODALITIES:-text,audio,image,both}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/SalmanAsif/RobbyMoseley/SAFE/SAFE/checkpoints/qwen_joint_both_from_scratch}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export EPOCHS="${EPOCHS:-10}"
export LR="${LR:-5e-5}"
export FUSION_GATE="${FUSION_GATE:-0.1}"
export SAFE_QWEN_QUANT="${SAFE_QWEN_QUANT:-none}"
export SAFE_GRAD_CKPT="${SAFE_GRAD_CKPT:-0}"
export USE_FP16="${USE_FP16:-0}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-SAFE-AVQA-Composition}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen_joint_both_from_scratch_${SLURM_JOB_ID:-local}}"
export WANDB_TAGS="${WANDB_TAGS:-neurips,qwen,music_avqa,joint,both,from_scratch}"

if [[ -n "${MAX_SAMPLES:-}" ]]; then
  export MAX_SAMPLES
fi

echo "[neurips-launch] SAFE_ROOT=$SAFE_ROOT"
echo "[neurips-launch] TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "[neurips-launch] MODEL_CONFIG=$MODEL_CONFIG TRAIN_MODALITY=$TRAIN_MODALITY EVAL_MODALITIES=$EVAL_MODALITIES"
echo "[neurips-launch] OUTPUT_DIR=$OUTPUT_DIR EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE LR=$LR FUSION_GATE=$FUSION_GATE"

exec "$TRAIN_SCRIPT"
