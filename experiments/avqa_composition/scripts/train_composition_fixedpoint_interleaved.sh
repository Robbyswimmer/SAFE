#!/bin/bash
#SBATCH --job-name=comp-fixpt
#SBATCH --output=logs/comp_fixpt_%j.out
#SBATCH --error=logs/comp_fixpt_%j.err
#SBATCH --time=72:00:00
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

MODEL_CONFIG="${MODEL_CONFIG:-composition_fixed_point}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/composition_fixed_point_interleaved}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-composition_fixed_point_${SLURM_JOB_ID:-local}}"
WANDB_TAGS="${WANDB_TAGS:-composition,fixed-point,interleaved}"

export MODEL_CONFIG OUTPUT_DIR WANDB_RUN_NAME WANDB_TAGS

"$SAFE_ROOT/experiments/avqa_composition/scripts/train_composition_interleaved.sh"
