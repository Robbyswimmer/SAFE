#!/bin/bash
#SBATCH --job-name=internvl-audio
#SBATCH --output=logs/internvl_audio_%j.out
#SBATCH --error=logs/internvl_audio_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
# pass --gres=gpu:1 at submit time

# InternVL 3.5-8B Audio Training with SAFE
#
# Uses InternVL 3.5-8B (InternViT-300M + Qwen3-8B) as frozen backbone,
# training only SAFE audio adapters (projector + fusion layers).
#
# Usage:
#   sbatch experiments/internvl_benchmark/scripts/train_audio.sh
#   # Or locally:
#   bash experiments/internvl_benchmark/scripts/train_audio.sh

set -euo pipefail

# Get SAFE root
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SAFE_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

# Conda environment
CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# InternVL-specific env vars
export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export FP16=0
export LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-8B}

# Training configuration
DATA_PATH=${DATA_PATH:-$SAFE_ROOT/experiments/full_training/data}
OUTPUT_DIR=${OUTPUT_DIR:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/audio_baseline}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_EPOCHS=${NUM_EPOCHS:-50}
SAFE_LR=${SAFE_LR:-5e-5}
HEAD_LR=${HEAD_LR:-1e-3}
FUSION_LAYERS=${FUSION_LAYERS:-"12,24,33"}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}

# W&B settings
WANDB_PROJECT=${WANDB_PROJECT:-"SAFE-InternVL-Benchmark"}
WANDB_NAME="internvl_audio_${SLURM_JOB_ID:-local}"
WANDB_TAGS=${WANDB_TAGS:-"internvl,audio,baseline"}

echo "========================================"
echo "InternVL 3.5-8B Audio Training"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "SAFE root: $SAFE_ROOT"
echo "Model path: $LLM_MODEL_PATH"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "SAFE LR: $SAFE_LR"
echo "Head LR: $HEAD_LR"
echo "Fusion layers: $FUSION_LAYERS"
echo "Audio tokens: $NUM_AUDIO_TOKENS"
echo "========================================"

cd "$SAFE_ROOT"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

python3 train_audio_llm_probe.py \
    --dataset esc50 \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model-config internvl \
    --fusion-layer-indices "$FUSION_LAYERS" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --safe-learning-rate "$SAFE_LR" \
    --head-learning-rate "$HEAD_LR" \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_NAME" \
    --wandb-tags "$WANDB_TAGS"

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================"
