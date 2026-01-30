#!/bin/bash
#SBATCH --job-name=esc50-baseline
#SBATCH --output=logs/esc50_baseline_%j.log
#SBATCH --error=logs/esc50_baseline_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

# ESC-50 Audio Classification - Baseline Training
#
# This script trains a single fold (fold 5 as val) for quick iteration.
# For full 5-fold CV, use train_5fold.sh
#
# Usage:
#   sbatch experiments/esc50_classification/scripts/train_baseline.sh
#   # Or with custom fold:
#   FOLD=1 sbatch experiments/esc50_classification/scripts/train_baseline.sh

set -e

# Configuration
FOLD=${FOLD:-5}  # Default: use fold 5 as validation
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Data and output paths
DATA_PATH="${DATA_PATH:-$SAFE_ROOT/experiments/full_training/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SAFE_ROOT/experiments/esc50_classification/outputs/baseline_fold${FOLD}}"

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_EPOCHS=${NUM_EPOCHS:-50}
LEARNING_RATE_PROJECTOR=${LR_PROJ:-1e-3}
LEARNING_RATE_ADAPTER=${LR_ADAPTER:-5e-4}
GRADIENT_ACCUMULATION=${GRAD_ACCUM:-8}
FUSION_LAYERS=${FUSION_LAYERS:-"1,5,9,13,17,21"}

# W&B settings
WANDB_PROJECT=${WANDB_PROJECT:-"ESC50-Classification"}
WANDB_NAME="baseline-fold${FOLD}"

echo "========================================"
echo "ESC-50 Classification - Baseline Training"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "SAFE root: $SAFE_ROOT"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Fold: $FOLD (train on others, val on fold $FOLD)"
echo "========================================"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "LR (projector): $LEARNING_RATE_PROJECTOR"
echo "LR (adapter): $LEARNING_RATE_ADAPTER"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Fusion layers: $FUSION_LAYERS"
echo "========================================"

# Activate conda environment
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate safe-env 2>/dev/null || conda activate safe 2>/dev/null || true
elif [ -f ~/.bashrc ]; then
    source ~/.bashrc
    conda activate safe-env 2>/dev/null || conda activate safe 2>/dev/null || true
fi

echo "Python: $(which python)"
echo "========================================"

cd "$SAFE_ROOT"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "experiments/esc50_classification/logs"

# Run training
python train_safe.py \
    --model-config phase1 \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --learning-rate-projector "$LEARNING_RATE_PROJECTOR" \
    --learning-rate-adapter "$LEARNING_RATE_ADAPTER" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION" \
    --fusion-layer-indices "$FUSION_LAYERS" \
    --fp16 \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "$WANDB_NAME" \
    --wandb-tags "esc50,baseline,fold${FOLD},preffn"

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================"
