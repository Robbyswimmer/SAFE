#!/bin/bash
#SBATCH --job-name=esc50-5fold
#SBATCH --output=logs/esc50_5fold_%j.log
#SBATCH --error=logs/esc50_5fold_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

# ESC-50 Audio Classification - 5-Fold Cross-Validation
#
# Standard ESC-50 evaluation protocol:
# - Train on 4 folds, validate on 1 fold
# - Repeat for all 5 folds
# - Report: mean accuracy ± std
#
# SOTA targets:
#   BEATs: 98.1%
#   CLAP: 96.7%
#   AST: 95.7%
#   Human: 81.3%
#   Our target: 90%+
#
# Usage:
#   sbatch experiments/esc50_classification/scripts/train_5fold.sh
#   # Or with custom config:
#   EXPERIMENT_NAME=ablation_more_layers sbatch experiments/esc50_classification/scripts/train_5fold.sh

set -e

# Configuration
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"baseline"}

# Get SAFE root - use SLURM_SUBMIT_DIR if available (submitted from SAFE root)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SAFE_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

# Data and output paths
DATA_PATH="${DATA_PATH:-$SAFE_ROOT/experiments/full_training/data}"
OUTPUT_BASE="${OUTPUT_DIR:-$SAFE_ROOT/experiments/esc50_classification/outputs/${EXPERIMENT_NAME}}"

# Training hyperparameters (can override via environment)
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_EPOCHS=${NUM_EPOCHS:-50}
LEARNING_RATE_PROJECTOR=${LR_PROJ:-1e-3}
LEARNING_RATE_ADAPTER=${LR_ADAPTER:-5e-4}
GRADIENT_ACCUMULATION=${GRAD_ACCUM:-8}
FUSION_LAYERS=${FUSION_LAYERS:-"1,5,9,13,17,21"}

# W&B settings
WANDB_PROJECT=${WANDB_PROJECT:-"ESC50-Classification"}

echo "========================================"
echo "ESC-50 Classification - 5-Fold CV"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Experiment: $EXPERIMENT_NAME"
echo "SAFE root: $SAFE_ROOT"
echo "Data path: $DATA_PATH"
echo "Output base: $OUTPUT_BASE"
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

# Create output directories
mkdir -p "$OUTPUT_BASE"
mkdir -p "$SAFE_ROOT/logs"

# Array to store fold accuracies
declare -a FOLD_ACCURACIES

# Train and evaluate each fold
for FOLD in 1 2 3 4 5; do
    echo ""
    echo "========================================"
    echo "FOLD $FOLD / 5"
    echo "========================================"
    echo "Training on folds: $(seq -s, 1 5 | sed "s/,$FOLD,/,/" | sed "s/^$FOLD,//" | sed "s/,$FOLD$//")"
    echo "Validating on fold: $FOLD"
    echo "========================================"

    FOLD_OUTPUT_DIR="${OUTPUT_BASE}/fold${FOLD}"
    mkdir -p "$FOLD_OUTPUT_DIR"

    # Run training for this fold
    python train_safe.py \
        --model-config phase1 \
        --data-path "$DATA_PATH" \
        --output-dir "$FOLD_OUTPUT_DIR" \
        --batch-size "$BATCH_SIZE" \
        --num-epochs "$NUM_EPOCHS" \
        --learning-rate-projector "$LEARNING_RATE_PROJECTOR" \
        --learning-rate-adapter "$LEARNING_RATE_ADAPTER" \
        --gradient-accumulation-steps "$GRADIENT_ACCUMULATION" \
        --fusion-layer-indices "$FUSION_LAYERS" \
        --fp16 \
        --wandb \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-name "${EXPERIMENT_NAME}-fold${FOLD}" \
        --wandb-tags "esc50,${EXPERIMENT_NAME},fold${FOLD},preffn,5fold"

    echo ""
    echo "Fold $FOLD training complete. Output: $FOLD_OUTPUT_DIR"
    echo "========================================"
done

echo ""
echo "========================================"
echo "5-Fold Cross-Validation Complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "To compute final accuracy, run:"
echo "  python experiments/esc50_classification/scripts/aggregate_results.py $OUTPUT_BASE"
echo ""
echo "Or manually check each fold:"
for FOLD in 1 2 3 4 5; do
    echo "  Fold $FOLD: $OUTPUT_BASE/fold${FOLD}/metrics.json"
done
echo ""
echo "========================================"
