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

# Get SAFE root - use SLURM_SUBMIT_DIR if available (submitted from SAFE root)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SAFE_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

# Data and output paths
DATA_PATH="${DATA_PATH:-$SAFE_ROOT/experiments/full_training/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SAFE_ROOT/experiments/esc50_classification/outputs/baseline_fold${FOLD}}"

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_EPOCHS=${NUM_EPOCHS:-50}
SAFE_LR=${SAFE_LR:-6e-5}
HEAD_LR=${HEAD_LR:-1e-3}
FUSION_LAYERS=${FUSION_LAYERS:-"1,5,9,13,17,21"}

# Regularization / Augmentation (set to 0 to disable)
MIXUP_ALPHA=${MIXUP_ALPHA:-0.0}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.0}
UNFREEZE_CLAP=${UNFREEZE_CLAP:-0}

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
echo "SAFE LR (projector+fusion): $SAFE_LR"
echo "Head LR: $HEAD_LR"
echo "Fusion layers: $FUSION_LAYERS"
echo "Mixup alpha: $MIXUP_ALPHA"
echo "Label smoothing: $LABEL_SMOOTHING"
echo "Unfreeze CLAP layers: $UNFREEZE_CLAP"
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
mkdir -p "$SAFE_ROOT/logs"

# Run training using train_audio_llm_probe.py for classification
python train_audio_llm_probe.py \
    --dataset esc50 \
    --data-path "$DATA_PATH" \
    --fold "$FOLD" \
    --output-dir "$OUTPUT_DIR" \
    --model-config phase1 \
    --fusion-layer-indices "$FUSION_LAYERS" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --safe-learning-rate "$SAFE_LR" \
    --head-learning-rate "$HEAD_LR" \
    --mixup-alpha "$MIXUP_ALPHA" \
    --label-smoothing "$LABEL_SMOOTHING" \
    --unfreeze-clap-layers "$UNFREEZE_CLAP" \
    --fp16 \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_NAME"

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================"
