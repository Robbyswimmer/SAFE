#!/bin/bash
#SBATCH --job-name=scannet-comp
#SBATCH --output=logs/scannet_composition_%j.log
#SBATCH --error=logs/scannet_composition_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

# ScanNet Point Cloud + Image Composition Experiment
#
# Tests whether combining point cloud and image modalities
# improves scene classification compared to either alone.
#
# Usage:
#   # Point cloud only
#   MODALITY="pointcloud" EXPERIMENT_NAME="pc_only" sbatch train_composition.sh
#
#   # Image only
#   MODALITY="image" EXPERIMENT_NAME="image_only" sbatch train_composition.sh
#
#   # Both (composition)
#   MODALITY="both" EXPERIMENT_NAME="pc_image" sbatch train_composition.sh

set -e

# Get SAFE root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SAFE_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

# Configuration
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"baseline"}
MODALITY=${MODALITY:-"both"}  # "pointcloud", "image", or "both"

# Data and output paths
DATA_PATH="${DATA_PATH:-$SAFE_ROOT/experiments/full_training/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SAFE_ROOT/experiments/scannet_composition/outputs/${EXPERIMENT_NAME}}"

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_EPOCHS=${NUM_EPOCHS:-100}
SAFE_LR=${SAFE_LR:-6e-5}
HEAD_LR=${HEAD_LR:-1e-3}
FUSION_LAYERS=${FUSION_LAYERS:-"1,5,9,13,17,21"}
NUM_TOKENS=${NUM_TOKENS:-8}
NUM_POINTS=${NUM_POINTS:-8192}

# Learning rate schedule
LR_SCHEDULER=${LR_SCHEDULER:-"constant"}

# Regularization
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.1}
MIXUP_ALPHA=${MIXUP_ALPHA:-0.0}

# Encoder settings
UNFREEZE_ENCODER=${UNFREEZE_ENCODER:-0}
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-$SAFE_ROOT/checkpoints/pointbert/pointbert_modelnet40_1024.pt}"

# W&B settings
WANDB_PROJECT=${WANDB_PROJECT:-"ScanNet-Composition"}
WANDB_TAGS=${WANDB_TAGS:-"scannet,composition"}

echo "========================================"
echo "ScanNet Composition Experiment"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Experiment: $EXPERIMENT_NAME"
echo "Modality: $MODALITY"
echo "SAFE root: $SAFE_ROOT"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "SAFE LR: $SAFE_LR"
echo "Head LR: $HEAD_LR"
echo "Fusion layers: $FUSION_LAYERS"
echo "Num tokens: $NUM_TOKENS"
echo "Num points: $NUM_POINTS"
echo "LR scheduler: $LR_SCHEDULER"
echo "Label smoothing: $LABEL_SMOOTHING"
echo "Encoder checkpoint: $ENCODER_CHECKPOINT"
echo "W&B tags: $WANDB_TAGS"
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

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SAFE_ROOT/logs"

# Run training
python train_scannet_composition.py \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --modality "$MODALITY" \
    --fusion-layer-indices "$FUSION_LAYERS" \
    --num-pointcloud-tokens "$NUM_TOKENS" \
    --num-points "$NUM_POINTS" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --safe-lr "$SAFE_LR" \
    --head-lr "$HEAD_LR" \
    --lr-scheduler "$LR_SCHEDULER" \
    --label-smoothing "$LABEL_SMOOTHING" \
    --unfreeze-encoder-last-n "$UNFREEZE_ENCODER" \
    --encoder-checkpoint "$ENCODER_CHECKPOINT" \
    --max-eval-batches 999 \
    --eval-every 1 \
    --fp16 \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "${EXPERIMENT_NAME}" \
    --wandb-tags "${WANDB_TAGS},${MODALITY},${EXPERIMENT_NAME}"

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================"
