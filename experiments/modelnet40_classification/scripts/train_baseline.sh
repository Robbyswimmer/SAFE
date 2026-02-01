#!/bin/bash
#SBATCH --job-name=mn40-baseline
#SBATCH --output=logs/modelnet40_baseline_%j.log
#SBATCH --error=logs/modelnet40_baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

# ModelNet40 Point Cloud Classification - Baseline Training
#
# Uses SAFE architecture with PointBERT encoder and LLM probe head.
# Same approach as ESC-50 audio classification.
#
# Usage:
#   sbatch experiments/modelnet40_classification/scripts/train_baseline.sh

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

# Data and output paths
DATA_PATH="${DATA_PATH:-$SAFE_ROOT/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SAFE_ROOT/experiments/modelnet40_classification/outputs/${EXPERIMENT_NAME}}"

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_EPOCHS=${NUM_EPOCHS:-100}
SAFE_LR=${SAFE_LR:-6e-5}
HEAD_LR=${HEAD_LR:-1e-3}
FUSION_LAYERS=${FUSION_LAYERS:-"1,5,9,13,17,21"}
NUM_TOKENS=${NUM_TOKENS:-8}
NUM_POINTS=${NUM_POINTS:-1024}

# Early stopping (patience in epochs)
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-5}

# Regularization / Augmentation
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.0}
MIXUP_ALPHA=${MIXUP_ALPHA:-0.0}

# Encoder unfreezing (0 = fully frozen)
UNFREEZE_ENCODER=${UNFREEZE_ENCODER:-0}

# Pre-trained PointBERT checkpoint
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-$SAFE_ROOT/checkpoints/pointbert/pointbert_modelnet40_1024.pt}"

# W&B settings
WANDB_PROJECT=${WANDB_PROJECT:-"ModelNet40-Classification"}
WANDB_TAGS=${WANDB_TAGS:-"modelnet40,baseline"}

echo "========================================"
echo "ModelNet40 Classification - Baseline"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Experiment: $EXPERIMENT_NAME"
echo "SAFE root: $SAFE_ROOT"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "SAFE LR: $SAFE_LR"
echo "Head LR: $HEAD_LR"
echo "Fusion layers: $FUSION_LAYERS"
echo "Num pointcloud tokens: $NUM_TOKENS"
echo "Num points: $NUM_POINTS"
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE"
echo "Label smoothing: $LABEL_SMOOTHING"
echo "Mixup alpha: $MIXUP_ALPHA"
echo "Unfreeze encoder layers: $UNFREEZE_ENCODER"
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
python train_pointcloud.py \
    --config modelnet40 \
    --phase classification \
    --llm-probe-head \
    --probe-pooling last \
    --probe-head-type linear \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --fusion-layer-indices "$FUSION_LAYERS" \
    --num-pointcloud-tokens "$NUM_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --safe-lr "$SAFE_LR" \
    --head-lr "$HEAD_LR" \
    --label-smoothing "$LABEL_SMOOTHING" \
    --mixup-alpha "$MIXUP_ALPHA" \
    --unfreeze-encoder-last-n "$UNFREEZE_ENCODER" \
    --encoder-checkpoint "$ENCODER_CHECKPOINT" \
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
    --max-eval-batches 999 \
    --eval-every 1 \
    --fp16 \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "${EXPERIMENT_NAME}" \
    --wandb-tags "${WANDB_TAGS},${EXPERIMENT_NAME}"

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================"
