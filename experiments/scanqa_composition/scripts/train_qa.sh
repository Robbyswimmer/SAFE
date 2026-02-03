#!/bin/bash
#SBATCH --job-name=scanqa-comp
#SBATCH --output=logs/scanqa_composition_%j.log
#SBATCH --error=logs/scanqa_composition_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

# ScanQA Point Cloud + Image QA Composition Experiment
#
# Tests whether combining point cloud and image modalities
# improves 3D question answering compared to either alone.
#
# Usage:
#   # Point cloud only
#   MODALITY="pointcloud" EXPERIMENT_NAME="pc_only" sbatch train_qa.sh
#
#   # Image only
#   MODALITY="image" EXPERIMENT_NAME="image_only" sbatch train_qa.sh
#
#   # Both (composition)
#   MODALITY="both" EXPERIMENT_NAME="pc_image" sbatch train_qa.sh

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
OUTPUT_DIR="${OUTPUT_DIR:-$SAFE_ROOT/experiments/scanqa_composition/outputs/${EXPERIMENT_NAME}}"

# Model
LLM_MODEL=${LLM_MODEL:-"llava-hf/llava-1.5-7b-hf"}  # 7B for generation
FUSION_LAYERS=${FUSION_LAYERS:-"1,5,9,13,17,21"}
NUM_TOKENS=${NUM_TOKENS:-8}
NUM_POINTS=${NUM_POINTS:-8192}

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}  # Effective batch = 16
NUM_EPOCHS=${NUM_EPOCHS:-20}
SAFE_LR=${SAFE_LR:-1e-5}
LR_SCHEDULER=${LR_SCHEDULER:-"cosine"}
WARMUP_STEPS=${WARMUP_STEPS:-200}

# Generation
MAX_ANSWER_TOKENS=${MAX_ANSWER_TOKENS:-32}

# Encoder settings
UNFREEZE_ENCODER=${UNFREEZE_ENCODER:-0}
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-$SAFE_ROOT/checkpoints/pointbert/pointbert_modelnet40_1024.pt}"
FREEZE_LLM=${FREEZE_LLM:-""}  # Empty = don't freeze (train for generation)

# W&B settings
WANDB_PROJECT=${WANDB_PROJECT:-"ScanQA-Composition"}
WANDB_TAGS=${WANDB_TAGS:-"scanqa,composition,qa"}

echo "========================================"
echo "ScanQA Composition Experiment"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Experiment: $EXPERIMENT_NAME"
echo "Modality: $MODALITY"
echo "SAFE root: $SAFE_ROOT"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"
echo "LLM: $LLM_MODEL"
echo "Batch size: $BATCH_SIZE (x$GRAD_ACCUM accum = $((BATCH_SIZE * GRAD_ACCUM)))"
echo "Epochs: $NUM_EPOCHS"
echo "LR: $SAFE_LR ($LR_SCHEDULER)"
echo "Fusion layers: $FUSION_LAYERS"
echo "Num tokens: $NUM_TOKENS"
echo "Max answer tokens: $MAX_ANSWER_TOKENS"
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

# Build command
CMD="python train_scanqa_composition.py \
    --data-path \"$DATA_PATH\" \
    --output-dir \"$OUTPUT_DIR\" \
    --modality \"$MODALITY\" \
    --llm-model \"$LLM_MODEL\" \
    --fusion-layer-indices \"$FUSION_LAYERS\" \
    --num-pointcloud-tokens \"$NUM_TOKENS\" \
    --num-points \"$NUM_POINTS\" \
    --batch-size \"$BATCH_SIZE\" \
    --gradient-accumulation-steps \"$GRAD_ACCUM\" \
    --num-epochs \"$NUM_EPOCHS\" \
    --safe-lr \"$SAFE_LR\" \
    --lr-scheduler \"$LR_SCHEDULER\" \
    --warmup-steps \"$WARMUP_STEPS\" \
    --max-answer-tokens \"$MAX_ANSWER_TOKENS\" \
    --unfreeze-encoder-last-n \"$UNFREEZE_ENCODER\" \
    --encoder-checkpoint \"$ENCODER_CHECKPOINT\" \
    --max-eval-samples 500 \
    --eval-every 1 \
    --fp16 \
    --wandb \
    --wandb-project \"$WANDB_PROJECT\" \
    --wandb-run-name \"${EXPERIMENT_NAME}\" \
    --wandb-tags \"${WANDB_TAGS},${MODALITY},${EXPERIMENT_NAME}\""

# Add freeze-llm flag if set
if [ -n "$FREEZE_LLM" ]; then
    CMD="$CMD --freeze-llm"
fi

# Run training
eval $CMD

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================"
