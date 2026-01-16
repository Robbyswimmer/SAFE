#!/bin/bash
#SBATCH --job-name=ave-clf
#SBATCH --output=logs/ave_classification_%j.log
#SBATCH --error=logs/ave_classification_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu

# AVE Audio Classification Training Script
# Uses EXACT SAME architecture as train_safe.py (CLAP + LLaVA 13B + multi-layer fusion)
# Task: Classification instead of captioning

set -e

# Configuration (override via environment)
DATA_PATH=${DATA_PATH:-"/data/SalmanAsif/AVE_Dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/ave_classification"}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_EPOCHS=${NUM_EPOCHS:-50}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
DROPOUT=${DROPOUT:-0.1}

# SAFE Architecture settings (uses same configs as train_safe.py)
MODEL_CONFIG=${MODEL_CONFIG:-"phase1"}  # phase1 = LLaVA 13B + CLAP
FUSION_LAYER_INDICES=${FUSION_LAYER_INDICES:-""}  # Empty = use config default, or e.g., "6,12,24"

# Conda environment
CONDA_ENV=${CONDA_ENV:-"safe-env"}

echo "========================================"
echo "AVE Audio Classification Training"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Num epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Model config: $MODEL_CONFIG (same as train_safe.py)"
echo "Fusion layers: ${FUSION_LAYER_INDICES:-'(from config)'}"
echo "========================================"

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh || source ~/.bashrc
conda activate "${CONDA_ENV}"

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Set WANDB
export WANDB_PROJECT="SAFE_2"
export WANDB_RUN_NAME="ave-clf-${SLURM_JOB_ID:-local}"

# Build command
CMD="python train_audio_classification.py"
CMD="$CMD --data-path $DATA_PATH"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --num-epochs $NUM_EPOCHS"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --model-config $MODEL_CONFIG"
CMD="$CMD --dropout $DROPOUT"
CMD="$CMD --fp16"
CMD="$CMD --wandb"
CMD="$CMD --wandb-project SAFE_2"
CMD="$CMD --wandb-run-name ave-clf-${SLURM_JOB_ID:-local}"
CMD="$CMD --num-workers 4"
CMD="$CMD --log-interval 10"
CMD="$CMD --save-frequency 10"

# Add fusion layer indices if specified
if [ -n "$FUSION_LAYER_INDICES" ]; then
    CMD="$CMD --fusion-layer-indices $FUSION_LAYER_INDICES"
fi

echo "Running: $CMD"
echo "========================================"

# Run training
$CMD

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
