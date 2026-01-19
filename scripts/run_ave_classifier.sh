#!/bin/bash
#SBATCH --job-name=ave-clf
#SBATCH --output=logs/ave_classifier_%j.log
#SBATCH --error=logs/ave_classifier_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

# AVE Audio Classification - Simple CLAP + MLP approach
# Clean research baseline: frozen CLAP encoder -> MLP classifier -> 28 classes

set -e

# Configuration
DATA_PATH=${DATA_PATH:-"/data/SalmanAsif/AVE_Dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/ave_classifier"}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_EPOCHS=${NUM_EPOCHS:-50}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
DROPOUT=${DROPOUT:-0.3}
HIDDEN_DIM=${HIDDEN_DIM:-512}

echo "========================================"
echo "AVE Audio Classification (CLAP + MLP)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Dropout: $DROPOUT"
echo "Hidden dim: $HIDDEN_DIM"
echo "========================================"

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# WANDB env diagnostics (helps debug missing uploads)
echo "WANDB_MODE: ${WANDB_MODE:-'(unset)'}"
echo "WANDB_DISABLED: ${WANDB_DISABLED:-'(unset)'}"
if [ -n "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY: set"
else
    echo "WANDB_API_KEY: NOT set"
fi

# Activate conda if available
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate safe-env
elif [ -f ~/.bashrc ]; then
    source ~/.bashrc
    conda activate safe-env 2>/dev/null || true
fi

echo "Python: $(which python)"

# Run training
# Set WANDB_API_KEY in your ~/.bashrc or pass --no-wandb to disable
USE_WANDB=${USE_WANDB:-1}
WANDB_MODE=${WANDB_MODE:-online}

WANDB_ARGS=""
if [ "$USE_WANDB" = "1" ]; then
    export WANDB_MODE
    WANDB_ARGS="--wandb --wandb-project AVE-Classification --wandb-run-name clap-mlp-${SLURM_JOB_ID:-local}"
    echo "WANDB enabled (mode=$WANDB_MODE)"
    python -c "import wandb; print('wandb version:', wandb.__version__)" || echo "wandb import failed in this env"
else
    echo "WANDB disabled (USE_WANDB=0)"
fi

python train_ave_classifier.py \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --dropout "$DROPOUT" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-workers 4 \
    $WANDB_ARGS

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
