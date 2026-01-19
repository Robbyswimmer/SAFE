#!/bin/bash
#SBATCH --job-name=kv-aug-clf
#SBATCH --output=logs/kv_augment_clf_%j.log
#SBATCH --error=logs/kv_augment_clf_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

# KV Augmentation Audio Classification Training
# Tests the new KV augmentation fusion mode on AVE 28-class classification

set -e

# Configuration
DATA_PATH=${DATA_PATH:-"/data/SalmanAsif/AVE_Dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/kv_augment_classification"}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_EPOCHS=${NUM_EPOCHS:-30}
LEARNING_RATE=${LEARNING_RATE:-5e-4}

echo "========================================"
echo "KV Augmentation Classification Training"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "========================================"

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Activate conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate safe-env
elif [ -f ~/.bashrc ]; then
    source ~/.bashrc
    conda activate safe-env 2>/dev/null || true
fi

echo "Python: $(which python)"

# WANDB
export WANDB_PROJECT="SAFE_2"
WANDB_RUN_NAME="kv-augment-clf-${SLURM_JOB_ID:-local}"

# Run training with kv_augment config
python train_audio_classification.py \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --model-config kv_augment \
    --fp16 \
    --wandb \
    --wandb-project SAFE_2 \
    --wandb-run-name "$WANDB_RUN_NAME" \
    --num-workers 4 \
    --log-interval 10 \
    --save-frequency 5

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
