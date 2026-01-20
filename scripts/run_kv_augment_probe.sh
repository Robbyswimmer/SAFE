#!/bin/bash
#SBATCH --job-name=kv-aug-probe
#SBATCH --output=logs/kv_augment_probe_%j.log
#SBATCH --error=logs/kv_augment_probe_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

# KV Augmentation Linear Probe Training
# Tests if KV augmentation enables the frozen LLM to produce discriminative hidden states

set -e

# Configuration
DATA_PATH=${DATA_PATH:-"/data/SalmanAsif/AVE_Dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/kv_augment_probe"}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_EPOCHS=${NUM_EPOCHS:-30}
# CRITICAL: SAFE LR > Head LR to force SAFE to learn (not head)
# Default was backwards (head=1e-3, safe=5e-4) which let head dominate
SAFE_LR=${SAFE_LR:-1e-3}
HEAD_LR=${HEAD_LR:-1e-4}
# ΔQ (query adapter) LR - needs to be balanced
# Too high (5e-3) → ΔQ explodes to 300%+ and destroys signal
# Too low → ΔQ stays near zero and doesn't learn
# Sweet spot: same as SAFE LR or slightly higher
DELTA_Q_LR=${DELTA_Q_LR:-1e-3}
# Freeze head for first N steps to force SAFE to learn discriminative features
HEAD_WARMUP_STEPS=${HEAD_WARMUP_STEPS:-500}
# audio_attn pools at positions with highest audio attention mass
# This is the most sensitive pooling for detecting if audio affects the LLM
POOLING=${POOLING:-"audio_attn"}
# Head type: "linear" or "mlp" (2-layer MLP may capture nonlinear mappings better)
HEAD_TYPE=${HEAD_TYPE:-"mlp"}
# Pool layers: concatenate hidden states from these layers (e.g., "16,24,32")
# Leave empty for last layer only
POOL_LAYERS=${POOL_LAYERS:-""}

echo "========================================"
echo "KV Augmentation Linear Probe Training"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "SAFE LR: $SAFE_LR (projector + K/V adapters)"
echo "Head LR: $HEAD_LR"
echo "ΔQ LR: $DELTA_Q_LR (query adapter - should be highest for fast attention learning)"
echo "Head warmup steps: $HEAD_WARMUP_STEPS (head frozen during warmup)"
echo "Pooling: $POOLING"
echo "Head type: $HEAD_TYPE"
echo "Pool layers: ${POOL_LAYERS:-'(last layer only)'}"
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
WANDB_RUN_NAME="kv-augment-probe-${SLURM_JOB_ID:-local}"

# Run training with kv_augment config
# Build optional args
POOL_LAYERS_ARG=""
if [ -n "$POOL_LAYERS" ]; then
    POOL_LAYERS_ARG="--pool-layers $POOL_LAYERS"
fi

python train_audio_llm_probe.py \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --safe-learning-rate "$SAFE_LR" \
    --head-learning-rate "$HEAD_LR" \
    --delta-q-learning-rate "$DELTA_Q_LR" \
    --head-warmup-steps "$HEAD_WARMUP_STEPS" \
    --model-config kv_augment \
    --fusion-layer-indices "16,24,32" \
    --pooling "$POOLING" \
    --head-type "$HEAD_TYPE" \
    $POOL_LAYERS_ARG \
    --fp16 \
    --wandb \
    --wandb-project SAFE_2 \
    --wandb-run-name "$WANDB_RUN_NAME" \
    --num-workers 4 \
    --log-interval 10

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
