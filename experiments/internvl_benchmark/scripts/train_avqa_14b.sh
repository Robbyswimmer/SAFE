#!/bin/bash
#SBATCH --job-name=internvl-14b
#SBATCH --output=logs/internvl_14b_%j.out
#SBATCH --error=logs/internvl_14b_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:3

# InternVL 3.5-14B MUSIC-AVQA Training with SAFE
#
# All 14B-specific settings (model path, fusion layers, hidden size) come
# from the internvl_14b config in configs/model_configs.py.
# Only override training hyperparams via env vars if needed.
#
# Usage:
#   sbatch experiments/internvl_benchmark/scripts/train_avqa_14b.sh
#   # With overrides:
#   sbatch --export=ALL,NUM_EPOCHS=5,MAX_SAMPLES=500 experiments/internvl_benchmark/scripts/train_avqa_14b.sh

set -euo pipefail

# Get SAFE root
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SAFE_ROOT="${SAFE_ROOT:-$SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SAFE_ROOT="${SAFE_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
fi

# Conda environment
CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# InternVL 14B env vars — model path baked in
export LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-14B}
export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export FP16=0
export SAFE_DEVICE_MAP=${SAFE_DEVICE_MAP:-auto}
export SAFE_MAX_MEMORY=${SAFE_MAX_MEMORY:-"0=46GiB,1=46GiB,2=46GiB,cpu=160GiB"}
export SAFE_OFFLOAD_FOLDER=${SAFE_OFFLOAD_FOLDER:-$SAFE_ROOT/.hf_offload}

# Fixed: model config and fusion layers for 14B (do NOT override via --export)
MODEL_CONFIG=internvl_14b
FUSION_LAYERS="5,11,17,23,29,35"

# Data paths
DATA_ROOT=${DATA_ROOT:-/data/SalmanAsif/RobbyMoseley/SAFE/SAFE/experiments/full_training/data/music_avqa}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$DATA_ROOT/manifests/train.jsonl}
VAL_MANIFEST=${VAL_MANIFEST:-$DATA_ROOT/manifests/validation.jsonl}
MEDIA_ROOT=${MEDIA_ROOT:-/}
OUTPUT_DIR=${OUTPUT_DIR:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/internvl14b_music}

# Training hyperparams
TRAIN_MODALITY=${TRAIN_MODALITY:-both}
EVAL_MODALITIES=${EVAL_MODALITIES:-"both,audio,image"}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-16}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}
FUSION_GATE=${FUSION_GATE:-0.2}
SEED=${SEED:-42}

# Optional features
LEARNED_GATE=${LEARNED_GATE:-0}
LEARNED_GATE_INIT=${LEARNED_GATE_INIT:-1.5}
GRAD_ATTRIBUTION=${GRAD_ATTRIBUTION:-1}
GRAD_LOG_EVERY=${GRAD_LOG_EVERY:-200}
MAX_SAMPLES=${MAX_SAMPLES:-0}

# W&B
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"SAFE-InternVL-AVQA"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"internvl14b_${SLURM_JOB_ID:-local}"}
WANDB_TAGS=${WANDB_TAGS:-"internvl,14b,vision+audio,avqa"}

echo "========================================"
echo "InternVL 3.5-14B AVQA Training"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "SAFE root: $SAFE_ROOT"
echo "Model path: $LLM_MODEL_PATH"
echo "Model config: $MODEL_CONFIG"
echo "SAFE_DEVICE_MAP: ${SAFE_DEVICE_MAP}"
echo "SAFE_MAX_MEMORY: ${SAFE_MAX_MEMORY}"
echo "Train modality: $TRAIN_MODALITY"
echo "Eval modalities: $EVAL_MODALITIES"
echo "Fusion layers: $FUSION_LAYERS"
echo "========================================"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Fusion gate: $FUSION_GATE"
echo "========================================"

cd "$SAFE_ROOT"
mkdir -p "$OUTPUT_DIR" logs "$SAFE_OFFLOAD_FOLDER"

# Build optional flags
EXTRA_FLAGS=""
if [ "$LEARNED_GATE" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --learned-gate --learned-gate-init $LEARNED_GATE_INIT"
fi
if [ "$GRAD_ATTRIBUTION" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --grad-attribution --grad-log-every $GRAD_LOG_EVERY"
fi
if [ "$MAX_SAMPLES" != "0" ] && [ -n "$MAX_SAMPLES" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --max-samples $MAX_SAMPLES"
fi

WANDB_ARGS=""
if [ "$WANDB" = "1" ]; then
    WANDB_ARGS="--wandb --wandb-project $WANDB_PROJECT --wandb-run-name $WANDB_RUN_NAME --wandb-tags $WANDB_TAGS"
fi

python3 experiments/avqa_composition/train_avqa_composition.py \
    --dataset music_avqa \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --media-root "$MEDIA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --model-config "$MODEL_CONFIG" \
    --train-modality "$TRAIN_MODALITY" \
    --eval-modalities "$EVAL_MODALITIES" \
    --fusion-layers "$FUSION_LAYERS" \
    --num-audio-tokens "$NUM_AUDIO_TOKENS" \
    --fusion-gate "$FUSION_GATE" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --lr-scheduler "$LR_SCHEDULER" \
    --warmup-ratio "$WARMUP_RATIO" \
    --min-lr-ratio "$MIN_LR_RATIO" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION" \
    --freeze-audio-encoder \
    --num-workers 2 \
    $WANDB_ARGS \
    $EXTRA_FLAGS

echo ""
echo "========================================"
echo "Training Complete!"
echo "Output: $OUTPUT_DIR"
echo "========================================"
