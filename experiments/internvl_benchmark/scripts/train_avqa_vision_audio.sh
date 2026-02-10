#!/bin/bash
#SBATCH --job-name=internvl-va
#SBATCH --output=logs/internvl_vision_audio_%j.out
#SBATCH --error=logs/internvl_vision_audio_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# InternVL 3.5-8B Vision+Audio Training with SAFE
#
# Uses InternVL 3.5-8B (InternViT-300M + Qwen3-8B) as frozen backbone,
# with built-in InternVL vision pipeline active alongside SAFE audio adapters.
# This enables full generative MUSIC-AVQA with both video frames and audio.
#
# Usage:
#   sbatch experiments/internvl_benchmark/scripts/train_avqa_vision_audio.sh
#   # Or locally:
#   bash experiments/internvl_benchmark/scripts/train_avqa_vision_audio.sh

set -euo pipefail

# Get SAFE root
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SAFE_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

# Conda environment
CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# InternVL-specific env vars
export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export FP16=0
export LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-8B}

# Model configuration
MODEL_CONFIG=${MODEL_CONFIG:-internvl}

# Modality: train on both vision+audio, evaluate on all combinations
TRAIN_MODALITY=${TRAIN_MODALITY:-both}
EVAL_MODALITIES=${EVAL_MODALITIES:-"both,audio,image"}

# Training configuration
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$SAFE_ROOT/data/music_avqa/manifests/train.jsonl}
VAL_MANIFEST=${VAL_MANIFEST:-$SAFE_ROOT/data/music_avqa/manifests/validation.jsonl}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT/data/music_avqa}
OUTPUT_DIR=${OUTPUT_DIR:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/vision_audio}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-16}
FUSION_LAYERS=${FUSION_LAYERS:-"12,24,33"}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}
FUSION_GATE=${FUSION_GATE:-1.0}

# Architectural features
LEARNED_GATE=${LEARNED_GATE:-0}          # 1 to enable per-layer learned gating
LEARNED_GATE_INIT=${LEARNED_GATE_INIT:-0.0}
GRAD_ATTRIBUTION=${GRAD_ATTRIBUTION:-0}  # 1 to log per-layer gradient norms
GRAD_LOG_EVERY=${GRAD_LOG_EVERY:-200}

# W&B settings
WANDB_PROJECT=${WANDB_PROJECT:-"SAFE-InternVL-AVQA"}
WANDB_NAME="internvl_va_${SLURM_JOB_ID:-local}"
WANDB_TAGS=${WANDB_TAGS:-"internvl,vision+audio,avqa"}

echo "========================================"
echo "InternVL 3.5-8B Vision+Audio AVQA Training"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "SAFE root: $SAFE_ROOT"
echo "Model path: $LLM_MODEL_PATH"
echo "Model config: $MODEL_CONFIG"
echo "Train modality: $TRAIN_MODALITY"
echo "Eval modalities: $EVAL_MODALITIES"
echo "========================================"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Fusion layers: $FUSION_LAYERS"
echo "Audio tokens: $NUM_AUDIO_TOKENS"
echo "Fusion gate: $FUSION_GATE"
echo "========================================"

cd "$SAFE_ROOT"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Build optional flags
EXTRA_FLAGS=""
if [ "$LEARNED_GATE" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --learned-gate --learned-gate-init $LEARNED_GATE_INIT"
    echo "Learned gating: ON (init=$LEARNED_GATE_INIT)"
fi
if [ "$GRAD_ATTRIBUTION" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --grad-attribution --grad-log-every $GRAD_LOG_EVERY"
    echo "Gradient attribution: ON (every $GRAD_LOG_EVERY steps)"
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
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION" \
    --freeze-audio-encoder \
    --num-workers 2 \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_NAME" \
    --wandb-tags "$WANDB_TAGS" \
    $EXTRA_FLAGS

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================"
