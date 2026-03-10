#!/bin/bash
#SBATCH --job-name=audiocaps-align
#SBATCH --output=logs/audiocaps_align_%j.out
#SBATCH --error=logs/audiocaps_align_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SAFE_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export FP16=0
export LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-8B}

MODEL_CONFIG=${MODEL_CONFIG:-rkca_joint_caption16}
DATA_PATH=${DATA_PATH:-$SAFE_ROOT/data}
OUTPUT_DIR=${OUTPUT_DIR:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/audiocaps_align_rkca_joint_caption16}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
VAL_SPLIT=${VAL_SPLIT:-val}
BATCH_SIZE=${BATCH_SIZE:-1}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-2}
NUM_EPOCHS=${NUM_EPOCHS:-5}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
LEARNING_RATE_PROJECTOR=${LEARNING_RATE_PROJECTOR:-1e-4}
LEARNING_RATE_ADAPTER=${LEARNING_RATE_ADAPTER:-5e-5}
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}
USE_WAVCAPS=${USE_WAVCAPS:-0}
WAVCAPS_RATIO=${WAVCAPS_RATIO:-0.5}

WANDB_PROJECT=${WANDB_PROJECT:-SAFE-InternVL-Caption}
WANDB_NAME=${WANDB_NAME:-audiocaps_align_${SLURM_JOB_ID:-local}}
WANDB_TAGS=${WANDB_TAGS:-internvl,rkca,captioning,audiocaps,prealign}

cd "$SAFE_ROOT"
mkdir -p "$OUTPUT_DIR" logs

echo "========================================"
echo "AudioCaps Pre-alignment: InternVL 3.5-8B + RKCA concat"
echo "========================================"
echo "SAFE root: $SAFE_ROOT"
echo "Model config: $MODEL_CONFIG"
echo "LLM path: $LLM_MODEL_PATH"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Grad accum: $GRADIENT_ACCUMULATION_STEPS"
echo "LR projector: $LEARNING_RATE_PROJECTOR"
echo "LR adapter: $LEARNING_RATE_ADAPTER"
echo "========================================"

ARGS=(
    --model-config "$MODEL_CONFIG"
    --data-path "$DATA_PATH"
    --output-dir "$OUTPUT_DIR"
    --train-split "$TRAIN_SPLIT"
    --val-split "$VAL_SPLIT"
    --num-epochs "$NUM_EPOCHS"
    --batch-size "$BATCH_SIZE"
    --val-batch-size "$VAL_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --num-workers "$NUM_WORKERS"
    --learning-rate-projector "$LEARNING_RATE_PROJECTOR"
    --learning-rate-adapter "$LEARNING_RATE_ADAPTER"
    --warmup-steps 500
    --proj-scale-min 0.5
    --max-new-tokens 32
    --eval-repetition-penalty 1.05
    --eval-no-repeat-ngram-size 3
    --wandb
    --wandb-project "$WANDB_PROJECT"
    --wandb-name "$WANDB_NAME"
    --wandb-tags "$WANDB_TAGS"
)

if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
    ARGS+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi
if [[ "$USE_WAVCAPS" == "1" ]]; then
    ARGS+=(--use-wavcaps --wavcaps-ratio "$WAVCAPS_RATIO")
fi

python3 train_safe.py "${ARGS[@]}"

BEST_CKPT="$OUTPUT_DIR/checkpoint_best.pt"
LAST_CKPT="$OUTPUT_DIR/checkpoint_last.pt"
REUSABLE_CKPT="$OUTPUT_DIR/audio_aligned_best.pt"

if [[ -f "$BEST_CKPT" ]]; then
    cp "$BEST_CKPT" "$REUSABLE_CKPT"
elif [[ -f "$LAST_CKPT" ]]; then
    cp "$LAST_CKPT" "$REUSABLE_CKPT"
else
    echo "FATAL: no checkpoint found in $OUTPUT_DIR" >&2
    exit 2
fi

printf '%s\n' "$REUSABLE_CKPT" > "$OUTPUT_DIR/latest_audio_aligned_checkpoint.txt"
echo "[save] reusable aligned checkpoint -> $REUSABLE_CKPT"
echo "[save] pointer -> $OUTPUT_DIR/latest_audio_aligned_checkpoint.txt"

