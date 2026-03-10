#!/bin/bash
#SBATCH --job-name=avqa-from-align
#SBATCH --output=logs/avqa_from_align_%j.out
#SBATCH --error=logs/avqa_from_align_%j.err
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
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$SAFE_ROOT/data/music_avqa/manifests/train.jsonl}
VAL_MANIFEST=${VAL_MANIFEST:-$SAFE_ROOT/data/music_avqa/manifests/validation.jsonl}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT/data/music_avqa}
OUTPUT_DIR=${OUTPUT_DIR:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/vision_audio_from_audiocaps_align_rkca_joint_caption16}
INIT_AUDIO_CKPT=${INIT_AUDIO_CKPT:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/audiocaps_align_rkca_joint_caption16/audio_aligned_best.pt}

TRAIN_MODALITY=${TRAIN_MODALITY:-both}
EVAL_MODALITIES=${EVAL_MODALITIES:-both,audio,image}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-8}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-16}
FUSION_GATE=${FUSION_GATE:-0.2}
SEED=${SEED:-42}
MAX_SAMPLES=${MAX_SAMPLES:-0}
MODALITY_AWARE_PROMPTS=${MODALITY_AWARE_PROMPTS:-0}

WANDB_PROJECT=${WANDB_PROJECT:-SAFE-InternVL-AVQA}
WANDB_NAME=${WANDB_NAME:-avqa_from_align_${SLURM_JOB_ID:-local}}
WANDB_TAGS=${WANDB_TAGS:-internvl,rkca,audiocaps-prealign,music-avqa}

cd "$SAFE_ROOT"
mkdir -p "$OUTPUT_DIR" logs

if [[ ! -f "$INIT_AUDIO_CKPT" ]]; then
    echo "FATAL: INIT_AUDIO_CKPT not found: $INIT_AUDIO_CKPT" >&2
    exit 2
fi

echo "========================================"
echo "MUSIC-AVQA Fine-tune from AudioCaps-aligned projector"
echo "========================================"
echo "SAFE root: $SAFE_ROOT"
echo "Model config: $MODEL_CONFIG"
echo "Init audio ckpt: $INIT_AUDIO_CKPT"
echo "Train manifest: $TRAIN_MANIFEST"
echo "Val manifest: $VAL_MANIFEST"
echo "Output dir: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Grad accum: $GRADIENT_ACCUMULATION"
echo "LR: $LEARNING_RATE"
echo "========================================"

EXTRA_FLAGS=()
if [[ "$MAX_SAMPLES" != "0" ]]; then
    EXTRA_FLAGS+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "$MODALITY_AWARE_PROMPTS" == "1" ]]; then
    EXTRA_FLAGS+=(--modality-aware-prompts)
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
    --init-audio-ckpt "$INIT_AUDIO_CKPT" \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_NAME" \
    --wandb-tags "$WANDB_TAGS" \
    "${EXTRA_FLAGS[@]}"

