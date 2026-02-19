#!/bin/bash
#SBATCH --job-name=lora-baseline-all
#SBATCH --time=72:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#
# Run all LoRA baseline stages sequentially in a single job:
#   Stage 0: text-only eval
#   Stage 1: audio LoRA training
#   Merge:   merge audio LoRA into base weights
#   Stage 2: vision LoRA training on merged model
#
# Usage:
#   sbatch run_all_stages.sh

set -euo pipefail

SAFE_ROOT="${SAFE_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
SEED="${SEED:-42}"
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

DATA_ROOT="${DATA_ROOT:-${SAFE_ROOT}/data/music_avqa}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${DATA_ROOT}/manifests/train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-${DATA_ROOT}/manifests/validation.jsonl}"
MEDIA_ROOT="${MEDIA_ROOT:-${DATA_ROOT}}"

LLM_MODEL="${LLM_MODEL:-models/Qwen_Qwen3-8B}"
OUTPUT_BASE="${OUTPUT_BASE:-${SAFE_ROOT}/checkpoints/lora_baseline}"

export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export FP16=0

TRAIN_SCRIPT="${SAFE_ROOT}/experiments/lora_baseline/train_lora_baseline.py"
MERGE_SCRIPT="${SAFE_ROOT}/experiments/lora_baseline/merge_and_continue.py"

COMMON_ARGS=(
    --train-manifest "${TRAIN_MANIFEST}"
    --val-manifest "${VAL_MANIFEST}"
    --media-root "${MEDIA_ROOT}"
    --llm-model "${LLM_MODEL}"
    --batch-size "${BATCH_SIZE}"
    --num-epochs "${NUM_EPOCHS}"
    --learning-rate "${LR}"
    --gradient-accumulation-steps "${GRAD_ACCUM}"
    --lora-rank "${LORA_RANK}"
    --lora-alpha "${LORA_ALPHA}"
    --lora-target-modules "${LORA_TARGET_MODULES}"
    --lora-dropout "${LORA_DROPOUT}"
    --seed "${SEED}"
    --num-workers "${NUM_WORKERS}"
    --bf16
    --wandb
    --wandb-project "SAFE-LoRA-Baseline"
)

echo "============================================"
echo " LoRA Baseline — Full Pipeline"
echo " $(date)"
echo "============================================"

# ---- Stage 0: text-only baseline ----
echo ""
echo "============================================"
echo " Stage 0: Text-only baseline evaluation"
echo "============================================"
python3 "${TRAIN_SCRIPT}" \
    --stage 0 \
    --output-dir "${OUTPUT_BASE}/stage0" \
    --train-modality audio \
    --eval-modalities text \
    "${COMMON_ARGS[@]}"

# ---- Stage 1: audio LoRA ----
echo ""
echo "============================================"
echo " Stage 1: Audio LoRA training"
echo "============================================"
python3 "${TRAIN_SCRIPT}" \
    --stage 1 \
    --output-dir "${OUTPUT_BASE}/stage1" \
    --train-modality audio \
    --eval-modalities text,audio \
    "${COMMON_ARGS[@]}"

# ---- Merge: audio LoRA -> base weights ----
echo ""
echo "============================================"
echo " Merge: audio LoRA into base weights"
echo "============================================"
python3 "${MERGE_SCRIPT}" \
    --base-model "${LLM_MODEL}" \
    --lora-checkpoint "${OUTPUT_BASE}/stage1/best_lora" \
    --output-dir "${OUTPUT_BASE}/stage1_merged" \
    --verify

# ---- Stage 2: vision LoRA on merged model ----
echo ""
echo "============================================"
echo " Stage 2: Vision LoRA on merged model"
echo "============================================"
python3 "${TRAIN_SCRIPT}" \
    --stage 2 \
    --output-dir "${OUTPUT_BASE}/stage2" \
    --train-modality image \
    --eval-modalities text,audio,image,both \
    --merged-model-path "${OUTPUT_BASE}/stage1_merged" \
    --audio-projector-path "${OUTPUT_BASE}/stage1/best_audio_projector.pt" \
    "${COMMON_ARGS[@]}"

# ---- Summary ----
echo ""
echo "============================================"
echo " All stages complete. Results:"
echo "============================================"
for f in "${OUTPUT_BASE}"/stage*/stage*_results.json "${OUTPUT_BASE}"/stage*/stage*_final_results.json; do
    if [ -f "$f" ]; then
        echo ""
        echo "--- $(basename "$f") ---"
        cat "$f"
    fi
done

echo ""
echo "[Done] $(date)"
