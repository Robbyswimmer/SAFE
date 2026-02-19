#!/bin/bash
#SBATCH --job-name=lora-baseline
#SBATCH --output=logs/lora_baseline_%j.out
#SBATCH --error=logs/lora_baseline_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#
# LoRA baseline experiment: single stage or merge
#
# Usage:
#   Stage 0:  STAGE=0 sbatch --gres=gpu:1 experiments/lora_baseline/scripts/run_lora_baseline.sh
#   Stage 1:  STAGE=1 sbatch --gres=gpu:1 experiments/lora_baseline/scripts/run_lora_baseline.sh
#   Merge:    sbatch --gres=gpu:1 experiments/lora_baseline/scripts/run_lora_baseline.sh merge
#   Stage 2:  STAGE=2 MERGED_MODEL_PATH=... AUDIO_PROJECTOR_PATH=... \
#               sbatch --gres=gpu:1 experiments/lora_baseline/scripts/run_lora_baseline.sh

set -euo pipefail

# ---- Cluster paths ----
SAFE_ROOT="${SAFE_ROOT:-/data/SalmanAsif/RobbyMoseley/SAFE/SAFE}"
if [[ ! -d "$SAFE_ROOT" ]]; then
  echo "ERROR: SAFE_ROOT does not exist: $SAFE_ROOT" >&2
  exit 1
fi

# ---- Conda activation ----
CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# ---- Configuration ----
STAGE="${STAGE:-1}"
SEED="${SEED:-42}"
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"

# LoRA hyperparameters
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Data paths
DATA_ROOT="${DATA_ROOT:-${SAFE_ROOT}/data/music_avqa}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${DATA_ROOT}/manifests/train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-${DATA_ROOT}/manifests/validation.jsonl}"
MEDIA_ROOT="${MEDIA_ROOT:-/}"

# Model paths
LLM_MODEL="${LLM_MODEL:-models/Qwen_Qwen3-8B}"
MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-}"
AUDIO_PROJECTOR_PATH="${AUDIO_PROJECTOR_PATH:-}"

# Output
OUTPUT_BASE="${OUTPUT_BASE:-${SAFE_ROOT}/checkpoints/lora_baseline}"

# Qwen-specific environment
export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export FP16=0

# ---- GPU sharding setup ----
GPU_COUNT=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _gpu_arr <<< "${CUDA_VISIBLE_DEVICES}"
  GPU_COUNT=${#_gpu_arr[@]}
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  if [[ "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
    GPU_COUNT=${SLURM_GPUS_ON_NODE}
  else
    GPU_COUNT=$(echo "${SLURM_GPUS_ON_NODE}" | grep -o '[0-9]\+' | head -n1 || echo 0)
  fi
fi
if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" -le 0 ]]; then
  GPU_COUNT=1
fi

if [[ -z "${SAFE_DEVICE_MAP:-}" ]]; then
  if [[ "${GPU_COUNT}" -le 1 ]]; then
    export SAFE_DEVICE_MAP=none
  else
    export SAFE_DEVICE_MAP=auto
  fi
fi

export SAFE_OFFLOAD_FOLDER=${SAFE_OFFLOAD_FOLDER:-$SAFE_ROOT/.hf_offload}

# ---- Create directories and cd ----
mkdir -p logs "${OUTPUT_BASE}"
mkdir -p "$SAFE_OFFLOAD_FOLDER"
cd "$SAFE_ROOT"

if [[ ! -f "$SAFE_ROOT/experiments/lora_baseline/train_lora_baseline.py" ]]; then
  echo "ERROR: training script missing at $SAFE_ROOT/experiments/lora_baseline/train_lora_baseline.py" >&2
  exit 1
fi
if [[ ! -f "$SAFE_ROOT/experiments/lora_baseline/merge_and_continue.py" ]]; then
  echo "ERROR: merge script missing at $SAFE_ROOT/experiments/lora_baseline/merge_and_continue.py" >&2
  exit 1
fi

# ---- CUDA check ----
python3 -c "import torch,sys; ok=torch.cuda.is_available() and torch.cuda.device_count()>0; print(f'[cuda_check] available={torch.cuda.is_available()} count={torch.cuda.device_count()}'); sys.exit(0 if ok else 2)"

echo "============================================"
echo " LoRA Baseline Experiment"
echo " SAFE_ROOT:  ${SAFE_ROOT}"
echo " STAGE:      ${STAGE}"
echo " SEED:       ${SEED}"
echo " LR:         ${LR}"
echo " BATCH_SIZE: ${BATCH_SIZE}"
echo " GRAD_ACCUM: ${GRAD_ACCUM}"
echo " NUM_EPOCHS: ${NUM_EPOCHS}"
echo " LORA_RANK:  ${LORA_RANK}"
echo " LORA_ALPHA: ${LORA_ALPHA}"
echo " DATA_ROOT:  ${DATA_ROOT}"
echo " MEDIA_ROOT: ${MEDIA_ROOT}"
echo " GPU_COUNT:  ${GPU_COUNT}"
echo "============================================"

# ---- Handle merge command ----
if [ "${1:-}" = "merge" ]; then
    LORA_CKPT="${LORA_CKPT:-${OUTPUT_BASE}/stage1/best_lora}"
    MERGE_OUTPUT="${MERGE_OUTPUT:-${OUTPUT_BASE}/stage1_merged}"

    echo "[Merge] Merging audio LoRA into base model..."
    echo "  Base model:    ${LLM_MODEL}"
    echo "  LoRA ckpt:     ${LORA_CKPT}"
    echo "  Output:        ${MERGE_OUTPUT}"

    python3 "${SAFE_ROOT}/experiments/lora_baseline/merge_and_continue.py" \
        --base-model "${LLM_MODEL}" \
        --lora-checkpoint "${LORA_CKPT}" \
        --output-dir "${MERGE_OUTPUT}" \
        --verify

    echo "[Merge] Done. Merged model at: ${MERGE_OUTPUT}"
    exit 0
fi

# ---- Run training/eval ----
OUTPUT_DIR="${OUTPUT_BASE}/stage${STAGE}"
mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()

case "${STAGE}" in
    0)
        echo "[Stage 0] Text-only baseline evaluation"
        TRAIN_MODALITY="audio"
        EXTRA_ARGS+=(--eval-modalities text)
        ;;
    1)
        echo "[Stage 1] Audio LoRA training"
        TRAIN_MODALITY="audio"
        EXTRA_ARGS+=(--eval-modalities text,audio)
        ;;
    2)
        echo "[Stage 2] Vision LoRA training (on merged model)"
        TRAIN_MODALITY="image"
        EXTRA_ARGS+=(--eval-modalities text,audio,image,both)

        if [ -z "${MERGED_MODEL_PATH}" ]; then
            echo "ERROR: MERGED_MODEL_PATH is required for stage 2"
            exit 1
        fi
        EXTRA_ARGS+=(--merged-model-path "${MERGED_MODEL_PATH}")

        if [ -n "${AUDIO_PROJECTOR_PATH}" ]; then
            EXTRA_ARGS+=(--audio-projector-path "${AUDIO_PROJECTOR_PATH}")
        fi
        ;;
    *)
        echo "ERROR: Unknown stage ${STAGE}"
        exit 1
        ;;
esac

python3 "${SAFE_ROOT}/experiments/lora_baseline/train_lora_baseline.py" \
    --stage "${STAGE}" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --val-manifest "${VAL_MANIFEST}" \
    --media-root "${MEDIA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --llm-model "${LLM_MODEL}" \
    --train-modality "${TRAIN_MODALITY}" \
    --batch-size "${BATCH_SIZE}" \
    --num-epochs "${NUM_EPOCHS}" \
    --learning-rate "${LR}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --lora-rank "${LORA_RANK}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-target-modules "${LORA_TARGET_MODULES}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --seed "${SEED}" \
    --num-workers "${NUM_WORKERS}" \
    --bf16 \
    --wandb \
    --wandb-project "SAFE-LoRA-Baseline" \
    "${EXTRA_ARGS[@]}"

echo "[Done] Stage ${STAGE} complete. Results in ${OUTPUT_DIR}"
