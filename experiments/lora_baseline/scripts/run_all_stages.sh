#!/bin/bash
#SBATCH --job-name=lora-baseline-all
#SBATCH --output=logs/lora_baseline_all_%j.out
#SBATCH --error=logs/lora_baseline_all_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
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
#   sbatch --gres=gpu:1 experiments/lora_baseline/scripts/run_all_stages.sh

set -euo pipefail

# ---- Cluster paths ----
SAFE_ROOT="/data/SalmanAsif/RobbyMoseley/SAFE/SAFE"

# ---- Conda activation ----
CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# ---- Hyperparameters ----
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

# ---- Data paths ----
DATA_ROOT="${DATA_ROOT:-${SAFE_ROOT}/data/music_avqa}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${DATA_ROOT}/manifests/train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-${DATA_ROOT}/manifests/validation.jsonl}"
MEDIA_ROOT="${MEDIA_ROOT:-/}"

# ---- Model paths ----
LLM_MODEL="${LLM_MODEL:-models/Qwen_Qwen3-8B}"
OUTPUT_BASE="${OUTPUT_BASE:-${SAFE_ROOT}/checkpoints/lora_baseline}"

# ---- Qwen-specific env vars ----
export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export FP16=0

# ---- GPU sharding setup (match composition scripts) ----
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

if [[ -z "${SAFE_MAX_MEMORY:-}" ]]; then
  SAFE_PER_GPU_MEMORY=${SAFE_PER_GPU_MEMORY:-46GiB}
  SAFE_CPU_MEMORY=${SAFE_CPU_MEMORY:-160GiB}
  _mem_entries=()
  for ((i=0; i<GPU_COUNT; i++)); do
    _mem_entries+=("${i}=${SAFE_PER_GPU_MEMORY}")
  done
  export SAFE_MAX_MEMORY="$(IFS=,; echo "${_mem_entries[*]}"),cpu=${SAFE_CPU_MEMORY}"
fi

export SAFE_OFFLOAD_FOLDER=${SAFE_OFFLOAD_FOLDER:-$SAFE_ROOT/.hf_offload}

# ---- Create directories and cd ----
mkdir -p logs "${OUTPUT_BASE}"
mkdir -p "$SAFE_OFFLOAD_FOLDER"
cd "$SAFE_ROOT"

# ---- CUDA check ----
python3 -c "import torch,sys; ok=torch.cuda.is_available() and torch.cuda.device_count()>0; print(f'[cuda_check] available={torch.cuda.is_available()} count={torch.cuda.device_count()}'); sys.exit(0 if ok else 2)"

# ---- Script paths ----
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
echo " SAFE_ROOT:  ${SAFE_ROOT}"
echo " DATA_ROOT:  ${DATA_ROOT}"
echo " MEDIA_ROOT: ${MEDIA_ROOT}"
echo " LLM_MODEL:  ${LLM_MODEL}"
echo " OUTPUT:     ${OUTPUT_BASE}"
echo " GPU_COUNT:  ${GPU_COUNT}"
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
