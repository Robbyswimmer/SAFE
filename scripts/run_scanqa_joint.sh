#!/bin/bash
#SBATCH --job-name=scanqa_joint
#SBATCH --output=logs/scanqa_joint_%j.out
#SBATCH --error=logs/scanqa_joint_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

# ScanQA Joint Training on BCC:
# PointBERT + native InternVL vision with modality=both.
#
# Usage:
#   sbatch scripts/run_scanqa_joint.sh

set -euo pipefail

# Get SAFE root
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SAFE_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SAFE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# Activate conda environment
CONDA_ENV=${CONDA_ENV:-safe-env}
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV" 2>/dev/null || conda activate safe 2>/dev/null || true
elif [ -f ~/.bashrc ]; then
    source ~/.bashrc
    conda activate "$CONDA_ENV" 2>/dev/null || conda activate safe 2>/dev/null || true
fi

# Configuration
LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-8B}
MODEL_CONFIG=${MODEL_CONFIG:-scanqa_internvl}
DATA_ROOT=${DATA_ROOT:-$SAFE_ROOT/data}
OUTPUT_DIR=${OUTPUT_DIR:-$SAFE_ROOT/outputs/scanqa_joint}
MODALITY=${MODALITY:-both}
BATCH_SIZE=${BATCH_SIZE:-1}
EPOCHS=${EPOCHS:-20}
LR=${LR:-5e-5}
SEED=${SEED:-42}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-SAFE-ScanQA}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-scanqa_joint_${SLURM_JOB_ID:-local}}
WANDB_TAGS=${WANDB_TAGS:-scanqa,joint,internvl,pointbert,multilayer,bcc}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-500}

# InternVL / HF env
export SAFE_QWEN_QUANT=${SAFE_QWEN_QUANT:-none}
export SAFE_GRAD_CKPT=${SAFE_GRAD_CKPT:-0}
export FP16=${FP16:-0}

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

mkdir -p "$SAFE_ROOT/logs" "$OUTPUT_DIR" "$SAFE_OFFLOAD_FOLDER"
cd "$SAFE_ROOT"

echo "============================================"
echo "ScanQA Joint (BCC)"
echo "============================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "SAFE_ROOT:   $SAFE_ROOT"
echo "DATA_ROOT:   $DATA_ROOT"
echo "OUTPUT_DIR:  $OUTPUT_DIR"
echo "MODEL_CONFIG:$MODEL_CONFIG"
echo "MODALITY:    $MODALITY"
echo "CONDA_ENV:   $CONDA_ENV"
echo "Python:      $(command -v python)"
echo "GPU_COUNT:   $GPU_COUNT"
echo "SAFE_MAXMEM: ${SAFE_MAX_MEMORY:-auto}"
echo "============================================"

python -c "import torch,sys; ok=torch.cuda.is_available() and torch.cuda.device_count()>0; print(f'[cuda_check] available={torch.cuda.is_available()} count={torch.cuda.device_count()}'); sys.exit(0 if ok else 2)"

WANDB_ARGS=()
if [[ "$WANDB" == "1" ]]; then
    WANDB_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME" --wandb-tags "$WANDB_TAGS")
fi

python "$SAFE_ROOT/train_scanqa_composition.py" \
    --model-config "$MODEL_CONFIG" \
    --data-path "$DATA_ROOT" \
    --modality "$MODALITY" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$EPOCHS" \
    --safe-lr "$LR" \
    --freeze-llm \
    --max-eval-samples "$MAX_EVAL_SAMPLES" \
    "${WANDB_ARGS[@]}"

