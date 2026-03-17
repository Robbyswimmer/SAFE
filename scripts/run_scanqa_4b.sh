#!/bin/bash
#SBATCH --job-name=scanqa_4b
#SBATCH --output=logs/scanqa_4b_%j.out
#SBATCH --error=logs/scanqa_4b_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

# ScanQA Joint Training — InternVL 4B
set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SAFE_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SAFE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

CONDA_ENV=${CONDA_ENV:-safe-env}
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV" 2>/dev/null || conda activate safe 2>/dev/null || true
elif [ -f ~/.bashrc ]; then
    source ~/.bashrc
    conda activate "$CONDA_ENV" 2>/dev/null || conda activate safe 2>/dev/null || true
fi

LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-4B}
MODEL_CONFIG=${MODEL_CONFIG:-scanqa_internvl_4b}
DATA_ROOT=${DATA_ROOT:-$SAFE_ROOT/data}
OUTPUT_DIR=${OUTPUT_DIR:-$SAFE_ROOT/outputs/scanqa_4b}
MODALITY=${MODALITY:-both}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
EPOCHS=${EPOCHS:-20}
LR=${LR:-2e-5}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-SAFE-ScanQA}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-scanqa_4b_${SLURM_JOB_ID:-local}}
WANDB_TAGS=${WANDB_TAGS:-scanqa,joint,internvl,4b,pointbert,scaling}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-500}
GATE_WARMUP_EPOCHS=${GATE_WARMUP_EPOCHS:-0}

export LLM_MODEL_PATH
export SAFE_QWEN_QUANT=${SAFE_QWEN_QUANT:-none}
export SAFE_GRAD_CKPT=${SAFE_GRAD_CKPT:-0}
export FP16=${FP16:-0}
export SAFE_DEVICE_MAP=${SAFE_DEVICE_MAP:-none}
export SAFE_OFFLOAD_FOLDER=${SAFE_OFFLOAD_FOLDER:-$SAFE_ROOT/.hf_offload}

mkdir -p "$SAFE_ROOT/logs" "$OUTPUT_DIR" "$SAFE_OFFLOAD_FOLDER"
cd "$SAFE_ROOT"

echo "============================================"
echo "ScanQA 4B Scaling Run"
echo "============================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "MODEL:       $LLM_MODEL_PATH"
echo "CONFIG:      $MODEL_CONFIG"
echo "DATA_ROOT:   $DATA_ROOT"
echo "OUTPUT_DIR:  $OUTPUT_DIR"
echo "BATCH_SIZE:  $BATCH_SIZE"
echo "GRAD_ACCUM:  $GRAD_ACCUM"
echo "EFF_BATCH:   $((BATCH_SIZE * GRAD_ACCUM))"
echo "LR:          $LR"
echo "GATE_WARMUP: $GATE_WARMUP_EPOCHS"
echo "============================================"

python -c "import torch,sys; print(f'[cuda] avail={torch.cuda.is_available()} count={torch.cuda.device_count()}'); sys.exit(0 if torch.cuda.is_available() else 2)"

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
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --num-epochs "$EPOCHS" \
    --safe-lr "$LR" \
    --gate-warmup-epochs "$GATE_WARMUP_EPOCHS" \
    --freeze-llm \
    --max-eval-samples "$MAX_EVAL_SAMPLES" \
    "${WANDB_ARGS[@]}"
