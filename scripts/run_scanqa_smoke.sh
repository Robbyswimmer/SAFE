#!/bin/bash
#SBATCH --job-name=scanqa_smoke
#SBATCH --output=logs/scanqa_smoke_%j.out
#SBATCH --error=logs/scanqa_smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

# ScanQA Smoke Test — verifies train + eval pipeline end-to-end
# Runs 10 train steps + 10 eval samples, prints loss/metrics, exits.
# Should finish in ~5-10 minutes depending on model load time.
#
# Usage:
#   sbatch scripts/run_scanqa_smoke.sh                          # 8B default
#   MODEL_SIZE=4b sbatch scripts/run_scanqa_smoke.sh            # 4B
#   MODEL_SIZE=1b sbatch scripts/run_scanqa_smoke.sh            # 1B

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

MODEL_SIZE=${MODEL_SIZE:-8b}

case "$MODEL_SIZE" in
    1b|1B)
        LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-1B}
        MODEL_CONFIG=scanqa_internvl_1b
        BATCH_SIZE=4
        ;;
    4b|4B)
        LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-4B}
        MODEL_CONFIG=scanqa_internvl_4b
        BATCH_SIZE=2
        ;;
    *)
        LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-8B}
        MODEL_CONFIG=scanqa_internvl
        BATCH_SIZE=1
        ;;
esac

DATA_ROOT=${DATA_ROOT:-$SAFE_ROOT/data}
OUTPUT_DIR=${OUTPUT_DIR:-$SAFE_ROOT/outputs/scanqa_smoke_${MODEL_SIZE}}

export LLM_MODEL_PATH
export SAFE_QWEN_QUANT=${SAFE_QWEN_QUANT:-none}
export SAFE_GRAD_CKPT=${SAFE_GRAD_CKPT:-0}
export FP16=${FP16:-0}
export SAFE_DEVICE_MAP=${SAFE_DEVICE_MAP:-none}
export SAFE_OFFLOAD_FOLDER=${SAFE_OFFLOAD_FOLDER:-$SAFE_ROOT/.hf_offload}

mkdir -p "$SAFE_ROOT/logs" "$OUTPUT_DIR" "$SAFE_OFFLOAD_FOLDER"
cd "$SAFE_ROOT"

echo "============================================"
echo "ScanQA SMOKE TEST — $MODEL_SIZE"
echo "============================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "MODEL:       $LLM_MODEL_PATH"
echo "CONFIG:      $MODEL_CONFIG"
echo "DATA_ROOT:   $DATA_ROOT"
echo "============================================"

python -c "import torch,sys; print(f'[cuda] avail={torch.cuda.is_available()} count={torch.cuda.device_count()}'); sys.exit(0 if torch.cuda.is_available() else 2)"

python "$SAFE_ROOT/train_scanqa_composition.py" \
    --model-config "$MODEL_CONFIG" \
    --data-path "$DATA_ROOT" \
    --modality both \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --freeze-llm \
    --smoke-test

echo ""
echo "============================================"
echo "SMOKE TEST PASSED"
echo "============================================"
