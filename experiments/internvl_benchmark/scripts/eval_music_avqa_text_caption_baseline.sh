#!/bin/bash
#SBATCH --job-name=avqa-cap-eval
#SBATCH --output=logs/avqa_cap_eval_%j.out
#SBATCH --error=logs/avqa_cap_eval_%j.err
#SBATCH --time=24:00:00
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
export SAFE_PREFER_FLASH2=0
export FP16=0
export LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-8B}

# Mirror the stable single-GPU InternVL training env.
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

if [[ -z "${SAFE_MAX_MEMORY:-}" && "${SAFE_DEVICE_MAP}" != "none" ]]; then
    SAFE_PER_GPU_MEMORY=${SAFE_PER_GPU_MEMORY:-46GiB}
    SAFE_CPU_MEMORY=${SAFE_CPU_MEMORY:-160GiB}
    _mem_entries=()
    for ((i=0; i<GPU_COUNT; i++)); do
        _mem_entries+=("${i}=${SAFE_PER_GPU_MEMORY}")
    done
    export SAFE_MAX_MEMORY="$(IFS=,; echo "${_mem_entries[*]}"),cpu=${SAFE_CPU_MEMORY}"
fi

export SAFE_OFFLOAD_FOLDER=${SAFE_OFFLOAD_FOLDER:-$SAFE_ROOT/.hf_offload}

MODEL_CONFIG=${MODEL_CONFIG:-rkca_joint}
CHECKPOINT=${CHECKPOINT:-}
MANIFEST=${MANIFEST:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/music_avqa_validation_audio_captions.jsonl}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT/data/music_avqa}
CAPTION_FIELD=${CAPTION_FIELD:-rich_audio_caption}
INPUT_MODE=${INPUT_MODE:-image,caption,both_null,both}
OUTPUT_DIR=${OUTPUT_DIR:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/text_caption_baseline_${CAPTION_FIELD}_${INPUT_MODE}}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_SAMPLES=${MAX_SAMPLES:-0}

cd "$SAFE_ROOT"
mkdir -p "$OUTPUT_DIR" logs
mkdir -p "$SAFE_OFFLOAD_FOLDER"

python3 experiments/internvl_benchmark/eval_music_avqa_text_caption_baseline.py \
  --model-config "$MODEL_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --media-root "$MEDIA_ROOT" \
  --caption-field "$CAPTION_FIELD" \
  --input-mode "$INPUT_MODE" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-samples "$MAX_SAMPLES"
