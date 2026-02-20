#!/bin/bash
#SBATCH --job-name=comp-diag
#SBATCH --output=logs/composability_diag_%j.out
#SBATCH --error=logs/composability_diag_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
# pass --gres=gpu:1 at submit time

set -euo pipefail

if [[ -z "${SAFE_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    SAFE_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SAFE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
  fi
fi

CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

MODEL_CONFIG=${MODEL_CONFIG:-composition_independent}
LLM_MODEL=${LLM_MODEL:-}
DATA_ROOT=${DATA_ROOT:-data/music_avqa}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/composability_diagnostics}
OUTPUT_JSON=${OUTPUT_JSON:-}

COMPOSE_AUDIO_CKPT=${COMPOSE_AUDIO_CKPT:-checkpoints/composition_audio_study/best_model.pt}
COMPOSE_VISION_CKPT=${COMPOSE_VISION_CKPT:-checkpoints/composition_vision_study/best_model.pt}

MAX_SAMPLES=${MAX_SAMPLES:-1000}
PROBE_SAMPLES=${PROBE_SAMPLES:-512}
SUBSPACE_RANK=${SUBSPACE_RANK:-8}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}
FUSION_GATE=${FUSION_GATE:-0.2}
MAX_ANSWER_TOKENS=${MAX_ANSWER_TOKENS:-16}
SEED=${SEED:-42}
SKIP_LAYER_PROBE=${SKIP_LAYER_PROBE:-0}

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
mkdir -p logs "$OUTPUT_DIR" "$SAFE_OFFLOAD_FOLDER"
cd "$SAFE_ROOT"

REQUIRE_CUDA=${REQUIRE_CUDA:-1}
if [[ "$REQUIRE_CUDA" == "1" ]]; then
  python3 -c "import torch,sys; ok=torch.cuda.is_available() and torch.cuda.device_count()>0; print(f'[cuda_check] available={torch.cuda.is_available()} count={torch.cuda.device_count()}'); sys.exit(0 if ok else 2)"
fi

echo "========================================"
echo "Composability Diagnostics"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "SAFE root: ${SAFE_ROOT}"
echo "Model config: ${MODEL_CONFIG}"
echo "LLM model: ${LLM_MODEL:-<from config>}"
echo "Audio ckpt: ${COMPOSE_AUDIO_CKPT}"
echo "Vision ckpt: ${COMPOSE_VISION_CKPT}"
echo "Samples: max=${MAX_SAMPLES} probe=${PROBE_SAMPLES}"
echo "Rank: ${SUBSPACE_RANK}"
echo "Fusion gate: ${FUSION_GATE}"
echo "GPU_COUNT: ${GPU_COUNT}"
echo "SAFE_DEVICE_MAP: ${SAFE_DEVICE_MAP}"
echo "SAFE_MAX_MEMORY: ${SAFE_MAX_MEMORY}"
echo "========================================"

PY_ARGS=(
  --train-manifest "$DATA_ROOT/manifests/train.jsonl"
  --val-manifest "$DATA_ROOT/manifests/validation.jsonl"
  --media-root "$MEDIA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --model-config "$MODEL_CONFIG"
  --num-audio-tokens "$NUM_AUDIO_TOKENS"
  --fusion-gate "$FUSION_GATE"
  --max-answer-tokens "$MAX_ANSWER_TOKENS"
  --compose-audio-ckpt "$COMPOSE_AUDIO_CKPT"
  --compose-vision-ckpt "$COMPOSE_VISION_CKPT"
  --max-samples "$MAX_SAMPLES"
  --probe-samples "$PROBE_SAMPLES"
  --subspace-rank "$SUBSPACE_RANK"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
)

if [[ -n "${LLM_MODEL}" ]]; then
  PY_ARGS+=(--llm-model "$LLM_MODEL")
fi
if [[ -n "${OUTPUT_JSON}" ]]; then
  PY_ARGS+=(--output-json "$OUTPUT_JSON")
fi
if [[ "${SKIP_LAYER_PROBE}" == "1" ]]; then
  PY_ARGS+=(--skip-layer-probe)
fi

python3 "$SAFE_ROOT/experiments/avqa_composition/analyze_composability.py" "${PY_ARGS[@]}"
