#!/bin/bash
#SBATCH --job-name=comp-indep-eval
#SBATCH --output=logs/composition_independent_eval_%j.out
#SBATCH --error=logs/composition_independent_eval_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
# pass --gres=gpu:3 at submit time

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
DATA_ROOT=${DATA_ROOT:-data/music_avqa}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/composition_independent_eval}

COMPOSE_AUDIO_CKPT=${COMPOSE_AUDIO_CKPT:-checkpoints/composition_audio_study/best_model.pt}
COMPOSE_VISION_CKPT=${COMPOSE_VISION_CKPT:-checkpoints/composition_vision_study/best_model.pt}

BATCH_SIZE=${BATCH_SIZE:-1}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}
FUSION_GATE=${FUSION_GATE:-0.1}
EVAL_MODALITIES=${EVAL_MODALITIES:-text,audio,image,both}
MAX_ANSWER_TOKENS=${MAX_ANSWER_TOKENS:-16}
MAX_SAMPLES=${MAX_SAMPLES:-0}

LAYER_ADDITIVITY_PROBE=${LAYER_ADDITIVITY_PROBE:-1}
LAYER_PROBE_SAMPLES=${LAYER_PROBE_SAMPLES:-512}
LAYER_PROBE_EVERY=${LAYER_PROBE_EVERY:-1}

WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-SAFE-Composition}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-composition_independent_eval_${SLURM_JOB_ID:-local}}
WANDB_TAGS=${WANDB_TAGS:-composition,independent,staggered,eval}

export SAFE_QWEN_QUANT=${SAFE_QWEN_QUANT:-none}
export SAFE_GRAD_CKPT=${SAFE_GRAD_CKPT:-0}
export FP16=${FP16:-0}

# Derive visible GPU count so sharding config matches the actual Slurm allocation.
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

WANDB_ARGS=()
if [[ "$WANDB" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME" --wandb-tags "$WANDB_TAGS")
fi

MAX_SAMPLES_ARGS=()
if [[ "$MAX_SAMPLES" != "0" ]]; then
  MAX_SAMPLES_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

PROBE_ARGS=()
if [[ "$LAYER_ADDITIVITY_PROBE" == "1" ]]; then
  PROBE_ARGS+=(--layer-additivity-probe --layer-probe-samples "$LAYER_PROBE_SAMPLES" --layer-probe-every "$LAYER_PROBE_EVERY")
fi

python3 "$SAFE_ROOT/experiments/avqa_composition/train_avqa_composition.py" \
  --dataset music_avqa \
  --model-config "$MODEL_CONFIG" \
  --train-manifest "$DATA_ROOT/manifests/train.jsonl" \
  --val-manifest "$DATA_ROOT/manifests/validation.jsonl" \
  --media-root "$MEDIA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs 0 \
  --num-audio-tokens "$NUM_AUDIO_TOKENS" \
  --train-modality both \
  --eval-modalities "$EVAL_MODALITIES" \
  --fusion-gate "$FUSION_GATE" \
  --max-answer-tokens "$MAX_ANSWER_TOKENS" \
  --freeze-audio-encoder \
  --compose-audio-ckpt "$COMPOSE_AUDIO_CKPT" \
  --compose-vision-ckpt "$COMPOSE_VISION_CKPT" \
  "${MAX_SAMPLES_ARGS[@]}" \
  "${PROBE_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
