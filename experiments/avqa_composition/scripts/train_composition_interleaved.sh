#!/bin/bash
#SBATCH --job-name=comp-interleaved
#SBATCH --output=logs/composition_interleaved_%j.out
#SBATCH --error=logs/composition_interleaved_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
# pass --gres=gpu:1 at submit time

# Composition experiment: Interleaved audio + vision training on Qwen3-8B
# Each epoch: train audio adapters → train vision adapters → evaluate 3 conditions:
#   1. text + audio (audio adapter only)
#   2. text + vision (vision adapter only)
#   3. text + audio + vision (composition)
# Tracks composition gain = accuracy(both) - max(audio, vision) per epoch.

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

MODEL_CONFIG=${MODEL_CONFIG:-composition_study}
DATA_ROOT=${DATA_ROOT:-data/music_avqa}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/composition_interleaved}
BATCH_SIZE=${BATCH_SIZE:-1}
EPOCHS=${EPOCHS:-10}
LR=${LR:-5e-5}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}
TRAIN_MODALITY=interleaved
EVAL_MODALITIES=${EVAL_MODALITIES:-text,audio,image,both}
FUSION_GATE=${FUSION_GATE:-0.2}
SEED=${SEED:-42}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-SAFE-Composition}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-composition_interleaved_${SLURM_JOB_ID:-local}}
WANDB_TAGS=${WANDB_TAGS:-composition,interleaved}
MAX_SAMPLES=${MAX_SAMPLES:-0}
EVAL_DEBUG_SAMPLES=${EVAL_DEBUG_SAMPLES:-0}
MAX_ANSWER_TOKENS=${MAX_ANSWER_TOKENS:-16}
LAYER_ADDITIVITY_PROBE=${LAYER_ADDITIVITY_PROBE:-1}
LAYER_PROBE_SAMPLES=${LAYER_PROBE_SAMPLES:-256}
LAYER_PROBE_EVERY=${LAYER_PROBE_EVERY:-1}

# Optional sequential composition objective (audio->vision compatibility regularizer)
COMPAT_REG_ENABLE=${COMPAT_REG_ENABLE:-0}
COMPAT_REG_LAMBDA=${COMPAT_REG_LAMBDA:-0.05}
COMPAT_REG_RANK=${COMPAT_REG_RANK:-8}
COMPAT_REG_AUDIO_SAMPLES=${COMPAT_REG_AUDIO_SAMPLES:-256}
COMPAT_REG_MIN_SAMPLES=${COMPAT_REG_MIN_SAMPLES:-64}
COMPAT_REG_REFRESH_EVERY=${COMPAT_REG_REFRESH_EVERY:-1}
COMPAT_REG_LAYERS=${COMPAT_REG_LAYERS:-}

# Qwen-specific env vars
export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export FP16=0

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

mkdir -p logs "$OUTPUT_DIR"
mkdir -p "$SAFE_OFFLOAD_FOLDER"

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

EVAL_DEBUG_ARGS=()
if [[ "$EVAL_DEBUG_SAMPLES" != "0" ]]; then
  EVAL_DEBUG_ARGS+=(--eval-debug-samples "$EVAL_DEBUG_SAMPLES")
fi

LAYER_PROBE_ARGS=()
if [[ "$LAYER_ADDITIVITY_PROBE" == "1" ]]; then
  LAYER_PROBE_ARGS+=(--layer-additivity-probe --layer-probe-samples "$LAYER_PROBE_SAMPLES" --layer-probe-every "$LAYER_PROBE_EVERY")
fi

COMPAT_REG_ARGS=()
if [[ "$COMPAT_REG_ENABLE" == "1" ]]; then
  COMPAT_REG_ARGS+=(
    --compat-reg-enable
    --compat-reg-lambda "$COMPAT_REG_LAMBDA"
    --compat-reg-rank "$COMPAT_REG_RANK"
    --compat-reg-audio-samples "$COMPAT_REG_AUDIO_SAMPLES"
    --compat-reg-min-samples "$COMPAT_REG_MIN_SAMPLES"
    --compat-reg-refresh-every "$COMPAT_REG_REFRESH_EVERY"
  )
  if [[ -n "$COMPAT_REG_LAYERS" ]]; then
    COMPAT_REG_ARGS+=(--compat-reg-layers "$COMPAT_REG_LAYERS")
  fi
fi

python3 "$SAFE_ROOT/experiments/avqa_composition/train_avqa_composition.py" \
  --dataset music_avqa \
  --model-config "$MODEL_CONFIG" \
  --train-manifest "$DATA_ROOT/manifests/train.jsonl" \
  --val-manifest "$DATA_ROOT/manifests/validation.jsonl" \
  --media-root "$MEDIA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs "$EPOCHS" \
  --learning-rate "$LR" \
  --num-audio-tokens "$NUM_AUDIO_TOKENS" \
  --fusion-gate "$FUSION_GATE" \
  --seed "$SEED" \
  --train-modality "$TRAIN_MODALITY" \
  --eval-modalities "$EVAL_MODALITIES" \
  --max-answer-tokens "$MAX_ANSWER_TOKENS" \
  --freeze-audio-encoder \
  "${MAX_SAMPLES_ARGS[@]}" \
  "${EVAL_DEBUG_ARGS[@]}" \
  "${LAYER_PROBE_ARGS[@]}" \
  "${COMPAT_REG_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
