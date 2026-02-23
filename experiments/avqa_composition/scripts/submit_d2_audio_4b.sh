#!/bin/bash
#SBATCH --job-name=d2-audio-4b
#SBATCH --output=logs/d2_audio_4b_%j.out
#SBATCH --error=logs/d2_audio_4b_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu

# D2: Audio-only adapter on Qwen3-4B
# Independent training — only audio adapter learns.
# Checkpoint used later for naive composition (D5) and diagnostics (E3).
# Save every epoch for transport energy analysis (E5).

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

MODEL_CONFIG=${MODEL_CONFIG:-composition_4b_staggered}
DATA_ROOT=${DATA_ROOT:-data/music_avqa}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/d2_audio_4b}
BATCH_SIZE=${BATCH_SIZE:-1}
EPOCHS=${EPOCHS:-10}
LR=${LR:-5e-5}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}
TRAIN_MODALITY=audio
EVAL_MODALITIES=${EVAL_MODALITIES:-text,audio}
SEED=${SEED:-42}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-SAFE-Composition}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-d2_audio_4b_s${SEED}_${SLURM_JOB_ID:-local}}
WANDB_TAGS=${WANDB_TAGS:-diagnostic,d2,audio_only,4b,independent}
MAX_SAMPLES=${MAX_SAMPLES:-0}
SLIM_PROJECTOR=${SLIM_PROJECTOR:-1}

export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export FP16=0

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

SLIM_ARGS=()
if [ "$SLIM_PROJECTOR" = "0" ]; then
  SLIM_ARGS+=(--no-slim-projector)
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
  --seed "$SEED" \
  --train-modality "$TRAIN_MODALITY" \
  --eval-modalities "$EVAL_MODALITIES" \
  --freeze-audio-encoder \
  "${SLIM_ARGS[@]}" \
  "${MAX_SAMPLES_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
