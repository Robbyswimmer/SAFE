#!/bin/bash
#SBATCH --job-name=rkca_sub
#SBATCH --output=logs/rkca_subspace_%j.out
#SBATCH --error=logs/rkca_subspace_%j.err
#SBATCH --time=168:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1

# HPCC-adapted RKCA + subspace avoidance experiment
# Target cluster: UCR HPCC (cluster.hpcc.ucr.edu)
# GPUs: A100 (80GB), H100 (80GB), Ada 6000 (48GB)
# Storage: /bigdata/asiflab/rmose009 (1TB)

set -euo pipefail

# ─── HPCC paths ──────────────────────────────────────────────────────
if [[ -z "${SAFE_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    SAFE_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SAFE_ROOT="/bigdata/asiflab/rmose009/SAFE/SAFE"
  fi
fi

# ─── HPCC module + conda setup ──────────────────────────────────────
module load cuda/12.4 2>/dev/null || true

CONDA_ENV=${CONDA_ENV:-safe-env}
# Try common HPCC conda locations
if [[ -f "$HOME/.conda/etc/profile.d/conda.sh" ]]; then
  source "$HOME/.conda/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
else
  module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
  eval "$(conda shell.bash hook 2>/dev/null)" || true
fi
conda activate "$CONDA_ENV"

# ─── Experiment config ───────────────────────────────────────────────
MODEL_CONFIG=${MODEL_CONFIG:-rkca_subspace}
DATA_ROOT=${DATA_ROOT:-${SAFE_ROOT}/data/music_avqa}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/rkca_subspace_avqa}
BATCH_SIZE=${BATCH_SIZE:-1}
EPOCHS=${EPOCHS:-10}
LR=${LR:-5e-5}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}
TRAIN_MODALITY=interleaved
EVAL_MODALITIES=${EVAL_MODALITIES:-text,audio,image,both}
FUSION_GATE=${FUSION_GATE:-1.0}
SEED=${SEED:-42}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-SAFE-Composition}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-rkca_subspace_${SLURM_JOB_ID:-local}}
WANDB_TAGS=${WANDB_TAGS:-rkca,subspace,composition,concat,interleaved,hpcc}
MAX_SAMPLES=${MAX_SAMPLES:-0}
EVAL_DEBUG_SAMPLES=${EVAL_DEBUG_SAMPLES:-0}
MAX_ANSWER_TOKENS=${MAX_ANSWER_TOKENS:-16}

# Subspace avoidance config
RKCA_SUBSPACE=${RKCA_SUBSPACE:-1}
RKCA_SUBSPACE_LAMBDA=${RKCA_SUBSPACE_LAMBDA:-0.05}
RKCA_SUBSPACE_BANK_SIZE=${RKCA_SUBSPACE_BANK_SIZE:-128}
RKCA_SUBSPACE_RANK=${RKCA_SUBSPACE_RANK:-8}

# Qwen-specific env vars
export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export FP16=0

# ─── GPU detection + memory ─────────────────────────────────────────
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

# HPCC A100/H100 = 80GB; Ada 6000 = 48GB
if [[ -z "${SAFE_MAX_MEMORY:-}" ]]; then
  SAFE_PER_GPU_MEMORY=${SAFE_PER_GPU_MEMORY:-76GiB}
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

SUBSPACE_ARGS=()
if [[ "$RKCA_SUBSPACE" == "1" ]]; then
  SUBSPACE_ARGS+=(--rkca-subspace-enable --rkca-subspace-lambda "$RKCA_SUBSPACE_LAMBDA" \
    --rkca-subspace-bank-size "$RKCA_SUBSPACE_BANK_SIZE" --rkca-subspace-rank "$RKCA_SUBSPACE_RANK")
fi

echo "============================================"
echo "  RKCA + Subspace (HPCC) — ${SLURM_JOB_ID:-local}"
echo "============================================"
echo "SAFE_ROOT:    $SAFE_ROOT"
echo "DATA_ROOT:    $DATA_ROOT"
echo "OUTPUT_DIR:   $OUTPUT_DIR"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "GPU_COUNT:    $GPU_COUNT"
echo "SAFE_MAX_MEM: ${SAFE_MAX_MEMORY:-auto}"
echo "SUBSPACE:     RKCA_SUBSPACE=$RKCA_SUBSPACE lambda=$RKCA_SUBSPACE_LAMBDA bank=$RKCA_SUBSPACE_BANK_SIZE rank=$RKCA_SUBSPACE_RANK"
echo "============================================"
echo ""

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
  --no-slim-projector \
  "${MAX_SAMPLES_ARGS[@]}" \
  "${EVAL_DEBUG_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${SUBSPACE_ARGS[@]}"
