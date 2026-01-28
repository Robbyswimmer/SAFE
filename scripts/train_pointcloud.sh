#!/bin/bash
#
# Point Cloud Training Script - SAFE Architecture on Point Clouds
#
# Usage:
#   bash scripts/train_pointcloud.sh                    # Single GPU
#   sbatch --gres=gpu:1 scripts/train_pointcloud.sh     # SLURM
#
# Notes:
# - For full SAFE + LLM probe classification (no decoding), set MODE=llm_probe (default).
# - Extra CLI args can be passed via: sbatch ... scripts/train_pointcloud.sh --some-arg ...

#SBATCH --job-name=SAFE-PointCloud
#SBATCH --output=logs/pointcloud_%j.txt
#SBATCH --error=logs/pointcloud_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH -p gpu

set -euo pipefail

# Memory optimization - prevent CUDA OOM from fragmentation during long runs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

# Environment setup
CONDA_ENV=${CONDA_ENV:-"safe-env"}

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

echo "Activating conda environment '${CONDA_ENV}'"
conda activate "${CONDA_ENV}"

# Verify environment
python --version
which python

# Configuration (override via environment variables)
CONFIG=${CONFIG:-"modelnet40"}
PHASE=${PHASE:-"classification"}
MODE=${MODE:-"llm_probe"}  # llm_probe | classification_head | generative
DATA_PATH=${DATA_PATH:-"data"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/pointcloud"}
NUM_EPOCHS=${NUM_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-4}
SAFE_LR=${SAFE_LR:-""}
HEAD_LR=${HEAD_LR:-""}
HEAD_WEIGHT_DECAY=${HEAD_WEIGHT_DECAY:-""}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-8}
WARMUP_STEPS=${WARMUP_STEPS:-100}
EVAL_EVERY=${EVAL_EVERY:-1}
MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-50}
SAVE_EVERY=${SAVE_EVERY:-5}
NUM_WORKERS=${NUM_WORKERS:-4}
LOG_EVERY=${LOG_EVERY:-10}
FP16=${FP16:-0}
DEBUG=${DEBUG:-0}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-""}
POINTBERT_CHECKPOINT=${POINTBERT_CHECKPOINT:-"checkpoints/pointbert/pointbert_shapenet.pt"}
UNFREEZE_ENCODER_LAST_N=${UNFREEZE_ENCODER_LAST_N:-0}
FUSION_MODE=${FUSION_MODE:-""}  # residual | film | kv_augment (optional override)

# Wandb (optional)
WANDB=${WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-"SAFE"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-""}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_GROUP=${WANDB_GROUP:-""}
WANDB_TAGS=${WANDB_TAGS:-""}
WANDB_NOTES=${WANDB_NOTES:-""}

# LLM probe settings (MODE=llm_probe)
PROBE_POOLING=${PROBE_POOLING:-"mean"}     # last | mean
PROBE_HEAD_TYPE=${PROBE_HEAD_TYPE:-"linear"}  # linear | mlp

# Create logs directory
mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "SAFE Point Cloud Training"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "========================================"
echo "Config: ${CONFIG}"
echo "Phase: ${PHASE}"
echo "Mode: ${MODE}"
echo "Data path: ${DATA_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LR}"
if [[ -n "${SAFE_LR}" ]]; then echo "SAFE LR: ${SAFE_LR}"; fi
if [[ -n "${HEAD_LR}" ]]; then echo "Head LR: ${HEAD_LR}"; fi
if [[ -n "${HEAD_WEIGHT_DECAY}" ]]; then echo "Head WD: ${HEAD_WEIGHT_DECAY}"; fi
echo "Gradient accumulation: ${GRADIENT_ACCUMULATION}"
echo "Log every: ${LOG_EVERY}"
echo "PointBERT checkpoint: ${POINTBERT_CHECKPOINT}"
echo "Unfreeze encoder last N: ${UNFREEZE_ENCODER_LAST_N}"
if [[ -n "${FUSION_MODE}" ]]; then echo "Fusion mode override: ${FUSION_MODE}"; fi
if [[ "${PHASE}" == "classification" && "${MODE}" == "llm_probe" ]]; then
  echo "Probe pooling: ${PROBE_POOLING}"
  echo "Probe head: ${PROBE_HEAD_TYPE}"
fi
if [[ "${WANDB}" == "1" ]]; then
  echo "Wandb: enabled project=${WANDB_PROJECT} run_name=${WANDB_RUN_NAME:-auto}"
fi
echo "========================================"

# Build command
CMD=(
  python train_pointcloud.py
  --config "${CONFIG}"
  --phase "${PHASE}"
  --data-path "${DATA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --num-epochs "${NUM_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --gradient-accumulation "${GRADIENT_ACCUMULATION}"
  --warmup-steps "${WARMUP_STEPS}"
  --eval-every "${EVAL_EVERY}"
  --max-eval-batches "${MAX_EVAL_BATCHES}"
  --save-every "${SAVE_EVERY}"
  --num-workers "${NUM_WORKERS}"
  --log-every "${LOG_EVERY}"
  --unfreeze-encoder-last-n "${UNFREEZE_ENCODER_LAST_N}"
)

if [[ -n "${FUSION_MODE}" ]]; then
  CMD+=(--fusion-mode "${FUSION_MODE}")
fi

if [[ -n "${SAFE_LR}" ]]; then
  CMD+=(--safe-lr "${SAFE_LR}")
fi
if [[ -n "${HEAD_LR}" ]]; then
  CMD+=(--head-lr "${HEAD_LR}")
fi
if [[ -n "${HEAD_WEIGHT_DECAY}" ]]; then
  CMD+=(--head-weight-decay "${HEAD_WEIGHT_DECAY}")
fi

if [[ "${PHASE}" == "classification" ]]; then
  if [[ "${MODE}" == "llm_probe" ]]; then
    CMD+=(--llm-probe-head --probe-pooling "${PROBE_POOLING}" --probe-head-type "${PROBE_HEAD_TYPE}")
  elif [[ "${MODE}" == "classification_head" ]]; then
    CMD+=(--classification-head)
  fi
fi

if [[ "${FP16}" == "1" ]]; then
  CMD+=(--fp16)
fi

if [[ "${DEBUG}" == "1" ]]; then
  CMD+=(--debug)
fi

if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  CMD+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
fi

if [[ -f "${POINTBERT_CHECKPOINT}" ]]; then
  CMD+=(--encoder-checkpoint "${POINTBERT_CHECKPOINT}")
  echo "Using PointBERT checkpoint: ${POINTBERT_CHECKPOINT}"
else
  echo "No PointBERT checkpoint found, training encoder from scratch"
fi

if [[ "${WANDB}" == "1" ]]; then
  CMD+=(--wandb --wandb-project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_RUN_NAME}" ]]; then
    CMD+=(--wandb-run-name "${WANDB_RUN_NAME}")
  fi
  if [[ -n "${WANDB_ENTITY}" ]]; then
    CMD+=(--wandb-entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_GROUP}" ]]; then
    CMD+=(--wandb-group "${WANDB_GROUP}")
  fi
  if [[ -n "${WANDB_TAGS}" ]]; then
    CMD+=(--wandb-tags "${WANDB_TAGS}")
  fi
  if [[ -n "${WANDB_NOTES}" ]]; then
    CMD+=(--wandb-notes "${WANDB_NOTES}")
  fi
fi

if [[ "$#" -gt 0 ]]; then
  echo "Extra args: $*"
  CMD+=("$@")
fi

echo "Running: ${CMD[*]}"
echo "========================================"

# Run training
"${CMD[@]}"

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
