#!/bin/bash
#SBATCH --job-name=sqa3d-composition
#SBATCH --output=logs/sqa3d_composition_%j.out
#SBATCH --error=logs/sqa3d_composition_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
# pass --gres=gpu:1 at submit time

set -euo pipefail

# Get SAFE root — prefer SLURM_SUBMIT_DIR (set by sbatch to the dir you ran sbatch from)
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SAFE_ROOT="${SAFE_ROOT:-$SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SAFE_ROOT="${SAFE_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
fi

CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# SQA3D defaults to checkpointing-off for stability. Override only if you
# explicitly want to debug/re-enable it.
SAFE_ENABLE_GRADIENT_CHECKPOINTING=0
export SAFE_ENABLE_GRADIENT_CHECKPOINTING

MODEL_CONFIG=${MODEL_CONFIG:-sqa3d_internvl_1b}

# Cluster-friendly defaults so a plain sbatch command works out of the box.
DATA_PATH="${DATA_PATH:-/data/SalmanAsif/RobbyMoseley/SAFE/SAFE/data}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/data/SalmanAsif/RobbyMoseley/SAFE/SAFE/checkpoints}"

case "$MODEL_CONFIG" in
  *1b*)
    MODEL_TAG="sqa3d_1b"
    DEFAULT_BATCH_SIZE=4
    ;;
  *4b*)
    MODEL_TAG="sqa3d_4b"
    DEFAULT_BATCH_SIZE=2
    ;;
  *8b*|*internvl)
    MODEL_TAG="sqa3d_8b"
    DEFAULT_BATCH_SIZE=1
    ;;
  *)
    MODEL_TAG="sqa3d_run"
    DEFAULT_BATCH_SIZE=2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE_DIR/$MODEL_TAG}"
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
EPOCHS=${EPOCHS:-20}
TRAIN_MODALITY=${TRAIN_MODALITY:-interleaved}
EVAL_MODALITIES=${EVAL_MODALITIES:-both,pointcloud,image,text}
FUSION_GATE=${FUSION_GATE:-1.0}
MAX_SAMPLES=${MAX_SAMPLES:-0}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-0}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-0}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-SAFE-SQA3D-Composition}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${MODEL_TAG}_${SLURM_JOB_ID:-local}}
WANDB_TAGS=${WANDB_TAGS:-sqa3d,composition}

mkdir -p logs "$OUTPUT_DIR"

cd "$SAFE_ROOT"

WANDB_ARGS=()
if [[ "$WANDB" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME" --wandb-tags "$WANDB_TAGS")
fi

SAMPLE_ARGS=()
if [[ "$MAX_SAMPLES" != "0" ]]; then
  SAMPLE_ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "$TRAIN_MAX_SAMPLES" != "0" ]]; then
  SAMPLE_ARGS+=(--train-max-samples "$TRAIN_MAX_SAMPLES")
fi
if [[ "$VAL_MAX_SAMPLES" != "0" ]]; then
  SAMPLE_ARGS+=(--val-max-samples "$VAL_MAX_SAMPLES")
fi

python3 "$SAFE_ROOT/experiments/sqa3d_composition/train_sqa3d_composition.py" \
  --model-config "$MODEL_CONFIG" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs "$EPOCHS" \
  --train-modality "$TRAIN_MODALITY" \
  --eval-modalities "$EVAL_MODALITIES" \
  --fusion-gate "$FUSION_GATE" \
  --include-situation \
  --fp16 \
  "${SAMPLE_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
