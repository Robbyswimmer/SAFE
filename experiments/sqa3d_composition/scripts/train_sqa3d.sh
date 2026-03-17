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

SAFE_ROOT="${SAFE_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"

CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

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
  "${WANDB_ARGS[@]}"
