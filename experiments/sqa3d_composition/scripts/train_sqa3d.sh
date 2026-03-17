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

DATA_PATH=${DATA_PATH:-./data}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/sqa3d_composition}
MODEL_CONFIG=${MODEL_CONFIG:-sqa3d_internvl_1b}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-20}
TRAIN_MODALITY=${TRAIN_MODALITY:-interleaved}
EVAL_MODALITIES=${EVAL_MODALITIES:-both,pointcloud,image,text}
FUSION_GATE=${FUSION_GATE:-0.2}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-SAFE-SQA3D-Composition}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-sqa3d_composition_${SLURM_JOB_ID:-local}}
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
