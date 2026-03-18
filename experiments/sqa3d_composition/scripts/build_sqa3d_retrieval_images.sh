#!/bin/bash
#SBATCH --job-name=sqa3d-retrieval
#SBATCH --output=logs/sqa3d_retrieval_%j.out
#SBATCH --error=logs/sqa3d_retrieval_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
# pass --gres=gpu:1 at submit time

set -euo pipefail

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

DATA_ROOT="${DATA_ROOT:-/data/SalmanAsif/RobbyMoseley/SAFE/SAFE/data}"
FRAMES_ROOT="${FRAMES_ROOT:-/home/csgrad/rmose009/bigdata/SAFE/SAFE/data/scannet/frames_25k}"
SPLIT="${SPLIT:-val}"
MAX_QUESTIONS="${MAX_QUESTIONS:-300}"
MAX_CANDIDATE_FRAMES="${MAX_CANDIDATE_FRAMES:-16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"

mkdir -p logs
cd "$SAFE_ROOT"

ARGS=(
  --data-root "$DATA_ROOT"
  --frames-root "$FRAMES_ROOT"
  --split "$SPLIT"
  --max-candidate-frames "$MAX_CANDIDATE_FRAMES"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
)

if [[ "$MAX_QUESTIONS" != "0" ]]; then
  ARGS+=(--max-questions "$MAX_QUESTIONS")
fi

echo "============================================"
echo "  SQA3D Retrieval Images — ${SLURM_JOB_ID:-local}"
echo "============================================"
echo "SAFE_ROOT:             $SAFE_ROOT"
echo "DATA_ROOT:             $DATA_ROOT"
echo "FRAMES_ROOT:           $FRAMES_ROOT"
echo "SPLIT:                 $SPLIT"
echo "MAX_QUESTIONS:         $MAX_QUESTIONS"
echo "MAX_CANDIDATE_FRAMES:  $MAX_CANDIDATE_FRAMES"
echo "BATCH_SIZE:            $BATCH_SIZE"
echo "DEVICE:                $DEVICE"
echo "============================================"

python3 "$SAFE_ROOT/experiments/sqa3d_composition/scripts/build_sqa3d_retrieval_images.py" "${ARGS[@]}"
