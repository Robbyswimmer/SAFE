#!/bin/bash
#
# Zero-Forgetting Verification Script
#
# This script verifies the core thesis: SAFE produces IDENTICAL outputs
# to the frozen baseline when audio input is absent.
#
# Usage:
#   bash safe/ablations/run_zero_forgetting.sh                    # Local
#   sbatch safe/ablations/run_zero_forgetting.sh                  # SLURM
#   NUM_SAMPLES=50 bash safe/ablations/run_zero_forgetting.sh     # Quick test

#SBATCH --job-name=SAFE-ZeroForget
#SBATCH --output=logs/zero_forgetting_%j.txt
#SBATCH --error=logs/zero_forgetting_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH -p gpu

set -euo pipefail

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

# Configuration
COCO_DIR=${COCO_DIR:-"$PWD/experiments/full_training/data/coco"}
NUM_SAMPLES=${NUM_SAMPLES:-100}
SEED=${SEED:-42}
USE_SYNTHETIC=${USE_SYNTHETIC:-0}

# Create logs directory
mkdir -p logs

# Log configuration
echo "========================================"
echo "Zero-Forgetting Verification"
echo "========================================"
echo "COCO dir: ${COCO_DIR}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Seed: ${SEED}"
echo "Use synthetic: ${USE_SYNTHETIC}"
echo "========================================"
echo ""

# Build arguments
ARGS=(
    --num_samples "${NUM_SAMPLES}"
    --seed "${SEED}"
)

if [[ "${USE_SYNTHETIC}" != "0" ]]; then
    ARGS+=(--synthetic)
elif [[ -d "${COCO_DIR}" ]]; then
    ARGS+=(--coco_dir "${COCO_DIR}")
else
    echo "[WARN] COCO dir not found, using synthetic samples"
    ARGS+=(--synthetic)
fi

# Run verification
echo "[INFO] Starting zero-forgetting verification..."
python safe/ablations/zero_forgetting_verify.py "${ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "========================================"
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "✓ VERIFICATION PASSED"
else
    echo "✗ VERIFICATION FAILED"
fi
echo "========================================"

exit ${EXIT_CODE}
