#!/bin/bash
#SBATCH --job-name=kv-sanity
#SBATCH --output=logs/sanity_check_%j.log
#SBATCH --error=logs/sanity_check_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

# KV Augmentation Sanity Check
# Run ONE forward pass to verify audio branch is connected before training

set -e

DATA_PATH=${DATA_PATH:-"/data/SalmanAsif/AVE_Dataset"}

echo "========================================"
echo "KV Augmentation Sanity Check"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Data path: $DATA_PATH"
echo "========================================"

mkdir -p logs

# Activate conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate safe-env
elif [ -f ~/.bashrc ]; then
    source ~/.bashrc
    conda activate safe-env 2>/dev/null || true
fi

echo "Python: $(which python)"
nvidia-smi

# Run sanity check
python scripts/sanity_check_kv_augment.py --data-path "$DATA_PATH"

EXIT_CODE=$?

echo "========================================"
echo "Finished: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SANITY CHECK PASSED - safe to train"
else
    echo "❌ SANITY CHECK FAILED - fix issues before training"
fi
echo "========================================"

exit $EXIT_CODE
