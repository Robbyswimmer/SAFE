#!/bin/bash
#SBATCH --job-name=inspect_ckpt
#SBATCH --output=logs/inspect_%j.out
#SBATCH --error=logs/inspect_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate safe-env

# Run ID argument
RUN_ID=$1
CHECKPOINT_NAME=$2

# Locate checkpoint
if [ -z "$CHECKPOINT_NAME" ]; then
    # Find latest checkpoint
    CHECKPOINT_PATH=$(ls -t experiments/full_training/runs/$RUN_ID/*/checkpoints/*.pt | head -n 1)
else
    CHECKPOINT_PATH="experiments/full_training/runs/$RUN_ID/*/checkpoints/$CHECKPOINT_NAME"
    # Resolve wildcard if needed
    CHECKPOINT_PATH=$(ls $CHECKPOINT_PATH | head -n 1)
fi

echo "Inspecting checkpoint: $CHECKPOINT_PATH"
python scripts/inspect_checkpoint.py "$CHECKPOINT_PATH"
