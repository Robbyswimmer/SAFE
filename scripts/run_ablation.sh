#!/bin/bash
#SBATCH --job-name=kv-ablation
#SBATCH --output=logs/ablation_%j.log
#SBATCH --error=logs/ablation_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

# KV Augmentation Ablation Study
# Tests: ΔQ enabled vs ΔQ=0 vs null audio
# Also verifies text path is identical when audio disabled
#
# Modes:
#   verify_text   - Just verify wrapped attention == original (no audio)
#   forward_check - Quick single-batch A/B/C (catches wiring issues fast)
#   ablation      - Full ablation on multiple samples
#   both          - verify_text + ablation
#   all           - verify_text + forward_check + ablation (RECOMMENDED)

set -e

DATA_PATH=${DATA_PATH:-"/data/SalmanAsif/AVE_Dataset"}
MODE=${MODE:-"all"}  # verify_text, forward_check, ablation, both, or all
NUM_SAMPLES=${NUM_SAMPLES:-10}

echo "========================================"
echo "KV Augmentation Ablation Study"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Data path: $DATA_PATH"
echo "Mode: $MODE"
echo "Num samples: $NUM_SAMPLES"
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

# Run ablation
python scripts/ablation_kv_augment.py --data-path "$DATA_PATH" --mode "$MODE" --num-samples "$NUM_SAMPLES"

EXIT_CODE=$?

echo "========================================"
echo "Finished: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ ABLATION PASSED"
else
    echo "❌ ABLATION FAILED or inconclusive"
fi
echo "========================================"

exit $EXIT_CODE
