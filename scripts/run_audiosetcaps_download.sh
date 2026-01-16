#!/bin/bash
#SBATCH --job-name=audiosetcaps-dl
#SBATCH --output=logs/audiosetcaps_download_%j.log
#SBATCH --error=logs/audiosetcaps_download_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH -p batch

# AudioSetCaps Download Script
# Runs in background on cluster, downloads to specified directory

set -e

# Configuration (override via environment)
OUTPUT_DIR=${OUTPUT_DIR:-"data/audiosetcaps"}
MAX_WORKERS=${MAX_WORKERS:-2}
MAX_SAMPLES=${MAX_SAMPLES:-""}  # Empty = download all
FRESH=${FRESH:-0}
VERBOSE=${VERBOSE:-0}
SKIP_PATHS=${SKIP_PATHS:-""}

# Conda environment
CONDA_ENV=${CONDA_ENV:-"safe-env"}

echo "========================================"
echo "AudioSetCaps Background Downloader"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Output dir: $OUTPUT_DIR"
echo "Max workers: $MAX_WORKERS"
echo "Max samples: ${MAX_SAMPLES:-unlimited}"
echo "========================================"

# Create logs directory if needed
mkdir -p logs

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh || source ~/.bashrc
conda activate "${CONDA_ENV}"

echo "Python: $(which python)"
echo "yt-dlp: $(which yt-dlp)"

# Build command
CMD="python scripts/download_audiosetcaps_robust.py"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --max-workers $MAX_WORKERS"

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

if [ "$FRESH" = "1" ]; then
    CMD="$CMD --fresh"
fi

if [ "$VERBOSE" = "1" ]; then
    CMD="$CMD --verbose"
fi

if [ -n "$SKIP_PATHS" ]; then
    CMD="$CMD --skip-paths $SKIP_PATHS"
fi

echo "Running: $CMD"
echo "========================================"

# Run the downloader
$CMD

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
