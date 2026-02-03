#!/bin/bash
#SBATCH --job-name=download_nuscenes_qa
#SBATCH --output=experiments/nuscenes_qa_composition/logs/download_%j.out
#SBATCH --error=experiments/nuscenes_qa_composition/logs/download_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

# Download NuScenes-QA dataset from HuggingFace
# No registration required!

set -e

CACHE_DIR=${CACHE_DIR:-"experiments/full_training/data/cache"}

echo "=============================================="
echo "Downloading NuScenes-QA Dataset"
echo "=============================================="
echo "Cache directory: ${CACHE_DIR}"
echo "=============================================="

mkdir -p "${CACHE_DIR}"
mkdir -p experiments/nuscenes_qa_composition/logs

# Activate environment (adjust as needed)
source ~/.bashrc
conda activate safe 2>/dev/null || true

python -c "
from safe.data.nuscenes_qa_dataset import download_nuscenes_qa
download_nuscenes_qa(cache_dir='${CACHE_DIR}')
"

echo ""
echo "Download complete!"
echo "Data cached to: ${CACHE_DIR}"
