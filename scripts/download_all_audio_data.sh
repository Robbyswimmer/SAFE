#!/bin/bash
#
# Download all audio captioning datasets for SAFE training
#
# Datasets:
#   - AudioCaps (~46K train) - YouTube audio with human captions
#   - WavCaps (~400K) - Large-scale with ChatGPT captions
#   - Clotho (~6K) - Freesound clips, 5 human captions each
#   - MACS (~3K) - Multi-annotator diverse sounds
#
# Usage:
#   bash scripts/download_all_audio_data.sh
#
# On SLURM cluster:
#   sbatch scripts/download_all_audio_data.sh
#
# Environment variables:
#   DATA_DIR: Override default data directory (default: experiments/full_training/data)

#SBATCH --job-name=download-audio
#SBATCH --output=logs/download_audio_%j.txt
#SBATCH --error=logs/download_audio_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

set -euo pipefail

# Configuration
DATA_DIR="${DATA_DIR:-$PWD/experiments/full_training/data}"
CONDA_ENV="${CONDA_ENV:-safe-env}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "SAFE Audio Dataset Download"
echo "========================================"
echo "Data directory: ${DATA_DIR}"
echo "========================================"

# Activate conda environment
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    module load anaconda &>/dev/null || true
    source "$HOME/.bashrc" 2>/dev/null || true
fi

echo "Activating conda environment: ${CONDA_ENV}"
conda activate "${CONDA_ENV}" || {
    echo -e "${RED}Failed to activate conda environment: ${CONDA_ENV}${NC}"
    echo "Make sure the environment exists with: conda create -n ${CONDA_ENV} python=3.10"
    exit 1
}

# Verify required packages
python -c "import datasets, requests, soundfile" 2>/dev/null || {
    echo -e "${YELLOW}Installing required packages...${NC}"
    pip install datasets requests soundfile librosa
}

# Create directories
mkdir -p "${DATA_DIR}"
mkdir -p logs

# Track success/failure
FAILED_DATASETS=()

#
# 1. Download AudioCaps
#
echo ""
echo -e "${GREEN}[1/4] Downloading AudioCaps...${NC}"
AUDIOCAPS_DIR="${DATA_DIR}/audiocaps"

if [[ -d "${AUDIOCAPS_DIR}/audio/train" ]] && [[ -f "${AUDIOCAPS_DIR}/AudioCaps_train.json" ]]; then
    echo "   AudioCaps already exists, skipping..."
else
    python scripts/download_audiocaps_hf.py \
        --output-dir "${AUDIOCAPS_DIR}" \
        --splits "train:train,validation:val,test:test" \
        || FAILED_DATASETS+=("AudioCaps")
fi

#
# 2. Download WavCaps (if script exists)
#
echo ""
echo -e "${GREEN}[2/4] Downloading WavCaps...${NC}"
WAVCAPS_DIR="${DATA_DIR}/wavcaps"

if [[ -d "${WAVCAPS_DIR}" ]] && [[ -n "$(ls -A ${WAVCAPS_DIR} 2>/dev/null)" ]]; then
    echo "   WavCaps already exists, skipping..."
elif [[ -f "scripts/download_wavcaps.py" ]]; then
    python scripts/download_wavcaps.py \
        --output-dir "${WAVCAPS_DIR}" \
        || FAILED_DATASETS+=("WavCaps")
else
    echo -e "${YELLOW}   WavCaps download script not found, skipping...${NC}"
    echo "   You may need to download WavCaps manually from:"
    echo "   https://github.com/XinhaoMei/WavCaps"
fi

#
# 3. Download Clotho
#
echo ""
echo -e "${GREEN}[3/4] Downloading Clotho...${NC}"
CLOTHO_DIR="${DATA_DIR}/clotho"

if [[ -d "${CLOTHO_DIR}/audio/train" ]] && [[ -f "${CLOTHO_DIR}/Clotho_train.json" ]]; then
    echo "   Clotho already exists, skipping..."
else
    python scripts/download_clotho.py \
        --output-dir "${CLOTHO_DIR}" \
        --splits "development,validation,evaluation" \
        || FAILED_DATASETS+=("Clotho")
fi

#
# 4. Download MACS
#
echo ""
echo -e "${GREEN}[4/4] Downloading MACS...${NC}"
MACS_DIR="${DATA_DIR}/macs"

if [[ -d "${MACS_DIR}/audio/train" ]] && [[ -f "${MACS_DIR}/MACS_train.json" ]]; then
    echo "   MACS already exists, skipping..."
else
    python scripts/download_macs.py \
        --output-dir "${MACS_DIR}" \
        --train-ratio 0.8 \
        || FAILED_DATASETS+=("MACS")
fi

#
# Summary
#
echo ""
echo "========================================"
echo "Download Summary"
echo "========================================"

# Count samples in each dataset
count_samples() {
    local json_pattern=$1
    local total=0
    for f in ${json_pattern} 2>/dev/null; do
        if [[ -f "$f" ]]; then
            count=$(python -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo "0")
            total=$((total + count))
        fi
    done
    echo $total
}

AUDIOCAPS_COUNT=$(count_samples "${AUDIOCAPS_DIR}/AudioCaps_*.json")
CLOTHO_COUNT=$(count_samples "${CLOTHO_DIR}/Clotho_*.json")
MACS_COUNT=$(count_samples "${MACS_DIR}/MACS_*.json")

echo "Dataset counts:"
echo "  AudioCaps: ${AUDIOCAPS_COUNT} samples"
echo "  Clotho:    ${CLOTHO_COUNT} samples"
echo "  MACS:      ${MACS_COUNT} samples"

# Check for WavCaps (different structure)
if [[ -d "${WAVCAPS_DIR}" ]]; then
    WAVCAPS_FILES=$(find "${WAVCAPS_DIR}" -name "*.json" 2>/dev/null | wc -l)
    echo "  WavCaps:   ${WAVCAPS_FILES} JSON files found"
fi

TOTAL=$((AUDIOCAPS_COUNT + CLOTHO_COUNT + MACS_COUNT))
echo ""
echo "Total (excluding WavCaps): ${TOTAL} samples"

if [[ ${#FAILED_DATASETS[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}Failed datasets: ${FAILED_DATASETS[*]}${NC}"
    echo "Check logs for details."
    exit 1
else
    echo ""
    echo -e "${GREEN}✅ All downloads completed successfully!${NC}"
fi

echo ""
echo "Data location: ${DATA_DIR}"
echo "========================================"
