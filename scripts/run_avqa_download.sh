#!/bin/bash
# Run MUSIC-AVQA download in background with proper settings for cluster use

set -e

# Configuration
DATA_DIR="${1:-./data/music_avqa}"
WORKERS="${2:-4}"
RATE_LIMIT="${3:-50K}"
SLEEP_TIME="${4:-3.0}"

echo "=================================================="
echo "MUSIC-AVQA Download Script"
echo "=================================================="
echo "Data directory: $DATA_DIR"
echo "Workers: $WORKERS"
echo "Rate limit: $RATE_LIMIT"
echo "Sleep between downloads: ${SLEEP_TIME}s"
echo "=================================================="
echo ""

# Check if yt-dlp is installed
if ! command -v yt-dlp &> /dev/null; then
    echo "ERROR: yt-dlp not found!"
    echo "Install with: pip install yt-dlp"
    exit 1
fi

# Check if Python script exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/download_music_avqa.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Create data directory
mkdir -p "$DATA_DIR"

# Determine if running interactively or in background
if [ -t 0 ]; then
    # Interactive mode
    echo "Running in INTERACTIVE mode..."
    echo "Press Ctrl+C to stop"
    echo ""

    python3 "$PYTHON_SCRIPT" \
        --data-dir "$DATA_DIR" \
        --workers "$WORKERS" \
        --rate-limit "$RATE_LIMIT" \
        --sleep "$SLEEP_TIME"
else
    # Background mode (e.g., via nohup)
    echo "Running in BACKGROUND mode..."
    echo "Logs will be written to: avqa_download.log"
    echo ""

    python3 -u "$PYTHON_SCRIPT" \
        --data-dir "$DATA_DIR" \
        --workers "$WORKERS" \
        --rate-limit "$RATE_LIMIT" \
        --sleep "$SLEEP_TIME"
fi

echo ""
echo "=================================================="
echo "Download complete!"
echo "Check data at: $DATA_DIR"
echo "=================================================="
