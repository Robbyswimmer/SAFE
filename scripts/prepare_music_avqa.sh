#!/bin/bash
#SBATCH --job-name=prep-music-avqa
#SBATCH --output=logs/prep_music_avqa_%j.out
#SBATCH --error=logs/prep_music_avqa_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH -p batch

set -euo pipefail

SAFE_ROOT="/data/SalmanAsif/RobbyMoseley/SAFE/SAFE"
DATA_ROOT="$SAFE_ROOT/experiments/full_training/data/music_avqa"
WORKERS=8

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe-env

cd "$SAFE_ROOT"
mkdir -p logs

echo "========================================"
echo "MUSIC-AVQA Data Preparation"
echo "========================================"
echo "Data root: $DATA_ROOT"
echo "Started: $(date)"
echo "========================================"

# Step 1: Extract audio from videos
echo ""
echo "Step 1: Extracting audio from videos..."
echo "========================================"

AUDIO_DIR="$DATA_ROOT/audio"
VIDEO_DIR="$DATA_ROOT/videos"
FRAMES_DIR="$DATA_ROOT/frames"

count=0
total=$(ls "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l)
echo "Total videos: $total"

for video in "$VIDEO_DIR"/*.mp4; do
    base=$(basename "$video" .mp4)
    wav="$AUDIO_DIR/${base}.wav"
    frame="$FRAMES_DIR/${base}.jpg"

    # Extract audio if not exists
    if [ ! -f "$wav" ]; then
        ffmpeg -i "$video" -ac 1 -ar 16000 -vn "$wav" -y -loglevel error 2>/dev/null || true
    fi

    # Extract middle frame if not exists
    if [ ! -f "$frame" ]; then
        duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$video" 2>/dev/null || echo "5")
        midpoint=$(echo "$duration / 2" | bc -l 2>/dev/null || echo "2.5")
        ffmpeg -i "$video" -ss "$midpoint" -vframes 1 "$frame" -y -loglevel error 2>/dev/null || true
    fi

    count=$((count + 1))
    if [ $((count % 100)) -eq 0 ]; then
        echo "  Processed $count / $total (audio: $(ls "$AUDIO_DIR"/*.wav 2>/dev/null | wc -l), frames: $(ls "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l))"
    fi
done

echo "Audio files: $(ls "$AUDIO_DIR"/*.wav 2>/dev/null | wc -l)"
echo "Frame files: $(ls "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l)"

# Step 2: Prepare manifests
echo ""
echo "Step 2: Preparing JSONL manifests..."
echo "========================================"

python3 experiments/avqa_composition/scripts/prepare_avqa_manifests.py \
    --dataset music_avqa \
    --train-json "$DATA_ROOT/metadata/avqa-train.json" \
    --val-json "$DATA_ROOT/metadata/avqa-val.json" \
    --output-root "$DATA_ROOT" \
    --media-root "$DATA_ROOT" \
    --audio-root "$DATA_ROOT/audio" \
    --image-root "$DATA_ROOT/frames" \
    --require-both

echo ""
echo "Manifest files:"
ls -la "$DATA_ROOT/manifests/"

echo ""
echo "========================================"
echo "Preparation complete: $(date)"
echo "========================================"
echo ""
echo "To train, run:"
echo "  DATA_ROOT=$DATA_ROOT sbatch --gres=gpu:1 experiments/avqa_composition/scripts/train_preffn_music_avqa.sh"
