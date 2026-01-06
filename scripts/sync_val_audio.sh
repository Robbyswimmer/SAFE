#!/bin/bash
# Sync validation audio from cluster to local machine
#
# Usage:
#   ./scripts/sync_val_audio.sh <remote_path>
#
# Examples:
#   ./scripts/sync_val_audio.sh user@cluster:/data/audiocaps/audio/val/
#   ./scripts/sync_val_audio.sh user@cluster:/scratch/audiocaps/audio/val_10s/

set -e

REMOTE_PATH="${1:-}"
LOCAL_DEST="data/audiocaps/audio/val"

if [ -z "$REMOTE_PATH" ]; then
    echo "Usage: $0 <remote_path>"
    echo ""
    echo "Examples:"
    echo "  $0 user@cluster:/data/audiocaps/audio/val/"
    echo "  $0 user@cluster:/scratch/project/audiocaps/audio/val_10s/"
    echo ""
    echo "This will sync audio files to: $LOCAL_DEST"
    exit 1
fi

mkdir -p "$LOCAL_DEST"

echo "============================================================"
echo "Syncing validation audio from cluster"
echo "============================================================"
echo "From: $REMOTE_PATH"
echo "To:   $LOCAL_DEST"
echo ""

rsync -avz --progress --include="*.wav" --include="*.flac" --include="*.mp3" "$REMOTE_PATH" "$LOCAL_DEST/"

echo ""
echo "============================================================"
echo "Done! Audio files synced to $LOCAL_DEST"
echo ""
echo "File count: $(ls -1 "$LOCAL_DEST"/*.wav 2>/dev/null | wc -l) WAV files"
echo ""
echo "Now run:"
echo "  python scripts/download_eval_audio.py \\"
echo "      --csv wandb_export_2026-01-05T10_34_54.909-08_00.csv \\"
echo "      --output eval_audio_review"
echo "============================================================"
