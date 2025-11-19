#!/bin/bash
# Download AudioSetCaps audio files directly from Google Drive

set -e

OUTPUT_DIR=${1:-"data/audiosetcaps"}
TEMP_DIR="${OUTPUT_DIR}_temp"

echo "Installing gdown if needed..."
pip install -q gdown

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

echo "Downloading tar files from Google Drive folder..."
echo "Folder: https://drive.google.com/drive/folders/1ZKyRZw3AhS3HkWivgMqtMODB0TkVPNk5"

# Download entire folder using gdown
# Note: This may require authentication for large folders
cd "$TEMP_DIR"

# Method 1: Try to download the folder directly
echo "Attempting to download folder..."
gdown --folder https://drive.google.com/drive/folders/1ZKyRZw3AhS3HkWivgMqtMODB0TkVPNk5 --remaining-ok

# Extract all tar files
echo ""
echo "Extracting tar files..."
for tarfile in *.tar; do
    if [ -f "$tarfile" ]; then
        echo "Extracting $tarfile..."
        tar -xf "$tarfile" -C "../${OUTPUT_DIR#*/}"
        echo "✓ Extracted $tarfile"

        # Optionally remove tar file to save space
        # rm "$tarfile"
    fi
done

echo ""
echo "✓ Download and extraction complete!"
echo "Audio files in: $OUTPUT_DIR"
echo "Tar files in: $TEMP_DIR (you can delete these to save space)"
