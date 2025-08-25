#!/bin/bash
# Setup script for downloading SAFE training datasets
# Run with: bash scripts/setup_datasets.sh

set -e  # Exit on any error

echo "=================================================="
echo "SAFE Dataset Setup Script"
echo "=================================================="

# Create data directories
echo "Creating data directory structure..."
mkdir -p data/{avqa,audiocaps,vqa}/{videos,audio,images,metadata,questions,annotations}
mkdir -p data/vqa/images/{train2014,val2014}

echo "✓ Directory structure created"

# Function to download file with progress
download_with_progress() {
    local url=$1
    local output=$2
    local description=$3
    
    echo "Downloading $description..."
    if command -v wget &> /dev/null; then
        wget --progress=bar:force:noscroll -O "$output" "$url"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$output" "$url"
    else
        echo "Error: Neither wget nor curl found. Please install one of them."
        exit 1
    fi
}

# Download VQA v2 Dataset
echo ""
echo "=================================================="
echo "DOWNLOADING VQA v2 DATASET"
echo "=================================================="

VQA_BASE_URL="https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa"

# Download VQA questions
echo "1. Downloading VQA v2 questions..."
download_with_progress "${VQA_BASE_URL}/v2_Questions_Train_mscoco.zip" "data/vqa/v2_Questions_Train_mscoco.zip" "VQA Train Questions"
download_with_progress "${VQA_BASE_URL}/v2_Questions_Val_mscoco.zip" "data/vqa/v2_Questions_Val_mscoco.zip" "VQA Val Questions"

# Download VQA annotations  
echo "2. Downloading VQA v2 annotations..."
download_with_progress "${VQA_BASE_URL}/v2_Annotations_Train_mscoco.zip" "data/vqa/v2_Annotations_Train_mscoco.zip" "VQA Train Annotations"
download_with_progress "${VQA_BASE_URL}/v2_Annotations_Val_mscoco.zip" "data/vqa/v2_Annotations_Val_mscoco.zip" "VQA Val Annotations"

# Extract VQA files
echo "3. Extracting VQA files..."
cd data/vqa
for file in *.zip; do
    echo "Extracting $file..."
    unzip -q "$file"
    rm "$file"
done
cd ../..

echo "✓ VQA v2 questions and annotations downloaded"

# Download COCO Images for VQA
echo ""
echo "4. Downloading COCO images for VQA..."
echo "Note: These are large files (13GB + 6GB)"
read -p "Download COCO images now? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    download_with_progress "http://images.cocodataset.org/zips/train2014.zip" "data/vqa/train2014.zip" "COCO Train 2014 Images (13GB)"
    download_with_progress "http://images.cocodataset.org/zips/val2014.zip" "data/vqa/val2014.zip" "COCO Val 2014 Images (6GB)"
    
    echo "Extracting COCO images..."
    cd data/vqa
    unzip -q train2014.zip -d images/
    unzip -q val2014.zip -d images/
    rm train2014.zip val2014.zip
    cd ../..
    echo "✓ COCO images extracted"
else
    echo "Skipping COCO images download. You can download later from:"
    echo "  http://images.cocodataset.org/zips/train2014.zip"
    echo "  http://images.cocodataset.org/zips/val2014.zip"
fi

# AudioCaps Dataset
echo ""
echo "=================================================="
echo "DOWNLOADING AUDIOCAPS DATASET"
echo "=================================================="

# Download AudioCaps metadata
echo "1. Downloading AudioCaps metadata..."
download_with_progress "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/train.csv" "data/audiocaps/metadata/train.csv" "AudioCaps Train CSV"
download_with_progress "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/val.csv" "data/audiocaps/metadata/val.csv" "AudioCaps Val CSV"
download_with_progress "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/test.csv" "data/audiocaps/metadata/test.csv" "AudioCaps Test CSV"

echo "✓ AudioCaps metadata downloaded"

# AudioCaps audio files (requires youtube-dl)
echo ""
echo "2. AudioCaps audio files:"
echo "AudioCaps uses YouTube audio. To download:"
echo ""
echo "# Install yt-dlp"
echo "pip install yt-dlp"
echo ""
echo "# Download audio files (this will take a while)"
echo "python scripts/download_audiocaps_audio.py"
echo ""

# MUSIC-AVQA Dataset
echo ""
echo "=================================================="
echo "MUSIC-AVQA DATASET"
echo "=================================================="

echo "MUSIC-AVQA requires manual download:"
echo ""
echo "1. Visit: https://github.com/gewu-lab/MUSIC-AVQA"
echo "2. Follow their download instructions"  
echo "3. Contact authors if needed for dataset access"
echo "4. Extract to: ./data/avqa/"
echo ""
echo "Expected structure:"
echo "  data/avqa/videos/        # Video files"
echo "  data/avqa/metadata/      # JSON metadata files"
echo ""

# Summary
echo ""
echo "=================================================="
echo "SETUP SUMMARY"
echo "=================================================="

echo "✓ Directory structure created"
echo "✓ VQA v2 questions and annotations downloaded"
if [ -d "data/vqa/images/train2014" ] && [ "$(ls -A data/vqa/images/train2014)" ]; then
    echo "✓ COCO images downloaded"
else
    echo "⚠ COCO images not downloaded (manual step)"
fi
echo "✓ AudioCaps metadata downloaded"
echo "⚠ AudioCaps audio requires youtube-dl (manual step)"
echo "⚠ MUSIC-AVQA requires manual download"

echo ""
echo "📁 Data location: $(pwd)/data"
echo "📊 To check status: ls -la data/*/"
echo ""
echo "🚀 Test with: python train_stage_a_curriculum.py --config demo --use-real-data"
echo ""
echo "=================================================="