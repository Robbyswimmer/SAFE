#!/bin/bash
# Download and prepare ESC-50 dataset for SAFE audio classification
#
# ESC-50: Environmental Sound Classification dataset
# - 2000 audio clips (5 seconds each)
# - 50 classes (40 clips per class)
# - 5 predefined folds for cross-validation
#
# Source: https://github.com/karolpiczak/ESC-50
#
# Usage:
#   bash experiments/esc50_classification/scripts/download_esc50.sh
#
# Output structure:
#   experiments/full_training/data/esc50/
#     meta/esc50.csv           # Original metadata
#     audio/*.wav              # All 2000 audio files
#     esc50_fold1_train.json   # Train split for fold 1 (folds 2,3,4,5)
#     esc50_fold1_val.json     # Val split for fold 1
#     ... (repeat for folds 2-5)

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-$SAFE_ROOT/experiments/full_training/data/esc50}"
ESC50_URL="https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"

echo "========================================"
echo "ESC-50 Dataset Download & Preparation"
echo "========================================"
echo "SAFE root: $SAFE_ROOT"
echo "Data directory: $DATA_DIR"
echo "========================================"

# Create directories
mkdir -p "$DATA_DIR/meta"
mkdir -p "$DATA_DIR/audio"

# Download ESC-50
TEMP_DIR=$(mktemp -d)
echo "Downloading ESC-50 to temporary directory..."
cd "$TEMP_DIR"

if command -v wget &> /dev/null; then
    wget -q --show-progress "$ESC50_URL" -O esc50.zip
elif command -v curl &> /dev/null; then
    curl -L "$ESC50_URL" -o esc50.zip
else
    echo "Error: Neither wget nor curl found. Please install one."
    exit 1
fi

echo "Extracting..."
unzip -q esc50.zip

# Copy files to data directory
echo "Copying audio files..."
cp ESC-50-master/audio/*.wav "$DATA_DIR/audio/"

echo "Copying metadata..."
cp ESC-50-master/meta/esc50.csv "$DATA_DIR/meta/"

# Cleanup temp directory
rm -rf "$TEMP_DIR"

echo "Creating fold split JSON files..."

# Python script to create JSON splits
python3 << 'PYTHON_SCRIPT'
import csv
import json
import os
from pathlib import Path
from collections import defaultdict

DATA_DIR = os.environ.get('DATA_DIR', 'experiments/full_training/data/esc50')
data_path = Path(DATA_DIR)

# ESC-50 class names (in order of target ID 0-49)
ESC50_CLASSES = [
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects",
    "sheep", "crow", "rain", "sea_waves", "crackling_fire", "crickets",
    "chirping_birds", "water_drops", "wind", "pouring_water", "toilet_flush",
    "thunderstorm", "crying_baby", "sneezing", "clapping", "breathing",
    "coughing", "footsteps", "laughing", "brushing_teeth", "snoring",
    "drinking_sipping", "door_wood_knock", "mouse_click", "keyboard_typing",
    "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner",
    "clock_alarm", "clock_tick", "glass_breaking", "helicopter", "chainsaw",
    "siren", "car_horn", "engine", "train", "church_bells", "airplane",
    "fireworks", "hand_saw"
]

# Read metadata
metadata = []
csv_path = data_path / "meta" / "esc50.csv"
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        metadata.append({
            'filename': row['filename'],
            'fold': int(row['fold']),
            'target': int(row['target']),
            'category': row['category'],
            'esc10': row['esc10'] == 'True',
            'src_file': row['src_file'],
            'take': row['take']
        })

print(f"Loaded {len(metadata)} samples from ESC-50 metadata")

# Group by fold
folds = defaultdict(list)
for item in metadata:
    folds[item['fold']].append(item)

print(f"Folds distribution: {dict((k, len(v)) for k, v in sorted(folds.items()))}")

# Create 5-fold cross-validation splits
for val_fold in range(1, 6):
    train_samples = []
    val_samples = []

    for fold_id, samples in folds.items():
        for sample in samples:
            entry = {
                'id': sample['filename'].replace('.wav', ''),
                'filename': sample['filename'],
                'audio_path': sample['filename'],
                'label': sample['category'],
                'label_id': sample['target'],
                'fold': sample['fold'],
                'esc10': sample['esc10']
            }

            if fold_id == val_fold:
                val_samples.append(entry)
            else:
                train_samples.append(entry)

    # Save train split
    train_path = data_path / f"esc50_fold{val_fold}_train.json"
    with open(train_path, 'w') as f:
        json.dump(train_samples, f, indent=2)
    print(f"Created {train_path.name}: {len(train_samples)} samples")

    # Save val split
    val_path = data_path / f"esc50_fold{val_fold}_val.json"
    with open(val_path, 'w') as f:
        json.dump(val_samples, f, indent=2)
    print(f"Created {val_path.name}: {len(val_samples)} samples")

# Also create a simple 80/20 split (folds 1-4 train, fold 5 val) for quick testing
simple_train = []
simple_val = []
for fold_id, samples in folds.items():
    for sample in samples:
        entry = {
            'id': sample['filename'].replace('.wav', ''),
            'filename': sample['filename'],
            'audio_path': sample['filename'],
            'label': sample['category'],
            'label_id': sample['target'],
            'fold': sample['fold'],
            'esc10': sample['esc10']
        }
        if fold_id == 5:
            simple_val.append(entry)
        else:
            simple_train.append(entry)

with open(data_path / "esc50_train.json", 'w') as f:
    json.dump(simple_train, f, indent=2)
print(f"Created esc50_train.json: {len(simple_train)} samples (folds 1-4)")

with open(data_path / "esc50_val.json", 'w') as f:
    json.dump(simple_val, f, indent=2)
print(f"Created esc50_val.json: {len(simple_val)} samples (fold 5)")

# Save class mapping
class_mapping = {i: name for i, name in enumerate(ESC50_CLASSES)}
with open(data_path / "class_mapping.json", 'w') as f:
    json.dump(class_mapping, f, indent=2)
print("Created class_mapping.json")

print("\nESC-50 dataset preparation complete!")
PYTHON_SCRIPT

echo ""
echo "========================================"
echo "ESC-50 Download Complete!"
echo "========================================"
echo "Data location: $DATA_DIR"
echo ""
echo "Files created:"
ls -la "$DATA_DIR"/*.json 2>/dev/null || echo "  (JSON files)"
echo ""
echo "Audio files: $(ls "$DATA_DIR/audio/"*.wav 2>/dev/null | wc -l | tr -d ' ') wav files"
echo ""
echo "5-fold CV splits:"
echo "  esc50_fold{1-5}_train.json - Training sets (1600 samples each)"
echo "  esc50_fold{1-5}_val.json   - Validation sets (400 samples each)"
echo ""
echo "Simple split (for quick testing):"
echo "  esc50_train.json - Folds 1-4 (1600 samples)"
echo "  esc50_val.json   - Fold 5 (400 samples)"
echo "========================================"
