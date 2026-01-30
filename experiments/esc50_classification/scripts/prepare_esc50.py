#!/usr/bin/env python3
"""
Download and prepare ESC-50 dataset for SAFE audio classification.

ESC-50: Environmental Sound Classification dataset
- 2000 audio clips (5 seconds each)
- 50 classes (40 clips per class)
- 5 predefined folds for cross-validation

Source: https://github.com/karolpiczak/ESC-50

Usage:
    python experiments/esc50_classification/scripts/prepare_esc50.py

    # Or with custom data directory:
    python experiments/esc50_classification/scripts/prepare_esc50.py --data-dir /path/to/data

Output structure:
    experiments/full_training/data/esc50/
        meta/esc50.csv           # Original metadata
        audio/*.wav              # All 2000 audio files
        esc50_fold1_train.json   # Train split for fold 1 (folds 2,3,4,5)
        esc50_fold1_val.json     # Val split for fold 1
        ... (repeat for folds 2-5)
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve


ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"

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


def download_progress(block_num, block_size, total_size):
    """Show download progress."""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
    print(f"\rDownloading: {percent}% ({downloaded // 1024 // 1024}MB)", end="", flush=True)


def download_esc50(data_dir: Path):
    """Download and extract ESC-50 dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "meta").mkdir(exist_ok=True)
    (data_dir / "audio").mkdir(exist_ok=True)

    # Check if already downloaded
    audio_files = list((data_dir / "audio").glob("*.wav"))
    if len(audio_files) >= 2000:
        print(f"ESC-50 already downloaded ({len(audio_files)} audio files found)")
        return True

    print("Downloading ESC-50 dataset...")

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / "esc50.zip"

        # Download
        try:
            urlretrieve(ESC50_URL, zip_path, reporthook=download_progress)
            print()  # Newline after progress
        except Exception as e:
            print(f"\nDownload failed: {e}")
            print("Trying with wget...")
            try:
                subprocess.run(["wget", "-q", "--show-progress", ESC50_URL, "-O", str(zip_path)], check=True)
            except subprocess.CalledProcessError:
                print("Trying with curl...")
                subprocess.run(["curl", "-L", ESC50_URL, "-o", str(zip_path)], check=True)

        # Extract
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        extracted_dir = Path(temp_dir) / "ESC-50-master"

        # Copy audio files
        print("Copying audio files...")
        for wav_file in (extracted_dir / "audio").glob("*.wav"):
            shutil.copy2(wav_file, data_dir / "audio" / wav_file.name)

        # Copy metadata
        print("Copying metadata...")
        shutil.copy2(extracted_dir / "meta" / "esc50.csv", data_dir / "meta" / "esc50.csv")

    print(f"Downloaded {len(list((data_dir / 'audio').glob('*.wav')))} audio files")
    return True


def create_splits(data_dir: Path):
    """Create JSON split files for 5-fold cross-validation."""
    csv_path = data_dir / "meta" / "esc50.csv"

    if not csv_path.exists():
        print(f"Error: Metadata file not found at {csv_path}")
        return False

    # Read metadata
    metadata = []
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
        train_path = data_dir / f"esc50_fold{val_fold}_train.json"
        with open(train_path, 'w') as f:
            json.dump(train_samples, f, indent=2)
        print(f"Created {train_path.name}: {len(train_samples)} samples")

        # Save val split
        val_path = data_dir / f"esc50_fold{val_fold}_val.json"
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

    with open(data_dir / "esc50_train.json", 'w') as f:
        json.dump(simple_train, f, indent=2)
    print(f"Created esc50_train.json: {len(simple_train)} samples (folds 1-4)")

    with open(data_dir / "esc50_val.json", 'w') as f:
        json.dump(simple_val, f, indent=2)
    print(f"Created esc50_val.json: {len(simple_val)} samples (fold 5)")

    # Save class mapping
    class_mapping = {i: name for i, name in enumerate(ESC50_CLASSES)}
    with open(data_dir / "class_mapping.json", 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print("Created class_mapping.json")

    return True


def main():
    parser = argparse.ArgumentParser(description="Download and prepare ESC-50 dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to store ESC-50 data (default: experiments/full_training/data/esc50)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, only create JSON splits (assumes audio already exists)"
    )
    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Find SAFE root
        script_dir = Path(__file__).parent
        safe_root = script_dir.parent.parent.parent
        data_dir = safe_root / "experiments" / "full_training" / "data" / "esc50"

    print("=" * 40)
    print("ESC-50 Dataset Download & Preparation")
    print("=" * 40)
    print(f"Data directory: {data_dir}")
    print("=" * 40)

    # Download
    if not args.skip_download:
        if not download_esc50(data_dir):
            print("Download failed!")
            return 1

    # Create splits
    print("\nCreating fold split JSON files...")
    if not create_splits(data_dir):
        print("Split creation failed!")
        return 1

    print()
    print("=" * 40)
    print("ESC-50 Download Complete!")
    print("=" * 40)
    print(f"Data location: {data_dir}")
    print()
    print("Files created:")
    for json_file in sorted(data_dir.glob("*.json")):
        print(f"  {json_file.name}")
    print()
    print(f"Audio files: {len(list((data_dir / 'audio').glob('*.wav')))} wav files")
    print()
    print("5-fold CV splits:")
    print("  esc50_fold{1-5}_train.json - Training sets (1600 samples each)")
    print("  esc50_fold{1-5}_val.json   - Validation sets (400 samples each)")
    print()
    print("Simple split (for quick testing):")
    print("  esc50_train.json - Folds 1-4 (1600 samples)")
    print("  esc50_val.json   - Fold 5 (400 samples)")
    print("=" * 40)

    return 0


if __name__ == "__main__":
    exit(main())
