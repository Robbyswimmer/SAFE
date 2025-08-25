# SAFE Dataset Setup Guide

This guide walks you through downloading and setting up the real datasets for SAFE training.

## Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Run the automated setup script
bash scripts/setup_datasets.sh

# Download AudioCaps audio files (optional, takes time)
pip install yt-dlp pandas
python scripts/download_audiocaps_audio.py
```

### Option 2: Manual Setup
Follow the individual dataset instructions below.

## Dataset Overview

| Dataset | Purpose | Size | Download Method |
|---------|---------|------|-----------------|
| **VQA v2** | Vision-Language baseline | ~19GB | Automated ✅ |
| **AudioCaps** | Audio captioning | ~5GB | Semi-automated ⚠️ |
| **MUSIC-AVQA** | Audio-Visual QA | ~15GB | Manual 🔧 |

## Detailed Instructions

### 1. VQA v2 Dataset ✅ **Fully Automated**

**Purpose**: Vision-language baseline evaluation and retention testing

**What it includes**:
- Questions: 1.1M questions on 200K images
- Annotations: Ground truth answers
- Images: COCO 2014 train/val images

**Download**:
```bash
# Questions and annotations (automatically downloaded)
# Images - you'll be prompted during setup
```

**Final structure**:
```
data/vqa/
├── images/
│   ├── train2014/     # 83K images (13GB)
│   └── val2014/       # 41K images (6GB)
├── questions/
│   ├── v2_OpenEnded_mscoco_train2014_questions.json
│   └── v2_OpenEnded_mscoco_val2014_questions.json
└── annotations/
    ├── v2_mscoco_train2014_annotations.json
    └── v2_mscoco_val2014_annotations.json
```

### 2. AudioCaps Dataset ⚠️ **Semi-Automated**

**Purpose**: Audio captioning for audio projector training

**What it includes**:
- 46K audio clips from YouTube (10 seconds each)
- Human-written captions
- Train/val/test splits

**Download**:
```bash
# Step 1: Metadata (automatically downloaded)
# Already done by setup script

# Step 2: Audio files (manual step)
pip install yt-dlp pandas
python scripts/download_audiocaps_audio.py

# Choose download size:
# - Test: 10 files per split (quick validation)  
# - Small: 100 files per split (demo training)
# - Full: All files (hours of downloading)
```

**Final structure**:
```
data/audiocaps/
├── metadata/
│   ├── train.csv      # 46K samples
│   ├── val.csv        # 495 samples  
│   └── test.csv       # 975 samples
└── audio/
    ├── train/         # .wav files
    ├── val/           # .wav files
    └── test/          # .wav files
```

### 3. MUSIC-AVQA Dataset 🔧 **Manual Download**

**Purpose**: Audio-visual question answering for main training

**What it includes**:
- 45K music performance videos
- 228K audio-visual QA pairs
- Difficulty levels and question types

**Download Steps**:

1. **Visit the official repository**:
   ```
   https://github.com/gewu-lab/MUSIC-AVQA
   ```

2. **Follow their instructions**:
   - May require contacting authors
   - Dataset access might need approval
   - Follow their specific download procedure

3. **Extract to the correct location**:
   ```bash
   # Extract videos to:
   data/avqa/videos/
   
   # Extract metadata to:  
   data/avqa/metadata/
   ```

**Final structure**:
```
data/avqa/
├── videos/
│   ├── sample_001.mp4
│   ├── sample_002.mp4
│   └── ...
└── metadata/
    ├── avqa_train.json
    ├── avqa_val.json
    └── avqa_test.json
```

## Verification

### Check Dataset Status
```bash
# Quick overview
ls -la data/*/

# Detailed file counts
python -c "
import os
from pathlib import Path

data_dir = Path('data')
for dataset in ['vqa', 'audiocaps', 'avqa']:
    path = data_dir / dataset
    if path.exists():
        files = list(path.rglob('*'))
        print(f'{dataset:<12}: {len(files)} files')
    else:
        print(f'{dataset:<12}: Not found')
"
```

### Test Real Data Pipeline
```bash
# Test with whatever data you have
python train_stage_a_curriculum.py --config demo --use-real-data --batch-size 2

# Should show:
# - "Creating real datasets..." 
# - Train/val sample counts
# - Either successful training or graceful fallback
```

## Minimal Setup for Testing

If you just want to validate the real data pipeline without downloading everything:

```bash
# Create synthetic demo data (instant)
python scripts/create_demo_data.py

# Test real data loading
python train_stage_a_curriculum.py --config demo --use-real-data --batch-size 2
```

## Storage Requirements

| Dataset | Metadata | Media Files | Total |
|---------|----------|-------------|-------|
| VQA v2 | ~50MB | ~19GB | ~19GB |
| AudioCaps | ~5MB | ~5GB | ~5GB |
| MUSIC-AVQA | ~100MB | ~15GB | ~15GB |
| **Total** | **~155MB** | **~39GB** | **~39GB** |

## Troubleshooting

### Common Issues

**"Could not load real datasets"**
- Expected behavior if datasets aren't downloaded yet
- Script will fall back to dummy data with clear instructions

**yt-dlp download failures**
- Some YouTube videos may be unavailable
- Script continues with available videos
- Consider using test/small download options first

**MUSIC-AVQA access**
- Dataset may require researcher approval
- Contact the original authors if needed
- Use dummy data or other datasets while waiting

### Performance Tips

1. **Start small**: Use test downloads to validate pipeline
2. **Parallel downloads**: VQA downloads can run while you handle AudioCaps
3. **Storage planning**: Ensure you have 50GB+ free space
4. **Network**: Large downloads work better with stable connections

## Usage After Setup

### Training Commands

```bash
# Demo model with real data
python train_stage_a_curriculum.py --config demo --use-real-data --batch-size 2

# Full model with real data (requires more GPU memory)
python train_stage_a_curriculum.py --config full --use-real-data --batch-size 4

# Mixed: real data with curriculum learning
python train_stage_a_curriculum.py --config demo --use-real-data --curriculum configs/curriculum/default_curriculum.yaml
```

### Data Verification

```bash
# Check what data is available
python -c "
from train_stage_a_curriculum import create_datasets
train_ds, val_ds = create_datasets(use_dummy=False)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}')
"
```

## Next Steps

Once you have datasets set up:

1. **Validate pipeline**: Run demo training to ensure everything works
2. **Monitor training**: Watch for data loading errors or tensor mismatches
3. **Scale up**: Move from demo to full model configuration
4. **Experiment**: Try different curriculum learning strategies

---

## Support

If you encounter issues:

1. **Check error messages**: The scripts provide detailed guidance
2. **Verify structure**: Ensure datasets match expected directory layout  
3. **Test components**: Try individual dataset downloads first
4. **Use fallbacks**: Dummy data works for pipeline validation

The training pipeline is designed to be robust - it will work with whatever data you have available and provide clear feedback about what's missing.