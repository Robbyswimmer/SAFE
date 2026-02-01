# ModelNet40 Point Cloud Classification

## Overview

**Phase 1B** of the SAFE research agenda: Point cloud classification to validate that the SAFE architecture generalizes beyond audio.

## Dataset: ModelNet40

- **40 classes** of 3D CAD models (airplane, bathtub, bed, chair, etc.)
- **12,311 total models** (9,843 train / 2,468 test)
- **Standard benchmark** for 3D understanding
- **No folds** - single train/test split (unlike ESC-50's 5-fold CV)

## SOTA Targets

| Model | Accuracy | Notes |
|-------|----------|-------|
| PointNeXt | 94.0% | Current SOTA |
| Point-MAE | 93.8% | Self-supervised pre-training |
| PointBERT | 93.2% | BERT-style pre-training |
| PointNet++ | 91.9% | Classic hierarchical |
| PointNet | 89.2% | Original baseline |
| **Our target** | **90%+** | Competitive with specialized models |

## Architecture

Same SAFE architecture as audio classification:
- **Encoder**: PointBERT (frozen, 768-dim, 64 group tokens)
- **Projector**: TokenSetProjector (trainable)
- **Fusion**: Pre-FFN residual at multiple layers (trainable)
- **LLM**: LLaVA-1.5-13B (frozen)
- **Head**: Linear classification head (trainable)

## Quick Start

```bash
# 1. Download ModelNet40 dataset
bash experiments/modelnet40_classification/scripts/download_modelnet40.sh

# 2. Run baseline training
sbatch experiments/modelnet40_classification/scripts/train_baseline.sh

# 3. Check results
cat logs/modelnet40_baseline_*.log | grep "Best Test Accuracy"
```

## Experiments

| Experiment | Status | Accuracy | Notes |
|------------|--------|----------|-------|
| Baseline | Pending | | Default config |
| More fusion layers (10) | Pending | | Based on ESC-50 results |
| More tokens (16→32) | Pending | | |
| More points (1024→2048) | Pending | | |
| Unfreeze encoder (2 layers) | Pending | | |

## Configuration

**Baseline Config**:
```
Encoder: PointBERT (frozen)
LLM: LLaVA-1.5-13B (frozen)
Fusion: Pre-FFN residual
Fusion layers: 1,5,9,13,17,21 (6 layers)
Num tokens: 8
Num points: 1024
Batch size: 16
Epochs: 50
SAFE LR: 6e-5
Head LR: 1e-3
```

## Data Location

```
experiments/full_training/data/modelnet40/
├── modelnet40_train.h5    # Training set (HDF5)
├── modelnet40_test.h5     # Test set (HDF5)
└── README.txt             # Dataset info
```

## Key Differences from ESC-50

| Aspect | ESC-50 (Audio) | ModelNet40 (Point Cloud) |
|--------|----------------|--------------------------|
| Classes | 50 | 40 |
| Samples | 2,000 | 12,311 |
| Evaluation | 5-fold CV | Single train/test split |
| Encoder | CLAP | PointBERT |
| Input | Audio spectrogram | XYZ coordinates |
