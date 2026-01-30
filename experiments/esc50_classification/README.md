# ESC-50 Audio Classification Experiment

**Phase 1A** of the SAFE Research Agenda: Audio classification with Pre-FFN fusion.

## Dataset

**ESC-50**: Environmental Sound Classification
- 2000 audio clips (5 seconds each, 44.1kHz)
- 50 classes (40 clips per class)
- 5 predefined folds for cross-validation
- Source: https://github.com/karolpiczak/ESC-50

### Classes (50 total)
Animals, Natural sounds, Human non-speech, Interior sounds, Exterior sounds

## SOTA Targets

| Model | Accuracy | Reference |
|-------|----------|-----------|
| BEATs | 98.1% | Current SOTA |
| CLAP | 96.7% | Audio-text pretrained |
| AST | 95.7% | Audio Spectrogram Transformer |
| Human | 81.3% | Human baseline |
| **Our target** | **90%+** | |

## Quick Start

```bash
# 1. Download and prepare ESC-50 dataset
bash experiments/esc50_classification/scripts/download_esc50.sh

# 2. Run baseline training (single fold, quick test)
sbatch experiments/esc50_classification/scripts/train_baseline.sh

# 3. Run full 5-fold cross-validation
sbatch experiments/esc50_classification/scripts/train_5fold.sh
```

## Directory Structure

```
experiments/esc50_classification/
├── README.md                    # This file
├── scripts/
│   ├── download_esc50.sh        # Download and prepare dataset
│   ├── train_baseline.sh        # Single fold training
│   └── train_5fold.sh           # Full 5-fold CV
├── outputs/                     # Training outputs (gitignored)
│   ├── baseline_fold5/          # Baseline single fold
│   └── baseline/                # Full 5-fold results
│       ├── fold1/
│       ├── fold2/
│       ├── fold3/
│       ├── fold4/
│       └── fold5/
└── logs/                        # SLURM logs (gitignored)
```

## Data Structure (after download)

```
experiments/full_training/data/esc50/
├── meta/
│   └── esc50.csv                # Original metadata
├── audio/                       # All 2000 wav files
│   ├── 1-100032-A-0.wav
│   ├── 1-100038-A-14.wav
│   └── ...
├── esc50_train.json             # Simple split: folds 1-4 (1600 samples)
├── esc50_val.json               # Simple split: fold 5 (400 samples)
├── esc50_fold1_train.json       # 5-fold CV: train on 2,3,4,5
├── esc50_fold1_val.json         # 5-fold CV: val on 1
├── esc50_fold2_train.json       # 5-fold CV: train on 1,3,4,5
├── esc50_fold2_val.json         # 5-fold CV: val on 2
├── ... (repeat for folds 3,4,5)
└── class_mapping.json           # Class ID to name mapping
```

## Training Configuration

### Baseline Settings

| Parameter | Value |
|-----------|-------|
| Model config | phase1 (Pre-FFN) |
| Fusion layers | 1,5,9,13,17,21 |
| Batch size | 4 |
| Gradient accumulation | 8 |
| Effective batch size | 32 |
| Epochs | 50 |
| LR (projector) | 1e-3 |
| LR (adapter) | 5e-4 |
| Precision | FP16 |

### Environment Variables

```bash
# Override defaults
FOLD=1 sbatch train_baseline.sh           # Train specific fold
BATCH_SIZE=8 sbatch train_5fold.sh        # Larger batch
NUM_EPOCHS=100 sbatch train_5fold.sh      # More epochs
LR_PROJ=2e-3 sbatch train_5fold.sh        # Higher projector LR
FUSION_LAYERS="1,5,9,13,17,21,25,29" sbatch train_5fold.sh  # More layers
```

## Evaluation Protocol

Standard ESC-50 5-fold cross-validation:
1. For each fold i (1-5):
   - Train on folds ≠ i
   - Validate on fold i
2. Report: **mean accuracy ± std** across 5 folds

## Planned Ablations

| Experiment | Status | Notes |
|------------|--------|-------|
| Baseline (default config) | | |
| More fusion layers (10 layers) | | |
| More tokens (8 → 16 → 32) | | |
| Unfreeze last 2 CLAP blocks | | |
| Unfreeze last 4 CLAP blocks | | |
| Higher LR sweep | | |
| Longer training (100 epochs) | | |
| SpecAugment | | |

## Results

See [RESEARCH_AGENDA.md](../../docs/RESEARCH_AGENDA.md) for detailed results logging.

### Best Configuration

```
TBD after experiments
```

### Final Accuracy

**____% ± ____%** (5-fold CV)

## Notes

- This experiment uses Pre-FFN fusion only (no KV-Aug until Phase 5)
- CLAP encoder is frozen by default
- Classification is formulated as QA: "What sound is this?" → class label
