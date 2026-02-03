# ScanQA Point Cloud + Image QA Composition Experiment

## Goal
Test whether combining point cloud and image modalities improves 3D question answering compared to either modality alone. This experiment is better suited to demonstrate composition benefits than scene classification because:
- QA requires spatial reasoning about specific objects
- Point clouds provide precise 3D geometry
- Images provide visual appearance and context
- Different questions may benefit from different modalities

## Dataset: ScanQA

- **41,363 QA pairs** from 800 ScanNet scenes
- **Train/Val/Test split**: ~32K train / ~4K val / ~5K test
- **Question types**: Object identification, spatial reasoning, counting, attributes
- **Answers**: Free-form text grounded in 3D scene objects
- **Evaluation metrics**: BLEU-1/4, ROUGE-L, METEOR, CIDEr

### Example Questions
- "What color is the chair next to the table?"
- "How many windows are in the room?"
- "What is on top of the desk?"
- "Is there a plant near the sofa?"

## Experiment Design

| Condition | Input | Expected Strength |
|-----------|-------|-------------------|
| Image only | RGB frames | Visual appearance, color, texture |
| Point cloud only | 3D points | Spatial layout, distances, sizes |
| **PC + Image** | Both | Combined reasoning (best of both) |

### Hypothesis
- **Spatial questions** ("next to", "behind", "how far"): PC should help
- **Visual questions** ("what color", "what type"): Image should help
- **Combined questions**: Composition should outperform either alone

## Download Instructions

### 1. Download ScanQA annotations
```bash
# Clone ScanQA repository
git clone https://github.com/ATR-DBI/ScanQA.git
cd ScanQA

# Download annotations
# Files needed: data/qa/ScanQA_v1.0_train.json, ScanQA_v1.0_val.json, ScanQA_v1.0_test.json
```

### 2. Download ScanNet (if not already done)
See `experiments/scannet_composition/README.md` for ScanNet download instructions.

### 3. Expected structure
```
experiments/full_training/data/
  scanqa/
    ScanQA_v1.0_train.json    # Training QA pairs
    ScanQA_v1.0_val.json      # Validation QA pairs
    ScanQA_v1.0_test.json     # Test QA pairs (answers withheld)
  scannet/
    pointclouds/              # Preprocessed point clouds
    images/                   # Extracted RGB frames
```

## Data Preparation

After downloading, run preprocessing:
```bash
python experiments/scanqa_composition/scripts/preprocess_scanqa.py \
  --scanqa-root /path/to/ScanQA/data \
  --scannet-root /path/to/scannet \
  --output-dir experiments/full_training/data
```

## Training Commands

```bash
# Image only (LLaVA baseline)
MODALITY="image" EXPERIMENT_NAME="image_only" \
  sbatch experiments/scanqa_composition/scripts/train_qa.sh

# Point cloud only (SAFE)
MODALITY="pointcloud" EXPERIMENT_NAME="pc_only" \
  sbatch experiments/scanqa_composition/scripts/train_qa.sh

# Point cloud + Image (composition)
MODALITY="both" EXPERIMENT_NAME="pc_image" \
  sbatch experiments/scanqa_composition/scripts/train_qa.sh
```

## Expected Results

| Condition | BLEU-4 | CIDEr | Notes |
|-----------|--------|-------|-------|
| Image only | ~15-20 | ~50-60 | Good for visual questions |
| Point cloud only | ~10-15 | ~40-50 | Good for spatial questions |
| **PC + Image** | **~20-25** | **~60-70** | Expected synergy |

### ScanQA Leaderboard Reference
- ScanQA baseline (2022): BLEU-4 ~18, CIDEr ~55
- 3D-LLM (2023): BLEU-4 ~21, CIDEr ~69
- Our goal: Demonstrate composition benefit

## Key Analysis

After training, analyze:
1. **Overall metrics**: Does PC+Image beat single modalities?
2. **Question type breakdown**: Which types benefit from composition?
3. **Error analysis**: What questions fail in each condition?
4. **Attention visualization**: Where does the model attend in each modality?

## Ablation Experiments
1. Baseline (default fusion layers)
2. Late fusion vs early fusion
3. Different fusion layer configurations
4. More/fewer point cloud tokens
5. Unfreeze encoder last N blocks
