# ScanNet Point Cloud + Image Composition Experiment

## Goal
Test whether combining point cloud and image modalities improves scene classification compared to either modality alone.

## Dataset: ScanNet
- **1,513 indoor scene scans** (1,201 train / 312 val)
- **Paired data**: RGB images + 3D point clouds + meshes
- **Task**: Scene type classification (13 classes)
- **Scene types**: apartment, bathroom, bedroom, bookstore, classroom, closet, conference room, copy room, dining room, game room, hallway, kitchen, laundry room, living room, lobby, office, storage

## Experiment Design

| Condition | Input | Model |
|-----------|-------|-------|
| Image only | RGB frames | LLaVA (frozen) + linear head |
| Point cloud only | 3D points | PointBERT → SAFE fusion → linear head |
| **PC + Image** | Both | PointBERT → SAFE fusion + LLaVA vision → linear head |

## Download Instructions

1. **Request access**: Fill out the [ScanNet Terms of Use](http://www.scan-net.org/ScanNet/) and email to scannet@googlegroups.com

2. **Download script** (after approval):
```bash
# Clone ScanNet repo
git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet

# Download with your provided script (you'll receive download_scannet.py)
python download_scannet.py -o /path/to/scannet --type .sens _vh_clean_2.ply _vh_clean_2.labels.ply
```

3. **Expected structure**:
```
scannet/
  scans/
    scene0000_00/
      scene0000_00.sens          # RGB-D sensor stream
      scene0000_00_vh_clean_2.ply  # Point cloud mesh
      scene0000_00_vh_clean_2.labels.ply  # Semantic labels
    scene0000_01/
    ...
  scans_test/
    ...
  scannetv2-labels.combined.tsv  # Label mappings
  scannetv2_train.txt            # Train split
  scannetv2_val.txt              # Val split
```

## Data Preparation

After download, run preprocessing:
```bash
python experiments/scannet_composition/scripts/preprocess_scannet.py \
  --scannet-root /path/to/scannet \
  --output-dir experiments/full_training/data/scannet
```

This will extract:
- Point clouds (downsampled to 8192 points)
- Representative RGB frames (1 per scene)
- Scene type labels

## Training Commands

```bash
# Image only (LLaVA baseline)
MODALITY="image" EXPERIMENT_NAME="image_only" \
  sbatch experiments/scannet_composition/scripts/train_composition.sh

# Point cloud only (SAFE)
MODALITY="pointcloud" EXPERIMENT_NAME="pc_only" \
  sbatch experiments/scannet_composition/scripts/train_composition.sh

# Point cloud + Image (composition)
MODALITY="both" EXPERIMENT_NAME="pc_image" \
  sbatch experiments/scannet_composition/scripts/train_composition.sh
```

## Expected Results

| Condition | Accuracy | Notes |
|-----------|----------|-------|
| Image only | ~70-80% | LLaVA vision baseline |
| Point cloud only | ~60-70% | SAFE with PointBERT |
| **PC + Image** | **?** | Does composition help? |

## Key Questions
1. Does adding point cloud to image improve accuracy?
2. Does adding image to point cloud improve accuracy?
3. Is there interference or synergy between modalities?
