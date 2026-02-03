# NuScenes-QA Composition Experiment

**Goal**: Test whether combining point cloud (LiDAR) and image (camera) modalities improves
3D question answering for autonomous driving scenarios.

## Why NuScenes-QA?

Unlike ScanNet/ScanQA which requires registration and approval, NuScenes-QA is:
- **Freely available** on Hugging Face (no registration required!)
- **Pre-processed** - ready to use with HuggingFace datasets library
- **Multi-modal** - includes 5D LiDAR + 6 camera views
- **Reasonably sized** - ~5.8K samples for fast iteration

## Dataset Details

| Split | Day | Night | Total |
|-------|-----|-------|-------|
| Train | 2,229 | 659 | 2,888 |
| Validation | 2,229 | 659 | 2,888 |
| **Total** | 4,458 | 1,318 | 5,776 |

**Modalities**:
- LIDAR_TOP: 5D point cloud (X, Y, Z, intensity, distance)
- 6 Camera views: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT

**Task**: Multi-hop reasoning questions about driving scenes
- Example: "There is a trailer; what number of traffic cones are to the back of it?"
- 29 answer classes

## Experiment Design

### Conditions
1. **PC-only**: Point cloud input only (LIDAR_TOP)
2. **Image-only**: Camera image only (CAM_FRONT by default)
3. **PC+Image** (Composition): Both modalities via true composition

### True Composition Architecture
```
LLaVA processes: input_ids + pixel_values (native image path)
SAFE injects: PC tokens as residuals at layers [1, 5, 9, 13, 17, 21]
Both modalities contribute in single forward pass
```

### Training Details
- **Trainable**: SAFE adapter (projector + fusion adapters) only
- **Frozen**: PointBERT encoder, LLaVA/LLM
- **LR**: 1e-5 with cosine decay
- **Epochs**: 20
- **Batch size**: 4 (effective 16 with gradient accumulation)

## Quick Start

### 1. Test Dataset Loading (no download needed)
```bash
# Downloads automatically on first use
python -c "from safe.data.nuscenes_qa_dataset import NuScenesQADataset; d = NuScenesQADataset(split='train', scene_type='day'); print(d[0])"
```

### 2. Run Training
```bash
# PC-only
python train_nuscenes_qa_composition.py \
    --modality pointcloud \
    --scene-type day \
    --output-dir experiments/nuscenes_qa_composition/outputs/pc_only

# Image-only (frozen LLaVA)
python train_nuscenes_qa_composition.py \
    --modality image \
    --scene-type day \
    --output-dir experiments/nuscenes_qa_composition/outputs/image_only

# Both (composition)
python train_nuscenes_qa_composition.py \
    --modality both \
    --scene-type day \
    --output-dir experiments/nuscenes_qa_composition/outputs/both
```

### 3. Run on Cluster
```bash
sbatch experiments/nuscenes_qa_composition/scripts/train_qa.sh
```

## Expected Results

| Condition | Expected Exact Match | Notes |
|-----------|---------------------|-------|
| Image-only | ~20-30% | Frozen LLaVA baseline |
| PC-only | ~15-25% | Train SAFE adapter |
| PC+Image | ~25-35% | True composition |

**Hypothesis**: Combining LiDAR depth information with camera images should improve spatial reasoning.

## File Structure
```
experiments/nuscenes_qa_composition/
  README.md                    # This file
  scripts/
    train_qa.sh                # SLURM training script
  configs/
    baseline.yaml              # Default config
  outputs/                     # Model checkpoints
  logs/                        # Training logs
```

## Dataset Source
- HuggingFace: https://huggingface.co/datasets/KevinNotSmile/nuscenes-qa-mini
- License: CC-BY-NC-SA-4.0
- Based on: nuScenes dataset (Boston/Singapore driving scenes)
