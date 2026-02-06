# ECCV Runbook (Pre-FFN Focus)

Last updated: 2026-02-06

## Goal
Ship a reproducible ECCV submission centered on the Pre-FFN residual addition path, with final tables for:
- ESC-50 audio classification
- ModelNet40 point cloud classification
- EPIC-SOUNDS AV-QA composition (Pre-FFN only for ECCV)

## 0) Environment Check

```bash
python3 --version
which python3
```

```bash
mkdir -p logs checkpoints/esc50 checkpoints/modelnet40 checkpoints/epic_sounds_avqa
```

## 1) ESC-50 (Audio Classification)

### 1.1 Data prep
```bash
bash experiments/esc50_classification/scripts/download_esc50.sh
```

### 1.2 Main 5-fold run (SLURM)
```bash
sbatch experiments/esc50_classification/scripts/train_5fold.sh
```

### 1.3 Optional controlled rerun (single fold)
```bash
FOLD=1 sbatch experiments/esc50_classification/scripts/train_baseline.sh
```

### 1.4 Direct CLI fallback (no SLURM)
```bash
python3 train_audio_llm_probe.py \
  --dataset esc50 \
  --data-path experiments/full_training/data/esc50 \
  --output-dir checkpoints/esc50/direct_run \
  --batch-size 16 \
  --num-epochs 50
```

## 2) ModelNet40 (Point Cloud Classification)

### 2.1 Data prep
```bash
bash experiments/modelnet40_classification/scripts/download_modelnet40.sh
```

### 2.2 Baseline run (SLURM)
```bash
sbatch experiments/modelnet40_classification/scripts/train_baseline.sh
```

### 2.3 Direct CLI fallback (no SLURM)
```bash
python3 train_pointcloud.py \
  --config modelnet40 \
  --phase classification \
  --data-path experiments/full_training/data/modelnet40 \
  --output-dir checkpoints/modelnet40/direct_run \
  --num-epochs 100
```

## 3) EPIC-SOUNDS AV-QA Composition (Pre-FFN)

### 3.1 Download metadata (+ optional videos)
```bash
python3 experiments/epic_sounds_avqa_composition/scripts/download_epic_sounds_data.py \
  --data-root experiments/epic_sounds_avqa_composition/data
```

```bash
python3 experiments/epic_sounds_avqa_composition/scripts/download_epic_sounds_data.py \
  --data-root experiments/epic_sounds_avqa_composition/data \
  --download-videos \
  --chunksize 100
```

### 3.2 Build manifests and media
```bash
python3 experiments/epic_sounds_avqa_composition/scripts/prepare_epic_sounds_avqa.py \
  --data-root experiments/epic_sounds_avqa_composition/data \
  --video-root experiments/epic_sounds_avqa_composition/data/raw_videos \
  --extract-media --skip-existing
```

### 3.3 Pre-FFN train/eval run (SLURM)
```bash
sbatch --gres=gpu:1 \
  --export=ALL,DATA_ROOT=experiments/epic_sounds_avqa_composition/data,OUTPUT_DIR=checkpoints/epic_sounds_avqa/preffn,TRAIN_MODALITY=both,EVAL_MODALITIES=both,audio,image \
  experiments/epic_sounds_avqa_composition/scripts/train_preffn.sh
```

### 3.4 Controls (audio-only, image-only)
```bash
sbatch --gres=gpu:1 \
  --export=ALL,DATA_ROOT=experiments/epic_sounds_avqa_composition/data,OUTPUT_DIR=checkpoints/epic_sounds_avqa/preffn_audio_only,TRAIN_MODALITY=audio,EVAL_MODALITIES=audio \
  experiments/epic_sounds_avqa_composition/scripts/train_preffn.sh
```

```bash
sbatch --gres=gpu:1 \
  --export=ALL,DATA_ROOT=experiments/epic_sounds_avqa_composition/data,OUTPUT_DIR=checkpoints/epic_sounds_avqa/preffn_image_only,TRAIN_MODALITY=image,EVAL_MODALITIES=image \
  experiments/epic_sounds_avqa_composition/scripts/train_preffn.sh
```

## 4) Aggregate Results and Fill Agenda

### 4.1 Summarize EPIC runs
```bash
python3 experiments/epic_sounds_avqa_composition/scripts/summarize_epic_sounds_results.py \
  --run epic_preffn=checkpoints/epic_sounds_avqa/preffn \
  --run epic_preffn_audio_only=checkpoints/epic_sounds_avqa/preffn_audio_only \
  --run epic_preffn_image_only=checkpoints/epic_sounds_avqa/preffn_image_only
```

### 4.2 Auto-update research agenda (EPIC section)
```bash
python3 experiments/epic_sounds_avqa_composition/scripts/summarize_epic_sounds_results.py \
  --run epic_preffn=checkpoints/epic_sounds_avqa/preffn \
  --run epic_preffn_audio_only=checkpoints/epic_sounds_avqa/preffn_audio_only \
  --run epic_preffn_image_only=checkpoints/epic_sounds_avqa/preffn_image_only \
  --update-research-agenda
```

## 5) Paper Asset Freeze Checklist

```bash
# Save git snapshot and branch info
git rev-parse HEAD
git status --short
```

```bash
# Collect key metrics files
find checkpoints/esc50 checkpoints/modelnet40 checkpoints/epic_sounds_avqa -name \"*.json\" | sort
```

```bash
# Ensure agenda contains no unresolved placeholders in final sections
rg -n \"TBA|____|TODO\" docs/RESEARCH_AGENDA.md
```

## 6) ECCV Main-Table Mapping

| Paper Table | Primary Source |
|-------------|----------------|
| Audio classification | `experiments/esc50_classification` outputs + `docs/RESEARCH_AGENDA.md` Phase 1A |
| Point cloud classification | `train_pointcloud.py` outputs + `docs/RESEARCH_AGENDA.md` Phase 1B |
| Composition QA | `checkpoints/epic_sounds_avqa/preffn/history.json` + EPIC summary markdown |
| Ablation table | Config sweep outputs listed in `docs/RESEARCH_AGENDA.md` |
| Qualitative examples | AV-QA predictions saved from EPIC run outputs |

## 7) Out-of-Scope for ECCV

- KV-augment is tracked for NeurIPS path and should not block ECCV Pre-FFN submission.
- New architecture proposals should be deferred until ECCV numbers are locked.
