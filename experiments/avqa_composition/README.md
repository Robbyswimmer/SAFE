# AVQA Composition (Pre-FFN)

This experiment package adds AVQA and MUSIC-AVQA composition runs for ECCV-focused
Pre-FFN residual fusion evaluation.

## 1) Prepare manifests

### General AVQA

```bash
python3 experiments/avqa_composition/scripts/prepare_avqa_manifests.py \
  --dataset avqa \
  --train-json /path/to/avqa/metadata/train_qa.json \
  --val-json /path/to/avqa/metadata/val_qa.json \
  --output-root data/avqa \
  --media-root data/avqa \
  --audio-root data/avqa/audio \
  --image-root data/avqa/frames \
  --require-both
```

### MUSIC-AVQA

```bash
python3 experiments/avqa_composition/scripts/prepare_avqa_manifests.py \
  --dataset music_avqa \
  --train-json /path/to/music_avqa/metadata/avqa-train.json \
  --val-json /path/to/music_avqa/metadata/avqa-val.json \
  --output-root data/music_avqa \
  --media-root data/music_avqa \
  --audio-root data/music_avqa/audio \
  --image-root data/music_avqa/frames \
  --require-both
```

## 2) Train

### General AVQA

```bash
sbatch --gres=gpu:1 experiments/avqa_composition/scripts/train_preffn_avqa.sh
```

### MUSIC-AVQA

```bash
sbatch --gres=gpu:1 experiments/avqa_composition/scripts/train_preffn_music_avqa.sh
```

## 3) Modality utilization controls

Audio-only:

```bash
sbatch --gres=gpu:1 \
  --export=ALL,TRAIN_MODALITY=audio,EVAL_MODALITIES=audio \
  experiments/avqa_composition/scripts/train_preffn_music_avqa.sh
```

Image-only:

```bash
sbatch --gres=gpu:1 \
  --export=ALL,TRAIN_MODALITY=image,EVAL_MODALITIES=image \
  experiments/avqa_composition/scripts/train_preffn_music_avqa.sh
```

