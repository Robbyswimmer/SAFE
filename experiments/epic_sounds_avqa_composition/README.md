# EPIC-SOUNDS AV-QA Composition Experiment

Goal: test whether SAFE audio fusion improves **vision-language question answering** on egocentric kitchen videos, and compare:
- `pre_ffn` residual fusion
- `kv_augment` attention-memory fusion

The target behavior is compositional reasoning, e.g. image cues identify the object (tomato) and audio cues identify the action/sound (washing), yielding a grounded answer.

## Task Design

We convert EPIC-SOUNDS + EPIC-KITCHENS-100 annotations into QA samples:
- `audio_event`: "What sound do you hear?" -> sound class
- `vision_object`: "Which object is being handled?" -> noun
- `av_composition`: "What is happening to the {noun}?" -> "{noun} is being {verb}"

Why QA: this repository already has stable QA training/eval loops, while long-form captioning has been less stable.

## Data Pipeline

1. Download annotations (+ optional raw videos):
```bash
python3 experiments/epic_sounds_avqa_composition/scripts/download_epic_sounds_data.py \
  --data-root experiments/epic_sounds_avqa_composition/data
```

2. Optional video download via official downloader wrapper:
```bash
python3 experiments/epic_sounds_avqa_composition/scripts/download_epic_sounds_data.py \
  --data-root experiments/epic_sounds_avqa_composition/data \
  --download-videos
```

3. Build manifests and extract 4s audio clips + keyframes:
```bash
python3 experiments/epic_sounds_avqa_composition/scripts/prepare_epic_sounds_avqa.py \
  --data-root experiments/epic_sounds_avqa_composition/data \
  --video-root experiments/epic_sounds_avqa_composition/data/raw_videos \
  --extract-media --skip-existing
```

Outputs:
- `experiments/epic_sounds_avqa_composition/data/manifests/train.jsonl`
- `experiments/epic_sounds_avqa_composition/data/manifests/validation.jsonl`
- `experiments/epic_sounds_avqa_composition/data/processed/audio/...`
- `experiments/epic_sounds_avqa_composition/data/processed/frames/...`

## Training

### Single architecture

Pre-FFN:
```bash
sbatch --gres=gpu:1 experiments/epic_sounds_avqa_composition/scripts/train_preffn.sh
```

KV-augment:
```bash
sbatch --gres=gpu:1 experiments/epic_sounds_avqa_composition/scripts/train_kvaugment.sh
```

### Ablation launcher

```bash
bash experiments/epic_sounds_avqa_composition/scripts/run_architecture_ablation.sh
```

## Main Script

`experiments/epic_sounds_avqa_composition/train_epic_sounds_avqa.py`

Key arguments:
- `--architecture pre_ffn|kv_augment`
- `--train-modality both|audio|image`
- `--eval-modalities both,audio,image`
- `--fusion-layers 1,5,9,13,17,21`

Validation reports:
- exact match
- token F1
- per-question-type breakdown (`audio_event`, `vision_object`, `av_composition`)

## Recommended Runs

1. `pre_ffn` + train `both`, eval `both,audio,image`
2. `kv_augment` + train `both`, eval `both,audio,image`
3. Optional stress tests: train `audio` only and `image` only

## Post-Run Summary

Generate markdown summary from run artifacts:

```bash
python3 experiments/epic_sounds_avqa_composition/scripts/summarize_epic_sounds_results.py
```

Update `docs/RESEARCH_AGENDA.md` automatically:

```bash
python3 experiments/epic_sounds_avqa_composition/scripts/summarize_epic_sounds_results.py \
  --update-research-agenda
```

## Success Criteria

- `both` outperforms single-modality (`audio` / `image`) on `av_composition`
- `kv_augment` vs `pre_ffn` comparison is reported under same manifest and training budget
- Qualitative samples show grounded object+action answers (not audio-only or vision-only shortcuts)
