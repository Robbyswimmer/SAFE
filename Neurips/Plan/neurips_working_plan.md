# NeurIPS Working Plan

Last updated: 2026-03-17

## Paper Direction

Working title direction:
- Incremental Modality Acquisition in Frozen Language Models
- Alternative: Composable Modality Adapters for Frozen Language Models

Core paper thesis:
- Start from a frozen text-only LLM hub.
- Add one modality at a time using the same gated residual adapter recipe.
- Measure unimodal performance, joint-data composition, and interference.
- Show that composition does not emerge for free, but improves under the right training/data/layering choices.

Current best framing:
- Main empirical story: incremental modality addition on Qwen3 with audio and vision.
- Main analysis story: interference/composition diagnostics explain why independent adapters fail and why staggered/joint training may help.
- Point cloud/SQA3D is supporting evidence, not the headline claim.

## Main Claims To Support

1. A frozen text LLM can acquire new modalities with a uniform adapter recipe.
2. Unimodal performance scales with data for each added modality.
3. Independently trained adapters do not automatically compose well.
4. Joint paired-data training and/or better layer placement improves composition.
5. Gated-off adapters preserve the base text pathway functionally in the no-modality setting.

## Primary Experimental Track

Backbone:
- Qwen3-8B frozen hub

Primary dataset:
- MUSIC-AVQA

Primary modalities:
- audio
- vision

Primary runs:
1. Audio-only adapter
2. Vision-only adapter
3. Joint paired-data training from scratch, shared-layer config
4. Joint paired-data training from scratch, staggered-layer config
5. Later: incremental initialization runs
   - start from vision adapter, add audio on paired data
   - start from audio adapter, add vision on paired data

Primary comparisons:
- text
- audio
- image
- both
- gain vs best single

## Layering Strategy

### Shared-layer joint run
Script:
- `/Users/robbymoseley/CascadeProjects/SAFE/Neurips/Scripts/run_qwen_joint_music_avqa.sh`

Config:
- `composition_study`

Layers:
- audio: `8,14,20,26`
- vision: `8,14,20,26`

Notes:
- same-layer residual addition
- learned per-layer gates enabled by config

### Staggered joint run
Script:
- `/Users/robbymoseley/CascadeProjects/SAFE/Neurips/Scripts/run_qwen_staggered_joint_music_avqa.sh`

Config:
- `composition_independent`

Layers:
- audio: `8,14,20,26`
- vision: `10,16,22,28`

Notes:
- disjoint/staggered residual addition
- better test of whether joint data plus separated layer placement improves composition

## Immediate Priority Order

1. Launch and monitor staggered Qwen joint-data run.
2. Launch shared-layer Qwen joint-data run.
3. Record unimodal and composed curves over training.
4. Run gated-off invariance checks on the frozen text path.
5. Compare shared vs staggered composition directly.
6. Only then decide whether to emphasize:
   - incremental modality addition first, or
   - interference diagnosis first.

## Decision Gates

Strong paper outcome:
- joint `both` clearly exceeds best single modality on MUSIC-AVQA
- staggered or paired-data training reduces the composition gap

Acceptable paper outcome:
- unimodal acquisition works strongly
- independent composition fails
- joint training or staggered placement partially recovers composition

Fallback story:
- modular modality addition works unimodally
- composition is fragile and interference is the main scientific finding

## Supporting Evidence Track

SQA3D:
- keep as secondary evidence for modality composition with point clouds
- useful for “recipe generality” if stable
- do not make it the main paper dependency

## Run Commands

Shared-layer Qwen joint run:
```bash
sbatch --gres=gpu:1 Neurips/Scripts/run_qwen_joint_music_avqa.sh
```

Staggered Qwen joint run:
```bash
sbatch --gres=gpu:1 Neurips/Scripts/run_qwen_staggered_joint_music_avqa.sh
```

Small debug override example:
```bash
MAX_SAMPLES=4000 EPOCHS=3 sbatch --gres=gpu:1 Neurips/Scripts/run_qwen_staggered_joint_music_avqa.sh
```
