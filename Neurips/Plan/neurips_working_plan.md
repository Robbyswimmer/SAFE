# NeurIPS Working Plan

Last updated: 2026-03-18

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

## Current Supported Claims

These are the claims the current evidence already supports well enough to build the paper around.

1. A frozen text-only Qwen3 backbone can acquire strong unimodal modality adapters.
2. Strong independently trained adapters do not compose automatically under naive residual addition.
3. Joint paired-data training can recover positive task-level composition on the same frozen backbone.
4. Recovering composition via joint training comes with a measurable unimodal-specialization penalty.
5. Sequential later-modality training can enter a stability-versus-strength bottleneck, with the new adapter pushed toward a near-silent solution.

## Claims Still Being Tested

1. Whether staggered layer placement materially improves composition under joint training.
2. Whether the sequential collapse persists through later epochs or eventually recovers.
3. Whether operator changes such as KV augmentation improve composition without paying the full joint-training penalty.
4. Whether gated-off functional invariance should be elevated as a main claim or left as supporting architecture detail.

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

1. Finish the sequential audio-after-vision readout and determine whether collapse persists.
2. Record the shared-layer joint result as the main recovered-composition baseline.
3. Compare shared vs staggered joint-data composition directly.
4. Run gated-off invariance checks on the frozen text path.
5. If the above is clean, try one operator-level intervention, with KV augmentation as the preferred next test.

## Decision Gates

Strong paper outcome:
- joint `both` clearly exceeds best single modality on MUSIC-AVQA
- independent and sequential settings fail in distinct, diagnosable ways
- one operator-level change shows whether residual addition itself is the main bottleneck

Acceptable paper outcome:
- unimodal acquisition works strongly
- independent composition fails
- joint training recovers composition and sequential addition shows collapse

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
