# NeurIPS Results Log

Last updated: 2026-03-17

## SQA3D Composition Result

Context:
- dataset: SQA3D
- training modality: `both`
- eval modalities: `both`, `pointcloud`, `image`, `text`
- image source: multiview when available

Validated result snapshot:

```text
[eval:both] accuracy=52.01 extracted=52.25 exact=52.01 f1=53.27 n=3261
[eval:pointcloud] accuracy=49.98 extracted=50.38 exact=49.98 f1=51.31 n=3261
[eval:image] accuracy=0.06 extracted=5.49 exact=0.03 f1=1.74 n=3261
[eval:text] accuracy=0.98 extracted=12.51 exact=0.83 f1=5.11 n=3261
[composition] text=12.51 pointcloud=50.38 image=5.49 both=52.25 gain_vs_best_single=+1.87
[checkpoint] saved best.pt (both extracted=52.25)
```

Interpretation:
- pointcloud is carrying most of the signal
- text-only prior is weak but nonzero
- image path remains very weak even after multiview upgrade
- full composition is positive: `both` beats the best single modality by `+1.87`
- this is credible supporting evidence for composition, but not the primary NeurIPS result

## SQA3D Notes

Takeaways:
- multiview scene images are now being used correctly
- the image branch is still underpowered for SQA3D
- likely next improvement would require better question-conditioned image retrieval rather than more tuning of the current single/montage image path

Role in paper:
- secondary modality-generalization evidence
- not the main story

## Archived Qwen3 / MUSIC-AVQA Baselines

Primary local sources:
- [qwen_legacy_results.md](/Users/robbymoseley/CascadeProjects/SAFE/Neurips/References/qwen_legacy_results.md)
- [10_composition_experiment.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/reference_notes/10_composition_experiment.md)
- [09_composition_theory.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/reference_notes/09_composition_theory.md)

Conservative same-layer independent-training baseline:

| Setting | Text | Audio | Vision | Both | Gain vs Best Single |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B, full data, same-layer | 22.96 | 58.80 | 64.48 | 59.30 | -5.18 |

Earlier pilot / epoch-1 same-layer snapshot:

| Setting | Text | Audio | Vision | Both | Gain vs Best Single |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B, pilot, same-layer | 27.86 | 64.72 | 65.92 | 53.21 | -12.71 |

Important note:
- the best archived staggered-layer Qwen figure-script value is `vision=73.95`
- the paired archived staggered composition reference from the same source is `both=71.12`

Do not confuse this with LoRA:
- the `71.4` image number found in [docs/composition_theory.md](/Users/robbymoseley/CascadeProjects/SAFE/docs/composition_theory.md) is a LoRA baseline, not the adapter result

## Qwen / MUSIC-AVQA Runs To Track

### Shared-layer joint-data run
Script:
- `/Users/robbymoseley/CascadeProjects/SAFE/Neurips/Scripts/run_qwen_joint_music_avqa.sh`

What it tests:
- paired audio+vision+text training from scratch
- same-layer composition under joint data

### Staggered joint-data run
Script:
- `/Users/robbymoseley/CascadeProjects/SAFE/Neurips/Scripts/run_qwen_staggered_joint_music_avqa.sh`

What it tests:
- paired audio+vision+text training from scratch
- disjoint audio/vision layer placement

Expected comparison:
- if staggered > shared, layer interference remains a central result
- if both recover composition strongly, joint data is the main mechanism
- if neither recovers composition, that is still publishable as a negative result about frozen text-only hubs

## Key Numbers To Record For Each Qwen Run

For every major checkpoint:
- text exact / extracted / F1
- audio exact / extracted / F1
- image exact / extracted / F1
- both exact / extracted / F1
- gain vs best single
- train modality
- layer placement
- gate value
- data size

## Open Questions

1. Does joint paired-data training from scratch recover composition on Qwen3-8B?
2. Does staggered placement outperform shared-layer placement under the same data?
3. Is incremental addition from an existing single-modality adapter better than from-scratch joint training?
