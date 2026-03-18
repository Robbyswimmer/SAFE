# NeurIPS Results Log

Last updated: 2026-03-18

## Check-In Synthesis

This section records only claims that currently have direct empirical support from runs already completed or partially validated.

### Finding 1: A frozen text-only Qwen3 backbone can support useful multimodal composition.

Direct support:
- joint paired training on MUSIC-AVQA gives `text=22.96`, `audio=41.74`, `vision=55.63`, `both=67.53`
- composition gain over the best single modality is `+11.90`

What we can claim:
- composition on a frozen text-only backbone is possible
- the strongest version of "Qwen cannot compose" is false

What we cannot yet claim:
- that composition emerges automatically from independent or sequential addition

### Finding 2: Strong independent unimodal adapters do not compose for free.

Direct support:
- conservative same-layer independent baseline on MUSIC-AVQA:
  - `text=22.96`, `audio=58.80`, `vision=64.48`, `both=59.30`
  - gain vs best single = `-5.18`
- earlier pilot showed an even larger failure:
  - `text=27.86`, `audio=64.72`, `vision=65.92`, `both=53.21`
  - gain vs best single = `-12.71`

What we can claim:
- independently trained adapters can each be strong on their own while still degrading when simultaneously activated
- naive residual composition is not reliable

### Finding 3: Joint training improves composition but weakens unimodal specialization.

Direct support:
- independent runs are stronger unimodally than the joint run
- joint run is much stronger compositionally than the independent same-layer baseline

Concrete contrast:
- independent same-layer: `audio=58.80`, `vision=64.48`, `both=59.30`
- joint paired training: `audio=41.74`, `vision=55.63`, `both=67.53`

What we can claim:
- there is an empirically supported tradeoff between unimodal specialization and composed compatibility
- joint training changes the learned injected directions enough to recover task-level composition

### Finding 4: Sequential addition can enter an interference-avoidance collapse regime.

Direct support:
- in the audio-after-vision run, audio-side gradients are strong while vision gradients are zero as expected
- despite this, audio remains at the text baseline in the recorded snapshot:
  - `text=22.96`
  - `audio=22.98`

What we can claim:
- the later adapter is not dead; it is being optimized
- nevertheless, it has not yet learned a useful modality signal
- this is consistent with a collapse regime where the easiest stable solution is near-silent residual injection

What we still need before making this a headline claim:
- one more checkpoint confirming that audio remains near baseline rather than recovering later

### Finding 5: The current evidence points to a geometry/operator problem, not a simple gate-scaling problem.

Direct support:
- strong independent adapters still fail under naive composition
- sequential training appears to collapse by suppressing the new signal
- joint training succeeds only after changing the learned adapter directions through co-training

What we can claim:
- changing only signal magnitude is unlikely to be the full solution
- the evidence so far is more consistent with interference in the shape/direction of injected updates than with a pure amplitude problem

What this implies for next experiments:
- prioritize operator-level changes such as KV augmentation or other geometry-changing composition rules over gate-only calibration

### Finding 6: Early KV-augmentation diagnostics suggest first-site dominance is positional, not layer-8-specific.

Direct support:
- two vision-only KV runs were launched with the same later layers but different first injection sites:
  - layer-8 run: `8,14,20,26`
  - layer-2 run: `2,14,20,26`
- in the layer-8 run, `kv:vision:8` is the largest vision-side gradient term by several-fold at steps `700-1000`
- in the layer-2 run, `kv:vision:2` becomes the largest vision-side gradient term by several-fold at steps `500-800`
- both runs evaluate cleanly, save checkpoints, and continue training without the earlier KV logging/pathology issues

What we can claim:
- the dominant learning signal in the current KV setup is concentrated at the first active vision injection site
- this concentration is not unique to layer `8`; it follows whichever layer is first
- the current KV operator is therefore likely biased toward earliest-site memory formation

What we cannot yet claim:
- that earlier-first KV placement improves final validation accuracy
- that first-site dominance is harmful rather than simply an efficient allocation pattern

What this implies for next experiments:
- treat layer placement in KV mode as a real operator design choice, not a cosmetic detail
- compare final validation curves for layer-2 versus layer-8 starts before fixing the NeurIPS claim

## Qwen3 / MUSIC-AVQA Joint-Training Result

Context:
- dataset: MUSIC-AVQA
- backbone: Qwen3-8B
- training modality: `both`
- training regime: joint paired `audio + image + text` from scratch
- source: user-provided converged training log

Validated result snapshot:

```text
[eval:text] raw_em=19.06 extracted_em=22.96 cat_f1=22.96 f1=19.06 n=4595
[eval:audio] raw_em=41.55 extracted_em=41.74 cat_f1=41.74 f1=41.56 n=4595
[eval:image] raw_em=55.50 extracted_em=55.63 cat_f1=55.63 f1=55.52 n=4595
[eval:both] raw_em=67.53 extracted_em=67.53 cat_f1=67.53 f1=67.53 n=4595
```

Derived composition summary:

| Setting | Text | Audio | Vision | Both | Gain vs Best Single | Synergy |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B, joint paired training | 22.96 | 41.74 | 55.63 | 67.53 | +11.90 | -6.88 |

Interpretation:
- joint training clearly recovers useful composition on a frozen text-only Qwen backbone
- `both` exceeds the best single modality by `+11.90`, so positive task-level composition is real in this setting
- at the same time, unimodal specialization is much weaker than in the independently trained runs
- the core tradeoff is now explicit:
  - independent training yields stronger unimodal adapters but poor composition
  - joint training yields weaker unimodal adapters but much better composition
- this means the main bottleneck is not simply "text-only backbones cannot compose"; composition can be learned, but the learned adapter directions become more compromise-oriented under joint training
- negative scalar synergy here should not be overread as "composition failure"; task-level gain over best single is still strongly positive

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

## Qwen3 Incremental Audio-After-Vision Snapshot

Context:
- run: incremental Qwen3 MUSIC-AVQA
- procedure: train vision first, then freeze vision and train audio with `audio + image + text` active
- source log: [qwen_audio_after_vision_245979.out](/Users/robbymoseley/CascadeProjects/SAFE/logs/qwen_audio_after_vision_245979.out)
- interpretation status: early-stage stage-2 audio learning, not final convergence

Recorded snapshot:

```text
[grad_attribution] step=8000 audio:14=9.580e-01 | audio:20=8.925e-01 | audio:26=4.532e-01 | audio:8=9.034e-01 | projector=9.797e-01 | vision:10=0.000e+00 | vision:16=0.000e+00 | vision:22=0.000e+00 | vision:28=0.000e+00
[eval:text] raw_em=19.06 extracted_em=22.96 cat_f1=22.96 f1=19.06 n=4595
[eval:audio] raw_em=19.24 extracted_em=22.98 cat_f1=22.98 f1=19.73 n=4595
```

Interpretation:
- vision is frozen and not being updated in stage 2, which is confirmed by zero vision-side gradient attribution in this snapshot
- audio-side parameters are receiving strong gradients, so the failure to improve audio is not a dead-training-path issue
- at this early point, `audio` is effectively still at the text baseline: `22.98` vs `22.96`
- this is consistent with a Jacobian-interference explanation: the new audio adapter is training on top of the already-perturbed hidden state induced by the vision adapter, so large early audio residuals would propagate through downstream Jacobians and risk damaging the preserved vision solution
- the practical consequence is a conservative regime in which the new adapter first learns to avoid harming vision before it learns a strong modality-specific signal
- this snapshot should therefore be cited as evidence for a stability-versus-strength tradeoff in incremental residual adaptation, not as the final incremental composition result

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

1. Does staggered placement outperform shared-layer placement under the same joint-data regime?
2. Does incremental addition from an existing strong vision adapter escape the collapse regime, or does it stay trapped near the text baseline for audio?
3. Can operator changes improve composition without paying the full joint-training unimodal penalty?
