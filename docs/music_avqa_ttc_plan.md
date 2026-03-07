## MUSIC-AVQA TTC Plan

Goal: show composed `both` performance from independently trained `audio` and `image`
adapters, with no paired composition training, using test-time computation over small
runtime gate controls.

### Pipeline

1. Train an audio-only adapter on MUSIC-AVQA.
2. Train a vision-only adapter on MUSIC-AVQA.
3. Load both checkpoints into a shared composition model with shared audio/vision
   fusion layers so stage-C interaction scalars have a well-defined location.
4. Evaluate `text`, `audio`, `image`, and `both`.
5. For `both`, optionally run TTC:
   - Stage B: optimize temporary per-layer runtime gate multipliers
   - Stage C: freeze the Stage-B gate solution and optimize one interaction scalar
     per shared fusion layer
   - keep the base model and adapter weights frozen
   - start gates at `1.0`
   - start interaction scalars at `0.0`
   - clamp both into small fixed ranges

### TTC Objectives

- `simple`
  - minimize next-token entropy for the composed answer
  - regularize gates toward `1.0`

- `entropy_noharm`
  - minimize next-token entropy for the composed answer
  - add hinge penalty if composed entropy exceeds the best single-modality entropy
  - regularize gates toward `1.0`

### Stage-C Interaction Rule

Stage C only runs if the interaction scalar gradients are alive before optimization.

Criterion:
- `||dL/dc^l|| / ||dL/dg|| > 0.1` for at least half of shared layers, or
- all shared layers are at least weakly alive with ratio `>= 0.01`

If ratios are below `0.01` everywhere, Stage C is skipped and recorded as inactive.

This makes the interaction claim conservative:
- Stage B = gate-only calibration / interference reduction
- Stage C = lower-bound estimate of extra benefit from a tiny interaction variable,
  because gates are frozen at the Stage-B solution

### Core claims this implementation supports

- Independently trained audio and vision adapters can be loaded into one frozen backbone.
- Additive composition can be improved at test time without joint composition training.
- Small runtime gate corrections are enough to move composed predictions toward lower
  uncertainty and, ideally, higher answer accuracy.

### First experiment ladder

1. Train audio-only adapter for 4 epochs.
2. Train vision-only adapter for 4 epochs.
3. Run composed eval without TTC.
4. Run gate-only TTC with `--ttc-objective simple`.
5. Run gate+interaction TTC with `--ttc-objective simple`.
6. Run gate-only TTC with `--ttc-objective entropy_noharm`.
7. Run gate+interaction TTC with `--ttc-objective entropy_noharm`.
8. Compare `both` against `max(audio, image)` and against the no-TTC composed baseline.
