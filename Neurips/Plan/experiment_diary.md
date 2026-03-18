# Experiment Diary

Last updated: 2026-03-18

Use this file as the chronological log for NeurIPS-relevant runs. Add one entry per run or milestone.

---

## 2026-03-17

### SQA3D full-validation checkpoint

Status:
- validated
- supporting evidence for multimodal composition

Source:
- cluster log snippet recorded by user

Metrics:
```text
[eval:both] accuracy=52.01 extracted=52.25 exact=52.01 f1=53.27 n=3261
[eval:pointcloud] accuracy=49.98 extracted=50.38 exact=49.98 f1=51.31 n=3261
[eval:image] accuracy=0.06 extracted=5.49 exact=0.03 f1=1.74 n=3261
[eval:text] accuracy=0.98 extracted=12.51 exact=0.83 f1=5.11 n=3261
[composition] text=12.51 pointcloud=50.38 image=5.49 both=52.25 gain_vs_best_single=+1.87
```

Interpretation:
- `both` is positive over the best single modality by `+1.87`
- pointcloud dominates the composition signal
- image remains weak and likely needs better question-conditioned retrieval

Paper role:
- secondary evidence
- recipe generality / pointcloud composition support

---

### Qwen shared-layer joint-data launcher created

Script:
- `/Users/robbymoseley/CascadeProjects/SAFE/Neurips/Scripts/run_qwen_joint_music_avqa.sh`

Purpose:
- paired `audio + vision + text` training from scratch on MUSIC-AVQA
- shared-layer composition setting

Config:
- `composition_study`

Layers:
- audio: `8,14,20,26`
- vision: `8,14,20,26`

Notes:
- wrapper updated to request `gpu` partition
- default gate set conservatively to `0.1`

---

### Qwen staggered joint-data launcher created

Script:
- `/Users/robbymoseley/CascadeProjects/SAFE/Neurips/Scripts/run_qwen_staggered_joint_music_avqa.sh`

Purpose:
- paired `audio + vision + text` training from scratch on MUSIC-AVQA
- staggered/disjoint layer placement to reduce direct interference

Config:
- `composition_independent`

Layers:
- audio: `8,14,20,26`
- vision: `10,16,22,28`

Notes:
- wrapper updated to request `gpu` partition
- this is currently the highest-priority Qwen run

---

### Legacy result sources linked into NeurIPS workspace

Copied / indexed:
- archived MUSIC-AVQA scaling results
- archived Qwen3 unimodal/interference results
- composition experiment notes
- competitive positioning note
- layer placement theory note

See:
- `/Users/robbymoseley/CascadeProjects/SAFE/Neurips/References/source_index.md`

---

### Archived Qwen3 unimodal / interference numbers recorded

Status:
- validated locally from notes and figure scripts
- not newly rerun in this turn

Recorded in:
- `/Users/robbymoseley/CascadeProjects/SAFE/Neurips/References/qwen_legacy_results.md`

Conservative canonical same-layer baseline:
- text `22.96`
- audio `58.80`
- vision `64.48`
- both `59.30`
- gain vs best single `-5.18`

Earlier pilot same-layer snapshot:
- text `27.86`
- audio `64.72`
- vision `65.92`
- both `53.21`
- gain vs best single `-12.71`

Higher archived staggered-layer figure-script values:
- vision `73.95`
- both `71.12`

---

## 2026-03-18

### Qwen3 joint paired-training converged result

Status:
- validated
- primary MUSIC-AVQA composition result on Qwen3

Source:
- user-provided converged training log

Setup:
- backbone: Qwen3-8B
- training modality: `both`
- training data: paired `audio + image + text`
- training style: from scratch, joint optimization of both adapters

Recorded snapshot:
```text
[eval:text] raw_em=19.06 extracted_em=22.96 cat_f1=22.96 f1=19.06 n=4595
[eval:audio] raw_em=41.55 extracted_em=41.74 cat_f1=41.74 f1=41.56 n=4595
[eval:image] raw_em=55.50 extracted_em=55.63 cat_f1=55.63 f1=55.52 n=4595
[eval:both] raw_em=67.53 extracted_em=67.53 cat_f1=67.53 f1=67.53 n=4595
```

Interpretation:
- this run establishes that a frozen text-only Qwen backbone can support positive audio+vision composition under joint paired-data training
- the composed result is strong: `both=67.53`, which is `+11.90` over the best single modality
- however, unimodal performance is substantially below the stronger independently trained adapters
- this creates the central tradeoff for the paper:
  - independent unimodal training gives strong single-modality performance but poor composition
  - joint training gives much better composition but weaker single-modality specialization
- this result rules out the strongest version of the claim that "Qwen cannot compose"; the more precise claim is that composition requires changing the learned injected directions, and joint training is one way to do that

---

### Qwen3 incremental audio-after-vision early stage-2 snapshot

Status:
- running
- informative early-stage result

Source:
- [qwen_audio_after_vision_245979.out](/Users/robbymoseley/CascadeProjects/SAFE/logs/qwen_audio_after_vision_245979.out)

Setup:
- stage 1: vision adapter trained first
- stage 2: vision frozen, audio trainable
- active inputs in stage 2: `audio + image + text`

Recorded snapshot:
```text
[grad_attribution] step=8000 audio:14=9.580e-01 | audio:20=8.925e-01 | audio:26=4.532e-01 | audio:8=9.034e-01 | projector=9.797e-01 | vision:10=0.000e+00 | vision:16=0.000e+00 | vision:22=0.000e+00 | vision:28=0.000e+00
[eval:text] raw_em=19.06 extracted_em=22.96 cat_f1=22.96 f1=19.06 n=4595
[eval:audio] raw_em=19.24 extracted_em=22.98 cat_f1=22.98 f1=19.73 n=4595
```

Interpretation:
- audio-side parameters are receiving strong gradients while vision remains frozen
- despite active optimization, the audio path is still effectively at text-baseline performance in this snapshot
- this is consistent with the Jacobian-transport view from the theory note: the later adapter is learning on top of the transported perturbation created by the earlier vision adapter, so the easiest early solution is to keep the new residual small enough not to damage the already-good vision pathway
- this run should be tracked as a key test of the incremental-addition hypothesis:
  - if audio later rises while vision is preserved, incremental addition is viable
  - if audio remains weak, that supports a stability-versus-strength bottleneck for later residual adapters
