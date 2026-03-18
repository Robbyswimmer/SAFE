# Experiment Diary

Last updated: 2026-03-17

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
