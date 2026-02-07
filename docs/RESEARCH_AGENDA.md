# SAFE Research Agenda

> **Principle**: Complete each experiment thoroughly before moving on. Validate results, push performance, and document findings at each stage.

---

## Phase 1: Classification (Pre-FFN)

**Goal**: Establish strong classification baselines for both modalities. Push accuracy as high as possible.

### 1A. Audio Classification - ESC-50

**Dataset**: ESC-50 (Environmental Sound Classification)
- 2000 audio clips (5 seconds each)
- 50 classes (40 clips per class)
- 5-fold cross-validation (standard protocol)

**SOTA Targets**:
| Model | Accuracy | Notes |
|-------|----------|-------|
| BEATs | 98.1% | Current SOTA |
| CLAP | 96.7% | Audio-text pretrained |
| AST | 95.7% | Audio Spectrogram Transformer |
| Human | 81.3% | Human baseline |
| **Our target** | **90%+** | Competitive with specialized models |

**Experiment Scripts**:
- Download: `experiments/esc50_classification/scripts/download_esc50.sh`
- Baseline (single fold): `experiments/esc50_classification/scripts/train_baseline.sh`
- Full 5-fold CV: `experiments/esc50_classification/scripts/train_5fold.sh`

| Experiment | Status | Accuracy (5-fold mean ± std) | Notes |
|------------|--------|------------------------------|-------|
| **kitchen_sink_8tok** | ✅ **COMPLETE** | **97.35% ± 0.45%** | **BEST** - unfreeze2+mixup+label_smooth |
| Baseline (batch=16) | 🔄 In Progress | 96.00% (fold 1), 94.50% (fold 5) | Train ~99%, see config below |
| Larger batch (batch=32) | ✅ Fold 5 done | 95.75% (fold 5) | epoch 32, val_loss 0.59 |
| more_layers (10 layers) | ✅ Fold 5 done | 95.75% (fold 5) | epoch 14, val_loss 0.46 |
| higher_lr (LR 2e-4) | ✅ Fold 5 done | 95.50% (fold 5) | Best epoch 26 |
| layers_lr_combo (10 layers + LR 2e-4) | ✅ Fold 5 done | 95.25% (fold 5) | Best epoch 46, higher LR hurt |
| Label smoothing (0.1) | 🔄 Running | 97.50% (fold 1) | Strong single-fold result |
| Mixup (alpha=0.3) | 🔄 Running | 97.00% (fold 1) | |
| Unfreeze 2 CLAP layers | 🔄 Running | 96.00% (fold 1) | |
| kitchen_sink_16tok | 🔄 Running | | 16 tokens version |

**Baseline Configuration (fold 5 result: 94.50% test, 99% train)**:
```
Script: train_audio_llm_probe.py --dataset esc50
Encoder: CLAP (frozen)
LLM: LLaVA-1.5-13B (frozen)
Fusion: Pre-FFN residual
Fusion layers: 1,5,9,13,17,21 (6 layers)
Num audio tokens: 8 (default)
Batch size: 16
Epochs: 50
SAFE LR (projector+fusion): 6e-5
Head LR: 1e-3
Pooling: last token
Head type: linear
FP16: enabled
```

**Completion Criteria**:
- [x] Best config identified and documented (kitchen_sink_8tok)
- [ ] Ablation table showing contribution of each trick
- [x] Final accuracy: **97.35% ± 0.45%** (5-fold mean ± std)
- [x] Training curves saved to `experiments/esc50_classification/outputs/`

**Best Configuration (kitchen_sink_8tok: 97.35% ± 0.45%)**:
```
Encoder: CLAP (unfreeze last 2 layers)
LLM: LLaVA-1.5-13B (frozen)
Fusion: Pre-FFN residual
Fusion layers: 1,5,9,13,17,21 (6 layers)
Num audio tokens: 8
Batch size: 16
Epochs: 50
SAFE LR: 6e-5
Head LR: 1e-3
Mixup alpha: 0.3
Label smoothing: 0.1
```

**Per-Fold Results (kitchen_sink_8tok)**:
| Fold | Accuracy | Best Epoch |
|------|----------|------------|
| 1 | 97.25% | 48 |
| 2 | 98.00% | 42 |
| 3 | 97.50% | 46 |
| 4 | 97.25% | 45 |
| 5 | 96.75% | 36 |
| **Mean ± Std** | **97.35% ± 0.45%** | |

### 1B. Point Cloud Classification (ModelNet40)

**Dataset**: ModelNet40 (3D CAD model classification)
- 12,311 models (9,843 train / 2,468 test)
- 40 classes (airplane, bathtub, bed, chair, etc.)
- Single train/test split (no cross-validation)

**SOTA Targets**:
| Model | Accuracy | Notes |
|-------|----------|-------|
| PointNeXt | 94.0% | Current SOTA |
| Point-MAE | 93.8% | Self-supervised |
| PointBERT | 93.2% | BERT-style pre-training |
| PointNet++ | 91.9% | Hierarchical |
| **Our target** | **90%+** | Competitive with specialized models |

**Experiment Scripts**:
- Download: `experiments/modelnet40_classification/scripts/download_modelnet40.sh`
- Baseline: `experiments/modelnet40_classification/scripts/train_baseline.sh`

| Experiment | Status | Accuracy | Notes |
|------------|--------|----------|-------|
| Baseline (cosine LR) | ✅ Done | ~81% | Plateaued, LR decaying too fast |
| Constant LR (500 epochs) | ✅ Done | ~84% | Best with kitchen_sink settings |
| kitchen_sink_1000ep | 🔄 Running | ~84% plateau | 1000 epochs, linear head, last pooling, 16 tok |
| **target_90plus** | 🔄 Running | TBD | MLP head + mean pool + 32 tok + cosine + 10 layers |

**Previous Best Configuration (84% plateau)**:
```
Script: train_pointcloud.py --llm-probe-head
Head: linear (5120 → 40)
Pooling: last token
Encoder: PointBERT (unfreeze last 8 blocks)
LLM: LLaVA-1.5-13B (frozen)
Fusion: Pre-FFN residual
Fusion layers: 1,5,9,13,17,21 (6 layers, first half only)
Num tokens: 16
Batch size: 16
Epochs: 1000
LR scheduler: constant
SAFE LR: 6e-5 | Head LR: 1e-3
Augmentation: rotate+scale+jitter+translate, label smoothing 0.1, mixup 0.3
FP16: enabled
```

**Current Run: target_90plus** (submitted 2026-02-07):
```
Script: scripts/train_pointcloud_90plus.sh
Head: MLP (4-layer, GELU + 0.1 dropout)          ← was linear
Pooling: mean (all non-pad tokens)                ← was last
Encoder: PointBERT (unfreeze last 8 blocks)
LLM: LLaVA-1.5-13B (frozen)
Fusion: Pre-FFN residual
Fusion layers: 1,5,9,13,17,21,25,29,33,37 (10 layers, full network) ← was 6 layers
Num tokens: 32                                    ← was 16
Batch size: 16
Epochs: 1000 (early stopping patience: 150)
LR scheduler: cosine (200-step warmup)            ← was constant
SAFE LR: 6e-5 | Head LR: 1e-3
Weight decay: 0.01 (head: 0.01)                   ← head was 0
Augmentation: rotate+scale+jitter+translate+point_dropout, label smoothing 0.1, mixup 0.3
FP16: enabled
W&B: ModelNet40-Classification / target_90plus_mlp_mean_32tok
```

**Completion Criteria**:
- [ ] Best config identified and documented
- [ ] Ablation table showing contribution of each trick
- [ ] Final accuracy: ____%
- [ ] Training curves saved to `experiments/modelnet40_classification/outputs/`

---

## Phase 2: Model Transfer Validation

**Goal**: Validate that SAFE architecture transfers to different base LLMs.

### 2A. Port to Qwen3-8B

| Task | Status | LLaVA-13B Acc | Qwen3-8B Acc | Notes |
|------|--------|---------------|--------------|-------|
| Audio Classification | | | | |
| PC Classification | | | | |

**Completion Criteria**:
- [ ] Qwen3-8B integration working
- [ ] Comparable accuracy to LLaVA-13B (within 5%)
- [ ] Document any architecture differences needed

---

## Phase 3: Captioning (Pre-FFN)

**Goal**: Train adapters for caption generation. Validate quality with standard metrics.

### 3A. Audio Captioning (AudioCaps / Clotho)

| Experiment | Status | CIDEr | BLEU-4 | Notes |
|------------|--------|-------|--------|-------|
| Baseline captioning | | | | |
| Best config from Phase 1 | | | | |
| 2-stage (alignment → captioning) | | | | |
| Longer training | | | | |

**Completion Criteria**:
- [ ] CIDEr score: ____
- [ ] BLEU-4 score: ____
- [ ] Sample outputs reviewed and qualitatively good
- [ ] Comparison to prior work (if available)

### 3B. Point Cloud Captioning (Cap3D)

| Experiment | Status | CIDEr | BLEU-4 | Notes |
|------------|--------|-------|--------|-------|
| Baseline captioning | | | | |
| Best config from Phase 1 | | | | |
| Longer training | | | | |

**Completion Criteria**:
- [ ] CIDEr score: ____
- [ ] BLEU-4 score: ____
- [ ] Sample outputs reviewed and qualitatively good

### 3C. Alternative Task (if captioning fails)

If captioning doesn't work well with our architecture, consider:
- [ ] VQA with constrained answers
- [ ] Multi-choice QA
- [ ] Retrieval tasks

---

## Phase 4: Modality Composition (Pre-FFN)

**Goal**: Test how multiple modalities interact when combined via SAFE fusion. Does composition help?

### 4A. MUSIC-AVQA Audio-Visual QA (Audio + Image) [PRIMARY ECCV BENCHMARK]

**Dataset**: MUSIC-AVQA (Li et al., CVPR 2022)
- 9,288 videos (7,422 Real + 1,866 Synthetic music performances)
- 45,867 QA pairs
- 22 instrument classes
- Question types: existential, counting, location, comparative, temporal

**Data Preparation**:
1. Raw videos obtained from MUSIC-AVQA authors
2. Audio extracted: 16kHz mono WAV via ffmpeg (standard for audio models)
3. Visual frames: single keyframe (middle frame) as JPEG per video
4. Manifests: standardized JSONL with `--require-both` (paired audio+image only)

**Data Preparation Script**: `scripts/prepare_music_avqa.sh`
**Training Script**: `experiments/avqa_composition/scripts/train_preffn_music_avqa.sh`
**Full Documentation**: `experiments/avqa_composition/README.md`

| Condition | Status | EM (%) | F1 (%) | Notes |
|-----------|--------|--------|--------|-------|
| Audio only | ⏳ Ready | | | Ablation: audio sufficiency |
| Image only | ⏳ Ready | | | Ablation: image sufficiency |
| **Both (composition)** | ⏳ Ready | | | True composition |

**Architecture (True Composition)**:
```
LLaVA: processes input_ids + pixel_values (native image path)
SAFE: injects audio tokens as Pre-FFN residuals at layers [1,5,9,13,17,21]
Both modalities contribute in single forward pass
```

**Training Config**: LLaVA-1.5-13B (frozen), CLAP audio encoder (frozen), 8 audio tokens, LR 5e-5, 10 epochs, batch 2, FP16

**Hypothesis**: Questions about sound sources ("Which instrument is playing?") need audio; questions about visual arrangement ("Where is the violin?") need image; composition should help on questions requiring both modalities.

**Completion Criteria**:
- [ ] All three conditions evaluated
- [ ] Per-question-type breakdown analysis
- [ ] Composition outperforms single-modality on cross-modal questions
- [ ] Qualitative examples demonstrating audio-visual grounding

---

### 4B. ScanNet Scene Classification (PC + Image)

**Dataset**: ScanNet
- 1,513 indoor scenes (1,201 train / 312 val)
- 17 scene types (apartment, bathroom, bedroom, etc.)
- Paired data: RGB images + 3D point clouds

**Experiment Scripts**:
- Preprocess: `experiments/scannet_composition/scripts/preprocess_scannet.py`
- Training: `experiments/scannet_composition/scripts/train_composition.sh`

| Condition | Status | Accuracy | Notes |
|-----------|--------|----------|-------|
| Image only | ⏳ Pending | | LLaVA vision baseline |
| Point cloud only | ⏳ Pending | | SAFE with PointBERT |
| **PC + Image** | ⏳ Pending | | Late fusion composition |

**Hypothesis**: Image likely dominates for scene classification (visual appearance is strong signal). May show neutral or small composition benefit.

**Completion Criteria**:
- [ ] All three conditions evaluated
- [ ] Statistical comparison of conditions
- [ ] Document whether composition helps or hurts

### 4B. NuScenes-QA 3D Question Answering (PC + Image) [RECOMMENDED]

**Dataset**: NuScenes-QA (via Hugging Face - NO REGISTRATION REQUIRED!)
- ~5,800 QA pairs from autonomous driving scenes (day + night)
- 5D LiDAR point clouds + 6-view camera images
- Multi-hop reasoning questions
- Evaluation: BLEU-1/4, METEOR, exact match

| Split | Day | Night | Total |
|-------|-----|-------|-------|
| Train | 2,229 | 659 | 2,888 |
| Validation | 2,229 | 659 | 2,888 |

**Why NuScenes-QA over ScanQA?**
- ✅ No registration required (direct HuggingFace download)
- ✅ Pre-processed and ready to use
- ✅ Multi-modal: LiDAR + 6 camera views
- ⚠️ Smaller dataset (~6K vs 41K)
- ⚠️ Different domain (outdoor driving vs indoor scenes)

**Experiment Scripts**:
- Training: `train_nuscenes_qa_composition.py`
- SLURM: `experiments/nuscenes_qa_composition/scripts/train_qa.sh`

| Condition | Status | Exact Match | Notes |
|-----------|--------|-------------|-------|
| Image only | ⏳ Ready | | Frozen LLaVA baseline |
| Point cloud only | ⏳ Ready | | Train SAFE adapter |
| **PC + Image** | ⏳ Ready | | True composition |

**Architecture (True Composition)**:
```
LLaVA: processes input_ids + pixel_values (native image path)
SAFE: injects PC tokens as residuals at layers [1,5,9,13,17,21]
Both modalities contribute in single forward pass
```

**Quick Start**:
```bash
# Test dataset (downloads automatically)
python -c "from safe.data.nuscenes_qa_dataset import NuScenesQADataset; d = NuScenesQADataset(split='train'); print(len(d))"

# Train composition
python train_nuscenes_qa_composition.py --modality both --scene-type day --output-dir outputs/nuscenes_qa
```

**Completion Criteria**:
- [ ] All three conditions evaluated
- [ ] Day vs night scene comparison
- [ ] Clear evidence of composition benefit (or lack thereof)

---

### 4C. ScanQA 3D Question Answering (PC + Image) [Alternative - requires registration]

**Dataset**: ScanQA
- 41,363 QA pairs from 800 ScanNet scenes
- Question types: spatial, color, counting, object identification
- Evaluation: BLEU-1/4, METEOR, exact match

**Note**: Requires ScanNet registration (http://www.scan-net.org/)

**Experiment Scripts**:
- Preprocess: `experiments/scanqa_composition/scripts/preprocess_scanqa.py`
- Training: `experiments/scanqa_composition/scripts/train_qa.sh`

| Condition | Status | BLEU-4 | METEOR | Notes |
|-----------|--------|--------|--------|-------|
| Image only | ⏳ Blocked | | | Needs ScanNet access |
| Point cloud only | ⏳ Blocked | | | Needs ScanNet access |
| **PC + Image** | ⏳ Blocked | | | Needs ScanNet access |

**Completion Criteria**:
- [ ] All three conditions evaluated
- [ ] Question-type breakdown analysis
- [ ] Clear evidence of composition benefit (or lack thereof)

### 4D. Key Questions

1. Does adding point cloud to image improve accuracy/metrics?
2. Does adding image to point cloud improve accuracy/metrics?
3. Is there interference or synergy between modalities?
4. Which question types benefit most from composition?

---

## Phase 5: KV Augmentation Architecture

**Goal**: Compare KV-Aug to Pre-FFN. Only start after Phases 1-4 complete.

> **DO NOT START UNTIL PHASES 1-4 ARE COMPLETE**

### 5A. KV-Aug Classification

| Task | Pre-FFN Acc | KV-Aug Acc | Notes |
|------|-------------|------------|-------|
| Audio Classification | | | |
| PC Classification | | | |

### 5B. KV-Aug Captioning

| Task | Pre-FFN CIDEr | KV-Aug CIDEr | Notes |
|------|---------------|--------------|-------|
| Audio Captioning | | | |
| PC Captioning | | | |

### 5C. KV-Aug Composition

| Condition | Pre-FFN Acc | KV-Aug Acc | Notes |
|-----------|-------------|------------|-------|
| Audio only | | | |
| PC only | | | |
| Both | | | |

**Completion Criteria**:
- [ ] Fair comparison with same hyperparameters
- [ ] Document any regularization needed (entropy, ablation hinge)
- [ ] Clear conclusion: KV-Aug vs Pre-FFN winner

---

## Current Focus

**Active Phase**: Phase 1B (Point Cloud Classification) + Phase 4 (Composition)

**Completed**:
- ✅ Phase 1A: ESC-50 Audio Classification - **97.35% ± 0.45%** (near-SOTA)
- ✅ MUSIC-AVQA data on cluster (9,288 videos: 7,422 Real + 1,866 Synthetic)
- ✅ NuScenes-QA composition infrastructure (ready to run!)

**Current Experiments**:
1. **ModelNet40 Point Cloud Classification** - Training in progress
   - Current: ~84% accuracy, stalling
   - Running 1000-epoch kitchen_sink + ablations (MLP head, mean pooling, 32 tokens)
   - Target: 90%+ accuracy

2. **MUSIC-AVQA Composition** [PRIMARY ECCV BENCHMARK] - Data prep in progress
   - 9,288 videos on cluster, extracting audio + frames
   - Prep script: `scripts/prepare_music_avqa.sh`
   - Training: `experiments/avqa_composition/scripts/train_preffn_music_avqa.sh`

3. **NuScenes-QA Composition** - Ready to run (supplementary)
   - Dataset: ~5.8K QA pairs (day + night driving scenes)
   - Training script: `train_nuscenes_qa_composition.py`

4. **ScanNet/ScanQA Composition** - Registration submitted, waiting
   - Requires ScanNet registration (http://www.scan-net.org/)

**Blocking Issues**:
- ModelNet40 accuracy plateau at ~84%
- ScanNet registration pending

**Next Steps**:
1. 🔄 Complete ModelNet40 ablations, push past 84%
2. 🔄 Finish MUSIC-AVQA data prep (audio extraction + manifests)
3. ⏳ Run MUSIC-AVQA composition: both, audio-only, image-only
4. ⏳ Analyze composition results by question type
5. ⏳ Run NuScenes-QA as supplementary composition benchmark

---

## Rules for This Agenda

1. **No skipping phases** - Complete each phase before moving on
2. **No parallel experiments across phases** - Focus on one phase at a time
3. **Document everything** - Fill in tables as experiments complete
4. **Push for best results** - Try multiple tricks before declaring "done"
5. **Weekly check-in** - Review this doc weekly, update status

---

## Experiment Log

### Week of 02/07/2026

**Completed**:
- MUSIC-AVQA videos transferred to cluster (9,288 videos: 7,422 Real + 1,866 Synthetic)
- NuScenes-QA dataset loader working (streaming mode, handles HF Arrow corruption)
- ModelNet40 1000-epoch run submitted + 4 ablation runs (MLP head, mean pooling, 32 tokens, combo)
- Data prep pipeline for MUSIC-AVQA created (`scripts/prepare_music_avqa.sh`)
- Updated experiment documentation with full methodology

**In Progress**:
- MUSIC-AVQA audio/frame extraction + manifest preparation
- ModelNet40 ablations running (targeting >84%)
- NuScenes-QA data download (streaming to pickle)
- ScanNet registration submitted

**Learnings**:
- MUSIC-AVQA is primary ECCV composition benchmark (Audio+Image QA)
- ModelNet40 plateau at ~84% likely due to: linear head, last-token pooling, or token count
- Single-frame extraction from video is standard practice in AV-QA (LAVISH, APE methods)
- HuggingFace datasets Arrow corruption workaround: use streaming=True + try/except

### Week of 02/03/2026

**Completed**:
- ESC-50 5-fold CV complete: **97.35% ± 0.45%** (kitchen_sink config)
- ModelNet40 baseline experiments started
- ScanNet composition experiment infrastructure created
- ScanQA QA composition experiment infrastructure created
- NuScenes-QA composition infrastructure created (no registration required!)

**In Progress**:
- ModelNet40 training with constant LR (targeting 90%+)
- NuScenes-QA composition experiments ready to run

**Blocked**:
- ScanNet requires registration (but NuScenes-QA is ready as alternative!)

**Learnings**:
- ESC-50: Mixup + label smoothing + unfreezing CLAP gave best results
- ModelNet40: Cosine LR scheduler caused plateau at ~81% (LR decayed too fast)
- Changed to constant LR with 500 epochs for point cloud experiments
- QA tasks may be better for demonstrating composition benefits than classification
- NuScenes-QA is excellent alternative to ScanQA - no registration, HuggingFace hosted

### Week of 01/30/2026

**Completed**:
- ESC-50 dataset download and preparation (2000 clips, 50 classes)
- ESC-50 baseline single-fold training (fold 5): **94.50% test accuracy**

**In Progress**:
- ESC-50 5-fold cross-validation running
- Planning ablations to push above 94.5%

**Blocked**:
- None

**Learnings**:
- 94.5% achieved with just 6 fusion layers and frozen CLAP - very promising
- Train/test gap (99% vs 94.5%) suggests room for regularization (mixup, label smoothing)
- Model converged quickly (~epoch 6), may benefit from early stopping or fewer epochs

---

### Week of ____/____/____ (copy this template for new weeks)

**Completed**:
-

**In Progress**:
-

**Blocked**:
-

**Learnings**:
-

---

## Results Log

### Phase 1A: Audio Classification Results

**Dataset**: ESC-50 (5-fold cross-validation)
- 2000 clips, 50 classes, 5-second audio
- Evaluation: Mean accuracy ± std across 5 folds

**SOTA Comparison**:
| Model | Accuracy | Notes |
|-------|----------|-------|
| BEATs | 98.1% | Current SOTA |
| **SAFE (ours)** | **97.35% ± 0.45%** | **Beats CLAP by 0.65%** |
| CLAP | 96.7% | Our encoder (frozen in baseline) |
| AST | 95.7% | Audio Spectrogram Transformer |
| Human | 81.3% | Human baseline |

**Best Configuration (kitchen_sink_8tok)**:
```
Encoder: CLAP (unfreeze last 2 layers)
Fusion layers: 1,5,9,13,17,21 (6 layers)
Num tokens: 8
Learning rate (SAFE): 6e-5
Learning rate (head): 1e-3
Epochs: 50
Batch size: 16
Mixup alpha: 0.3
Label smoothing: 0.1
```

**Ablation Results**:

| Setting | Accuracy (mean ± std) | Delta | Notes |
|---------|----------------------|-------|-------|
| Baseline | | - | |
| + More layers | | | |
| + More tokens | | | |
| + Unfreeze encoder | | | |
| + Data augmentation | | | |
| **Best combo** | | | |

**Per-Fold Results** (baseline config, batch=16):

| Fold | Val Accuracy | Train Accuracy | Best Epoch | Notes |
|------|--------------|----------------|------------|-------|
| Fold 1 | **96.00%** | 98.75% | 8 | 5-fold CV run |
| Fold 2 | | | | In progress |
| Fold 3 | | | | |
| Fold 4 | | | | |
| Fold 5 | 94.50% | 99% | ~6 | Single-fold test run |
| **Mean ± Std** | | | | 5-fold CV in progress |

**Per-Fold Results** (larger_batch config, batch=32):

| Fold | Val Accuracy | Train Accuracy | Best Epoch | Notes |
|------|--------------|----------------|------------|-------|
| Fold 1 | | | | In progress |
| Fold 2 | | | | |
| Fold 3 | | | | |
| Fold 4 | | | | |
| Fold 5 | **95.75%** | 100% | 32 | Val loss 0.59 @ best |
| **Mean ± Std** | | | | |

**Training Curves**: `experiments/esc50_classification/outputs/`

**Key Findings**:
-
-
-

**Final Accuracy**: ____% ± ____%

---

### Phase 1B: Point Cloud Classification Results

**Dataset**: ModelNet40
- 12,311 models (9,843 train / 2,468 test)
- 40 classes
- Single train/test split

**SOTA Comparison**:
| Model | Accuracy |
|-------|----------|
| PointNeXt | 94.0% |
| Point-MAE | 93.8% |
| PointBERT | 93.2% |
| PointNet++ | 91.9% |
| **SAFE (ours)** | ____% |

**Best Configuration**:
```
Encoder: PointBERT (frozen/unfrozen: ___)
Fusion layers:
Num tokens:
Num points:
Learning rate:
Epochs:
Other settings:
```

**Ablation Results**:

| Setting | Accuracy | Delta | Training Time |
|---------|----------|-------|---------------|
| Baseline | | - | |
| + More layers | | | |
| + More tokens | | | |
| + More points | | | |
| + Unfreeze encoder | | | |
| **Best combo** | | | |

**Training Curves**: `outputs/pc_classification/best_run/`

**Key Findings**:
-
-
-

**Final Accuracy**: ____%

---

### Phase 2: Model Transfer Results

**Comparison: LLaVA-1.5-13B vs Qwen3-8B**

| Task | LLaVA-13B | Qwen3-8B | Difference |
|------|-----------|----------|------------|
| Audio Classification | | | |
| PC Classification | | | |

**Architecture Changes Needed**:
-

**Key Findings**:
-
-

**Conclusion**: Transfer successful? Yes / No

---

### Phase 3A: Audio Captioning Results

**Dataset**: AudioCaps / Clotho (specify)

**Best Configuration**:
```
Training: single-stage / 2-stage
Fusion layers:
Num tokens:
Learning rate:
Epochs:
Other settings:
```

**Metrics**:

| Experiment | CIDEr | BLEU-4 | METEOR | ROUGE-L |
|------------|-------|--------|--------|---------|
| Baseline | | | | |
| Best config | | | | |
| + 2-stage | | | | |

**Sample Outputs**:

| Audio | Prediction | Ground Truth |
|-------|------------|--------------|
| sample1.wav | | |
| sample2.wav | | |
| sample3.wav | | |

**Key Findings**:
-
-
-

---

### Phase 3B: Point Cloud Captioning Results

**Dataset**: Cap3D

**Best Configuration**:
```
Fusion layers:
Num tokens:
Learning rate:
Epochs:
Other settings:
```

**Metrics**:

| Experiment | CIDEr | BLEU-4 | METEOR | ROUGE-L |
|------------|-------|--------|--------|---------|
| Baseline | | | | |
| Best config | | | | |

**Sample Outputs**:

| Object | Prediction | Ground Truth |
|--------|------------|--------------|
| obj1 | | |
| obj2 | | |
| obj3 | | |

**Key Findings**:
-
-
-

---

### Phase 4: Modality Composition Results

**Evaluation**: MCUB

**Adapter Training**: Classification / Captioning (specify)

**Composition Matrix**:

| Condition | Accuracy | Std | N |
|-----------|----------|-----|---|
| Vision only (baseline) | | | |
| Audio only | | | |
| Point cloud only | | | |
| Vision + Audio | | | |
| Vision + PC | | | |
| Audio + PC | | | |
| Vision + Audio + PC | | | |

**Statistical Significance**:

| Comparison | p-value | Significant? |
|------------|---------|--------------|
| Audio+PC vs Audio only | | |
| Audio+PC vs PC only | | |
| All three vs Vision only | | |

**Key Findings**:
-
-
-

**Conclusion**: Does composition help? Yes / No / Mixed

---

### Phase 5: KV-Aug vs Pre-FFN Comparison

> Only fill in after Phases 1-4 complete

### Phase 5A: EPIC-SOUNDS Vision-Language-Audio QA Composition

**Objective**:
- Validate vision+language+audio composition on egocentric kitchen videos.
- Compare `pre_ffn` vs `kv_augment` under the same QA task and data manifest.
- Measure modality utilization by evaluating `both`, `image`, and `audio` modes.

**ECCV Dataset Lock (Pre-FFN Only)**:
- Primary composition benchmark: `MUSIC-AVQA` (main table for ECCV).
- Secondary composition benchmark: `EPIC-SOUNDS AVQA` (egocentric transfer/realism track).
- Supplemental benchmark: `General AVQA` (appendix or robustness table if time allows).
- KV-augment runs are excluded from ECCV main claims and tracked for NeurIPS follow-up.

**Experiment Package**:
- `experiments/epic_sounds_avqa_composition/README.md`
- `experiments/epic_sounds_avqa_composition/train_epic_sounds_avqa.py`
- `experiments/epic_sounds_avqa_composition/scripts/download_epic_sounds_data.py`
- `experiments/epic_sounds_avqa_composition/scripts/prepare_epic_sounds_avqa.py`

**Task**: QA (captioning avoided for stability reasons)
- `audio_event`: \"What sound do you hear?\" -> sound class
- `vision_object`: \"Which object is being handled?\" -> noun
- `av_composition`: \"What is happening to the {noun}?\" -> \"{noun} is being {verb}\"

**Run Matrix (Architectures x Modalities)**:

| Run ID | Architecture | Train Modality | Eval Modalities | Status | Notes |
|--------|--------------|----------------|-----------------|--------|-------|
| epic_preffn_v1 | pre_ffn | both | both,audio,image | Planned | Baseline composition run |
| epic_kvaug_v1 | kv_augment | both | both,audio,image | Planned | KV augmentation run |
| epic_preffn_audio_only_v1 | pre_ffn | audio | audio | Planned | Shortcut/control |
| epic_preffn_image_only_v1 | pre_ffn | image | image | Planned | Shortcut/control |

**Primary Metrics**:
- Exact Match (%)
- Token F1 (%)
- Per-question-type Exact Match/F1 for:
- `audio_event`
- `vision_object`
- `av_composition`

**Result Log**:

| Date | Run ID | EM (both) | F1 (both) | EM (av_composition) | F1 (av_composition) | Artifact Path | Notes |
|------|--------|-----------|-----------|----------------------|----------------------|---------------|-------|
| 2026-02-06 | epic_preffn_v1 | TBA | TBA | TBA | TBA | `checkpoints/epic_sounds_avqa/preffn` | Pending run |
| 2026-02-06 | epic_kvaug_v1 | TBA | TBA | TBA | TBA | `checkpoints/epic_sounds_avqa/kv_augment` | Pending run |

**Success Criteria for Composition**:
- `both` outperforms `audio` and `image` on `av_composition` QA.
- Qualitative predictions include both object + action grounding (e.g., \"tomato is being washed\").
- Architecture comparison is based on matched data, prompt format, and evaluation settings.

<!-- EPIC_AVQA_RESULTS_START -->
### Phase 5A EPIC-SOUNDS Result Snapshot

Updated: 2026-02-06

| Run | Architecture | Best Epoch | EM (both) | F1 (both) | EM (av_composition) | F1 (av_composition) | EM (audio eval) | EM (image eval) | Artifact |
|-----|--------------|------------|-----------|-----------|----------------------|----------------------|------------------|------------------|----------|
| epic_preffn | pre_ffn | TBA | TBA | TBA | TBA | TBA | TBA | TBA | `checkpoints/epic_sounds_avqa/preffn` |
| epic_kvaug | kv_augment | TBA | TBA | TBA | TBA | TBA | TBA | TBA | `checkpoints/epic_sounds_avqa/kv_augment` |
<!-- EPIC_AVQA_RESULTS_END -->

**Classification Comparison**:

| Task | Pre-FFN | KV-Aug | Winner |
|------|---------|--------|--------|
| Audio | | | |
| Point Cloud | | | |

**Captioning Comparison**:

| Task | Pre-FFN CIDEr | KV-Aug CIDEr | Winner |
|------|---------------|--------------|--------|
| Audio | | | |
| Point Cloud | | | |

**Composition Comparison**:

| Condition | Pre-FFN | KV-Aug | Winner |
|-----------|---------|--------|--------|
| Audio only | | | |
| PC only | | | |
| Audio + PC | | | |

**KV-Aug Specific Notes**:
- Regularization used: entropy weight = ___, ablation hinge weight = ___
- Collapse observed? Yes / No
- Training stability:

**Key Findings**:
-
-
-

**Final Conclusion**: Pre-FFN vs KV-Aug recommendation:

---

## ECCV Finalization Checklist (Current -> Submission)

**Scope Lock (by February 10, 2026)**
- [ ] Freeze ECCV method scope to Pre-FFN residual only (no new architecture additions).
- [ ] Freeze ECCV datasets/tasks: ESC-50, ModelNet40, MUSIC-AVQA (primary composition), EPIC-SOUNDS AVQA (secondary composition).
- [ ] Freeze evaluation metrics and prompt formats for all ECCV tables.

**Experiment Lock (by February 12, 2026)**
- [ ] Define final run matrix with exact configs/seeds in one file (`configs/` + script args).
- [ ] Define run IDs and output directories for every main table row.
- [ ] Define ablation subsets that are in-scope for ECCV and defer all others.

**Phase 1A Audio (Finalize by February 18, 2026)**
- [ ] Fill final ESC-50 5-fold mean ± std in `Phase 1A` table.
- [ ] Fill per-fold table fully (fold1..fold5) and best-epoch metadata.
- [ ] Fill ablation table deltas (layers/tokens/unfreeze/mixup/label smoothing).
- [ ] Archive training curves under `experiments/esc50_classification/outputs/`.

**Phase 1B Point Cloud (Finalize by February 22, 2026)**
- [ ] Fill final ModelNet40 test accuracy in `Phase 1B` section.
- [ ] Fill point cloud ablation table with deltas and training-time notes.
- [ ] Select one final config and mark all non-winning runs as exploratory.

**Composition QA (Finalize by February 26, 2026)**
- [ ] Run composition matrix with controlled modalities (image/audio/both).
- [ ] Fill composition table with exact metrics and sample counts.
- [ ] Add qualitative examples that explicitly demonstrate cross-modal grounding.
- [ ] Confirm at least one “object + action” success case in outputs.

**Statistical and Robustness Checks (by February 28, 2026)**
- [ ] Add confidence intervals or seed variance for primary claims.
- [ ] Run significance tests for core comparisons and fill p-value table.
- [ ] Document known failure modes and one negative result case.

**Paper Assets Freeze (by March 2, 2026)**
- [ ] Export final figures (architecture, curves, qualitative grid).
- [ ] Export final tables (main + appendix) from run artifacts, not manual edits.
- [ ] Generate one reproducibility manifest (configs, seeds, checkpoints, commit hash).

**Writing and Internal Review (March 3, 2026 to March 13, 2026)**
- [ ] Draft complete narrative around Pre-FFN contribution and limitations.
- [ ] Add related work positioning specific to residual fusion and multimodal adapters.
- [ ] Complete appendix: hyperparameters, compute budget, dataset prep details.
- [ ] Perform one full internal technical review pass and resolve all blocking comments.

**Submission Readiness (March 16, 2026 to March 20, 2026)**
- [ ] Freeze all ECCV numbers and lock manuscript.
- [ ] Verify every claim maps to a table/figure or appendix citation.
- [ ] Verify anonymity and supplementary package completeness.
- [ ] Submit ECCV package.

**Owner Execution Board**

| Workstream | Output Artifact | Status |
|------------|------------------|--------|
| Audio classification final | `docs/RESEARCH_AGENDA.md` Phase 1A tables filled | ⬜ |
| Point cloud classification final | `docs/RESEARCH_AGENDA.md` Phase 1B tables filled | ⬜ |
| Composition QA final | `docs/RESEARCH_AGENDA.md` Phase 4 table + examples | ⬜ |
| Statistical validation | p-value table + variance notes | ⬜ |
| Paper assets | figures/tables + reproducibility manifest | ⬜ |
| Manuscript | full draft + appendix + final polish | ⬜ |

**Definition of Done for ECCV**
- [ ] Every placeholder in Phase 1A/1B/4 tables is replaced with a final number.
- [ ] Main claim is supported by at least one statistically validated comparison.
- [ ] Reproducibility manifest is complete and points to exact run artifacts.
- [ ] Final PDF is submission-ready with no TODO markers.

---

## Final Summary

**Best Audio Pipeline**:
- Encoder:
- Fusion: Pre-FFN / KV-Aug
- Layers:
- Tokens:
- Best task: Classification / Captioning

**Best Point Cloud Pipeline**:
- Encoder:
- Fusion: Pre-FFN / KV-Aug
- Layers:
- Tokens:
- Best task: Classification / Captioning

**Composition Verdict**:
- Does multi-modal help?
- Best combination:

**Paper-Ready Results**:
- [ ] All tables filled
- [ ] Statistical tests done
- [ ] Training curves plotted
- [ ] Sample outputs curated
