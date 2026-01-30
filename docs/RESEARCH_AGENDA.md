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
| Baseline (default config) | 🔄 In Progress | | Layers 1,5,9,13,17,21 |
| More fusion layers (1,5,9,13,17,21,25,29,33,37) | | | |
| More tokens (8 → 16 → 32) | | | |
| Unfreeze last 2 encoder blocks | | | |
| Unfreeze last 4 encoder blocks | | | |
| Higher learning rate sweep | | | |
| Longer training (more epochs) | | | |
| Data augmentation (SpecAugment, noise) | | | |

**Completion Criteria**:
- [ ] Best config identified and documented
- [ ] Ablation table showing contribution of each trick
- [ ] Final accuracy: ____% (5-fold mean ± std)
- [ ] Training curves saved to `experiments/esc50_classification/outputs/`

### 1B. Point Cloud Classification (ModelNet40 / ScanObjectNN)

| Experiment | Status | Accuracy | Notes |
|------------|--------|----------|-------|
| Baseline (default config) | | | |
| More fusion layers | | | |
| More tokens (8 → 16 → 32) | | | |
| Unfreeze last 2 encoder blocks | | | |
| Unfreeze last 4 encoder blocks | | | |
| More points (1024 → 2048 → 4096) | | | |
| Higher learning rate sweep | | | |
| Longer training | | | |

**Completion Criteria**:
- [ ] Best config identified and documented
- [ ] Ablation table showing contribution of each trick
- [ ] Final accuracy: ____%
- [ ] Training curves saved

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

**Goal**: Test how audio + point cloud interact when combined. Does composition help?

### 4A. MCUB Evaluation

| Condition | Status | Accuracy | Notes |
|-----------|--------|----------|-------|
| Audio only | | | |
| Point cloud only | | | |
| Audio + PC (both) | | | |
| Vision only (baseline) | | | |
| Audio + Vision | | | |
| PC + Vision | | | |
| All three | | | |

**Key Questions**:
- Does adding modalities improve accuracy?
- Is there interference between modalities?
- Which combinations work best?

**Completion Criteria**:
- [ ] Full composition matrix evaluated
- [ ] Statistical significance tested
- [ ] Clear conclusions about composition effects

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

**Active Phase**: Phase 1A - Audio Classification

**Current Experiment**: ESC-50 Baseline (5-fold CV)

**Blocking Issues**:
- None

**Next Steps**:
1. Download ESC-50 dataset: `bash experiments/esc50_classification/scripts/download_esc50.sh`
2. Run baseline training: `sbatch experiments/esc50_classification/scripts/train_5fold.sh`
3. Analyze results and plan ablations based on baseline performance

---

## Rules for This Agenda

1. **No skipping phases** - Complete each phase before moving on
2. **No parallel experiments across phases** - Focus on one phase at a time
3. **Document everything** - Fill in tables as experiments complete
4. **Push for best results** - Try multiple tricks before declaring "done"
5. **Weekly check-in** - Review this doc weekly, update status

---

## Experiment Log

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
| Model | Accuracy |
|-------|----------|
| BEATs | 98.1% |
| CLAP | 96.7% |
| AST | 95.7% |
| Human | 81.3% |
| **SAFE (ours)** | ____% |

**Best Configuration**:
```
Encoder: CLAP (frozen/unfrozen: ___)
Fusion layers:
Num tokens:
Learning rate (projector):
Learning rate (adapter):
Epochs:
Batch size:
Gradient accumulation:
Other settings:
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

**Per-Fold Results** (for best config):

| Fold | Val Accuracy | Notes |
|------|--------------|-------|
| Fold 1 | | |
| Fold 2 | | |
| Fold 3 | | |
| Fold 4 | | |
| Fold 5 | | |
| **Mean ± Std** | | |

**Training Curves**: `experiments/esc50_classification/outputs/`

**Key Findings**:
-
-
-

**Final Accuracy**: ____% ± ____%

---

### Phase 1B: Point Cloud Classification Results

**Dataset**: ModelNet40 / ScanObjectNN (specify)

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
