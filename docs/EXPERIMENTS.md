# SAFE Training Experiments & Validation Plan

This document outlines a comprehensive experimental validation plan for SAFE (Safe Audio-Visual Enhancement) training. Each experiment has clear success criteria ("Go" conditions) that must be met before proceeding.

## Experiment Categories

### 🔍 **Data Validation** → 🧠 **Model Safety** → 📊 **Learning Validation** → ⚡ **Efficiency** → 🎯 **Full Pipeline**

---

## 1. Data & Pipeline Preflight

### Schema & Distribution Audit
**Objective**: Validate data quality per dataset and difficulty level

**Tasks**:
- Class balance analysis for each dataset (AudioCaps, VQA, MUSIC-AVQA)
- Length/SNR histograms for audio samples
- Duplicate detection within and across datasets
- Label/ID leakage detection between train/val/test splits

**Go Criteria**:
- ✅ No label/ID leakage detected
- ✅ ≥95% of files load successfully 
- ✅ Class imbalance <5:1 unless intentionally designed
- ✅ No duplicate samples across splits

### A/V Alignment Sanity Check
**Objective**: Verify audio-visual temporal alignment

**Tasks**:
- Sample 1k mixed audio-visual pairs
- Check event time overlap between audio and visual content
- Verify speech presence vs. transcripts correlation
- Manual spot-check for obviously mismatched pairs

**Go Criteria**:
- ✅ ≥90% alignment score on curated validation checks
- ✅ <2% obviously mismatched audio-visual pairs

### YAML Configuration Validation
**Objective**: Ensure curriculum sampling matches configuration

**Tasks**:
- Verify sampling ratios match curriculum specifications
- Test stage difficulty filters are applied correctly
- Validate batch composition statistics

**Go Criteria**:
- ✅ Sampled batch statistics match configuration ±2%
- ✅ Stage transitions occur at specified thresholds

---

## 2. Base-Parity (Safety) Checks

### Parameter Freeze Audit
**Objective**: Verify only intended parameters are trainable

**Tasks**:
- Print trainable vs. frozen parameter counts
- Verify backbone models (BLIP-2, CLAP) are 100% frozen
- Confirm only projector + LoRA fusion layers are trainable

**Go Criteria**:
- ✅ Trainable parameters ≈ 10–50M (0.2% of total)
- ✅ Backbone models 100% frozen
- ✅ Only projector and LoRA layers show `requires_grad=True`

### Forward Equivalence Test
**Objective**: Ensure SAFE with gating OFF = base VL model

**Tasks**:
- Run 1k VL-only samples through both models
- Compare logits with audio gating disabled
- Verify accuracy metrics are identical

**Go Criteria**:
- ✅ Mean absolute logit difference <1e-5
- ✅ VQA accuracy identical between base and SAFE (gate OFF)
- ✅ No computational overhead when gate OFF

---

## 3. Tiny Overfit (Learning Path Sanity)

### Small-Scale Learning Validation
**Objective**: Verify model can learn on tiny dataset

**Tasks**:
- Train on 512–1k mixed examples (always-on audio)
- Run for a few hundred steps until overfitting
- Measure training loss convergence and accuracy

**Go Criteria**:
- ✅ Training loss approaches near-zero
- ✅ Audio QA accuracy >90% on tiny training split
- ✅ VL retention unchanged with gate OFF
- ✅ Clear learning curve progression

---

## 4. Projector + Fusion Pilot (Stage-A Warm Start)

### Medium-Scale Training Pilot
**Objective**: Validate Stage-A training approach

**Tasks**:
- Train on 50k mixed samples
- Train projector + LoRA only (backbone frozen)
- Evaluate on held-out validation sets
- Measure both audio gains and VL retention

**Go Criteria**:
- ✅ VQA/GQA performance drop ≤0.5% (gate OFF)
- ✅ Audio task improvement ≥+3–5% vs. base model
- ✅ No latency spikes >+10% when audio enabled
- ✅ Stable training curves with no divergence

### Architecture Hyperparameter Ablation
**Objective**: Find optimal projector and LoRA configuration

**Tasks**:
- Test audio token counts: k ∈ {4, 8, 12}
- Test LoRA ranks: r ∈ {4, 8, 16}  
- Evaluate efficiency vs. performance trade-offs

**Go Criteria**:
- ✅ Select smallest config meeting performance targets
- ✅ Memory usage fits within hardware constraints
- ✅ Latency increase acceptable for gains achieved

---

## 5. Gating Policy Smoke Tests

### Pre-RL Gating Validation
**Objective**: Test audio consultation policies before RL training

**Tasks**:
- Implement heuristic gates (entropy, confidence thresholds)
- Compare always-on vs. always-off vs. heuristic gating
- Measure efficiency gains and performance trade-offs

**Go Criteria**:
- ✅ Heuristic gate achieves ≥20% audio skips
- ✅ Audio task performance drop ≤1%
- ✅ VL performance unchanged (gate OFF path)
- ✅ Clear efficiency vs. accuracy trade-off curve

---

## 6. Efficiency & Resource Profiling

### Hardware Performance Characterization
**Objective**: Understand computational requirements and limits

**Tasks**:
- Profile throughput vs. batch size curves
- Measure VRAM usage with/without audio
- Compare on-the-fly vs. cached audio embeddings
- Benchmark per-sample latency with different audio token counts

**Go Criteria**:
- ✅ Target batch size fits in VRAM with ≥10% headroom
- ✅ Audio processing adds acceptable latency (define threshold)
- ✅ Memory usage scales predictably with batch size
- ✅ Throughput meets training time requirements

---

## 7. Evaluation Harness Dry Runs

### Metrics Validation
**Objective**: Verify evaluation metrics are computed correctly

**Tasks**:
- Compare VQA accuracy against known baselines
- Validate CIDEr/SPIDEr scores on standard datasets
- Test retention KL divergence calculations
- Cross-check with literature benchmarks

**Go Criteria**:
- ✅ Baseline model scores match published results
- ✅ All metrics stable across multiple runs
- ✅ No obvious computational errors in metric calculation

### LOMO (Leave One Modality Out) Testing
**Objective**: Validate audio dependency detection

**Tasks**:
- Run evaluation with audio muted on audio-dependent questions
- Measure performance drop on truly audio-reliant items
- Test on VL-only questions (should show no drop)

**Go Criteria**:
- ✅ ≥80% of audio-dependent items show performance drop when muted
- ✅ VL-only items show no performance degradation
- ✅ Clear separation between audio-dependent and VL-only performance

### Irrelevance Stress Testing
**Objective**: Test false consultation resistance

**Tasks**:
- Pair VL-only questions with random/noisy audio
- Measure false consultation rate under different gating policies
- Validate robustness to audio distractors

**Go Criteria**:
- ✅ False consultation rate ≤5% under heuristic gating
- ✅ Performance on VL-only tasks unaffected by noise audio
- ✅ Gating policy robust to audio quality variations

---

## 8. Curriculum Logic Rehearsal

### End-to-End Curriculum Testing
**Objective**: Validate curriculum learning implementation

**Tasks**:
- Run 1 epoch per curriculum stage with small dataset
- Test both normal progression and extension scenarios
- Verify checkpoint/resume functionality between stages
- Validate all logging and metrics collection

**Go Criteria**:
- ✅ No crashes during stage transitions
- ✅ Stage transition criteria applied correctly
- ✅ Identical metrics after checkpoint resume
- ✅ All curriculum logging behaves as specified

---

## 9. Retention Mechanisms Stress Test

### Catastrophic Forgetting Prevention
**Objective**: Validate retention loss effectiveness

**Tasks**:
- Compare training with/without distillation loss
- Measure VL performance drift over training
- Test null-space projection if implemented
- Ablate retention loss weight values

**Go Criteria**:
- ✅ Distillation reduces VL drift by ≥50% vs. no distillation
- ✅ Null-space projection maintains or improves retention
- ✅ Optimal retention weight found through systematic testing

---

## 10. Pilot Hyperparameter Sweep

### Coarse Grid Search
**Objective**: Find stable hyperparameter ranges

**Tasks**:
- Test projector learning rates: {5e-5, 1e-4, 2e-4}
- Test LoRA learning rates: {2e-5, 5e-5, 1e-4}
- Test retention loss weights: {0.5, 1.0, 2.0}
- Evaluate loss curve stability and convergence

**Go Criteria**:
- ✅ Select smallest LR/λ combo meeting Stage-A criteria
- ✅ Stable loss curves without oscillation or divergence
- ✅ Consistent performance across hyperparameter ranges

---

## 11. Variance & Reproducibility

### Multi-Seed Validation
**Objective**: Ensure training stability and reproducibility

**Tasks**:
- Run 3 different random seeds on best configuration
- Measure standard deviation of key metrics
- Verify deterministic behavior of frozen paths

**Go Criteria**:
- ✅ Standard deviation ≤0.5% on audio accuracy
- ✅ Standard deviation ≤0.2% on VL retention metrics
- ✅ Identical checksums for frozen-path outputs across seeds

---

## 12. (Optional) RL Gate Micro-Pilot

### Reinforcement Learning Validation
**Objective**: Test RL-based audio consultation policies

**Tasks**:
- Implement bandit for consultation decisions {0,1}
- Test token count optimization k ∈ {0, 4, 8}
- Train on 50–100k examples with reward = score - α·latency
- Maintain VL retention constraints

**Go Criteria**:
- ✅ Achieve ≥30–40% audio skips with ≤1% audio task drop
- ✅ VL constraint never violated (≥baseline - 0.5%)
- ✅ Clear Pareto frontier between efficiency and accuracy

---

## 🚦 Launch Checklist (Green-Light Criteria)

Before scaling to full model + full dataset, ensure ALL criteria are met:

- ✅ **Base-parity holds**: Gate OFF path ≡ base model performance
- ✅ **Pilot shows gains**: Clear audio improvements with no VL regression  
- ✅ **Efficiency validated**: Hardware requirements and SLOs achievable
- ✅ **Curriculum verified**: Stage transitions and checkpoints working
- ✅ **Metrics validated**: LOMO and irrelevance tests behave as expected
- ✅ **Reproducible**: Low seed variance with locked configurations

## 📊 Experiment Tracking

### Recommended Tools
- **Weights & Biases**: For experiment logging and hyperparameter tracking
- **TensorBoard**: For real-time loss and metric visualization  
- **MLflow**: For model versioning and experiment comparison
- **Custom Scripts**: For dataset analysis and validation checks

### Key Metrics to Track
- **Audio Task Performance**: Accuracy on AudioCaps and MUSIC-AVQA
- **VL Retention**: VQA v2 and GQA accuracy with gate OFF
- **Efficiency Metrics**: Tokens used, consultation rate, latency
- **Training Stability**: Loss curves, gradient norms, learning rates

---

## 🔄 Execution Strategy

1. **Sequential Execution**: Complete experiments 1-8 before advancing
2. **Parallel Where Possible**: Run efficiency profiling alongside learning validation
3. **Documentation**: Log all results with clear pass/fail status
4. **Iteration**: If any experiment fails, diagnose and repeat before proceeding

**Once all experiments show green checkmarks, you can confidently scale to full training with minimal surprises!**