# SAFE: Scalable Incremental Learning for Multimodal Foundation Models

## Research Plan: Proving Modality-Agnostic Incremental Learning

---

## Executive Summary

**Core Thesis**: SAFE (Selectively Augmenting Frozen Encoders) demonstrates that new modalities can be incrementally added to frozen foundation models with minimal trainable parameters (~2M) while preserving original capabilities. This research plan outlines the systematic validation needed to establish SAFE as a general-purpose incremental learning framework.

**Current State**:
- Audio path implemented with CIDEr ~50, METEOR ~0.19 on limited data
- Architecture validated but ablations incomplete
- Single modality proven; scalability undemonstrated

**Target Outcomes**:
1. Comprehensive ablation study establishing optimal hyperparameters
2. Multi-modality experiments proving framework generalization
3. Scaling laws for incremental modality addition
4. Publication-ready empirical validation

---

## Phase 1: Audio Baseline Consolidation (Weeks 1-3)

### 1.1 Complete Missing Ablations

**Priority: CRITICAL** - These establish the foundation for all multi-modal claims.

#### A. Projector Architecture Ablations

| Experiment | Variables | Metrics | Runs |
|------------|-----------|---------|------|
| **Layer Depth** | num_layers ∈ {1, 2, 3, 4} | CIDEr, METEOR, VL Retention | 3 seeds each |
| **Hidden Dimension** | bottleneck_dim ∈ {512, 1024, 2048, 4096} | Same + param count | 3 seeds each |
| **Token Count** | num_audio_tokens ∈ {4, 8, 16, 32} | Same + inference speed | 3 seeds each |
| **Activation** | {gelu, relu, silu, tanh} | Convergence speed, final metrics | 3 seeds each |

```python
# Suggested config structure for projector ablations
projector_ablation_configs = {
    "depth_study": {
        "base": {"bottleneck_dim": 1024, "num_audio_tokens": 8},
        "sweep": {"num_layers": [1, 2, 3, 4]}
    },
    "width_study": {
        "base": {"num_layers": 2, "num_audio_tokens": 8},
        "sweep": {"bottleneck_dim": [512, 1024, 2048, 4096]}
    },
    "token_study": {
        "base": {"num_layers": 2, "bottleneck_dim": 1024},
        "sweep": {"num_audio_tokens": [4, 8, 16, 32]}
    }
}
```

#### B. Fusion Adapter Ablations

| Experiment | Variables | Metrics |
|------------|-----------|---------|
| **LoRA Rank** | r ∈ {4, 8, 16, 32, 64} | Performance vs params |
| **LoRA Alpha** | α ∈ {8, 16, 32} (with fixed rank) | Stability, final perf |
| **Residual Scale** | scale ∈ {0.01, 0.05, 0.1, 0.2} | Retention vs capability |
| **Fusion Layers** | {embed_only, [6,12], [6,12,24], all} | Performance vs compute |
| **Target Modules** | {Q, V}, {Q, K, V}, {Q, K, V, O} | Expressiveness |

#### C. Retention Strategy Comparison

| Variant | Config | Expected Outcome |
|---------|--------|------------------|
| `no_retention` | λ_ret = 0 | Upper bound audio, lower bound VL |
| `soft_retention` | KL only, λ = 0.1 | Balanced baseline |
| `fisher_retention` | KL + Fisher, λ = 0.1 | Selective preservation |
| `nullspace_retention` | KL + Nullspace | Gradient-aware preservation |
| `full_retention` | All mechanisms | Maximum VL safety |

**Deliverable**: Table showing (CIDEr, METEOR, VL_Retention) for each variant.

### 1.2 Scale Up Audio Training

**Goal**: Establish best-case audio performance with full data.

#### A. Dataset Expansion

| Dataset | Size | Type | Current | Target |
|---------|------|------|---------|--------|
| AudioCaps | 46K | Captioning | ✅ Using | Full training |
| WavCaps | 400K+ | Captioning | ✅ Partial | Full training |
| Clotho | 6K | Captioning | ❌ | Add |
| AudioSet-SL | 20K | Strong labels | ❌ | Add |
| VALOR-32K | 32K | Audio-visual | ❌ | Add for AVQA |

#### B. Training Scale Study

```yaml
scale_study:
  small:
    data: AudioCaps only (46K)
    epochs: 5
    expected_cider: ~50 (current)

  medium:
    data: AudioCaps + WavCaps (450K)
    epochs: 3
    expected_cider: ~65-75

  large:
    data: All audio datasets (500K+)
    epochs: 2
    expected_cider: ~80-90
```

### 1.3 Establish Strong Baselines

**Baselines Required**:

1. **Original VL (No Audio)**: Baseline performance on VQA/GQA
2. **Full Fine-tuning**: Unfreeze base model, fine-tune on audio
3. **LoRA-only**: LoRA without gating mechanism
4. **Linear Probe**: Simple linear projection (no MLP)
5. **Concatenation**: Audio tokens concatenated vs. cross-attention fused

---

## Phase 2: Multi-Modal Expansion (Weeks 4-8)

### 2.1 Video Modality (Weeks 4-5)

**Hypothesis**: The same projector + fusion architecture transfers to video with minimal modification.

#### A. Video Encoder Selection

| Encoder | Dim | Strength | Integration Effort |
|---------|-----|----------|-------------------|
| CLIP ViT | 768 | Already in ecosystem | Low |
| VideoMAE | 768 | Temporal understanding | Medium |
| InternVideo | 768 | State-of-the-art | Medium |
| LanguageBind-Video | 768 | Aligned embeddings | Low |

**Recommendation**: Start with CLIP ViT (frame-based) for direct comparison, then VideoMAE for temporal.

#### B. Video Projector Configuration

```python
video_projector_config = {
    # Transfer from audio (baseline)
    "audio_transfer": {
        "architecture": "same_as_audio",
        "num_layers": 2,  # From audio ablation winner
        "bottleneck_dim": 1024,
        "num_video_tokens": 8,  # Start same as audio
    },

    # Video-specific (temporal)
    "temporal_aware": {
        "architecture": "temporal_pooling",
        "frame_tokens": 4,
        "num_frames": 8,
        "temporal_aggregation": "attention",  # vs mean, max
    }
}
```

#### C. Video Datasets

| Dataset | Size | Task | Priority |
|---------|------|------|----------|
| MSRVTT | 10K | Captioning | P0 |
| MSVD | 2K | Captioning | P0 |
| ActivityNet-Captions | 20K | Dense captioning | P1 |
| VATEX | 41K | Multilingual | P2 |

#### D. Video Experiments

| Experiment | Goal | Metric |
|------------|------|--------|
| **Transfer Test** | Audio projector → Video | CIDEr gap |
| **Architecture Ablation** | Video-specific vs generic | CIDEr, VL Retention |
| **Temporal Ablation** | Frames: 4, 8, 16, 32 | Quality vs compute |

### 2.2 Depth Modality (Weeks 5-6)

**Hypothesis**: Dense prediction modalities (depth, segmentation) follow same pattern.

#### A. Depth Encoder Selection

| Encoder | Output | Integration |
|---------|--------|-------------|
| DINOv2 (depth head) | Dense features | Medium |
| Depth Anything | 384-dim | Low |
| MiDaS | Dense | Medium |

#### B. Depth Projector Design

```python
depth_projector_config = {
    "encoder": "depth_anything",
    "input_dim": 384,
    "architecture": "spatial_pooling",  # Pool spatial grid → tokens
    "num_depth_tokens": 8,
    "spatial_reduction": "adaptive_avg_pool",  # 14x14 → 4x4 → flatten
}
```

#### C. Depth Datasets

| Dataset | Size | Task |
|---------|------|------|
| NYU Depth V2 | 1.4K | Depth estimation QA |
| KITTI | 15K | Driving scenes |
| SUN RGB-D | 10K | Indoor scenes |

### 2.3 Tactile/Touch Modality (Weeks 6-7)

**Hypothesis**: Even exotic modalities can be integrated with same framework.

#### A. Tactile Encoder

| Encoder | Modality | Availability |
|---------|----------|--------------|
| T3 (Touch-Text-Tactile) | Tactile images | Pretrained available |
| Touch and Go | Tactile + proprioception | Research |

#### B. Tactile Datasets

| Dataset | Size | Sensor |
|---------|------|--------|
| Touch and Go | 15K | GelSight |
| ObjectFolder | 1K | Synthetic |
| YCB-Tactile | 100+ objects | Multi-sensor |

### 2.4 Multi-Modal Combination (Weeks 7-8)

**Hypothesis**: Multiple modalities can be added sequentially without forgetting.

#### A. Sequential Addition Experiments

```
Experiment Chain:
1. Base VL → +Audio (SAFE_A)
2. SAFE_A → +Video (SAFE_AV)
3. SAFE_AV → +Depth (SAFE_AVD)
4. SAFE_AVD → +Tactile (SAFE_AVDT)

Measure at each step:
- New modality performance
- All previous modality retention
- Base VL retention
- Total trainable params
```

#### B. Parallel vs Sequential Training

| Strategy | Description | Hypothesis |
|----------|-------------|------------|
| Sequential | Add one modality at a time, freeze previous | Better retention |
| Parallel | Train all modalities jointly | Better cross-modal |
| Curriculum | Gradually introduce modalities | Best of both |

---

## Phase 3: Scaling Laws & Theoretical Analysis (Weeks 9-11)

### 3.1 Scaling Experiments

#### A. Model Size Scaling

| Base VL Model | Params | Audio Performance | Retention |
|---------------|--------|-------------------|-----------|
| LLaVA-7B | 7B | Baseline | Baseline |
| LLaVA-13B | 13B | Expected +5% | Expected same |
| LLaVA-34B | 34B | Expected +8% | Expected same |

**Key Question**: Do benefits of SAFE scale with base model size?

#### B. Projector Size Scaling

```
Scaling Law Hypothesis:
Performance ∝ log(projector_params) * sqrt(data_size)

Test:
- Fixed data, vary projector: 0.5M, 1M, 2M, 4M, 8M params
- Fixed projector, vary data: 10K, 50K, 100K, 500K samples
```

#### C. Modality Count Scaling

| # Modalities | Total Trainable | Per-Modality Overhead |
|--------------|-----------------|----------------------|
| 1 (audio) | ~2M | 2M |
| 2 (+ video) | ~4M | 2M |
| 3 (+ depth) | ~6M | 2M |
| 4 (+ tactile) | ~8M | 2M |

**Key Claim**: Linear scaling in trainable parameters with modality count.

### 3.2 Theoretical Analysis

#### A. Prove Retention Guarantee

**Theorem (Informal)**: When gate = 0, SAFE output is functionally equivalent to base VL.

**Proof Sketch**:
1. Audio tokens have zero influence when gate = 0
2. LoRA adapters contribute zero when gate scales to 0
3. Therefore output = base_vl(input)

**Empirical Validation**: Gate=0 accuracy matches base VL within numerical precision.

#### B. Capacity Analysis

**Question**: How much capacity does each modality require?

**Experiment**:
- Measure rank of learned projector weights
- Analyze attention patterns in fusion adapter
- Compare effective dimensionality across modalities

### 3.3 Efficiency Analysis

#### A. Compute Overhead

| Component | FLOPs | % of Base |
|-----------|-------|-----------|
| Audio Encoder | ~1G | <1% |
| Audio Projector | ~4M | <0.01% |
| Fusion Adapter | ~100M | <0.1% |
| **Total Overhead** | ~1.1G | **<2%** |

#### B. Memory Overhead

| Component | Memory | % of Base |
|-----------|--------|-----------|
| Audio Encoder (frozen) | ~300MB | +4% |
| Trainable Params | ~8MB | +0.1% |
| Activations | ~100MB | +2% |
| **Total Overhead** | ~400MB | **<7%** |

---

## Phase 4: Robustness & Safety Validation (Weeks 11-12)

### 4.1 Robustness Testing

#### A. Distribution Shift

| Test | Description | Metric |
|------|-------------|--------|
| Domain shift | Train AudioCaps, test Clotho | CIDEr gap |
| Noise injection | Add noise to audio | Degradation curve |
| Missing modality | Audio unavailable at inference | Graceful degradation |

#### B. Adversarial Testing

| Attack | Target | Defense |
|--------|--------|---------|
| Audio adversarial | Fool audio encoder | Measure VL impact |
| Cross-modal | Audio attacks affecting VL | Gate=0 fallback |

### 4.2 Safety Validation

#### A. Catastrophic Forgetting Analysis

```
Forgetting Metric:
F = (VL_accuracy_before - VL_accuracy_after) / VL_accuracy_before

Targets:
- F < 0.5% for any single modality addition
- F < 2% for all modalities combined
```

#### B. Interference Analysis

| Modality Pair | Potential Interference | Test |
|---------------|----------------------|------|
| Audio + Video | Temporal confusion | AV sync tasks |
| Video + Depth | Spatial competition | Scene understanding |
| All modalities | Capacity saturation | Joint evaluation |

---

## Phase 5: Publication Preparation (Weeks 12-14)

### 5.1 Main Results Tables

#### Table 1: Audio Captioning (Primary Result)

| Method | CIDEr | METEOR | BLEU-4 | VL Ret. | Params |
|--------|-------|--------|--------|---------|--------|
| Full Fine-tune | X.X | X.XX | X.XX | 85% | 7B |
| LoRA-only | X.X | X.XX | X.XX | 95% | 4M |
| Linear Probe | X.X | X.XX | X.XX | 100% | 0.5M |
| **SAFE (Ours)** | **X.X** | **X.XX** | **X.XX** | **99.5%** | **2M** |

#### Table 2: Multi-Modal Scaling

| Modalities | Audio | Video | Depth | Tactile | VL Ret. |
|------------|-------|-------|-------|---------|---------|
| Base VL | - | - | - | - | 100% |
| +Audio | 80+ | - | - | - | 99.5% |
| +Video | 80+ | 45+ | - | - | 99.2% |
| +Depth | 80+ | 45+ | 0.85 | - | 99.0% |
| +Tactile | 80+ | 45+ | 0.85 | 0.80 | 98.8% |

#### Table 3: Ablation Summary

| Ablation | Best Config | Improvement |
|----------|-------------|-------------|
| Projector depth | 2 layers | +5% over 1 layer |
| LoRA rank | 8 | Best param/perf |
| Token count | 8 | Optimal |
| Retention strategy | Fisher | +2% VL retention |

### 5.2 Key Figures

1. **Architecture Diagram**: SAFE framework with frozen/trainable components
2. **Scaling Curve**: Performance vs. trainable parameters
3. **Modality Addition**: Sequential addition with retention tracking
4. **Ablation Heatmaps**: Projector depth × width, LoRA rank × alpha
5. **Attention Visualization**: Cross-modal attention patterns

### 5.3 Claims Checklist

| Claim | Evidence Required | Status |
|-------|-------------------|--------|
| "Adds modalities with <2M params" | Parameter count table | ⬜ |
| "Maintains 99%+ VL performance" | Retention experiments | ⬜ |
| "Generalizes across modalities" | 4+ modality results | ⬜ |
| "Linear scaling with modality count" | Scaling experiments | ⬜ |
| "Provably safe at gate=0" | Theorem + experiments | ⬜ |

---

## Experiment Tracking & Infrastructure

### Compute Requirements

| Phase | GPU-Hours | Hardware |
|-------|-----------|----------|
| Phase 1 (Audio) | ~500 | 4× A100 |
| Phase 2 (Multi-modal) | ~800 | 4× A100 |
| Phase 3 (Scaling) | ~400 | 8× A100 |
| Phase 4 (Robustness) | ~200 | 4× A100 |
| **Total** | **~1900** | - |

### Experiment Naming Convention

```
safe_{modality}_{encoder}_{projector}_{retention}_{seed}

Examples:
- safe_audio_clap_mlp2_fisher_42
- safe_video_clipvit_temporal_soft_123
- safe_multimodal_avd_joint_full_0
```

### Checkpoints & Logging

```yaml
logging:
  platform: wandb
  project: safe-incremental-learning

checkpoints:
  save_every: 1000 steps
  keep_best: 3
  metric: "val/cider"

metrics:
  primary: [cider, meteor, vl_retention]
  secondary: [bleu4, bertscore, gate_zero_acc]
```

---

## Risk Mitigation

| Risk | Mitigation | Contingency |
|------|------------|-------------|
| Audio perf plateaus | More data, larger projector | Focus on retention story |
| VL degradation > 1% | Stronger retention loss | Gate=0 fallback emphasis |
| Video underperforms | Temporal-aware architecture | Frame-based baseline |
| Compute constraints | Prioritize ablations | Reduce seed count |
| Negative results | Document failure modes | Pivot to analysis paper |

---

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-3 | Audio Consolidation | Ablation tables, scaled results |
| 4-5 | Video Expansion | Video baseline, transfer results |
| 5-6 | Depth Expansion | Depth results, cross-modal |
| 6-7 | Tactile Expansion | Full modality coverage |
| 7-8 | Multi-Modal Integration | Sequential addition results |
| 9-11 | Scaling & Theory | Scaling laws, proofs |
| 11-12 | Robustness | Safety validation |
| 12-14 | Publication | Paper draft, figures |

---

## Success Criteria

### Minimum Viable Publication
- [ ] Audio CIDEr > 70 with full data
- [ ] VL retention > 99%
- [ ] 2+ modalities demonstrated
- [ ] Complete ablation table
- [ ] Baselines comparison

### Strong Publication
- [ ] Audio CIDEr > 80
- [ ] 4 modalities demonstrated
- [ ] Scaling laws established
- [ ] Theoretical retention guarantee
- [ ] Comprehensive robustness

### Best Case
- [ ] State-of-the-art audio captioning
- [ ] General-purpose incremental learning framework
- [ ] Open-source release with pretrained weights
- [ ] Novel theoretical contributions

---

## Next Steps (Immediate)

1. **Today**: Run projector depth ablation (1, 2, 3, 4 layers)
2. **This Week**: Complete all projector ablations
3. **Next Week**: Scale up audio training with full data
4. **Week 3**: Begin video encoder integration

---

*Document Version: 1.0*
*Last Updated: January 2025*
*Authors: [Your Team]*
