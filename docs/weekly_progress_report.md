# Weekly Progress Report: SAFE Architecture Development

**Date:** Week of January 20-27, 2025
**Project:** Scalable Audio Fusion for LLMs (SAFE)

---

## Executive Summary

This week focused on three main objectives: (1) validating the SAFE architecture on audio understanding tasks, (2) demonstrating modality generalization by adding point cloud support, and (3) investigating fusion mechanisms for multi-modal composition. Key results include 91% audio classification accuracy on AVE, 65 CIDEr on audio captioning, and 80% point cloud classification accuracy—achieved with minimal architectural changes when switching modalities.

---

## 1. Audio Classification (Complete)

### Task
Audio-visual event classification on the AVE dataset (28 event categories).

### Approach
- Frozen CLAP encoder (512-dim audio embeddings)
- Lightweight projector mapping to LLM token space
- Single-layer fusion at early decoder layer
- Classification framed as text generation: "What sound is this?" → "dog barking"

### Results
| Metric | Score |
|--------|-------|
| Test Accuracy | **91%** |

### Notes
- Fixed earlier OOM issues during contrastive pre-training by optimizing batch handling
- Single fusion layer sufficient for classification (layer 1 of 40)
- Confirms the projector successfully aligns audio embeddings to LLM representation space

---

## 2. Audio Captioning (In Progress)

### Task
Generate natural language descriptions of audio content (AudioCaps benchmark).

### Approach
- Same frozen CLAP encoder as classification
- Multi-layer fusion at layers [12, 24, 36] for richer semantic integration
- Trained with cross-entropy loss on caption tokens
- Evaluating with standard captioning metrics (CIDEr, BLEU, METEOR)

### Current Results
| Metric | Current | SOTA | Gap |
|--------|---------|------|-----|
| CIDEr | **65** | 84 | 19 pts |

### Ongoing Work
- Running extended training with larger data mixture (AudioCaps + WavCaps + AudioSetCaps)
- Experimenting with number of audio tokens (8 → 16)
- Investigating SCST (Self-Critical Sequence Training) for CIDEr optimization

### Analysis
The 65 CIDEr result was achieved with limited training data. The gap to SOTA (84) is expected to narrow with:
1. More diverse training data
2. Longer training schedule
3. Potential SCST fine-tuning phase

---

## 3. Point Cloud Modality (New)

### Motivation
To validate that SAFE's architecture is truly **modality-agnostic**, I added point cloud as a second modality. Point clouds were chosen because they are structurally distinct from audio:
- Audio: 1D temporal signal → 2D spectrogram → sequence
- Point clouds: Unordered 3D coordinate sets with geometric structure

If the same projector and fusion architecture works for both, it demonstrates the design generalizes beyond audio-specific inductive biases.

### Architecture Changes
| Component | Audio | Point Cloud | Change Required |
|-----------|-------|-------------|-----------------|
| Encoder | CLAP (frozen) | PointBERT (frozen) | Swap encoder |
| Embed dim | 512 | 768 | Update projector input dim |
| Projector | AudioProjector | Same class | None (reused) |
| Fusion | MultiLayerFusionAdapter | Same class | None (reused) |
| LLM | LLaVA-1.5-13B (frozen) | Same | None |

**Key finding:** The projector and fusion modules required zero modification—only the encoder was swapped.

### Classification Results (ModelNet40)

ModelNet40: 40-class 3D object classification (airplane, chair, table, etc.)

| Metric | Score | PointBERT Baseline |
|--------|-------|-------------------|
| Val Accuracy | **80%** | 93% |

### Analysis
- 80% accuracy confirms the fusion architecture successfully transfers point cloud information to the LLM
- Gap to PointBERT baseline (93%) is expected since we're using a generative LLM rather than a discriminative classifier
- The LLM must generate the class name token-by-token rather than direct softmax classification

### Next Steps
- Set up Cap3D dataset (~660K captioned 3D objects from Objaverse)
- Train point cloud captioning model
- Compare captioning performance across modalities

---

## 4. Fusion Architecture Investigation

### Research Question
How does the choice of fusion injection point affect modality integration and composition?

### Two Approaches Under Evaluation

#### A. Pre-FFN Additive Fusion (Current Default)
```
Self-Attention → Add&Norm → [FUSION] → FFN → Add&Norm
```
- Modality information injected as residual delta before feed-forward network
- Simple additive combination: H' = H + g · Δ(H, M)
- Gate g controls modality strength

#### B. KV Augmentation (Pre-Attention)
```
[FUSION via K,V augmentation] → Self-Attention → Add&Norm → FFN → Add&Norm
```
- Modality tokens become additional key-value memory for attention
- LLM queries explicitly attend to modality information
- Dual-branch design: text attention (frozen) + modality attention (learned)

### Preliminary Observations

| Aspect | Pre-FFN Additive | KV Augmentation |
|--------|------------------|-----------------|
| Text preservation | May drift with large deltas | Exact (frozen branch) |
| Gradient signal | Can be suppressed by frozen LLM | Direct via attention weights |
| Compositionality | Additive mixing | Selective attention |
| Implementation | Forward hooks on MLP | Module replacement |
| Training stability | Sensitive to delta scale | More stable (zero-init ΔQ) |

### Hypothesis
KV augmentation may offer advantages for multi-modal composition because:
1. The LLM cannot trivially ignore attention-based injection
2. Multiple modalities can contribute independent K/V pairs
3. Attention weights provide interpretable modality usage patterns

### Planned Experiments
- Compare pre-FFN vs KV augmentation on audio captioning (matched hyperparameters)
- Analyze attention patterns to understand which layers use modality information
- Test simultaneous audio + point cloud input (true multi-modal)

---

## 5. Summary Table

| Task | Modality | Status | Key Result |
|------|----------|--------|------------|
| Classification | Audio | Complete | 91% (AVE) |
| Captioning | Audio | In Progress | 65 CIDEr (target: 84) |
| Classification | Point Cloud | Complete | 80% (ModelNet40) |
| Captioning | Point Cloud | Setting Up | - |
| Fusion Analysis | Both | In Progress | Comparing pre-FFN vs KV-aug |

---

## 6. Next Week Goals

1. **Audio Captioning:** Complete extended training run, report updated CIDEr
2. **Point Cloud Captioning:** Train on Cap3D, establish baseline metrics
3. **Fusion Ablation:** Quantitative comparison of fusion strategies
4. **Multi-Modal:** Initial experiments with audio + point cloud simultaneously

---

## 7. Technical Artifacts

### Code Structure
```
safe/
├── models/
│   ├── safe_model.py              # Audio SAFE model
│   ├── safe_pointcloud_model.py   # Point cloud SAFE model
│   ├── audio_encoders.py          # CLAP encoder
│   ├── pointcloud_encoders.py     # PointBERT encoder
│   ├── projectors.py              # Modality-agnostic projector
│   ├── fusion_adapter.py          # Multi-layer fusion
│   └── kv_augmentation.py         # KV augmentation implementation
├── data/
│   ├── datasets.py                # Audio datasets
│   └── pointcloud_datasets.py     # Point cloud datasets
```

### Key Design Decisions
- **Frozen encoders:** Pre-trained representations (CLAP, PointBERT) are kept frozen
- **Frozen LLM:** Base LLaVA model is not fine-tuned
- **Trainable components:** Only projector + fusion adapters (~1% of total parameters)
- **Parameter efficiency:** Full model ~13B params, trainable ~100M params

---

## Appendix: Detailed Architecture

### SAFE Information Flow
```
Input Modality (audio/pointcloud)
        │
        ▼
┌───────────────────┐
│  Frozen Encoder   │  (CLAP-512d / PointBERT-768d)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│    Projector      │  (Trainable, produces N tokens)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Fusion Adapter   │  (Injects at selected LLM layers)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Frozen LLM      │  (LLaVA-1.5-13B)
└─────────┬─────────┘
          │
          ▼
    Generated Text
```
