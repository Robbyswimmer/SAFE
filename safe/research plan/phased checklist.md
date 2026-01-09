# SAFE Phased Checklist (Jan–Jun 2026)

Last updated: 2026-01-08
Companion doc: `safe/research plan/research plan.md`

## Core Claim (Refined 2026-01-08)

***Composable, independently-trainable modality adapters with guaranteed zero interference.***

**Why this is novel:**
1. No prior work trains modality adapters independently then composes them
2. Practical deployment: VLM → add audio → add depth → no retraining
3. The architectural guarantee enables composition (without it, adapters interfere)

**The key result we need:** Train audio adapter. Train depth adapter separately. Load both. Show both work + VL unchanged.

---

## Minimum Experiments for Top-Venue Submission

| # | Experiment | Purpose | Effort | Status |
|---|------------|---------|--------|--------|
| 1 | Full audio training | Competitive CIDEr | 1-2 days | 🟡 Exploratory done |
| 2 | Point cloud adapter (ScanNet) | Prove generality (3D modality) | 1 week | ⬜ Not started |
| 3 | Composition test | Audio + point cloud together | 1 day | ⬜ Blocked on #2 |
| 4 | Retention suite (COCO/VQAv2) | Zero forgetting proof | 2-3 days | 🟡 Partial |
| 5 | One baseline (EWC) | Show alternative fails | 3-4 days | ⬜ Not started |

**Why Point Cloud over Depth?**
- Undeniably different: 3D sparse vs 2D dense vs 1D temporal
- Stronger generality claim (depth is "just another 2D visual modality")
- Clean story: Vision (2D) + Audio (temporal) + Point Cloud (3D)

**Timeline:** NeurIPS 2026 (deadline ~May 2026)
- Jan-Feb: Complete experiments 1-5
- Mar: Write full draft
- Apr: Internal review, fill gaps
- May: Submit

---

## Phase 0 — Architecture Lock-In ✅ COMPLETE

**Reference:** `safe/research plan/notes/architecture_lock_in.md`

- [x] Frozen VL backbone (LLaVA 1.5 13B)
- [x] Audio encoder (CLAP, frozen)
- [x] Audio projector (~34M params)
- [x] Cross-attention LoRA fusion (~2M params)
- [x] Three conditions formalized (frozen + additive + gated bypass)
- [x] Zero-forgetting verified on COCO val (100% exact match)

---

## Phase 1 — Audio Modality (Jan Weeks 2-3) 🟡 IN PROGRESS

**Goal:** Competitive audio captioning performance.

### 1.1 Exploratory Ablations (Quick, 10% data)
- [x] Layer ablation bug fixed (2026-01-08)
- [ ] Layer sweep: 1L, 2L, 3L, 4L at layers 8, 16, 24, 32 — RUNNING
- [ ] Identify best layer configuration

### 1.2 Full-Scale Training
- [ ] Train best config on full data (AudioCaps + WavCaps + Clotho + MACS)
- [ ] Target: CIDEr > 60 on AudioCaps test
- [ ] Run `scripts/benchmark_eval.py` for paper metrics

### 1.3 Audio Retention Verification
- [x] Verify COCO outputs identical when audio=None ✅
- [ ] Verify VQAv2 outputs identical when audio=None

**Done when:** CIDEr > 60, retention verified.

---

## Phase 2 — Point Cloud Modality (Jan Week 3 – Feb Week 1) ⬜ HIGH PRIORITY

**Goal:** Second modality to prove generality. START ASAP—this is highest risk.

**Why Point Cloud over Depth:**
- Undeniably different: 3D sparse vs 2D dense (vision) vs 1D temporal (audio)
- Stronger generality claim than depth (which is "just another 2D visual modality")
- Clean story: Vision (2D) + Audio (temporal) + Point Cloud (3D)

### 2.1 Point Cloud Encoder Integration
- [ ] Select encoder: PointNet++ or Point-BERT (recommend Point-BERT for better features)
- [ ] Implement `PointCloudEncoder` class (similar to `CLAPEncoder`)
- [ ] Implement `PointCloudProjector` (encoder_dim → 5120×T, same architecture as audio)
- [ ] Add point cloud fusion adapter (same pattern as audio)

### 2.2 Dataset Preparation
- [ ] Download ScanNet dataset (3D indoor scenes) or ModelNet (3D objects)
- [ ] Create `ScanNetDataset` class with point cloud + caption/description
- [ ] Define task: "Describe this 3D scene" or 3D object classification

### 2.3 Point Cloud Training
- [ ] Train point cloud adapter on ScanNet (independent of audio—don't load audio weights)
- [ ] Verify training stability
- [ ] Evaluate on 3D understanding task

### 2.4 Point Cloud Retention Verification
- [ ] Verify COCO outputs identical when point_cloud=None
- [ ] Verify VQAv2 outputs identical when point_cloud=None

**Done when:** Point cloud adapter works, retention verified.

**Risk mitigation:** If point cloud doesn't work, can fall back to depth. Start immediately.

---

## Phase 3 — Composition Test (Feb Week 2) ⬜ THE KEY RESULT

**Goal:** Demonstrate independently trained adapters compose without interference.

### 3.1 Load Both Adapters
- [ ] Load audio adapter weights (trained in Phase 1)
- [ ] Load point cloud adapter weights (trained in Phase 2)
- [ ] Verify both load without conflict

### 3.2 Composition Verification
- [ ] Test audio-only input → audio captioning works
- [ ] Test point cloud-only input → 3D task works
- [ ] Test image-only input → matches frozen baseline EXACTLY
- [ ] Test audio + image → audio captioning works
- [ ] Test point cloud + image → 3D task works
- [ ] Optional: Test audio + point cloud + image together

### 3.3 Quantitative Results
- [ ] AudioCaps metrics with both adapters loaded (should match audio-only)
- [ ] ScanNet metrics with both adapters loaded (should match point cloud-only)
- [ ] COCO/VQAv2 metrics (should match frozen baseline)

**Done when:** Table showing all combinations work, no interference.

---

## Phase 4 — Retention Suite (Feb Week 2-3) 🟡 PARTIAL

**Goal:** Formal retention metrics on standard VL benchmarks.

### 4.1 Baseline Establishment
- [ ] Run frozen LLaVA on COCO Captions val → save outputs + metrics
- [ ] Run frozen LLaVA on VQAv2 val → save outputs + metrics

### 4.2 Retention Verification Table
| Model | COCO CIDEr | VQAv2 Acc | Δ from Baseline |
|-------|------------|-----------|-----------------|
| Frozen baseline | X | Y | — |
| + Audio (no audio input) | ? | ? | Must be 0.0% |
| + Point Cloud (no PC input) | ? | ? | Must be 0.0% |
| + Audio + Point Cloud (neither input) | ? | ? | Must be 0.0% |

- [ ] Fill in all cells
- [ ] Verify Δ = 0.0% for all rows

**Done when:** Table complete, all Δ = 0.0%.

---

## Phase 5 — Baseline Comparison: Efficiency + Composition (Feb Week 3-4) ⬜ P1

**Goal:** Show we achieve *better* composition with *fewer* parameters.

**The story:** "250x fewer parameters than fine-tuning, yet perfect composition where they fail."

### 5.1 Sequential Fine-tuning Baseline (Start here—simplest)
- [ ] Fine-tune LLM on audio (unfreeze all, ~7B params)
- [ ] Fine-tune same LLM on point cloud
- [ ] Measure: audio degradation, VL degradation
- [ ] Record total trainable params: ~7B (100%)

### 5.2 EWC Baseline (More rigorous)
- [ ] Implement EWC loss (Fisher Information penalty)
- [ ] Train audio with EWC (unfreeze LLM + EWC penalty)
- [ ] Compute Fisher matrix, train point cloud with EWC
- [ ] Load both → measure interference
- [ ] Record total trainable params: ~7B (100%)

### 5.3 LoRA Fine-tuning Baseline (Parameter-efficient but still fails)
- [ ] Standard LoRA on LLM self-attention (not isolated adapters)
- [ ] Train audio with LoRA, then point cloud with LoRA
- [ ] Show: still has interference despite fewer params
- [ ] Record total trainable params: ~70-140M (1-2%)

### 5.4 Full Comparison Table
| Method | Trainable Params | Audio | 3D | COCO Δ | VQAv2 Δ | Composable? |
|--------|------------------|-------|-----|--------|---------|-------------|
| Sequential FT | 100% (~7B) | ↓ degraded | ✓ | -5-10% | -5-10% | ❌ |
| EWC | 100% (~7B) | ~ok | ✓ | -2-5% | -2-5% | ❌ |
| LoRA FT | 1-2% (~100M) | ? | ? | -1-3% | -1-3% | ❌ |
| **Ours** | **0.4%** (~35M) | ✓ | ✓ | **0.0%** | **0.0%** | ✅ |

### 5.5 Efficiency Summary
| Metric | Full FT | EWC | LoRA FT | Ours |
|--------|---------|-----|---------|------|
| Trainable params | 7B | 7B | 100M | **35M** |
| Replay buffer | Maybe | Maybe | Maybe | **No** |
| Fisher computation | No | Yes | No | **No** |
| Retention λ tuning | No | Yes | No | **No** |
| Zero interference | ❌ | ❌ | ❌ | **✅** |

**Done when:** Table filled, efficiency + composition advantage clear.

---

## Phase 6 — Paper Writing (Mar-Apr)

### 6.1 Required Figures
- [ ] Figure 1: Architecture diagram (frozen backbone + gated adapters)
- [ ] Figure 2: Composition demonstration (audio + depth loaded together)
- [ ] Figure 3: Retention proof (Δ = 0 for all configurations)

### 6.2 Required Tables
- [ ] Table 1: Zero-interference proof (retention metrics)
- [ ] Table 2: Audio captioning results (AudioCaps test)
- [ ] Table 3: Point cloud task results (ScanNet)
- [ ] Table 4: Composition results (both adapters loaded)
- [ ] Table 5: Comparison to baselines

### 6.3 Paper Sections
- [ ] Abstract (~150 words)
- [ ] Introduction (composition story, practical value)
- [ ] Related Work (see `notes/related_work_analysis.md`)
- [ ] Method (three conditions, architecture)
- [ ] Experiments (audio, depth, composition, baselines)
- [ ] Analysis & Discussion
- [ ] Conclusion

**Done when:** Complete draft ready for review.

---

## Phase 7 — Submission (May)

- [ ] Internal review and revisions
- [ ] Camera-ready figures
- [ ] Supplementary material
- [ ] Code release preparation
- [ ] Submit to NeurIPS 2026

---

## Running Log

| Date | Description | Key Result | Notes |
|------|-------------|------------|-------|
| 2026-01-05 | Audio training | CIDEr ~49 | Initial baseline on limited data |
| 2026-01-06 | Zero-forgetting verification | **100% exact match** | Core guarantee proven |
| 2026-01-08 | Layer ablation bug fix | CLI now works | Nested config priority issue |
| 2026-01-08 | Extended datasets | +6K Clotho, +3.9K MACS | Download scripts ready |
| 2026-01-08 | Benchmark eval script | Full metric suite | `scripts/benchmark_eval.py` |
| 2026-01-08 | **Contribution refined** | Composition story | See research plan |
| 2026-01-08 | Layer ablation restart | 1L/2L/3L/4L running | 3K samples, bug fixed |

---

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Point cloud doesn't work | Generality claim fails | Fall back to depth (simpler), start ASAP |
| Audio CIDEr too low | "Why care about guarantee?" | Full data training, may need SCST |
| Composition has interference | Core claim fails | Should work by design—verify early |
| PointNet integration complex | Delays schedule | Use pretrained Point-BERT, freeze encoder |
| Not enough time | Miss deadline | Prioritize P0 experiments, skip nice-to-haves |

---

## Key Code Pointers

- **Architecture Lock-In:** `safe/research plan/notes/architecture_lock_in.md`
- **Related Work Analysis:** `safe/research plan/notes/related_work_analysis.md`
- **Paper Draft:** `safe/research plan/publication/paper_draft.md`
- **Paper Outline:** `safe/research plan/publication/paper_outline.md`
- Fusion adapter: `safe/models/fusion_adapter.py`
- Audio projector: `safe/models/projectors.py`
- SAFE model: `safe/models/safe_model.py`
- Training: `train_safe.py`
- Benchmark eval: `scripts/benchmark_eval.py`
- Dataset downloads: `scripts/download_clotho.py`, `scripts/download_macs.py`

---

## Recent Bugs & Fixes

| Date | Bug | Fix | File |
|------|-----|-----|------|
| 2026-01-08 | `--fusion-layer-indices` CLI not applied | Update nested config | `train_safe.py:2991-3000` |
| 2026-01-07 | LoRA params frozen incorrectly | Check param names | `fusion_adapter.py` |
| 2026-01-06 | DDP slower than single GPU | Deferred | - |
