# SAFE Research Plan (Jan–Jun 2026)

Owner: Robby Moseley
Last updated: 2026-01-06
Repo: `SAFE/`

## 0) Executive Summary

**Core thesis:** We propose an architecture for **incremental modality learning with guaranteed zero forgetting** of the base model's original capabilities. By keeping the base vision-language model completely frozen and introducing new modalities through isolated, additive adapter pathways, we achieve **architectural guarantees** against catastrophic forgetting—not just empirical mitigation.

**Key insight:** Unlike prior continual learning approaches that rely on regularization (EWC, distillation, replay) to *reduce* forgetting, our architecture makes forgetting *impossible by construction*: the base model's forward pass is unchanged when audio tokens are absent, and audio adapters only add residual information without modifying existing representations.

**Validation strategy:** We demonstrate this approach on **two modalities**:
1. **Audio** (primary): Audio captioning on AudioCaps/WavCaps
2. **Depth** (secondary): Depth-conditioned image understanding on NYUv2/SUN RGB-D

By showing the same architectural pattern works for both modalities with zero retention loss, we establish a **general methodology** for incremental modality expansion.

This document defines:

- The architectural guarantee and how we formalize/prove it.
- A two-modality experimental validation program.
- Ablations showing the tradeoffs within our constrained design space.
- A phased timeline and deliverables (paired with `phased checklist.md`).

### Working directory layout (this folder)

- `safe/research plan/research plan.md` — the “paper plan” and experimental design.
- `safe/research plan/phased checklist.md` — the living execution checklist + run index.
- `safe/research plan/papers/` — PDFs and paper notes (use consistent filenames).
- `safe/research plan/notes/` — running notes, meeting notes, decision logs.
- `safe/research plan/runs/` — run manifests, exported metrics, ablation tables, plots.

## 1) Current Status (Starting Point)

### Baseline snapshot

- Task: Audio captioning (AudioCaps + WavCaps mixture).
- Current performance (example): CIDEr ~49, ROUGE-L ~0.49, METEOR ~0.18–0.19, BLEU ~0.22.
- Data regime: “limited data” mixture (e.g., ~50% AudioCaps + ~70% WavCaps; confirm exact sampling logic).
- Model: Frozen VL backbone (e.g., LLaVA 13B), frozen audio encoder (CLAP), trainable audio projector + LoRA cross-attention fusion (multi-layer).

### What this implies

1) Audio path works (signal can flow and be used by the frozen LLM).  
2) Next gains are likely from: better audio tokenization (temporal detail), metric-aligned training, and carefully designed continual learning/retention experiments.

## 2) Target Contribution (Top-Conference, H1 2026)

### Primary Claim: Architectural Guarantee Against Forgetting

**Zero-forgetting modality expansion by construction:** Our architecture guarantees that adding a new modality adapter cannot degrade the base model's performance on its original tasks. This is achieved through:

1. **Complete backbone freezing:** All base model parameters remain frozen—no gradients flow through them.
2. **Additive-only fusion:** New modality information is injected via residual cross-attention that adds to (never replaces) existing hidden states.
3. **Gated bypass:** When the new modality is absent, the adapter pathway outputs zero, making the forward pass mathematically identical to the original model.

**Formal statement:** Let $f_\theta$ be the frozen base model and $g_\phi$ be the modality adapter. For any input $x$ without the new modality:
$$f_{\theta+\phi}(x) = f_\theta(x)$$

This is not an empirical observation—it is a *structural property* of the architecture.

### Secondary Claims (Supporting)

- **Generality:** The same architectural pattern applies to multiple modalities (audio, depth) with identical retention guarantees.
- **Efficiency:** Competitive task performance with <3% additional parameters over the frozen base.
- **Composability:** Multiple modality adapters can be added incrementally without interference (each adapter is independently gated).

### What We Explicitly Claim vs. Don't Claim

**We claim:**
- Zero forgetting *by design* (not "reduced" or "mitigated")
- Demonstrated on two distinct modalities (audio + depth)
- Competitive performance on modality-specific tasks

**We do NOT claim:**
- SOTA on any single benchmark (our focus is the learning paradigm, not benchmark racing)
- That this approach is optimal for joint multi-modal learning (it's for *incremental* addition)
- That adapters trained separately will exhibit emergent cross-modal reasoning

## 3) Research Questions & Hypotheses

### RQ1 — Architectural Guarantee Verification

Does our architecture truly achieve zero forgetting, and what are the conditions?

- **H1:** With completely frozen backbone + gated residual adapters, retention metrics are *exactly* equal to the frozen baseline (within numerical precision) when the new modality is absent.
- **H2:** Even when the new modality is present, retention on original tasks remains within noise (±0.1%) because adapters only add information, never modify existing representations.

### RQ2 — Generalization Across Modalities

Does the same architectural pattern work for different modalities?

- **H3:** Audio and depth adapters, trained with identical architecture but different encoders/projectors, both achieve zero forgetting.
- **H4:** Adapters can be composed (audio + depth simultaneously) without interference—each operates independently.

### RQ3 — Performance Within the Constrained Design Space

Given the frozen backbone constraint, what design choices maximize new-modality task performance?

- **H5:** Fusion layer depth affects performance: mid-to-late layers capture better semantic alignment.
- **H6:** More modality tokens (temporal resolution) matters more than adapter capacity (LoRA rank) for performance.
- **H7:** Gate warmup enables stable training without compromising the zero-forgetting guarantee.

### RQ4 — Comparison to Regularization-Based Approaches

How does our architectural guarantee compare to traditional continual learning methods?

- **H8:** Regularization methods (EWC, distillation) reduce but do not eliminate forgetting; our method eliminates it by construction.
- **H9:** Our approach requires no replay buffer, hyperparameter tuning for retention, or forgetting-performance tradeoffs.

## 4) Experimental Design Overview (What We Will Run)

We structure experiments in four blocks:

### Block A: Architectural Guarantee Verification
Prove that our architecture achieves *exact* zero forgetting.
- Run frozen baseline on retention suite (COCO captions, VQAv2)
- Run model with audio adapter on same retention suite (audio input absent)
- Demonstrate outputs are bitwise identical (or within fp16 precision)

### Block B: Audio Modality (Primary)
Demonstrate competitive performance on audio captioning.
- Train audio adapter on AudioCaps + WavCaps
- Evaluate on AudioCaps test (CIDEr, BLEU, METEOR, ROUGE-L)
- Ablate: fusion layers, token count, projector size, gate warmup

### Block C: Depth Modality (Secondary Validation)
Demonstrate the same pattern works for a different modality.
- Train depth adapter on NYUv2 or SUN RGB-D
- Task: depth-conditioned captioning or depth QA
- Show identical zero-forgetting property

### Block D: Comparison to Regularization Baselines
Show our architectural approach is superior to empirical mitigation.
- Implement: EWC, distillation, replay buffer baselines
- Show they *reduce* but don't *eliminate* forgetting
- Our approach: zero forgetting, no hyperparameter sensitivity

Each experiment must define:
- Model config (fusion layers, rank, tokens, projector)
- Data regime (datasets, splits)
- Training regime (LRs, warmup, epochs)
- Metrics: task performance + retention suite

## 5) Datasets, Splits, and Data Mixtures

### Audio Modality Datasets

**Primary:**
- **AudioCaps:** Standard train/val/test splits (~46K/495/975 samples). Ensure no YouTube ID leakage.
- **WavCaps:** Large-scale audio captions (~400K). Use as training augmentation.

**Evaluation:**
- AudioCaps test set (primary)
- Clotho (out-of-domain robustness, optional)

### Depth Modality Datasets

**Primary:**
- **NYUv2:** ~1.4K RGB-D images with dense depth + semantic labels. Standard train/test split.
- **SUN RGB-D:** ~10K RGB-D images. Larger scale alternative.

**Tasks:**
- Depth-conditioned image captioning: "Describe this scene given the depth information"
- Depth QA: "What is closest to the camera?" / "How far is the chair?"

**Note:** Depth is a good second modality because:
1. It's structurally different from audio (spatial vs. temporal)
2. It pairs naturally with vision (same scene, different sensor)
3. Existing RGB-D datasets are clean and well-defined
4. The encoder (e.g., DPT, MiDaS) outputs spatial features that test different fusion dynamics

### Vision-Language Retention Suite

Goal: Prove zero forgetting on original VL capabilities.

**Required (minimum):**
- **COCO Captions val:** Image captioning (CIDEr, BLEU, METEOR)
- **VQAv2 val:** Visual QA (accuracy)

**Optional (strengthens claim):**
- Text-only perplexity on a held-out corpus
- LLaVA-Bench or similar instruction-following benchmark

**Key requirement:** Retention metrics must be *identical* to frozen baseline when new modality input is absent. This is the architectural guarantee.

## 6) Model Variants To Study (Ablations)

This is the core of the paper: controlled experiments isolating the impact of each design choice.

### 6.1 Fusion depth (layer indices)

Study at fixed total capacity:

- Single-layer fusion at {early, mid, late}.
- 3-layer fusion distributed by depth (e.g., 30%/60%/90%).
- Optional: dense fusion (every N layers) if compute allows.

Metrics:

- Audio captioning metrics (CIDEr etc.).
- Retention suite metrics.
- Audio conditionality probes (see Section 8).

### 6.2 Injection point

- `pre_ffn` vs `post_layer` (already supported in config).

Hypothesis: `pre_ffn` often increases usefulness but can be more invasive; gating/warmup may restore retention.

### 6.3 Number of fusion sites

- K ∈ {1, 2, 3, 6} fusion insertion points with matched total LoRA parameter count (or explicitly report parameters).

### 6.4 LoRA rank + target modules

At fixed fusion topology:

- Rank r ∈ {4, 8, 16, 32}.
- Target modules: {Q,V} vs {Q,K,V,O} (O = output dense).
- LoRA alpha and dropout: small grid after rank sweep.

Track:

- Training stability (loss curves, gradient norms).
- Attention probe summary (mean/max attention to audio tokens).

### 6.5 Projector capacity (bottleneck)

Projector bottleneck dim ∈ {256, 512, 1024, 2048, none}.

Also vary:

- 1-layer vs 2-layer MLP.
- Output norm + learned scale initialization (verify stable scale learning).

### 6.6 Audio token count (“bandwidth”)

For pooled encoders (CLAP pooled embedding):

- Tokens T ∈ {4, 8, 16, 32}. Interpret as “capacity tokens.”

For sequence encoders (Whisper / frame-level / intermediate CLAP tokens):

- Tokens T ∈ {8, 16, 32, 64} where T corresponds to temporal resolution (downsample strategy must be specified).

### 6.7 Gating strategy

- Global scalar gate (and warmup schedule).
- Tokenwise gate (`use_tokenwise_gate`).
- Silence-aware bypass (already present): verify it fully bypasses audio for silent rows.

### 6.8 What to keep frozen (controlled unfreezing)

Minimal variants to justify “safe”:

- Fully frozen backbone (default).
- LoRA on a small subset of LLM blocks (e.g., last N layers) + retention penalty.
- Unfreeze only layer norms (if supported) as a low-risk adaptation baseline.

## 7) Architectural Guarantee: Formal Treatment

### 7.1 The Zero-Forgetting Property

**Definition:** A modality adapter has the *zero-forgetting property* if, for any input $x$ without the new modality:
$$\text{output}(x; \theta_{\text{base}}, \phi_{\text{adapter}}) = \text{output}(x; \theta_{\text{base}})$$

**How we achieve this:**

1. **Frozen backbone:** $\theta_{\text{base}}$ receives no gradients during adapter training.

2. **Gated residual injection:** The adapter adds a residual to hidden states:
   $$h' = h + \alpha \cdot g(h, m)$$
   where $m$ is the new modality input, $g$ is the cross-attention fusion, and $\alpha$ is a learnable gate.

3. **Silence bypass:** When $m = \emptyset$ (no modality input), the gate outputs zero:
   $$g(h, \emptyset) = 0 \implies h' = h$$

**Implication:** The model's behavior on original tasks is *unchanged by construction*—not "approximately preserved" or "regularized toward preservation."

### 7.2 Comparison to Regularization Approaches

| Method | Forgetting | Hyperparameters | Guarantees |
|--------|------------|-----------------|------------|
| EWC | Reduced | Fisher weight, λ | None (empirical) |
| Distillation | Reduced | Temperature, loss weight | None (empirical) |
| Replay | Reduced | Buffer size, sampling | None (empirical) |
| **Ours** | **Zero** | **None for retention** | **Architectural** |

We will demonstrate this empirically by:
1. Running regularization baselines and showing non-zero forgetting
2. Running our approach and showing exact-zero forgetting

### 7.3 Two-Modality Validation Protocol

To establish *generality*, we apply the same architecture to two modalities:

**Step 1: Audio adapter**
- Train on AudioCaps/WavCaps
- Verify zero forgetting on COCO/VQAv2

**Step 2: Depth adapter**
- Train on NYUv2 (completely separate from audio)
- Verify zero forgetting on COCO/VQAv2

**Step 3: Composition (optional but powerful)**
- Load both adapters simultaneously
- Verify: audio-only inputs work, depth-only inputs work, VL-only inputs match baseline exactly
- This demonstrates *modular composability*

### 7.4 Metrics

**For the architectural guarantee:**
- Retention Δ = 0.0% (exact match to baseline, within numerical precision)

**For task performance:**
- Standard metrics per task (CIDEr for captioning, accuracy for QA)
- We do NOT need Pareto curves because there is no forgetting-performance tradeoff

## 8) Diagnostic Probes (To Strengthen the Story)

These reduce “it works” ambiguity and help reviewers trust mechanism.

### 8.1 Audio conditionality checks

- “Audio swap” test: same prompt, swap audio across batch, measure change in caption metrics or CLAP similarity.
- “Silence ablation”: replace audio with silence/zeros; model output should degrade appropriately (not remain unchanged).
- “Counterfactual audio”: artificially mix two audios; analyze attention to tokens and output changes.

### 8.2 Attention probes

Use existing attention summaries:

- Track attention entropy and mean attention mass to audio tokens across training.
- Compare across fusion layers (early vs late) and injection points.

### 8.3 Calibration checks

- Track audio token norm distribution vs LLM hidden state norms.
- Ensure audio projector scale does not drift to extremes; if it does, add constraints or scheduled clipping.

## 9) Training & Optimization Plan

### 9.1 Baseline training recipe (frozen backbone)

- Use current best configuration as the “anchor baseline”.
- Run at least 3 seeds for final-table results; 1 seed for broad sweeps.

### 9.2 SCST (metric-aligned finetuning)

We already have SCST support configured in `configs/training/full.yaml`.

Plan:

- Run SCST only after XE plateau.
- Evaluate retention before and after SCST.
- Report: CIDEr gain vs any retention loss or increased hallucination/generic captions.

### 9.3 Contrastive auxiliary objective (optional)

After stable captioning:

- Add a small-weight audio-text contrastive term to improve grounding (guard against “generic caption” collapse).
- Carefully validate it does not harm retention.

## 10) Baselines to Include (Minimum Publishable Set)

Audio task baselines:

- Frozen backbone + no audio (should fail audio captioning; sanity).
- Projector-only (no cross-attention) vs projector+fusion.
- Single-layer fusion vs multi-layer fusion.

Continual learning baselines:

- No retention constraint.
- Distillation-only retention.
- Fisher/EWC-style retention.
- Replay buffer baseline (small).

Optional external baselines (only if feasible and fair):

- A known audio-captioning model not using LLaVA (report separately).
- A generic multimodal LLM with audio if available (ensure matched evaluation).

## 11) Reporting, Tables, and Figures (What Reviewers Expect)

### Must-Have Figures

1. **Architecture diagram:** Show the frozen backbone, modality encoder, projector, gated cross-attention, and residual injection. Highlight the "bypass when absent" property.

2. **Zero-forgetting proof table:**
   | Model | COCO CIDEr | VQAv2 Acc | Δ from Baseline |
   |-------|------------|-----------|-----------------|
   | Frozen baseline | X | Y | — |
   | + Audio adapter (no audio input) | X | Y | 0.0% |
   | + Depth adapter (no depth input) | X | Y | 0.0% |

3. **Comparison to regularization methods:**
   | Method | Audio CIDEr | COCO Δ | VQAv2 Δ |
   |--------|-------------|--------|---------|
   | EWC | ... | -1.2% | -0.8% |
   | Distillation | ... | -0.7% | -0.5% |
   | **Ours** | ... | **0.0%** | **0.0%** |

4. **Two-modality validation:** Show audio and depth both work with the same pattern.

5. **Ablation table:** Fusion layers, token count, projector size—focused on *task performance*, not retention (retention is always 0).

### Nice-to-Have

- Training dynamics (loss curves, attention patterns)
- Qualitative examples with audio playback
- Composition demo (audio + depth + image simultaneously)

## 12) Timeline (Jan 6 → Jun 30)

### January (Weeks 1–4): Foundation + Architectural Guarantee Proof

**Goal:** Prove the zero-forgetting property with audio.

- [ ] Lock evaluation harness for audio captioning + retention suite
- [ ] Run frozen baseline retention metrics (COCO, VQAv2)
- [ ] Train audio adapter, verify *exact* retention match
- [ ] Document the architectural guarantee with code pointers

### February (Weeks 5–8): Audio Ablations + Depth Setup

**Goal:** Optimize audio performance; prepare second modality.

- [ ] Run audio ablations (fusion layers, tokens, projector)
- [ ] Finalize best audio configuration
- [ ] Set up depth encoder (DPT or MiDaS)
- [ ] Prepare NYUv2 dataloader and depth captioning task

### March (Weeks 9–12): Depth Validation + Composition

**Goal:** Prove generality with second modality.

- [ ] Train depth adapter with same architecture pattern
- [ ] Verify zero forgetting on retention suite
- [ ] Test audio + depth composition (both adapters loaded)
- [ ] Document: same pattern, same guarantee, different modality

### April (Weeks 13–16): Baselines + Full Results

**Goal:** Complete the comparison story.

- [ ] Implement regularization baselines (EWC, distillation)
- [ ] Show they have non-zero forgetting (vs our zero)
- [ ] Run multi-seed experiments for final numbers
- [ ] Generate all main tables and figures

### May (Weeks 17–20): Paper Draft

**Goal:** Complete, reviewable draft.

- [ ] Write: intro, method, experiments, results, discussion
- [ ] Create figures: architecture diagram, zero-forgetting proof, ablations
- [ ] Internal review and red-teaming
- [ ] Address feedback with targeted experiments

### June (Weeks 21–26): Polish + Submission

**Goal:** Camera-ready submission.

- [ ] Final figures and tables
- [ ] Supplementary material (extended ablations, code)
- [ ] Submission to target venue

## 13) Reproducibility & Experiment Hygiene

Non-negotiables:

- Fixed seeds and logged configs for every reported number.
- Exact decoding settings saved alongside checkpoints.
- Dataset versioning (hash of metadata files + split lists).
- Leakage checks recorded (especially YouTube ID overlap).

Suggested experiment naming:

`YYYYMMDD__task__cfg__layers-12-24-36__inj-preffn__tok-8__rank-16__projbn-1024__mix-ac0.5-wc0.7__seed-42`

## 14) Experiment Log & Findings

### 2026-01-08: LoRA Bug Fix & Layer Ablation Study

**Bug Discovery:** Found that LoRA weights in fusion adapters were being frozen incorrectly. In `fusion_adapter.py`, the code `for param in base_model.parameters(): param.requires_grad = bool(train_base_cross_attention)` was freezing ALL parameters including LoRA weights when `train_base_cross_attention=False`.

**Fix Applied:** Updated to check parameter names - LoRA weights (`lora_` in name) and `residual_scale` are always trainable, base layers controlled by flag.

**Surprising Finding:** Despite frozen LoRA, we achieved CIDEr ~49 and METEOR ~0.19 using only:
- Audio projector (~42M params)
- Token gates (~0.03M params)
- Residual scales (~0.00M params)

This serves as an **ablation baseline** showing what's achievable without cross-attention adaptation.

**Current Trainable Params (with LoRA fix):**
- audio_projector: 42.52M
- fusion_adapter/lora: 1.97M (now unfrozen!)
- fusion_adapter/token_gate: 0.03M
- Total trainable: ~44.5M

### Current Experiment: Layer Ablation (Running)

Testing how fusion layer depth impacts accuracy. Using layers at 8, 16, 24, 32 (evenly distributed through 40-layer LLM):

| Run | Layers | Hypothesis |
|-----|--------|------------|
| 1L | [8] | Early-only fusion - tests if semantic alignment needs depth |
| 2L | [8, 16] | Early+mid - progressive refinement |
| 3L | [8, 16, 24] | Distributed fusion |
| 4L | [8, 16, 24, 32] | Full depth coverage |

**Metrics tracked:**
- `val/cider`, `val/meteor` - validation performance (per epoch)
- `train_acc/cider`, `train_acc/meteor` - training accuracy on fixed 300 samples (every 500 steps)

**Training config:**
- Single GPU (DDP had issues, deferred)
- WavCaps 50% mix
- 20 epochs
- LoRA rank 8

### Multi-GPU Status

Attempted DDP support with 3 GPUs but encountered performance issues (slower than single GPU). Likely causes:
- `find_unused_parameters=True` overhead (fixed in code but not yet tested)
- Communication overhead with large frozen backbone

Deferred to single-GPU runs for current experiments. Multi-GPU optimization is future work.

### Infrastructure Improvements

1. **`--fusion-layer-indices` CLI arg** - Override fusion layers without changing config files
2. **Training accuracy logging** - `train_acc/cider`, `train_acc/meteor` logged to W&B every 500 steps on fixed 300 training samples
3. **LoRA parameter detection** - Startup now warns if LoRA params are missing from trainable set

## 15) Open Decisions (Resolve Early)

1) Primary backbone for paper (LLaVA 13B vs smaller for sweeps).
2) Retention suite scope (COCO/VQA minimum; add more if feasible).
3) Audio tokenization approach for "bandwidth" study (pooled vs temporal).
4) Final continual learning protocol (A vs B vs both).

## 16) Links Into Repo (to keep aligned with implementation)

- Model presets: `configs/model_configs.py`
- Training configs: `configs/training/full.yaml`
- Fusion: `safe/models/fusion_adapter.py`
- Projectors: `safe/models/projectors.py`
- Trainer/retention losses: `safe/training/stage_a.py`, `safe/training/losses.py`
- Phase 1 launcher: `scripts/train_phase1.sh`
