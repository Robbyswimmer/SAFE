# SAFE Paper Outline - NeurIPS Style

**Working Title:** Zero-Forgetting Modality Expansion via Architectural Guarantees

**Target:** NeurIPS 2026 (or ICML/ICLR)

---

## 1. Abstract (~150 words)

### Requirements:
- [ ] One sentence: Problem (adding modalities causes forgetting)
- [ ] One sentence: Our key insight (architectural guarantee vs empirical mitigation)
- [ ] One sentence: Method summary (frozen backbone + gated residual adapters)
- [ ] One sentence: Main result (zero forgetting by construction, demonstrated on 2 modalities)
- [ ] One sentence: Significance (first architectural guarantee for continual multimodal learning)

### Questions to answer:
- What is our single most impressive quantitative result?
- What makes this fundamentally different from prior work?

---

## 2. Introduction (~1 page)

### Requirements:
- [ ] Opening hook: The forgetting problem in multimodal learning
- [ ] Current approaches and their limitations (regularization reduces but doesn't eliminate)
- [ ] Our key insight: Make forgetting impossible by design, not unlikely by training
- [ ] Contribution bullets (3-4 crisp claims)
- [ ] Paper roadmap

### Questions to answer:
- What real-world scenario motivates this? (e.g., deploying VLMs that need to add audio later)
- Why haven't others done this? What's the perceived tradeoff we overcome?
- What's the "aha" moment for the reader?

### Key figure:
- [ ] Figure 1: Architecture diagram showing frozen backbone + gated bypass

---

## 3. Related Work (~0.75 page)

### Subsections needed:
1. **Continual Learning** - EWC, distillation, replay, PackNet, etc.
2. **Multimodal Foundation Models** - LLaVA, BLIP-2, Flamingo, etc.
3. **Modality Adapters** - LoRA, adapters, prefix tuning
4. **Audio-Language Models** - SALMONN, Qwen-Audio, etc.

### Requirements:
- [ ] Position our work clearly against each category
- [ ] Identify the gap: No prior work offers *guarantees* against forgetting
- [ ] Be fair to prior work while highlighting our unique contribution

### Questions to answer:
- Which continual learning papers are most relevant to cite?
- Are there any architectural guarantee papers in other domains we should reference?
- What's the closest competitor and how do we differentiate?

---

## 4. Method (~2 pages)

### 4.1 Problem Formulation
- [ ] Define the incremental modality learning setting
- [ ] Define forgetting formally: performance drop on original tasks
- [ ] State what we want: zero forgetting guarantee

### 4.2 Architectural Guarantee
- [ ] Formal definition of zero-forgetting property
- [ ] Three conditions that achieve it:
  1. Frozen backbone (no gradient flow)
  2. Additive-only fusion (residual injection)
  3. Gated bypass (zero output when modality absent)
- [ ] Mathematical statement: f(x; θ+φ) = f(x; θ) when modality absent
- [ ] Proof sketch or argument for why this holds

### 4.3 SAFE Architecture
- [ ] Modality encoder (CLAP for audio)
- [ ] Projector (maps encoder → LLM hidden dim)
- [ ] Gated cross-attention fusion with LoRA
- [ ] Multi-layer injection strategy
- [ ] Training objective (standard captioning loss)

### 4.4 Two-Modality Validation Protocol
- [ ] Why two modalities matter for generality claim
- [ ] Audio adapter setup
- [ ] Depth adapter setup (same architecture pattern)

### Questions to answer:
- How do we make the "guarantee" rigorous without being pedantic?
- What's the minimal math needed to be convincing?
- Do we need pseudocode or is the architecture figure sufficient?

### Key figures:
- [ ] Architecture diagram (detailed)
- [ ] Gating mechanism visualization

---

## 5. Experiments (~2.5 pages)

### 5.1 Experimental Setup
- [ ] Base model: LLaVA 13B (or which variant?)
- [ ] Audio encoder: CLAP
- [ ] Depth encoder: DPT/MiDaS
- [ ] Datasets: AudioCaps, WavCaps, NYUv2, COCO, VQAv2
- [ ] Training details: LRs, epochs, batch size, hardware

### 5.2 Zero-Forgetting Verification (Block A)
**This is the core claim - must be bulletproof**

- [ ] Table: Retention metrics (COCO CIDEr, VQAv2 acc) for:
  - Frozen baseline
  - + Audio adapter (no audio input) → must be identical
  - + Depth adapter (no depth input) → must be identical
- [ ] Statistical significance / variance across seeds
- [ ] Discuss numerical precision (fp16 vs fp32)

### Questions to answer:
- How do we show "identical" convincingly? Bit-wise? Within epsilon?
- Do we need multiple random seeds?
- What if there's tiny numerical drift - how do we address?

### 5.3 Audio Captioning Performance (Block B)
- [ ] Main results table: CIDEr, BLEU-4, METEOR, ROUGE-L on AudioCaps test
- [ ] Comparison to audio captioning baselines (if fair comparison exists)
- [ ] Qualitative examples

### Questions to answer:
- What's SOTA on AudioCaps? Can we compare fairly?
- If we're not SOTA, how do we frame this? (We trade peak performance for guarantees)

### 5.4 Depth Modality Validation (Block C)
- [ ] Same architecture applied to depth
- [ ] Zero-forgetting verification (same as 5.2)
- [ ] Task performance on depth-conditioned captioning/QA

### Questions to answer:
- What's a good depth task that's clearly different from audio?
- Do we have depth results yet?

### 5.5 Comparison to Regularization Methods (Block D)
- [ ] Table comparing:
  - EWC: reduced forgetting, non-zero
  - Distillation: reduced forgetting, non-zero
  - Replay: reduced forgetting, non-zero
  - **Ours: zero forgetting**
- [ ] Show the fundamental difference (ours is 0.0%, others are >0%)

### Questions to answer:
- Which baselines are essential vs nice-to-have?
- How much effort to implement EWC/distillation properly?

### 5.6 Ablation Studies
- [ ] Fusion layer depth (1L, 2L, 3L, 4L)
- [ ] LoRA rank (if it matters)
- [ ] Audio token count (if it matters)
- [ ] Gate warmup (if it matters)

### Questions to answer:
- Which ablations tell the most interesting story?
- What if ablations show "nothing matters much"? (Could be a finding: robustness)

### ⚠️ IMPORTANT: Ablation Data Requirements

**For paper ablations, we need FULL DATA training, not exploratory sweeps.**

| Type | Data | Epochs | Purpose |
|------|------|--------|---------|
| Exploratory sweeps | 10% (~5K samples) | 3-5 | Quick ranking to guide decisions |
| **Paper ablations** | **100% (AudioCaps + WavCaps)** | **20** | **Final numbers for publication** |

**Process:**
1. Run exploratory sweeps at 10% data to find promising configs (current runs)
2. Select top configs + interesting comparisons
3. Re-run winners at full scale for paper tables
4. Reviewers expect converged models, not early stopping

**Current status:** Layer ablation running at 10% data (exploratory). Will need full-scale runs for Table 4.

### Key tables:
- [ ] Table 1: Zero-forgetting proof (retention metrics)
- [ ] Table 2: Audio captioning results
- [ ] Table 3: Comparison to regularization methods
- [ ] Table 4: Ablations (MUST BE FULL DATA)

---

## 6. Analysis & Discussion (~1 page)

### 6.1 Why Does This Work?
- [ ] Intuition beyond the math
- [ ] The LLM as a "universal interface" that adapters learn to speak to

### 6.2 Emergent Properties Investigation

#### Robustness to Modality Degradation (investigate for paper)
Our gated architecture should enable graceful degradation when modality quality drops.

**Experiment:**
- Systematically degrade audio quality (add noise, reduce bandwidth, silence segments)
- Measure caption quality degradation curve
- Compare to joint-trained models that may fail catastrophically

**Expected finding:** Smooth degradation, not cliff-edge failure

**Why this matters:** Real-world deployments face noisy/missing modalities

#### Attention Specialization Across Fusion Layers (investigate for paper)
Different fusion layers may specialize for different aspects of modality integration.

**Experiment:**
- Use existing attention probes at each fusion layer
- Analyze: Do early layers handle feature alignment? Late layers semantic integration?
- Visualize attention patterns across layers for different input types

**Potential figure:** Attention heatmaps showing layer specialization

### 6.3 Limitations
- [ ] Not SOTA on individual benchmarks (trade performance for guarantees)
- [ ] Requires modular adapter design
- [ ] No cross-modal reasoning between adapters (they're independent)
- [ ] Performance ceiling from frozen backbone

### Questions to answer:
- What are the honest limitations?
- What would break our approach?
- Is there a performance ceiling from keeping backbone frozen?

---

## 7. Conclusion & Future Work (~0.5 page)

### Requirements:
- [ ] Restate main contribution
- [ ] Broader impact: Safer deployment of multimodal AI

### Future Work:

#### Implicit Cross-Modal Grounding (mention as future work)
Even though adapters don't directly interact, they both influence the same LLM hidden states.
- Could the LLM learn to resolve ambiguity using both signals?
- E.g., video shows two people, audio helps identify speaker
- Hard to test rigorously, but interesting direction

#### Modality Contribution Attribution (mention as future work)
Our architecture enables clean measurement of per-modality contribution to answers:

```
h' = h_base + α_audio * audio_residual + α_depth * depth_residual
```

**Potential experiments:**
1. **Modality knockout:** Remove one modality → measure accuracy drop per question type
2. **Contribution curves:** Scale α from 0→1, plot answer confidence
3. **Layer-wise attribution:** Which fusion layer contributes most per question type?
4. **Conflict resolution:** Give contradictory modality signals, see which "wins"

| Question Type | Expected Dominant Modality |
|--------------|---------------------------|
| "What color is the car?" | Vision |
| "What sound is playing?" | Audio |
| "How far is the table?" | Depth |
| "Is the dog barking at the cat?" | Vision + Audio |

**Why this matters:** Unlike integrated gradients or SHAP on entangled models, we have true modularity—attribution without approximation.

**Note:** This could be its own follow-up paper: "Interpretable Multimodal Attribution via Modular Adapters"

#### Other Future Directions
- [ ] More modalities (video, tactile, etc.)
- [ ] Scaling to larger backbones
- [ ] Few-shot modality adaptation

---

## 8. Supplementary Material

### Required:
- [ ] Full training hyperparameters
- [ ] Extended ablation tables
- [ ] More qualitative examples
- [ ] Code release plan

### Optional:
- [ ] Attention visualizations
- [ ] Audio samples (if submission allows)
- [ ] Proofs/derivations

---

## Critical Experiments Still Needed

### Must-have for submission:
1. [ ] Zero-forgetting verification with exact numbers
2. [ ] Audio captioning final results (full training)
3. [ ] Depth modality results (proves generality)
4. [ ] At least one regularization baseline comparison

### Nice-to-have:
5. [ ] Multiple seeds for key results
6. [ ] Composition demo (audio + depth simultaneously)
7. [ ] SCST finetuning results
8. [ ] Extensive ablations

---

## Open Questions for Paper

1. **Framing:** "Zero forgetting" vs "Guaranteed retention" vs "Architectural preservation"?
2. **Scope:** How much do we emphasize audio vs the general principle?
3. **Baselines:** Which regularization methods are essential to implement?
4. **Depth modality:** What task exactly? Captioning? QA? Both?
5. **Negative results:** What if depth doesn't work as well - how do we handle?
6. **Compute:** Do we have enough GPU time for all experiments?

---

## Timeline to Submission

| Phase | Weeks | Deliverable |
|-------|-------|-------------|
| Experiments | 1-8 | All tables filled in |
| First draft | 9-12 | Complete paper draft |
| Internal review | 13-14 | Feedback and revisions |
| Camera ready | 15-16 | Final polish |

---

*Last updated: 2026-01-08*
