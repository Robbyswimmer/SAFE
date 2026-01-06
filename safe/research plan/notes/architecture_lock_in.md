# SAFE Architecture Lock-In Document

**Version:** 1.0
**Date:** 2026-01-06
**Status:** Phase 0 — Formal Specification

---

## 1. Executive Summary

SAFE (Safe Audio Fusion Extension) adds audio understanding to a frozen vision-language model via **gated residual adapters**. The core guarantee: when audio input is absent, model outputs are **identical** to the frozen baseline—by construction, not regularization.

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SAFE Model                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────────────┐ │
│  │   Audio     │    │  Audio          │    │     Gated Residual           │ │
│  │   Input     │───▶│  Encoder        │───▶│     Fusion Adapter           │ │
│  │  (waveform) │    │  (CLAP, frozen) │    │  (LoRA cross-attention)      │ │
│  └─────────────┘    └────────┬────────┘    └──────────────┬───────────────┘ │
│                              │                            │                  │
│                              ▼                            │                  │
│                     ┌─────────────────┐                   │                  │
│                     │  Audio          │                   │                  │
│                     │  Projector      │───────────────────┘                  │
│                     │  (trainable)    │                                      │
│                     └─────────────────┘                                      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Frozen Vision-Language Backbone                       ││
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐  ││
│  │  │   Image     │    │   Vision    │    │     LLaVA 13B               │  ││
│  │  │   Input     │───▶│   Encoder   │───▶│     (40 decoder layers)     │  ││
│  │  │             │    │   (CLIP)    │    │     FROZEN                  │  ││
│  │  └─────────────┘    └─────────────┘    └─────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│                                    │                                         │
│                                    ▼                                         │
│                           ┌─────────────────┐                                │
│                           │  Text Output    │                                │
│                           │  (caption/VQA)  │                                │
│                           └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Specifications

### 3.1 Base Vision-Language Model (FROZEN)

| Parameter | Value |
|-----------|-------|
| Model | LLaVA 1.5 13B (`llava-hf/llava-1.5-13b-hf`) |
| LLM Backend | Vicuna 13B (LLaMA architecture) |
| Hidden Size | **d = 5120** |
| Attention Heads | **40** |
| Decoder Layers | **40** |
| Vision Encoder | CLIP ViT-L/14 (1024-dim) |
| Vocabulary Size | 32,000 |
| Training Status | **Completely frozen** |

**Critical:** All 13B parameters are frozen. No gradients flow into the base model.

---

### 3.2 Audio Encoder (FROZEN)

| Parameter | Value |
|-----------|-------|
| Model | LAION CLAP (`laion/larger_clap_music_and_speech`) |
| Output Dimension | **d_audio = 512** |
| Sample Rate | 48,000 Hz |
| Input Duration | 10 seconds (480,000 samples) |
| Training Status | **Frozen** |

**CLAP Embedding:** Single 512-dim vector per audio clip capturing semantic audio content.

```python
# Audio encoding (frozen)
audio_embedding = CLAP.get_audio_embedding(waveform)  # → (B, 512)
```

---

### 3.3 Audio Projector (TRAINABLE)

**Purpose:** Map 512-dim CLAP embedding to T audio tokens in LLM space (5120-dim).

| Parameter | Value |
|-----------|-------|
| Input Dimension | 512 (CLAP output) |
| Output Dimension | 5120 × T (LLM hidden size × num tokens) |
| Bottleneck Dimension | **1024** |
| Number of Audio Tokens | **T = 8** (current), 16 (full config) |
| Activation | GELU |
| Dropout | 0.1 |

**Architecture:**

```
Input: (B, 512)
    │
    ▼
┌─────────────────────────────┐
│  LayerNorm(512)             │  ← Input normalization
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Linear(512 → 1024)         │  ← Bottleneck projection
│  GELU                       │
│  Dropout(0.1)               │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Linear(1024 → 5120 × T)    │  ← Expand to LLM space
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Reshape to (B, T, 5120)    │
│  LayerNorm(5120)            │  ← Per-token normalization
│  × output_scale             │  ← Learnable scale (init=1.0)
└─────────────────────────────┘

Output: (B, T, 5120)  ← T audio tokens per sample
```

**Parameter Count:**
- Input norm: 2 × 512 = 1,024
- First linear: 512 × 1024 + 1024 = 525,312
- Second linear: 1024 × (5120 × T) = 1024 × 40,960 = 41,943,040 (for T=8)
- Output norm: 2 × 5120 = 10,240
- Output scale: 1
- **Total (T=8): ~42.5M parameters**

---

### 3.4 Cross-Attention Fusion Adapter (TRAINABLE)

**Purpose:** Inject audio information into LLM hidden states via cross-attention.

#### 3.4.1 Cross-Attention Mechanism

```
Query: LLM hidden states h ∈ ℝ^(B × L × 5120)
Key/Value: Audio tokens a ∈ ℝ^(B × T × 5120)

Attention(Q, K, V) = softmax(QK^T / √d_head) × V
```

| Parameter | Value |
|-----------|-------|
| Hidden Size | 5120 |
| Attention Heads | 40 |
| Head Dimension | 5120 / 40 = **128** |
| Q/K/V Projections | Linear(5120 → 5120) |
| Output Projection | Linear(5120 → 5120) |

**Cross-Attention Math:**

Given hidden states $h \in \mathbb{R}^{B \times L \times d}$ and audio tokens $a \in \mathbb{R}^{B \times T \times d}$:

$$Q = h W_Q, \quad K = a W_K, \quad V = a W_V$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

$$\delta = W_O \cdot \text{Attention}(Q, K, V)$$

Where:
- $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$ (5120 × 5120)
- $d_k = 128$ (head dimension)
- $\delta$ is the residual update

#### 3.4.2 LoRA Adaptation

**LoRA (Low-Rank Adaptation)** reduces trainable parameters while preserving expressivity.

| Parameter | Value |
|-----------|-------|
| LoRA Rank | **r = 16** |
| LoRA Alpha | **α = 16** |
| Target Modules | query, key, value, output_dense |
| LoRA Dropout | 0.0 |

**LoRA Math:**

For a frozen weight matrix $W_0 \in \mathbb{R}^{d \times d}$:

$$W = W_0 + \frac{\alpha}{r} \cdot BA$$

Where:
- $B \in \mathbb{R}^{d \times r}$ (down-projection)
- $A \in \mathbb{R}^{r \times d}$ (up-projection)
- $\frac{\alpha}{r} = 1.0$ scales the adaptation

**LoRA Parameter Count (per projection):**
- $B$: 5120 × 16 = 81,920
- $A$: 16 × 5120 = 81,920
- Per projection: 163,840
- All 4 projections: 655,360 per fusion layer

#### 3.4.3 Gated Residual Injection

**This is the zero-forgetting guarantee mechanism.**

```python
# Residual update with learnable scale
residual_scale = nn.Parameter(torch.tensor(0.1))  # Init conservative
residual_scale = clamp(residual_scale, 0.0, 5.0)  # Bounded for stability

delta = cross_attention(h, audio_tokens)
delta = layer_norm(delta)
delta = residual_scale * delta

# Final output (THE KEY EQUATION)
output = h + gate * delta
```

**Gate Behavior:**
- `gate = 1.0`: Full audio fusion
- `gate = 0.0`: Pure passthrough (output = h exactly)
- Audio absent → attention mask zeros all tokens → effective gate = 0

---

### 3.5 Multi-Layer Fusion

Audio is injected at **3 decoder layers** for hierarchical fusion:

| Fusion Point | Layer Index | Depth (%) | Purpose |
|--------------|-------------|-----------|---------|
| Early | 12 | 30% | Low-level feature alignment |
| Mid | 24 | 60% | Semantic integration |
| Late | 36 | 90% | High-level reasoning |

**Injection Point:** Pre-FFN (before feed-forward network within each layer)

```
Decoder Layer Structure:
    │
    ▼
┌─────────────────────┐
│  Self-Attention     │
│  + Add & Norm       │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  FUSION INJECTION   │  ← Audio cross-attention here
│  (pre-FFN)          │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  FFN                │
│  + Add & Norm       │
└─────────────────────┘
    │
    ▼
```

---

## 4. Zero-Forgetting Mathematical Guarantee

### 4.1 Formal Definition

Let:
- $f_\theta$: Frozen VL model with parameters $\theta$
- $f_{\theta,\phi}$: SAFE model with additional trainable parameters $\phi$
- $x$: Input (text + optional image)
- $a$: Audio input (may be null)

**Zero-Forgetting Property:**

$$\forall x: \quad f_{\theta,\phi}(x, a=\emptyset) = f_\theta(x)$$

When audio is absent ($a = \emptyset$), SAFE produces **identical** outputs to the frozen baseline.

### 4.2 Proof by Construction

**Claim:** The gated residual architecture guarantees zero forgetting.

**Proof:**

1. **Audio Encoding:** When $a = \emptyset$:
   - Audio encoder receives no input
   - Audio tokens $\mathbf{a} = \mathbf{0}$ (zero tensor)
   - Audio attention mask = 0 (all positions masked)

2. **Cross-Attention Bypass:** With all audio tokens masked:
   - Softmax over empty set → undefined, handled by masking
   - Attention output $\delta = \mathbf{0}$
   - Even if $\delta \neq 0$, the gate mechanism ensures bypass

3. **Gated Residual:**
   ```python
   if audio_absent:
       gate = 0.0  # Forced to zero when audio mask is all-zero

   output = h + gate * delta
          = h + 0 * delta
          = h  # Pure passthrough
   ```

4. **Layer-wise Propagation:** Since each fusion layer is a pure passthrough when audio is absent, the full forward pass reduces to:
   ```
   f_{θ,φ}(x, a=∅) = f_θ(x)
   ```

**QED.** The output is bitwise identical (within floating-point precision).

### 4.3 Implementation Details

```python
# From safe_model.py forward()
no_audio = (audio_tokens is None or audio_tokens.numel() == 0)

if no_audio:
    # TRUE VL PASSTHROUGH: Use base embeddings + pixel_values
    base_embeddings_layer = self.base_vl.llm.get_input_embeddings()
    inputs_embeds = base_embeddings_layer(input_ids)

    # Call LLaVA directly (no fusion hooks)
    outputs = self.base_vl.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels
    )
    return outputs  # Identical to frozen baseline
```

---

## 5. Parameter Summary

### 5.1 Frozen Components (No Training)

| Component | Parameters |
|-----------|------------|
| LLaVA 13B (LLM) | ~13B |
| CLIP ViT-L (Vision) | ~304M |
| CLAP (Audio Encoder) | ~89M |
| **Total Frozen** | **~13.4B** |

### 5.2 Trainable Components

| Component | Parameters | Notes |
|-----------|------------|-------|
| Audio Projector | ~42.5M | Bottleneck MLP |
| Fusion Adapters (×3 layers) | ~2.0M | LoRA only |
| Audio Token Embeddings | ~10K | 2 special tokens |
| **Total Trainable** | **~44.5M** | **0.33%** of total |

---

## 6. Training Configuration

### 6.1 Optimization

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 2e-4 |
| Weight Decay | 0.01 |
| LR Schedule | Cosine with warmup |
| Warmup Steps | 500 |
| Batch Size (effective) | 128 |
| Gradient Accumulation | 16 (batch_size=1) |
| Mixed Precision | FP16 |
| Max Epochs | 10 |

### 6.2 Data

| Dataset | Split | Samples | Task |
|---------|-------|---------|------|
| AudioCaps | Train | ~46K | Audio captioning |
| AudioCaps | Val | ~2.2K | Evaluation |
| AudioCaps | Test | ~957 | Final metrics |

### 6.3 Evaluation Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| CIDEr | >45 | Primary metric |
| BLEU-4 | >10 | N-gram precision |
| METEOR | >15 | Semantic similarity |
| ROUGE-L | >30 | Longest common subsequence |

---

## 7. Input/Output Specifications

### 7.1 Input Format

```python
# Audio input
waveform: torch.Tensor  # Shape: (B, samples) or (B, 1, samples)
                        # Sample rate: 48kHz
                        # Duration: 10 seconds (480,000 samples)

# Text input (for captioning)
prompt: str  # "USER: Question: Describe the audio. ASSISTANT:"

# Image input (optional, for multimodal)
pixel_values: torch.Tensor  # Shape: (B, 3, 336, 336)
```

### 7.2 Output Format

```python
# Generated caption
output: str  # "A dog is barking while birds chirp in the background."

# Training output
{
    "logits": torch.Tensor,  # (B, L, vocab_size)
    "loss": torch.Tensor,    # Scalar cross-entropy loss
    "hidden_states": None    # Not returned by default
}
```

---

## 8. Token Flow Analysis

### 8.1 Sequence Structure

```
[BOS] [prompt tokens...] [ASSISTANT:] [generated tokens...] [EOS]
       ↑                       ↑              ↑
       L_prompt               L_response     L_total

Audio tokens are NOT in sequence - they are fused via cross-attention.
```

### 8.2 Audio Token Details

| Property | Value |
|----------|-------|
| Number of Tokens | T = 8 (or 16) |
| Token Dimension | 5120 |
| Position in Sequence | None (cross-attention only) |
| Attention Pattern | All LLM tokens attend to all audio tokens |

---

## 9. Numerical Stability Measures

### 9.1 Input Sanitization

```python
# Clean NaN/Inf values
audio_features = torch.nan_to_num(audio_features, nan=0.0, posinf=0.0, neginf=0.0)
```

### 9.2 Attention Stability

```python
# Score clamping
attention_scores = torch.clamp(attention_scores, min=-50.0, max=50.0)

# Softmax in FP32
attention_probs = F.softmax(attention_scores.float(), dim=-1)
```

### 9.3 Residual Scaling

```python
# Bounded learnable scale
residual_scale = torch.clamp(self.residual_scale, 0.0, 5.0)
```

---

## 10. Key Design Decisions

### 10.1 Why Gated Residual?

- **Guaranteed passthrough:** Mathematical proof of zero forgetting
- **Soft start:** Initial scale = 0.1 prevents destabilization
- **Learnable:** Model can increase/decrease audio influence

### 10.2 Why LoRA on Cross-Attention?

- **Parameter efficiency:** ~2M vs ~100M for full cross-attention
- **Stable training:** Low-rank updates are more stable
- **Composability:** Multiple adapters can be loaded/merged

### 10.3 Why Multi-Layer Fusion?

- **Hierarchical integration:** Different layers capture different abstractions
- **Gradient flow:** Multiple injection points improve training signal
- **Flexibility:** Can ablate individual layers

### 10.4 Why Pre-FFN Injection?

- **After normalization:** Hidden states are normalized post-self-attention
- **Before transformation:** FFN can integrate fused information
- **Cleaner separation:** Self-attention (text) vs cross-attention (audio)

---

## 11. File References

| Component | File | Key Lines |
|-----------|------|-----------|
| SAFEModel | `safe/models/safe_model.py` | Full model class |
| Audio Projector | `safe/models/projectors.py:6-158` | AudioProjector class |
| Fusion Adapter | `safe/models/fusion_adapter.py:245-427` | LoRAFusionAdapter |
| Cross-Attention | `safe/models/fusion_adapter.py:11-242` | CrossAttentionBlock |
| Layer Hooks | `safe/models/layer_hooks.py` | Hook registration |
| Audio Encoder | `safe/models/audio_encoders.py:11-251` | CLAPAudioEncoder |
| Training Loop | `train_safe.py` | Main training script |
| Config | `configs/model_configs.py:181-248` | PHASE1_CONFIG |

---

## 12. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-06 | Initial architecture lock-in |

---

## 13. Appendix: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $B$ | Batch size |
| $L$ | Sequence length |
| $T$ | Number of audio tokens |
| $d$ | Hidden dimension (5120) |
| $d_k$ | Attention head dimension (128) |
| $r$ | LoRA rank (16) |
| $\alpha$ | LoRA scaling factor (16) |
| $h$ | Hidden states |
| $a$ | Audio tokens |
| $\delta$ | Cross-attention residual |
| $g$ | Gate value |
| $\theta$ | Frozen model parameters |
| $\phi$ | Trainable adapter parameters |
