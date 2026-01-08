# SAFE Paper Draft - Methodology Section

**Working Title:** Zero-Forgetting Modality Expansion via Architectural Guarantees

*Last updated: 2026-01-08*

---

## 4. Method

### 4.1 Problem Formulation

We consider the problem of **incremental modality expansion**: given a pretrained vision-language model $f_\theta: \mathcal{X}_v \times \mathcal{X}_t \rightarrow \mathcal{Y}$ that maps visual and textual inputs to outputs, we seek to extend it to process an additional modality (e.g., audio) $\mathcal{X}_a$ while preserving its original capabilities.

**Definition 1 (Catastrophic Forgetting).** Let $\mathcal{D}_\text{VL}$ be a distribution over vision-language tasks. A model exhibits catastrophic forgetting if, after training on modality expansion data $\mathcal{D}_\text{audio}$, its performance on $\mathcal{D}_\text{VL}$ degrades:

$$\mathbb{E}_{(x,y) \sim \mathcal{D}_\text{VL}}[\mathcal{L}(f_{\theta'}(x), y)] > \mathbb{E}_{(x,y) \sim \mathcal{D}_\text{VL}}[\mathcal{L}(f_\theta(x), y)] + \epsilon$$

where $\theta' = \text{Train}(\theta, \mathcal{D}_\text{audio})$ and $\epsilon > 0$ is a non-negligible threshold.

**Definition 2 (Zero-Forgetting Guarantee).** A modality expansion method provides a zero-forgetting guarantee if, for any input $(x_v, x_t) \in \mathcal{X}_v \times \mathcal{X}_t$ without the new modality present:

$$f_{\theta + \phi}(x_v, x_t, \emptyset) = f_\theta(x_v, x_t)$$

where $\phi$ represents the newly added parameters and $\emptyset$ denotes the absence of the new modality. This equality must hold **exactly by construction**, not approximately through regularization.

### 4.2 Architectural Guarantee: Three Conditions

We identify three sufficient conditions that, when satisfied jointly, provide a zero-forgetting guarantee:

**Condition 1: Frozen Backbone.** The base model parameters $\theta$ receive zero gradient updates during modality expansion training:
$$\nabla_\theta \mathcal{L}_\text{audio} = 0$$

This is achieved by setting `requires_grad=False` for all parameters in the pretrained vision-language backbone, including the LLM decoder, vision encoder, and vision projector.

**Condition 2: Additive-Only Fusion.** New modality information is integrated through purely additive residual connections. At each fusion layer $\ell$, the hidden state transformation is:

$$h^{(\ell)} \leftarrow h^{(\ell)} + \Delta h^{(\ell)}_\text{audio}$$

where $\Delta h^{(\ell)}_\text{audio}$ is the fusion residual computed from the audio representation. Critically, the original forward path $h^{(\ell)}$ is never modified multiplicatively or replaced.

**Condition 3: Gated Bypass with Zero Default.** The fusion residual is gated such that it produces exactly zero contribution when the new modality is absent:

$$\Delta h^{(\ell)}_\text{audio} = g \cdot \text{CrossAttn}(h^{(\ell)}, m_\text{audio})$$

where $g \in [0, 1]$ is a gate value and $m_\text{audio}$ represents the audio token embeddings. When no audio is present, the entire fusion path is bypassed, yielding $\Delta h^{(\ell)}_\text{audio} = 0$.

**Theorem 1 (Zero-Forgetting).** *If Conditions 1-3 are satisfied, then for any VL input without audio:*
$$f_{\theta + \phi}(x_v, x_t, \emptyset) = f_\theta(x_v, x_t)$$

*Proof sketch.* With audio absent, the gated bypass (Condition 3) ensures all fusion residuals are exactly zero. By Condition 2, the hidden states follow the original computation path: $h^{(\ell)} = h^{(\ell)} + 0 = h^{(\ell)}$. Since the backbone is frozen (Condition 1), the parameters producing $h^{(\ell)}$ are identical to the original model. By induction over layers, the final output is unchanged. $\square$

### 4.3 SAFE Architecture

We instantiate these principles in **SAFE** (Simple Adaptive Fusion Extension), illustrated in Figure 1. The architecture consists of four trainable components:

#### 4.3.1 Audio Encoder

We employ CLAP (Contrastive Language-Audio Pretraining) as the audio encoder, specifically the `larger_clap_music_and_speech` variant. CLAP provides semantically rich audio representations aligned with natural language through contrastive pretraining on large-scale audio-text pairs.

- **Input**: Raw waveform at 48 kHz, up to 10 seconds
- **Output**: 512-dimensional audio embedding $e_\text{audio} \in \mathbb{R}^{512}$
- **Training**: Frozen during SAFE training (pretrained representations)

The choice of a frozen, pretrained audio encoder is deliberate: it provides stable, high-quality audio representations from initialization, allowing the trainable projection and fusion layers to focus on cross-modal alignment rather than audio understanding.

#### 4.3.2 Audio Projector

The audio projector maps the CLAP embedding to a sequence of tokens in the LLM's hidden space. We use a parameter-efficient bottleneck MLP:

$$\text{AudioProjector}(e) = \text{LayerNorm}(\text{MLP}(\text{LayerNorm}(e))) \cdot \alpha$$

**Architecture Details:**
- Input LayerNorm: Stabilizes CLAP embeddings
- Layer 1: Linear($d_\text{audio}$, $d_\text{bottleneck}$) + GELU + Dropout(0.1)
- Layer 2: Linear($d_\text{bottleneck}$, $d_\text{LLM} \times k$)
- Output LayerNorm: Per-token normalization
- Learnable scale $\alpha$: Initialized to 1.0, trained to match LLM hidden state magnitudes

**Hyperparameters:**
- $d_\text{audio} = 512$ (CLAP embedding dimension)
- $d_\text{bottleneck} = 1024$ (80% parameter reduction vs. direct projection)
- $d_\text{LLM} = 4096$ (LLaVA 1.5 13B hidden dimension)
- $k = 8$ audio tokens per sample

The bottleneck architecture reduces parameters from $512 \times 4096 \times 8 \approx 16.8$M to $512 \times 1024 + 1024 \times 4096 \times 8 \approx 33.8$M total projector parameters. The output is reshaped to $(B, k, d_\text{LLM})$ to produce $k$ audio token embeddings per input.

**Parameter Count:**
- Layer 1: $512 \times 1024 + 1024 = 525,312$
- Layer 2: $1024 \times (4096 \times 8) + (4096 \times 8) = 33,587,200$
- LayerNorms: $2 \times 512 + 2 \times 4096 = 9,216$
- Scale: 1
- **Total Projector: ~34.1M parameters**

#### 4.3.3 Gated Cross-Attention Fusion with LoRA

The core of SAFE is a cross-attention mechanism that allows LLM hidden states to attend to audio token embeddings. We apply Low-Rank Adaptation (LoRA) to all projection matrices for parameter efficiency.

**Cross-Attention Block:**
$$\text{Query} = W_Q h + B_Q A_Q h \quad \text{(with LoRA)}$$
$$\text{Key} = W_K m + B_K A_K m \quad \text{(with LoRA)}$$
$$\text{Value} = W_V m + B_V A_V m \quad \text{(with LoRA)}$$

where $h \in \mathbb{R}^{L \times d}$ are LLM hidden states, $m \in \mathbb{R}^{k \times d}$ are audio tokens, and $(A_*, B_*)$ are the LoRA low-rank matrices with rank $r$.

**Attention Computation:**
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_\text{head}}}\right) V$$

**Residual Output with Gating:**
$$\Delta h = g \cdot \gamma \cdot \text{LayerNorm}(\text{Linear}(\text{Attn}(Q, K, V)))$$

where:
- $g \in [0, 1]$ is the external gate (0 when audio absent, 1 during training/inference with audio)
- $\gamma$ is a learnable residual scale, initialized to 0.1 and clamped to $[0, 5]$

**LoRA Configuration:**
- Rank $r = 8$
- Alpha $\alpha_\text{LoRA} = 16$ (scaling factor)
- Target modules: `query`, `key`, `value`, `output_dense`
- Dropout: 0.0 (disabled for stable gradient flow during training)

**Attention Hyperparameters:**
- Number of heads: 8
- Head dimension: $4096 / 8 = 512$
- Attention dropout: 0.1

**Parameter Count per Fusion Layer:**
- Base cross-attention (frozen by default):
  - Q, K, V projections: $3 \times (4096 \times 4096) = 50.3$M
  - Output projection: $4096 \times 4096 = 16.8$M
- LoRA adapters (trainable):
  - Per projection: $2 \times (4096 \times 8) = 65,536$
  - 4 projections: $4 \times 65,536 = 262,144$
- LayerNorm: $2 \times 4096 = 8,192$
- Residual scale: 1
- **Total per layer: ~270K trainable parameters** (+ 67M frozen base)

#### 4.3.4 Multi-Layer Fusion Strategy

Rather than fusing audio at a single layer, we inject audio information at multiple decoder layers. This allows the model to integrate audio at different levels of abstraction:

$$h^{(\ell)} \leftarrow h^{(\ell)} + \Delta h^{(\ell)}_\text{audio}, \quad \forall \ell \in \mathcal{F}$$

where $\mathcal{F}$ is the set of fusion layer indices. Based on empirical ablations (Section 5.6), we use:

$$\mathcal{F} = \{8, 16, 24, 32\}$$

for a 40-layer LLM (LLaVA 1.5 13B uses Vicuna-13B with 40 transformer layers). This distributes fusion across early (layer 8), mid-early (layer 16), mid-late (layer 24), and late (layer 32) processing stages.

Each fusion layer has its own independent LoRA cross-attention adapter, allowing specialization: early layers may focus on low-level audio-text alignment while later layers handle semantic integration.

**Total Fusion Parameters:**
- 4 fusion layers x ~270K trainable = ~1.08M trainable LoRA parameters
- 4 fusion layers x ~67M frozen = ~268M frozen base parameters

### 4.4 Training Objective

We train SAFE with standard cross-entropy language modeling loss on audio captioning data:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t | y_{<t}, x_\text{audio}, x_\text{prompt}; \theta, \phi)$$

where $y$ is the target caption, $x_\text{audio}$ is the audio input, $x_\text{prompt}$ is the text prompt (e.g., "Describe the audio:"), and $\phi$ represents all trainable SAFE parameters.

**Training Configuration:**
- Optimizer: AdamW with decoupled weight decay
- Learning rates: $10^{-3}$ (projector), $5 \times 10^{-4}$ (fusion adapters)
- Weight decay: 0.01
- Warmup: 2000 steps (linear)
- Schedule: Cosine decay to 10% of peak LR
- Batch size: 128 (effective, via gradient accumulation)
- Precision: Mixed FP16 (autocast)
- Gradient checkpointing: Enabled for memory efficiency

### 4.5 Two-Modality Validation Protocol

To demonstrate that our architectural guarantee generalizes beyond audio, we apply the identical SAFE architecture to a second modality: **depth**. This serves two purposes:

1. **Generality Claim**: The zero-forgetting guarantee holds for any new modality, not just audio
2. **Composition Potential**: Independent adapters can be loaded simultaneously

For depth, we use DPT/MiDaS as the depth encoder (producing 384-dimensional depth embeddings) with the same projector -> multi-layer LoRA fusion architecture. The only changes are encoder-specific:
- Input: RGB image -> depth map
- Encoder: DPT-Large pretrained on mixed datasets
- Embedding dimension: 384 (projected to 4096 via bottleneck MLP)

### 4.6 Comparison to Regularization-Based Methods

Traditional continual learning approaches mitigate forgetting through:

1. **Elastic Weight Consolidation (EWC)**: Adds quadratic penalty based on Fisher information
2. **Knowledge Distillation**: Matches output distributions with frozen teacher
3. **Experience Replay**: Maintains buffer of old-task examples

These methods **reduce** forgetting but cannot **eliminate** it -- regularization strength trades off against new-task learning capacity. Our architectural approach is fundamentally different: by construction, no information about the original task can be lost because no original parameters are modified.

| Method | Forgetting | Mechanism | Guarantee |
|--------|-----------|-----------|-----------|
| EWC | Reduced | Penalty on important weights | Probabilistic |
| Distillation | Reduced | Output matching | Empirical |
| Replay | Reduced | Interleaved training | Statistical |
| **SAFE (Ours)** | **Zero** | **Frozen backbone + gated bypass** | **Exact** |

---

## Trainable Parameter Summary

| Component | Parameters | Status |
|-----------|-----------|--------|
| Base VL Model (LLaVA 13B) | ~13B | Frozen |
| CLAP Audio Encoder | ~86M | Frozen |
| Audio Projector | ~34.1M | **Trainable** |
| Fusion Adapters (4 layers) | ~1.08M | **Trainable** |
| Audio Token Embeddings | ~8K | **Trainable** |
| **Total Trainable** | **~35.2M** | 0.27% of total |

This represents a highly parameter-efficient approach: we add audio understanding capability by training only 0.27% of the total model parameters.

---

## Implementation Notes

**VL Passthrough Logic**: When audio is absent (`audio_tokens is None` or `audio_tokens.numel() == 0`), the model executes a **true VL passthrough**: the forward pass uses the base model's embedding layer directly without any SAFE-specific processing. This ensures bit-exact outputs matching the original VL model.

**Numerical Precision**: All cross-attention computations are performed in FP32 for numerical stability, even when the model runs in FP16 mixed precision. Attention scores are clamped to $[-50, 50]$ before softmax to prevent overflow.

**Silent Audio Detection**: Audio samples with maximum waveform amplitude below $10^{-4}$ are treated as absent, triggering the VL passthrough path. This prevents near-silent audio from corrupting VL outputs.

---

## Code Reference

Key implementation files:
- `safe/models/safe_model.py` - Main SAFE model with VL passthrough and fusion hooks
- `safe/models/fusion_adapter.py` - CrossAttentionBlock, LoRAFusionAdapter, MultiLayerFusionAdapter
- `safe/models/projectors.py` - AudioProjector with bottleneck MLP
- `safe/models/audio_encoders.py` - CLAPAudioEncoder wrapper
- `safe/models/layer_hooks.py` - Hook manager for mid-layer fusion injection

---

*This section will be expanded with additional implementation details, pseudocode, and architecture diagrams for the final submission.*
