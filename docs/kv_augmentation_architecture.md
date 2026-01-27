# KV Augmentation for Multimodal Fusion in Large Language Models

## Abstract

This document describes the KV augmentation architecture implemented in the SAFE (Scalable Audio Fusion for LLMs) codebase. Unlike additive fusion approaches that inject modality information as residual perturbations to hidden states, KV augmentation provides modality tokens as explicit key-value memory that the LLM's attention mechanism can query. This architectural choice offers improved compositionality across modalities and stronger gradient signal, as the frozen LLM cannot trivially suppress attention-based information injection.

---

## 1. Motivation

### 1.1 Limitations of Additive Fusion

In the standard pre-FFN additive fusion path, modality information is injected by adding a learned delta to the LLM's hidden states at selected layers:

$$H' = H + g \cdot \Delta(H, M)$$

where $H \in \mathbb{R}^{B \times L \times d}$ represents the LLM hidden states, $M$ denotes modality tokens, and $g$ is a gating scalar.

While simple and effective, this approach has a key limitation: a frozen LLM can learn to suppress or ignore small additive perturbations, particularly when the delta is noisy or when the model finds text-only solutions sufficient during early training.

### 1.2 The KV Augmentation Alternative

KV augmentation takes a fundamentally different approach. Rather than perturbing hidden states, it injects modality information directly into the self-attention computation by providing additional key-value pairs derived from modality tokens. The LLM's attention mechanism explicitly queries this modality "memory," making it architecturally difficult to ignore.

This design offers two primary advantages:
1. **Stronger coupling**: Attention weights provide direct, interpretable access to modality information
2. **Natural compositionality**: Multiple modalities can contribute independent K/V pairs that the model attends to selectively

---

## 2. Background: Standard LLM Self-Attention

For reference, we briefly review standard self-attention as implemented in LLaMA-style models.

Given hidden states $H \in \mathbb{R}^{B \times L \times d}$, a single attention layer computes:

$$Q = HW_q, \quad K = HW_k, \quad V = HW_v$$

After applying rotary positional embeddings (RoPE) to $Q$ and $K$:

$$A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_h}} + \text{mask}\right)$$

$$O = AV$$

$$\text{output} = OW_o$$

For grouped-query attention (GQA), $Q$ has `num_heads` heads while $K, V$ have `num_key_value_heads` heads; the latter are repeated to match query head count.

---

## 3. KV Augmentation Architecture

### 3.1 Design Overview: Dual-Branch Attention

The implementation does **not** concatenate text and modality keys into a single attention computation. Instead, it computes two separate attention branches that are combined via gated addition:

```
                    ┌─────────────────────────┐
                    │     Hidden States H     │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
        ┌───────────┐    ┌───────────┐    ┌───────────────┐
        │  Q_text   │    │  K_text   │    │    V_text     │
        │ (frozen)  │    │ (frozen)  │    │   (frozen)    │
        └─────┬─────┘    └─────┬─────┘    └───────┬───────┘
              │                │                  │
              │                └────────┬─────────┘
              │                         │
              ▼                         ▼
        ┌─────────────────────────────────────┐
        │   BRANCH A: Text Self-Attention     │
        │   O_text = Attn(Q_text, K_text,     │
        │                 V_text, causal_mask) │
        └──────────────────┬──────────────────┘
                           │
                           │
        ┌──────────────────┼──────────────────┐
        │                                      │
        │     ┌────────────────────────┐       │
        │     │   Modality Tokens M    │       │
        │     └───────────┬────────────┘       │
        │                 │                    │
        │     ┌───────────┴───────────┐        │
        │     ▼                       ▼        │
        │  ┌────────┐           ┌────────┐     │
        │  │ K_mod  │           │ V_mod  │     │
        │  │(adapt.)│           │(adapt.)│     │
        │  └───┬────┘           └───┬────┘     │
        │      └──────────┬─────────┘          │
        │                 │                    │
        │      ┌──────────┴──────────┐         │
        │      │    Q_mod = Q_text   │         │
        │      │      + ΔQ(H)        │         │
        │      └──────────┬──────────┘         │
        │                 │                    │
        │                 ▼                    │
        │  ┌─────────────────────────────┐     │
        │  │  BRANCH B: Modality Attn    │     │
        │  │  O_mod = Attn(Q_mod, K_mod, │     │
        │  │              V_mod)          │     │
        │  └──────────────┬──────────────┘     │
        │                 │                    │
        └─────────────────┼────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ O = O_text + g·O_mod  │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   output = O·W_o      │
              │      (frozen)         │
              └───────────────────────┘
```

### 3.2 Branch A: Text Attention (Frozen)

Text attention proceeds identically to the original LLM:

1. Project hidden states to $Q_{\text{text}}, K_{\text{text}}, V_{\text{text}}$ using frozen LLM weights
2. Apply RoPE to $Q_{\text{text}}$ and $K_{\text{text}}$
3. Compute attention with causal masking:
   $$O_{\text{text}} = \text{Attn}(Q_{\text{text}}, K_{\text{text}}, V_{\text{text}})$$

This branch preserves the base model's text processing capabilities exactly.

### 3.3 Branch B: Modality Attention (Trainable)

The modality branch introduces learned components to enable cross-modal attention:

**Step 1: Query Adaptation**

The frozen text queries cannot directly attend to novel modality representations. We introduce a learned query delta:

$$\Delta Q = f_\theta(H)$$

where $f_\theta$ is a low-rank adapter. RoPE is applied to $\Delta Q$ to maintain positional consistency:

$$Q_{\text{mod}} = Q_{\text{text}} + \text{RoPE}(\Delta Q)$$

**Step 2: Modality Key-Value Projection**

Modality tokens $M \in \mathbb{R}^{B \times T_m \times d_m}$ are projected to key-value space:

$$K_{\text{mod}}, V_{\text{mod}} = g_\phi(M)$$

Notably, RoPE is **not** applied to modality keys—they function as position-agnostic prefix memory, always accessible to all query positions.

**Step 3: Modality Attention**

$$O_{\text{mod}} = \text{Attn}(Q_{\text{mod}}, K_{\text{mod}}, V_{\text{mod}})$$

### 3.4 Output Combination

The final attention output combines both branches via gated addition:

$$O_{\text{combined}} = O_{\text{text}} + g \cdot O_{\text{mod}}$$

where $g \in [0, 1]$ is a learnable or fixed gate. The combined output passes through the frozen output projection $W_o$.

---

## 4. Trainable Components

### 4.1 KVAugmentationAdapter

**Location**: `safe/models/kv_augmentation.py`

This module projects modality tokens to key-value space:

```python
class KVAugmentationAdapter(nn.Module):
    def __init__(self, input_dim, hidden_size, num_heads,
                 num_key_value_heads, head_dim, bottleneck_dim):
        self.audio_norm = nn.LayerNorm(input_dim)
        self.audio_k_proj = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, num_kv_heads * head_dim)
        )
        self.audio_v_proj = nn.Sequential(...)  # Same structure
        self.audio_scale = nn.Parameter(torch.ones(1))
```

**Design considerations**:
- Bottleneck architecture reduces parameter count
- Input dimension matches modality encoder output (e.g., 512 for CLAP, 768 for PointBERT)
- Learnable scale provides per-layer modality strength control

### 4.2 AudioQueryAdapter (ΔQ Adapter)

**Location**: `safe/models/kv_augmentation.py`

This module enables the frozen LLM queries to attend to modality memory:

```python
class AudioQueryAdapter(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, rank):
        self.down_proj = nn.Linear(hidden_size, rank)
        self.up_proj = nn.Linear(rank, num_heads * head_dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

        # Critical: initialize up_proj to zeros for stable training start
        nn.init.zeros_(self.up_proj.weight)
```

**Key insight**: Without $\Delta Q$, frozen LLM queries exist in a subspace that was never trained to align with modality keys. The query adapter learns the minimal adjustment needed to enable cross-modal attention. Zero initialization ensures $\Delta Q \approx 0$ initially, preserving text behavior.

---

## 5. Implementation Details

### 5.1 Attention Module Replacement

Unlike hook-based approaches, KV augmentation requires modifying attention internals. The `KVAugmentationHookManager` replaces attention modules:

```python
def wrap_attention_modules(self):
    for layer_idx in self.fusion_layer_indices:
        layer = self.get_decoder_layer(layer_idx)
        original_attn = layer.self_attn
        wrapped_attn = KVAugmentedAttention(
            original_attn,
            self.kv_adapters[layer_idx],
            layer_idx
        )
        layer.self_attn = wrapped_attn
```

This is fundamentally different from pre-FFN additive fusion, which uses forward hooks on MLP modules.

### 5.2 Token Injection Protocol

Before each forward pass, modality tokens must be injected:

```python
self.kv_hook_manager.inject_audio(
    audio_tokens=modality_tokens,
    audio_mask=attention_mask,
    gate=1.0
)

outputs = self.base_vl.llm(inputs_embeds=inputs_embeds, ...)
```

**Critical warning**: Tokens should **not** be cleared after forward. Gradient checkpointing replays forward during backward; clearing tokens would break gradient flow.

### 5.3 Masking Behavior

| Branch | Mask Type | Behavior |
|--------|-----------|----------|
| Text | Causal + padding | Standard autoregressive masking |
| Modality | Optional padding only | Position-agnostic; all queries can attend to all modality tokens |

The modality branch does not use causal masking—modality tokens function as globally-visible prefix memory.

### 5.4 HuggingFace Compatibility

HF attention modules return varying tuple formats:
- `(attn_output, attn_weights)`
- `(attn_output, attn_weights, past_key_value)`
- `attn_output` (rare)

`KVAugmentedAttention` probes the original module once to detect the return format and matches it, preventing decoder layer unpacking errors.

---

## 6. KV Caching Considerations

KV augmentation interacts poorly with standard KV caching because:

1. Modality K/V must be augmented consistently across generation steps
2. Beam expansion requires careful handling of modality states
3. Modality keys should not be treated as causal text positions

The current implementation disables caching (`use_cache=False`) for correctness. This impacts generation speed but ensures correct behavior.

---

## 7. Diagnostics and Monitoring

### 7.1 Key Metrics

The implementation logs several diagnostic metrics:

| Metric | Computation | Interpretation |
|--------|-------------|----------------|
| `rms_ratio` | $\frac{\text{RMS}(O_{\text{mod}})}{\text{RMS}(O_{\text{text}})}$ | Modality contribution strength |
| `attn_entropy` | Entropy of attention weights over modality tokens | Distribution uniformity |
| `delta_q_ratio` | $\frac{\text{RMS}(\Delta Q)}{\text{RMS}(Q)}$ | Query adapter strength |

### 7.2 Healthy Ranges

- `rms_ratio`: 0.01–0.3 (too low = modality ignored; too high = text disruption)
- `attn_entropy`: Should not collapse to 0 (attending to single token)
- `delta_q_ratio`: 0.05–0.2 (too high suggests unstable training)

### 7.3 Regularization: MinAudioAttentionLoss

An optional loss term enforces minimum attention to modality tokens:

$$\mathcal{L}_{\text{min-attn}} = \max(0, \tau - \bar{a}_{\text{answer}})$$

where $\bar{a}_{\text{answer}}$ is mean attention to modality tokens at answer positions and $\tau$ is the threshold. Use sparingly—excessive strength can harm text generation quality.

---

## 8. Compositionality with Multiple Modalities

KV augmentation naturally extends to multiple modalities. Each modality contributes independent K/V pairs:

**Option A: Separate branches** (current design)
$$O = O_{\text{text}} + g_A \cdot O_A + g_B \cdot O_B$$

**Option B: Concatenated modality memory**
$$K_{\text{combined}} = [K_A; K_B], \quad V_{\text{combined}} = [V_A; V_B]$$
$$O_{\text{mod}} = \text{Attn}(Q_{\text{mod}}, K_{\text{combined}}, V_{\text{combined}})$$

Either approach enables selective attention based on which modality memory is relevant for predicting the current token.

---

## 9. Summary: Frozen vs. Trainable Parameters

| Component | Status | Notes |
|-----------|--------|-------|
| LLM q/k/v/o projections | Frozen | Preserves text capabilities |
| LLM FFN, embeddings | Frozen | Preserves text capabilities |
| Text attention path | Frozen | Exact original computation |
| Modality encoder | Frozen | Pre-trained representations |
| Modality projector | Trainable | Maps encoder → token space |
| KVAugmentationAdapter | Trainable | Modality → K/V projection |
| AudioQueryAdapter | Trainable | Hidden → ΔQ adapter |
| Gates and scales | Trainable | Per-layer modality strength |

**Total trainable parameters**: Typically <1% of LLM parameters, enabling efficient modality adaptation without catastrophic forgetting.

---

## References

- Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (2019) — Adapter architecture foundations
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) — Low-rank adaptation
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021) — RoPE
- Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning" (2022) — Cross-attention for multimodal fusion
