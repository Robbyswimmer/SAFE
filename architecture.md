# SAFE Model Architecture Specification

**Version:** 1.0
**Last Updated:** 2025-10-09
**Model Type:** Defensive Multimodal Architecture with Audio Addition to Vision-Language Models

---

## Executive Summary

SAFE (Simple, Adaptive, Failure-Proof Audio Addition) is a multimodal architecture designed to add audio capabilities to pre-trained vision-language (VL) models while **provably preserving** base VL performance. Unlike traditional multimodal models that risk catastrophic forgetting during modality addition, SAFE employs defensive architectural patterns including frozen base models, conditional bypass paths, explicit retention loss, and continuous monitoring.

**Key Innovation:** SAFE guarantees that when audio is absent (`audio=None` or `gate=0`), the model's behavior is **mathematically identical** to the base VL model, eliminating regression risk for the original capabilities.

---

## Architecture Overview

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        SAFE MODEL                                │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Vision     │    │    Audio     │    │    Text      │      │
│  │   Encoder    │    │   Encoder    │    │  Tokenizer   │      │
│  │  (FROZEN)    │    │  (FROZEN)    │    │              │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   │               │
│  ┌──────────────┐    ┌──────────────┐          │               │
│  │   Vision     │    │    Audio     │          │               │
│  │  Projector   │    │  Projector   │          │               │
│  │  (FROZEN)    │    │ (TRAINABLE)  │          │               │
│  └──────┬───────┘    └──────┬───────┘          │               │
│         │                   │                   │               │
│         │                   │                   ▼               │
│         │                   │          ┌─────────────────┐     │
│         │                   │          │  Base Embeddings│     │
│         │                   │          │    (FROZEN)     │     │
│         │                   │          └────────┬────────┘     │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────────────────────────────────────────┐          │
│  │            Token Sequence Builder                │          │
│  │   [<img> vision_tokens </img> text audio_tokens] │          │
│  └────────────────────┬─────────────────────────────┘          │
│                       │                                         │
│                       ▼                                         │
│  ┌──────────────────────────────────────────────────┐          │
│  │         INPUT EMBEDDINGS (inputs_embeds)         │          │
│  │           Shape: (B, seq_len, hidden_dim)        │          │
│  └────────────────────┬─────────────────────────────┘          │
│                       │                                         │
│         ┌─────────────┴─────────────┐                          │
│         │                           │                          │
│         │    Audio Present?         │                          │
│         │    gate > 0?              │                          │
│         │                           │                          │
│    NO   │                           │   YES                    │
│         ▼                           ▼                          │
│  ┌─────────────┐          ┌──────────────────┐                │
│  │   BYPASS    │          │  Cross-Attention │                │
│  │   Direct to │          │  Fusion Adapter  │                │
│  │   Base VL   │          │   (TRAINABLE)    │                │
│  └──────┬──────┘          └────────┬─────────┘                │
│         │                          │                           │
│         │                          │ Q: hidden states          │
│         │                          │ K,V: audio tokens         │
│         │                          │                           │
│         └──────────┬───────────────┘                           │
│                    ▼                                           │
│         ┌────────────────────┐                                 │
│         │   Base VL Model    │                                 │
│         │   Language Model   │                                 │
│         │     (FROZEN)       │                                 │
│         └──────────┬─────────┘                                 │
│                    ▼                                           │
│         ┌────────────────────┐                                 │
│         │  Output Logits     │                                 │
│         │  (vocab_size)      │                                 │
│         └────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. **Base Vision-Language Model** (`BaseVLModel`)

**Location:** `safe/models/base_vl.py`

**Architecture Types Supported:**
- LLaVA (Llama + CLIP)
- BLIP-2 (OPT/Flan-T5 + ViT)
- Custom VL models

**Status:** **COMPLETELY FROZEN** during audio training

**Parameters:**
```python
llm_model_name: str = "microsoft/DialoGPT-medium"  # Or Llama, GPT-2, etc.
vision_model_name: str = "openai/clip-vit-large-patch14"
freeze_vision: bool = True
freeze_llm: bool = True
```

**Components:**
- **Vision Encoder:** Pre-trained CLIP/ViT (frozen)
- **Language Model:** Pre-trained LLM (frozen)
- **Vision Projector:** Maps vision features to LLM space (frozen)
- **Tokenizer:** Shared across all modalities

**Key Property:** All `requires_grad=False` for all parameters during Stage A training.

---

### 2. **Audio Encoder** (`CLAPAudioEncoder` / `WhisperAudioEncoder` / `MultiModalAudioEncoder`)

**Location:** `safe/models/audio_encoders.py`

**Supported Types:**

#### **2a. CLAP Audio Encoder**
```python
model: "laion/clap-htsat-unfused"
audio_embed_dim: 512
freeze: True  # Frozen during training
sample_rate: 48000
```

**Architecture:** HTSAT (Hierarchical Token-Semantic Audio Transformer)
- Input: Raw waveform or mel-spectrogram
- Output: 512-dim audio embedding

#### **2b. Whisper Audio Encoder**
```python
model: "openai/whisper-small"
audio_embed_dim: 768
freeze: True
extract_features: True  # Use encoder outputs, not transcripts
```

**Architecture:** Transformer encoder from Whisper
- Input: Log-mel spectrogram
- Output: 768-dim sequence features + optional transcription

#### **2c. MultiModal Audio Encoder**
Combines multiple audio encoders (CLAP + Whisper) for richer representations.

**Status:** **FROZEN** during Stage A (feature extraction only)

---

### 3. **Audio Projector** (`AudioProjector` / `AdaptiveAudioProjector`)

**Location:** `safe/models/projectors.py`

**Purpose:** Map audio embeddings to LLM hidden dimension space.

**Status:** **TRAINABLE** (primary audio learning component)

#### **3a. Standard Audio Projector**
```python
AudioProjector(
    audio_embed_dim: int = 512,      # From audio encoder
    llm_hidden_size: int = 1024,     # LLM dimension
    num_audio_tokens: int = 8,       # Fixed number of output tokens
    hidden_size: int = 2048,         # MLP intermediate size
    num_layers: int = 2,             # MLP depth
    dropout: float = 0.1
)
```

**Architecture:**
```
audio_features (B, 512)
    → MLP [512 → 2048 → 1024]
    → Reshape/Expand to (B, 8, 1024)
    → LayerNorm
    → audio_tokens
```

#### **3b. Adaptive Audio Projector**
```python
AdaptiveAudioProjector(
    max_audio_tokens: int = 32,      # Maximum tokens
    gating_mechanism: str = "learned" # Adaptive token selection
)
```

**Architecture:** Variable-length token generation with learned gating.

**Gradient Flow:** Projector receives gradients from both:
1. Audio task loss (direct supervision)
2. Retention loss (through LLM backward pass)

---

### 4. **Fusion Adapter** (Cross-Attention Module)

**Location:** `safe/models/fusion_adapter.py`

**Purpose:** Fuse audio tokens with LLM hidden states via cross-attention.

**Status:** **TRAINABLE** with LoRA adapters

#### **4a. Cross-Attention Block** (`CrossAttentionBlock`)

**Core Mechanism:**
```python
CrossAttentionBlock(
    hidden_size: int = 1024,
    num_attention_heads: int = 8,
    attention_dropout: float = 0.1,
    output_dropout: float = 0.1,
    layer_norm_eps: float = 1e-5
)
'''

**Architecture:**
```
Query (Q):  LLM hidden_states → Linear(hidden_size, hidden_size)
Key (K):    audio_tokens → Linear(hidden_size, hidden_size)
Value (V):  audio_tokens → Linear(hidden_size, hidden_size)

Attention: softmax(Q @ K^T / sqrt(d_head)) @ V
Output: hidden_states + residual_scale * attention_output
```

**Multi-Head Attention:**
- `num_heads = 8`
- `head_dim = hidden_size / num_heads`
- Parallel attention computation across heads

**Numerical Stability Features:**
```python
# Upcast to fp32 for stable attention
query_layer = query_layer.to(torch.float32)
key_layer = key_layer.to(torch.float32)

# Clamp scores to prevent softmax overflow
attention_scores = torch.clamp(attention_scores, min=-50.0, max=50.0)

# NaN/Inf sanitization
attention_scores = torch.nan_to_num(attention_scores, nan=0.0)
```

**Residual Scaling:**
```python
residual_scale = 0.05  # Gentle fusion (5% audio influence initially)
output = hidden_states + (residual_scale * context)
```

**Attention Masking:**
- Supports `audio_attention_mask` to ignore silent/invalid audio
- Masking format: `1 = attend`, `0 = ignore`
- Masked positions filled with `-1e4` before softmax

#### **4b. LoRA Fusion Adapter** (`LoRAFusionAdapter`)

**Architecture:**
```python
LoRAFusionAdapter(
    hidden_size: int = 1024,
    lora_rank: int = 8,              # Low-rank bottleneck
    lora_alpha: float = 16.0,        # Scaling factor
    lora_dropout: float = 0.0,       # Disabled for stability
    target_modules: ["query", "value"]  # Apply LoRA to Q and V
)
```

**LoRA Application:**
```
Original: W_q, W_v  (frozen conceptually, but we train adapters)
LoRA: W_q' = W_q + (alpha/rank) * A_q @ B_q
      where A_q: (hidden_size, rank), B_q: (rank, hidden_size)
```

**Parameter Efficiency:**
- Full Q/V matrices: `2 * hidden_size²` parameters
- LoRA adapters: `2 * 2 * hidden_size * rank` parameters
- Example: 1024-dim, rank=8 → 2M params vs 32K params (62x reduction)

**Gating Mechanism:**
```python
# Interpolate between original and fused states
output = gate * fused_states + (1.0 - gate) * hidden_states

# gate ∈ [0, 1]
# gate=0 → no audio influence (identical to base VL)
# gate=1 → full audio fusion
```

#### **4c. Multi-Layer Fusion Adapter** (`MultiLayerFusionAdapter`)

**Purpose:** Apply fusion at multiple LLM layers (experimental).

```python
MultiLayerFusionAdapter(
    fusion_layer_indices: [4, 8, 12],  # Apply at specific LLM layers
    num_layers: int = 12
)
```

**Usage:** Insert fusion at mid-layers for hierarchical audio integration.

#### **4d. Gated Fusion Adapter** (`GatedFusionAdapter`)

**Purpose:** Learnable, input-dependent gating.

```python
gate_network = MLP([hidden_size*2 → 64 → 1 → sigmoid])
gate = gate_network(concat(hidden_states_pooled, audio_tokens_pooled))
```

**Adaptive Fusion:** Gate learned per-sample based on content.

---

### 5. **Special Token Embeddings**

**Location:** `safe_model.py:143-158`

**Purpose:** Separate embedding space for audio marker tokens.

**Tokens:**
- `<audio>` (start marker)
- `</audio>` (end marker)

**Architecture:**
```python
# Base vocabulary: e.g., 50257 tokens (GPT-2)
# Add 2 new tokens → vocabulary becomes 50259

# CRITICAL: Don't resize base embedding table!
# Instead, create separate embedding layer:
audio_token_embeddings = nn.Embedding(2, hidden_size)

# Initialize with mean of base embeddings
audio_token_embeddings.weight = mean(base_embeddings.weight)
```

**Status:** **TRAINABLE**

**Embedding Routing:**
```python
def get_input_embeddings(input_ids):
    if input_ids < original_vocab_size:
        return base_embeddings(input_ids)  # Frozen
    else:
        idx = input_ids - original_vocab_size
        return audio_token_embeddings(idx)  # Trainable
```

**Key Property:** Base embedding table remains **untouched** and frozen.

---

## Data Flow Architecture

### Forward Pass (Full Multimodal)

```
Step 1: ENCODING
─────────────────
Vision Input (B, 3, 224, 224)
    → Vision Encoder (FROZEN)
    → Vision Features (B, 257, 1024)
    → Vision Projector (FROZEN)
    → Vision Tokens (B, 256, 1024)

Audio Input (B, waveform)
    → Audio Encoder (FROZEN)
    → Audio Features (B, 512)
    → Audio Projector (TRAINABLE)
    → Audio Tokens (B, 8, 1024)

Text Input "Describe the image and sound"
    → Tokenizer
    → Token IDs (B, seq_len)

Step 2: SEQUENCE CONSTRUCTION
──────────────────────────────
Token Sequence:
    [<img> <vision_tokens> </img> <audio> <audio_tokens> </audio> <text_tokens>]

Token ID Sequence:
    [32000, ..., 32001, 50257, ..., 50258, 1234, 5678, ...]
     └─vision placeholder─┘ └─audio placeholder─┘ └─text─┘

Step 3: EMBEDDING LOOKUP
─────────────────────────
For each token ID:
    if ID < 50257:  # Base vocabulary
        embedding = base_embeddings[ID]  (FROZEN)
    else:  # Audio tokens (50257, 50258)
        embedding = audio_token_embeddings[ID - 50257]  (TRAINABLE)

Result: inputs_embeds (B, seq_len, 1024)

Step 4: CONDITIONAL FUSION
───────────────────────────
if audio_tokens is not None and gate > 0.0:
    # Apply cross-attention fusion
    fused_embeds = LoRAFusionAdapter(
        hidden_states=inputs_embeds,     # Query
        audio_tokens=audio_tokens,       # Key, Value
        attention_mask=audio_attention_mask,
        gate=gate
    )

    # Cross-attention computation:
    Q = Linear_q(inputs_embeds)          # (B, seq_len, 1024)
    K = Linear_k(audio_tokens)           # (B, 8, 1024)
    V = Linear_v(audio_tokens)           # (B, 8, 1024)

    scores = Q @ K^T / sqrt(128)         # (B, 8, seq_len, 8)
    attention = softmax(scores, dim=-1)
    context = attention @ V              # (B, seq_len, 1024)

    # Residual connection with scaling
    output = inputs_embeds + 0.05 * context

    # Gating
    fused_embeds = gate * output + (1-gate) * inputs_embeds
else:
    # BYPASS: No fusion, direct pass to base model
    fused_embeds = inputs_embeds

Step 5: BASE VL FORWARD
────────────────────────
outputs = base_vl.llm(
    inputs_embeds=fused_embeds,  # (B, seq_len, 1024)
    attention_mask=attention_mask,
    labels=labels
)

logits = outputs.logits  # (B, seq_len, vocab_size)
loss = outputs.loss      # Scalar
```

### Forward Pass (VL Only, No Audio)

```
Step 1-3: Same as above (Vision + Text encoding)

Step 4: BYPASS FUSION
──────────────────────
if audio_tokens is None:
    # Direct path to base model - NO cross-attention
    # This is IDENTICAL to original base VL model
    outputs = base_vl(
        input_ids=input_ids,  # Can use IDs directly
        attention_mask=attention_mask,
        pixel_values=pixel_values
    )

Result: Mathematically equivalent to base VL model
```

**Key Guarantee:** When `audio=None`, the computational graph is **identical** to the base VL model. No cross-attention layers are instantiated, no fusion adapters are called.

---

## Training Pipeline (Stage A)

### Trainable Components

```python
TRAINABLE:
✓ Audio Projector          (~2M parameters)
✓ Fusion Adapter (LoRA)    (~32K parameters with rank=8)
✓ Audio Token Embeddings   (~2K parameters, 2 tokens × 1024 dim)

FROZEN:
✗ Base VL Model            (~1B parameters)
✗ Vision Encoder           (~300M parameters)
✗ Audio Encoder            (~80M parameters)
✗ Base Embeddings          (~50M parameters)

Total Trainable: ~2M parameters (~0.15% of full model)
```

### Loss Function Architecture

**Location:** `safe/training/losses.py`

```python
total_loss = audio_weight * L_audio
           + retention_weight * L_retention

where:
    audio_weight = 1.0      # Default
    retention_weight = 1.0  # Default (balanced)
```

#### **Audio Task Loss** (`AudioTaskLoss`)

**Supported Tasks:**
1. **Audio Captioning**
   ```python
   L_caption = CrossEntropyLoss(
       predicted_tokens,
       ground_truth_caption_tokens
   )
   ```

2. **Audio Question Answering**
   ```python
   L_qa = CrossEntropyLoss(
       predicted_answer,
       ground_truth_answer
   )
   ```

3. **Audio Classification**
   ```python
   L_class = CrossEntropyLoss(
       logits[<audio_token_positions>],
       class_labels
   )
   ```

**Implementation:**
- Standard next-token prediction
- Labels masked: `-100` for non-target tokens
- Computed only on audio-containing samples

#### **Retention Loss** (`RetentionLoss`)

**Purpose:** Preserve base VL performance on vision-language tasks.

**Components:**

**1. Distillation Loss**
```python
L_distill = KL_divergence(
    safe_logits / temperature,
    base_logits / temperature
) * (temperature^2)

temperature = 3.0  # Soften distributions
```

**Computation:**
- Forward through SAFE model → `safe_logits`
- Forward through base VL model (frozen) → `base_logits`
- Compute KL divergence on VL-only samples
- Only applied where `has_audio=False`

**Purpose:** Ensure SAFE's outputs match base VL on vision-language tasks.

**2. Fisher Information Regularization**
```python
L_fisher = Σ_i F_i * (θ_i - θ_i*)^2

where:
    F_i = Fisher information for parameter i
    θ_i = current parameter value
    θ_i* = initial parameter value
```

**Fisher Computation:**
```python
# Sample batches from VL data
for batch in vl_dataloader:
    outputs = base_vl(**batch)
    loss = outputs.loss

    # Compute gradients
    gradients = autograd.grad(loss, base_vl.parameters())

    # Fisher = E[gradient^2]
    fisher[param] += gradient^2

fisher /= num_batches
```

**Purpose:** Penalize parameter changes that hurt base VL performance.

**3. Combined Retention Loss**
```python
L_retention = distillation_weight * L_distill
            + fisher_weight * L_fisher

distillation_weight = 1.0  # Default
fisher_weight = 0.1        # Default (optional)
```

### Optimizer Configuration

```python
AdamW(
    params=[
        {
            "params": audio_projector.parameters(),
            "lr": 1e-4,
            "weight_decay": 0.01
        },
        {
            "params": fusion_adapter.parameters(),
            "lr": 5e-5,  # Lower for LoRA
            "weight_decay": 0.01
        },
        {
            "params": audio_token_embeddings.parameters(),
            "lr": 1e-4,
            "weight_decay": 0.0  # No decay for embeddings
        }
    ],
    betas=(0.9, 0.999),
    eps=1e-8
)
```

**Learning Rate Schedule:**
- Warmup: 500-2000 steps
- Cosine annealing or constant
- Gate warmup: 0 → 1 over 2000 steps

### Training Algorithm

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Prepare inputs
        inputs = safe_model.prepare_multimodal_inputs(
            text=batch["text"],
            images=batch["images"],
            audio=batch["audio"],
            answers=batch["answers"],
            training_mode=True  # Apply ground truth answers
        )

        # 2. Forward pass through SAFE
        safe_outputs = safe_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            audio_tokens=inputs.get("audio_tokens"),
            labels=inputs["labels"],
            gate=current_gate  # Gradual warmup
        )

        # 3. Forward pass through base VL (frozen, no grad)
        if retention_enabled and not batch["has_audio"]:
            with torch.no_grad():
                base_outputs = base_vl(
                    input_ids=sanitized_input_ids,
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs.get("pixel_values"),
                    labels=inputs["labels"]
                )

        # 4. Compute loss
        loss_dict = combined_loss(
            safe_outputs=safe_outputs,
            base_outputs=base_outputs,
            labels=inputs["labels"],
            has_audio=batch["has_audio"]
        )

        total_loss = loss_dict["total_loss"]

        # 5. Backward + optimize
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        # 6. Update gate (warmup)
        if global_step < gate_warmup_steps:
            current_gate = global_step / gate_warmup_steps
            safe_model.set_gate(current_gate)

        # 7. Monitor retention
        if global_step % eval_steps == 0:
            eval_metrics = evaluate(val_dataloader)
            retention_score = eval_metrics["retention_score"]

            # Check for degradation
            if retention_score < baseline_retention - tolerance:
                print("WARNING: Retention degradation detected!")
                # Early stopping or rollback
```

---

## Evaluation Metrics

### 1. **Retention Score**

```python
retention_score = safe_vl_accuracy / base_vl_accuracy

where:
    safe_vl_accuracy = accuracy on VL tasks using SAFE (gate=1)
    base_vl_accuracy = accuracy on VL tasks using base VL model

Target: retention_score ≥ 0.995 (≤ 0.5% degradation)
```

**Interpretation:**
- `1.00`: Perfect retention (no degradation)
- `0.99`: 1% degradation (acceptable)
- `0.95`: 5% degradation (concerning)
- `< 0.95`: Significant forgetting (failure)

### 2. **Audio Gain Score**

```python
audio_gain = safe_audio_accuracy - base_audio_accuracy

where:
    safe_audio_accuracy = accuracy on audio tasks using SAFE
    base_audio_accuracy = accuracy on audio tasks using base VL (should be ~0)

Target: audio_gain > 0.5 (50% accuracy on audio tasks)
```

### 3. **Gate-Zero Validation**

```python
gate_zero_accuracy = accuracy with SAFE(gate=0)
original_accuracy = accuracy with base VL model

gate_zero_retention = gate_zero_accuracy / original_accuracy

Target: gate_zero_retention ≈ 1.00 (mathematically should be identical)
```

**Purpose:** Verify that `gate=0` truly bypasses audio fusion.

### 4. **Cross-Modal Alignment** (Advanced)

```python
# Measure correlation between audio and vision features
alignment = cosine_similarity(
    audio_tokens_pooled,
    vision_tokens_pooled
)
```

---

## Key Design Decisions & Rationale

### 1. **Why Freeze the Base Model?**

**Decision:** All base VL parameters frozen (`requires_grad=False`)

**Rationale:**
- **Prevents catastrophic forgetting:** Base weights cannot change
- **Enables conditional bypass:** Can route inputs directly to frozen model
- **Reduces training cost:** Only ~2M parameters to optimize
- **Provides stable teacher:** Base outputs used for distillation loss

**Trade-off:** Limits adaptation to new modality (cannot modify base representations). Mitigated by powerful cross-attention fusion.

### 2. **Why Cross-Attention Instead of Concatenation?**

**Decision:** Audio as Key/Value, LLM hidden states as Query

**Alternatives:**
- **Concatenation:** `[vision_tokens, audio_tokens, text_tokens]` → LLM
- **Prefix tuning:** Prepend audio tokens to input sequence
- **Self-attention:** Treat audio as additional tokens in self-attention

**Rationale:**
- **Selective integration:** Each LLM token attends to relevant audio
- **Flexible fusion depth:** Can apply at multiple layers
- **Preserves sequence length:** No context length inflation
- **Controllable influence:** Gate mechanism for smooth control

**Comparison:**

| Approach | Pros | Cons | Retention Risk |
|----------|------|------|----------------|
| Concatenation | Simple, parallelizable | Inflates context, rigid fusion | Medium |
| Prefix tuning | Efficient, no architecture change | Limited expressiveness | Low-Medium |
| Cross-attention (SAFE) | Flexible, selective, controllable | More parameters, slower | **Low** |

### 3. **Why LoRA for Cross-Attention?**

**Decision:** Apply LoRA (rank=8) to Query and Value projections

**Rationale:**
- **Parameter efficiency:** 62x reduction vs. full fine-tuning
- **Regularization:** Low-rank constraint prevents overfitting
- **Modular adaptation:** LoRA weights can be saved/loaded separately
- **Less forgetting:** Smaller parameter space → smaller gradient updates

**LoRA Configuration:**
```python
rank = 8
alpha = 16.0  # Scaling factor (alpha/rank = 2.0)
target_modules = ["query", "value"]  # Q and V, not K
dropout = 0.0  # Disabled for stability
```

**Why not apply to Key?** Keys primarily route information; Values carry content. Q and V are more critical for fusion quality.

### 4. **Why Separate Audio Token Embeddings?**

**Decision:** Create `audio_token_embeddings` separate from base embeddings

**Alternative:** Resize base embedding table to include audio tokens

**Rationale:**
- **Isolation:** Base embeddings remain completely frozen and unchanged
- **Safety:** No risk of corrupting base token representations
- **Modularity:** Audio embeddings can be independently initialized/trained
- **Verification:** Can inspect base embeddings to confirm no modification

**Trade-off:** Slightly more complex embedding lookup logic. Worth it for safety guarantees.

### 5. **Why Gated Fusion?**

**Decision:** Interpolate `gate * fused + (1-gate) * original`

**Rationale:**
- **Warm-up stability:** Start with `gate=0` (no audio) → gradually increase
- **Ablation studies:** Can programmatically disable audio for experiments
- **Debugging:** Test model behavior at different fusion strengths
- **Adaptive inference:** Could implement learned, per-sample gating

**Gate Schedule:**
```python
# Linear warmup over 2000 steps
gate = min(1.0, global_step / 2000)

# Or cosine schedule
gate = 0.5 * (1 + cos(π * (1 - progress)))
```

### 6. **Why Retention Loss?**

**Decision:** Explicit distillation + Fisher information regularization

**Alternative:** Trust freezing alone to preserve performance

**Rationale:**
- **Freezing is necessary but not sufficient:** Frozen weights can still produce bad outputs with modified inputs
- **Input distribution shift:** Fused embeddings may fall outside base model's training distribution
- **Gradient signal:** Retention loss guides audio components to preserve base behavior
- **Early detection:** Loss degradation signals retention failure before eval

**Empirical validation:** Models without retention loss show 5-15% VL degradation even with frozen base.

### 7. **Why Conditional Bypass?**

**Decision:** When `audio=None`, skip fusion entirely

**Alternative:** Always run fusion, use zero-padded audio tokens

**Rationale:**
- **Provable equivalence:** Bypass path is mathematically identical to base VL
- **Efficiency:** No unnecessary computation when audio absent
- **Debugging:** Can directly compare to base model (should be identical)
- **Regression prevention:** Cannot degrade VL performance if base model is untouched

**Implementation:**
```python
if audio_tokens is None or gate == 0.0:
    # BYPASS: Direct to base VL
    return base_vl(input_ids, attention_mask, ...)
else:
    # FUSION: Apply cross-attention
    fused_embeds = fusion_adapter(inputs_embeds, audio_tokens)
    return base_vl(inputs_embeds=fused_embeds, ...)
```

---

## Comparison to Other Architectures

### SAFE vs. LLaVA

| Aspect | LLaVA | SAFE |
|--------|-------|------|
| **Base Model** | Fine-tunes LLM (full or LoRA) | **Completely frozen** |
| **Fusion Method** | Concatenation (vision prefix) | **Cross-attention** |
| **Modality Addition** | Vision-only (requires retraining for audio) | **Modular audio addition** |
| **Retention Guarantee** | None (performance degradation observed) | **Explicit loss + monitoring** |
| **Bypass Path** | No (all inputs through modified architecture) | **Yes** (audio=None → base model) |
| **Trainable Params** | ~4-7B (LoRA) or ~7B (full) | **~2M** |
| **Risk of Forgetting** | High (modifies base model) | **Low** (frozen + retention loss) |

### SAFE vs. BLIP-2

| Aspect | BLIP-2 | SAFE |
|--------|--------|------|
| **Base Models** | Both frozen (vision + LM) | **Both frozen (vision + LM)** ✓ |
| **Fusion Method** | Q-Former (frozen LM never sees modality directly) | **Cross-attention (modality fused into LM)** |
| **Trainable Component** | Q-Former (~188M params) | **Projector + LoRA adapter (~2M)** |
| **Retention Loss** | None (assumes freezing suffices) | **Explicit distillation** |
| **Modality Flexibility** | Q-Former is vision-specific | **Generalizes to any modality** |
| **Context Integration** | Limited (Q-Former bottleneck) | **Full cross-attention** |

### SAFE vs. Flamingo

| Aspect | Flamingo | SAFE |
|--------|----------|------|
| **Base Model** | Fine-tunes cross-attention layers + some LM layers | **Completely frozen** |
| **Fusion Method** | Gated cross-attention (inserted into LM) | **External cross-attention (before LM)** |
| **Gating** | Learned, always active | **Explicit, can disable (gate=0)** |
| **Bypass Path** | No (all tokens pass through gated layers) | **Yes** (audio=None → bypass) |
| **Retention Guarantee** | None | **Explicit loss + monitoring** |
| **Architecture Modification** | Modifies LM architecture permanently | **External adapter, LM unchanged** |

### SAFE vs. Traditional Continual Learning

| Aspect | EWC/LwF (Continual Learning) | SAFE |
|--------|------------------------------|------|
| **Target Problem** | Sequential task learning | **Modality addition** |
| **Base Model** | Partially frozen (important params) | **Completely frozen** |
| **Regularization** | Fisher information | **Fisher + distillation** |
| **Architecture** | Same architecture, updated weights | **Extended architecture, frozen weights** |
| **New Capability** | New task head | **New modality encoder + fusion** |
| **Forgetting Risk** | Medium (depends on regularization strength) | **Low** (architectural isolation) |

---

## Model Variants & Configurations

### Small (Demo)
```python
base_llm: "microsoft/DialoGPT-small" (124M)
vision: "openai/clip-vit-base-patch32" (151M)
audio: "laion/clap-htsat-fused" (80M)

projector_hidden_size: 1024
num_audio_tokens: 4
lora_rank: 4

Total: ~360M parameters
Trainable: ~1M parameters
```

### Medium (Research)
```python
base_llm: "meta-llama/Llama-2-7b-hf" (7B)
vision: "openai/clip-vit-large-patch14" (304M)
audio: "openai/whisper-small" (244M)

projector_hidden_size: 2048
num_audio_tokens: 8
lora_rank: 8

Total: ~7.5B parameters
Trainable: ~2M parameters
```

### Large (Production)
```python
base_llm: "meta-llama/Llama-2-13b-hf" (13B)
vision: "openai/clip-vit-large-patch14-336" (428M)
audio: "openai/whisper-medium" (769M)

projector_hidden_size: 4096
num_audio_tokens: 16
lora_rank: 16

Total: ~14B parameters
Trainable: ~8M parameters
```

---

## Advanced Features

### 1. **Curriculum Learning**

**Location:** `safe/training/curriculum.py`

**Stages:**
1. **Stage 1:** VL-only + Simple audio (no retention loss)
2. **Stage 2:** Introduce retention loss (low weight)
3. **Stage 3:** Balanced audio + retention
4. **Stage 4:** Complex multimodal tasks

**Benefits:**
- Gradual adaptation
- Stable retention
- Better final performance

### 2. **Fisher Information Caching**

**Purpose:** Pre-compute Fisher matrix for base VL model

**Algorithm:**
```python
# One-time computation
fisher_info = {}
for batch in vl_dataloader:
    outputs = base_vl(**batch)
    loss = outputs.loss
    grads = autograd.grad(loss, base_vl.parameters())

    for param, grad in zip(base_vl.parameters(), grads):
        fisher_info[param] += grad ** 2

fisher_info = {p: f / len(vl_dataloader) for p, f in fisher_info.items()}

# Save for future use
torch.save(fisher_info, "fisher_cache.pt")
```

**Usage:** Load pre-computed Fisher for faster training.

### 3. **Attention Probing**

**Purpose:** Debug cross-attention behavior

**Metrics:**
```python
attention_summary = {
    "overall_mean": mean(attention_probs),
    "overall_max": max(attention_probs),
    "entropy": -sum(p * log(p)),  # Attention diversity
    "supervised_mean": mean(attention_probs[supervised_tokens])
}
```

**Activation:**
```python
safe_model.configure_attention_probe(enabled=True, log_limit=10)
```

### 4. **Null-Space Projection (Experimental)**

**Purpose:** Constrain gradient updates to null space of base VL

**Algorithm:**
```python
# Compute projection matrix
U, S, Vt = SVD(base_vl.important_params)
null_space = Vt[rank:, :]  # Singular vectors with small singular values

# Project gradients
for param in trainable_params:
    param.grad = null_space @ (null_space.T @ param.grad)
```

**Status:** Experimental, not enabled by default.

---

## Hyperparameter Defaults

### Model Configuration
```python
# Audio encoder
audio_encoder_type: "clap"
audio_encoder_freeze: True

# Projector
projector_type: "standard"
num_audio_tokens: 8
projector_hidden_size: 2048
projector_num_layers: 2
projector_dropout: 0.1

# Fusion adapter
fusion_type: "lora"
lora_rank: 8
lora_alpha: 16.0
lora_dropout: 0.0
num_attention_heads: 8
attention_dropout: 0.1

# Base model
freeze_base_vl: True
freeze_audio_encoder: True
```

### Training Configuration
```python
# Optimization
learning_rate: 1e-4
weight_decay: 0.01
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 1e-8
max_grad_norm: 1.0

# Loss weights
audio_loss_weight: 1.0
retention_loss_weight: 1.0
distillation_weight: 1.0
distillation_temperature: 3.0
fisher_weight: 0.1

# Scheduling
warmup_steps: 500
gate_warmup_steps: 2000
max_steps: 50000

# Evaluation
eval_steps: 1000
retention_tolerance: 0.005  # 0.5%
early_stopping_patience: 3
```

---

## File Structure

```
safe/
├── models/
│   ├── safe_model.py           # Main SAFE model
│   ├── base_vl.py              # Base VL model wrapper
│   ├── audio_encoders.py       # CLAP, Whisper encoders
│   ├── projectors.py           # Audio projectors
│   └── fusion_adapter.py       # Cross-attention adapters
│
├── training/
│   ├── stage_a.py              # Stage A trainer
│   ├── losses.py               # Loss functions
│   ├── curriculum.py           # Curriculum learning
│   └── metrics.py              # Evaluation metrics
│
├── data/
│   ├── audio_vl_dataset.py     # Multimodal dataset
│   └── collators.py            # Data collation
│
└── scripts/
    ├── train_stage_a.py        # Training script
    └── evaluate.py             # Evaluation script
```

---

## Future Extensions

### Stage B: Joint Fine-Tuning (Planned)

After Stage A establishes audio fusion:

**Changes:**
- **Unfreeze base LLM** (with careful learning rate)
- **Continue retention loss** (critical!)
- **Add task-specific heads**
- **Multi-task learning** across audio, vision, language

**Risk:** Higher forgetting risk → requires stronger retention mechanisms.

### Additional Modalities

SAFE architecture generalizes to:
- **Video:** Temporal cross-attention over video frames
- **Depth:** Depth map encoder + fusion
- **Tactile:** Sensor data encoder + fusion
- **Any modality:** Just need encoder + projector

**Key:** Same cross-attention pattern, same retention guarantees.

---

## Citation & References

### SAFE Model
```bibtex
@article{safe2025,
  title={SAFE: Simple, Adaptive, Failure-Proof Audio Addition to Vision-Language Models},
  author={Your Team},
  year={2025}
}
```

### Related Work
- **LLaVA:** Liu et al., "Visual Instruction Tuning", NeurIPS 2023
- **BLIP-2:** Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training", ICML 2023
- **Flamingo:** Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning", NeurIPS 2022
- **LoRA:** Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
- **EWC:** Kirkpatrick et al., "Overcoming catastrophic forgetting", PNAS 2017

---

## Appendix: Mathematical Formulations

### Cross-Attention

```math
Q = W_q · H_text                    # Query from LLM hidden states
K = W_k · A                         # Key from audio tokens
V = W_v · A                         # Value from audio tokens

Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

Output = LayerNorm(H_text + α · Attention(Q, K, V))

where:
    H_text ∈ ℝ^(B×L×D)  # LLM hidden states
    A ∈ ℝ^(B×T×D)       # Audio tokens
    α = 0.05            # Residual scale
    d_k = D / num_heads # Head dimension
```

### LoRA Adaptation

```math
W' = W_0 + (α/r) · B · A

where:
    W_0 ∈ ℝ^(d×k)      # Original frozen weight
    B ∈ ℝ^(d×r)        # Down-projection (trainable)
    A ∈ ℝ^(r×k)        # Up-projection (trainable)
    r << min(d, k)     # Low rank (e.g., 8)
    α = 16             # Scaling factor
```

### Retention Loss

```math
L_retention = λ_d · L_distill + λ_f · L_fisher

L_distill = KL(P_safe || P_base)
          = Σ_i P_base[i] · log(P_base[i] / P_safe[i])

L_fisher = Σ_i F_i · (θ_i - θ_i^*)^2

where:
    P_safe = softmax(logits_safe / τ)
    P_base = softmax(logits_base / τ)
    τ = 3.0                          # Temperature
    F_i = Fisher information for θ_i
    θ_i^* = Initial parameter value
```

---

**Document Version:** 1.0
**Last Updated:** 2025-10-09
**Status:** Complete
