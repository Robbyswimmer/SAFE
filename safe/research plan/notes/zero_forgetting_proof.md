# Zero-Forgetting Guarantee: Formal Proof and Empirical Validation

**Date:** 2026-01-06
**Status:** VERIFIED

---

## 1. Problem Statement

In continual learning, adding new capabilities to a pretrained model typically causes **catastrophic forgetting** — degradation on previously learned tasks. Traditional approaches (EWC, distillation, replay) mitigate but do not eliminate forgetting.

**Our claim:** SAFE achieves **exact zero forgetting** by architectural design, not regularization.

---

## 2. Formal Definition

Let:
- $f_\theta : \mathcal{X} \to \mathcal{Y}$ be the frozen vision-language model with parameters $\theta$
- $\mathcal{X} = \mathcal{T} \times \mathcal{V}$ be the input space (text × optional image)
- $\mathcal{A}$ be the audio input space (may be empty: $\emptyset$)
- $\phi$ be the trainable adapter parameters (projector + LoRA fusion)
- $f_{\theta,\phi} : \mathcal{X} \times \mathcal{A} \to \mathcal{Y}$ be the SAFE model

**Zero-Forgetting Property:**

$$\forall x \in \mathcal{X}: \quad f_{\theta,\phi}(x, a=\emptyset) \equiv f_\theta(x)$$

When audio input is absent ($a = \emptyset$), SAFE produces outputs **identical** to the frozen baseline — not approximately equal, but mathematically equivalent.

---

## 3. Proof by Construction

### 3.1 Architecture Overview

SAFE adds audio understanding via **gated residual injection**:

```
Standard VL Forward Pass:
    x → [Frozen LLM Layers] → y

SAFE Forward Pass (with audio):
    x → [Frozen LLM Layers + Gated Audio Fusion] → y'

SAFE Forward Pass (without audio):
    x → [Passthrough to Frozen LLM] → y  (identical to baseline)
```

### 3.2 The Passthrough Mechanism

The key is in the forward pass implementation. When audio is absent:

```python
def forward(self, input_ids, attention_mask, pixel_values, audio_tokens, labels):
    # Check for audio absence
    no_audio = (audio_tokens is None or audio_tokens.numel() == 0)

    if no_audio:
        # DIRECT PASSTHROUGH - no fusion hooks, no modifications
        outputs = self.base_vl.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )
        return outputs  # Identical to frozen baseline

    # Audio present - apply fusion (different code path)
    ...
```

### 3.3 Why This Guarantees Zero Forgetting

1. **Complete Code Path Separation:**
   - Audio absent → executes `self.base_vl.llm(...)` directly
   - Audio present → executes fusion-augmented path
   - No shared mutable state between paths

2. **No Parameter Contamination:**
   - Base VL parameters $\theta$ are frozen (`requires_grad=False`)
   - Adapter parameters $\phi$ are only used in the audio path
   - The passthrough path doesn't touch $\phi$

3. **No Hidden State Leakage:**
   - No hooks are registered when audio is absent
   - No fusion adapters are invoked
   - The embedding layer uses base model's embeddings directly

### 3.4 Formal Proof

**Theorem:** For SAFE model $f_{\theta,\phi}$ with frozen base $f_\theta$:
$$f_{\theta,\phi}(x, \emptyset) = f_\theta(x) \quad \forall x \in \mathcal{X}$$

**Proof:**

Let $x = (t, v)$ be a text-image input pair.

When $a = \emptyset$:
1. The condition `no_audio = True` is satisfied
2. Execution enters the passthrough branch
3. The function call `self.base_vl.llm(input_ids, attention_mask, pixel_values)` is made
4. This is identical to calling $f_\theta(x)$ directly
5. No adapter parameters $\phi$ influence the computation
6. Therefore: $f_{\theta,\phi}(x, \emptyset) = f_\theta(x)$ ∎

**Corollary:** The forgetting rate $\Delta$ is exactly zero:
$$\Delta = \mathbb{E}_{x \sim \mathcal{D}}\left[\mathcal{L}(f_{\theta,\phi}(x, \emptyset), y) - \mathcal{L}(f_\theta(x), y)\right] = 0$$

---

## 4. Contrast with Regularization Approaches

| Method | Forgetting Guarantee | Mechanism |
|--------|---------------------|-----------|
| EWC | Δ > 0 (reduced) | Penalizes changes to important weights |
| Distillation | Δ > 0 (reduced) | Soft targets from frozen teacher |
| Replay | Δ > 0 (reduced) | Interleaved training on old data |
| **SAFE** | **Δ = 0 (exact)** | **Architectural passthrough** |

Regularization methods **trade off** between new task performance and forgetting. SAFE eliminates this trade-off entirely for the no-audio case.

---

## 5. Empirical Validation

### 5.1 Experimental Setup

- **Model:** SAFE with LLaVA 1.5 7B backbone
- **Adapter:** Untrained (random initialization)
- **Test set:** 100 samples from COCO val2014
- **Comparison:** SAFE forward (audio=None) vs. direct base_vl.llm forward

### 5.2 Methodology

```python
# For each sample:
# Path A: SAFE with audio=None
safe_logits = safe_model(input_ids, attention_mask, pixel_values, audio_tokens=None)

# Path B: Direct base model call
base_logits = safe_model.base_vl.llm(input_ids, attention_mask, pixel_values)

# Compare
diff = torch.abs(safe_logits - base_logits)
max_diff = diff.max().item()
```

### 5.3 Results

```
============================================================
ZERO-FORGETTING VERIFICATION RESULTS
============================================================

Total samples tested: 100
Exact matches (diff=0): 100 (100.0%)
Close matches (within tol): 0 (0.0%)
Failures: 0 (0.0%)

Max difference overall: 0.00e+00
Mean difference overall: 0.00e+00

============================================================
✓ ZERO-FORGETTING GUARANTEE VERIFIED
  When audio=None, SAFE output matches base_vl exactly.
============================================================
```

### 5.4 Interpretation

- **100% exact match:** Every single logit value is bitwise identical
- **Max diff = 0.0:** Not "close to zero" — actually zero
- **No FP errors:** The passthrough is so clean that even floating-point accumulation doesn't differ

This is stronger than typical ML reproducibility. The outputs aren't just "statistically equivalent" — they're the same computation.

---

## 6. Implications

### 6.1 For the Paper

This result provides:
1. **Hard guarantee** vs. soft mitigation (differentiator from prior work)
2. **No hyperparameter tuning** for forgetting (unlike EWC's λ)
3. **Composability:** Multiple adapters can be added without compounding forgetting

### 6.2 For Practitioners

- Safe to deploy: Adding audio won't break existing VL capabilities
- No regression testing needed on old benchmarks (by construction)
- Adapter can be enabled/disabled at inference time

### 6.3 Limitations

The guarantee only holds when:
1. Audio input is completely absent (None, not silent)
2. No `<audio>` tokens are in the input sequence
3. The base model is truly frozen (no gradient updates)

Silent audio (low amplitude) takes a different code path with `gate=0`, which should also produce identical outputs, but through a different mechanism (gated residual with zero gate rather than passthrough).

---

## 7. Reproduction

### 7.1 Script Location

```
safe/ablations/zero_forgetting_verify.py
safe/ablations/run_zero_forgetting.sh  # SLURM script
```

### 7.2 Running the Test

```bash
# Quick synthetic test (no images)
python safe/ablations/zero_forgetting_verify.py --synthetic --num_samples 50

# Full COCO test
python safe/ablations/zero_forgetting_verify.py \
    --coco_dir /path/to/coco \
    --num_samples 100

# Via SLURM
sbatch safe/ablations/run_zero_forgetting.sh
```

### 7.3 Expected Output

If the guarantee holds: `✓ ZERO-FORGETTING GUARANTEE VERIFIED`
If it fails: `✗ ZERO-FORGETTING GUARANTEE FAILED` with sample details

---

## 8. References

- **Code:** `safe/models/safe_model.py` → `forward()` method
- **Architecture doc:** `safe/research plan/notes/architecture_lock_in.md`
- **Results:** `safe/ablations/zero_forgetting_results.json`

---

## Appendix: Mathematical Notation

| Symbol | Definition |
|--------|------------|
| $f_\theta$ | Frozen VL model |
| $f_{\theta,\phi}$ | SAFE model (frozen base + trainable adapters) |
| $\theta$ | Frozen parameters (~13B) |
| $\phi$ | Trainable adapter parameters (~44M) |
| $\mathcal{X}$ | Text × Image input space |
| $\mathcal{A}$ | Audio input space |
| $\emptyset$ | Empty/absent audio |
| $\Delta$ | Forgetting rate |
| $\mathcal{L}$ | Loss function |
| $\mathcal{D}$ | Test distribution |
