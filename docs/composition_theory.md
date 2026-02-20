# Modality Adapter Composition: Theory and Diagnostics

## 1. Motivation

Large vision-language models (LVLMs) are extended to new modalities (audio, 3D, etc.) by attaching lightweight adapters to a frozen backbone. A natural question arises: if we train an audio adapter and a vision adapter **independently**, do they compose at inference time? That is, can we activate both adapters simultaneously and get correct audio-visual answers without ever training on paired audio-visual data?

Empirically, the answer is: **sometimes, but unreliably.** We observe:

- **InternVL binding experiment (epoch 1):** Audio adapter trained for 1 epoch on audio-only data. At eval, activating frozen InternVL vision alongside the audio adapter yields `both=64.3% > audio=63.6%` -- positive synergy (+0.7pp). The frozen backbone appears to bind modalities without explicit joint training.

- **InternVL binding experiment (epoch 9):** Same setup, longer training. Now `both=53.0%` while `audio=66.4%` and `image=57.7%` -- massive negative synergy (-13.4pp). Longer audio training destroyed the composition that existed at epoch 1.

- **LoRA baseline (Stage 2):** Audio LoRA merged into weights, vision LoRA trained on image-only data. Result: `both=69.7% < image=71.4%` -- the audio adapter actively *hurts* when combined with vision (-1.7pp).

These results demonstrate that composition is not guaranteed and degrades with training. **Why?** And can we predict, measure, and correct this interference?


## 2. The Interference Equation

### 2.1 Setup

Let `s(theta)` be a scalar score function (e.g., accuracy) evaluated at model parameters `theta`. Consider a frozen backbone with parameters `u`, and two adapter perturbations:

- `delta_a`: parameter change induced by the audio adapter
- `delta_v`: parameter change induced by the vision adapter

When both adapters are active, the effective parameters are `u + delta_a + delta_v`.

> **Note on "parameter perturbation" for cross-attention adapters:** For architectures like MICA where adapters inject via cross-attention (not by modifying weights), `delta_a` and `delta_v` represent the *functional* perturbation to the hidden state computation, not literal weight changes. The Taylor expansion applies to the function composition, not just weight space. The key property is that each adapter's effect is additive in hidden-state space: `h' = h + adapter_a(h) + adapter_v(h)`.


### 2.2 Second-Order Taylor Expansion

Expand `s` around the unperturbed backbone `u`:

```
s(u + delta_a + delta_v)  =  s(u)
                            + grad_s(u)^T delta_a
                            + grad_s(u)^T delta_v
                            + (1/2) delta_a^T H delta_a
                            + (1/2) delta_v^T H delta_v
                            + delta_a^T H delta_v
                            + O(||delta||^3)
```

where `H = nabla^2 s(u)` is the Hessian of the score at the unperturbed point.

### 2.3 Identifying Known Quantities

Recognize that the single-adapter scores have their own Taylor expansions:

```
s(u + delta_a)  ~=  s(u) + grad_s^T delta_a + (1/2) delta_a^T H delta_a
s(u + delta_v)  ~=  s(u) + grad_s^T delta_v + (1/2) delta_v^T H delta_v
```

Substituting back:

```
s(u + delta_a + delta_v)  ~=  s(u + delta_a) + s(u + delta_v) - s(u) + delta_a^T H delta_v
```

### 2.4 The Synergy Equation

Rearranging:

```
                                    ┌─────────────────────┐
s(u + delta_a + delta_v)            │                     │
  - s(u + delta_a)          =      │  delta_a^T H delta_v│
  - s(u + delta_v)                  │                     │
  + s(u)                            └─────────────────────┘
```

Or in terms of observable accuracy values:

```
synergy  =  y_av - y_a - y_v + y_0  =  delta_a^T H delta_v  +  O(||delta||^3)
```

where:
- `y_0`  = text-only baseline (no adapters)
- `y_a`  = audio adapter only
- `y_v`  = vision adapter only
- `y_av` = both adapters active

**This is the central result.** The synergy metric -- the interaction effect in a 2x2 factorial design -- equals the Hessian coupling between adapter perturbations, to second order. It is directly measurable from four eval runs.

### 2.5 Interpretation of the Coupling Term

The term `delta_a^T H delta_v` captures how the *curvature* of the score landscape mediates interference between adapters:

| Sign of `delta_a^T H delta_v` | Meaning |
|-------------------------------|---------|
| **= 0** | Perfect composition. Adapters don't interact. Superposition holds. |
| **> 0** | Positive synergy. Adapters reinforce each other through shared curvature. |
| **< 0** | Negative synergy (interference). Adapters degrade each other. |

Crucially, this term depends on:
1. The **directions** of `delta_a` and `delta_v` in parameter/function space
2. The **curvature** `H` of the score landscape at the operating point

Two adapters can each be individually excellent (`s(u + delta_a)` high, `s(u + delta_v)` high) yet still interfere if their perturbation directions couple through negative curvature.


## 3. From Parameters to Hidden States

### 3.1 The Non-Additivity Residual

The Hessian `H` is not directly observable (it lives in parameter space and is astronomically large). However, its effects manifest in **hidden-state space**, which we can measure.

At each layer `l` of the backbone, define the hidden-state shift caused by each adapter:

```
Delta_a^l  =  h_l(audio) - h_l(text)       # audio adapter's effect at layer l
Delta_v^l  =  h_l(image) - h_l(text)       # vision adapter's effect at layer l
Delta_av^l =  h_l(both)  - h_l(text)       # composed effect at layer l
```

If composition were perfectly additive (superposition), we would have:

```
Delta_av^l  =  Delta_a^l + Delta_v^l        # (superposition)
```

The **non-additivity residual** measures deviation from superposition:

```
epsilon_l  =  Delta_av^l - Delta_a^l - Delta_v^l
           =  h_l(both) - h_l(audio) - h_l(image) + h_l(text)
```

This is the hidden-state analog of the output-level synergy metric. Its norm `||epsilon_l||` at each layer quantifies where in the network composition breaks down.

### 3.2 Connecting Residual to Hessian

The non-additivity residual `epsilon_l` is the layer-wise manifestation of the Hessian coupling term `delta_a^T H delta_v`. When the Hessian coupling is zero, `epsilon_l = 0` at all layers. When it is nonzero, `epsilon_l` reveals **which layers** contribute most to the interference and **how large** the effect is relative to the individual adapter shifts.

We normalize to get a scale-free measure:

```
rho_l  =  ||epsilon_l|| / (||Delta_a^l|| + ||Delta_v^l||)
```

A value of `rho_l = 0` means perfect additivity. Values approaching 1 indicate severe non-linear interaction.


## 4. Subspace Geometry: Why Composition Fails

### 4.1 The Overlap Decomposition

Measuring `||epsilon_l||` tells us **how much** composition fails at each layer. To understand **why**, we analyze the geometry of the adapter shift subspaces.

For each layer `l`, collect the hidden-state shifts `{Delta_a^l}` and `{Delta_v^l}` across many samples. Apply SVD to get low-rank subspace bases:

```
U_a^l  =  top-k left singular vectors of  [Delta_a^l]       (rank-k audio subspace)
U_v^l  =  top-k left singular vectors of  [Delta_v^l]       (rank-k vision subspace)
```

Now decompose the vision shift into components parallel and perpendicular to the audio subspace:

```
Delta_v_parallel  =  U_a (U_a^T Delta_v)       # vision shift projected onto audio subspace
Delta_v_perp      =  Delta_v - Delta_v_parallel  # orthogonal component
```

### 4.2 The Overlap Scale m*

If composition were additive, the composed shift would be:

```
Delta_av  =  Delta_a + Delta_v_perp + 1.0 * Delta_v_parallel
```

In practice, the optimal reconstruction is:

```
Delta_av  ~=  Delta_a + Delta_v_perp + m* * Delta_v_parallel
```

where `m*` is fitted by least squares to minimize the reconstruction error. This **overlap scale** has a clean interpretation:

| Value of `m*` | Interpretation |
|---------------|----------------|
| `m* = 1` | Additive in the overlap subspace. The parallel component composes correctly. |
| `m* < 1` | Over-coupling. The overlap component is *suppressed* during composition (interference). |
| `m* > 1` | Under-coupling. The overlap component is *amplified* during composition. |
| `m* = 0` | Complete cancellation of the overlap component. |

### 4.3 Additional Subspace Metrics

**Principal cosines** between `U_a` and `U_v`:
```
sigma_i  =  singular values of  (U_a^T U_v)
```
These measure the alignment between the two subspaces. `sigma_max` close to 1 means the subspaces share a nearly identical direction. All `sigma_i` close to 0 means the subspaces are nearly orthogonal.

**Vision-parallel ratio:**
```
r_parallel  =  mean_samples( ||Delta_v_parallel|| / ||Delta_v|| )
```
The fraction of the vision shift that lives in the audio subspace. High values mean the adapters operate in overlapping regions of hidden-state space.


## 5. What We Measure (Composability Diagnostics)

Given independently trained audio and vision adapter checkpoints, the diagnostic runs **one-time** (not per training step) and produces:

### 5.1 Output-Level Metrics
Run four eval conditions on the validation set:

| Condition | Adapters Active | Score |
|-----------|----------------|-------|
| text | none | `y_0` |
| audio | audio only | `y_a` |
| image | vision only | `y_v` |
| both | audio + vision | `y_av` |

Derived:
- **Synergy** = `y_av - y_a - y_v + y_0` (the Hessian coupling term)
- **Gain vs best single** = `y_av - max(y_a, y_v)` (practical composition benefit)

### 5.2 Per-Layer Representation Diagnostics
For each fusion layer `l`:

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| Audio shift norm | `||Delta_a^l||` | Magnitude of audio adapter's effect |
| Vision shift norm | `||Delta_v^l||` | Magnitude of vision adapter's effect |
| Both shift norm | `||Delta_av^l||` | Magnitude of composed effect |
| Additivity residual | `||epsilon_l||` | How much composition deviates from superposition |
| Normalized add. error | `||epsilon_l|| / (||Delta_a|| + ||Delta_v||)` | Scale-free non-additivity |
| Cosine(Delta_a, Delta_v) | Direction alignment of raw shifts | |
| Subspace overlap (Frobenius) | `||U_a^T U_v||_F / sqrt(k)` | Overall subspace alignment |
| Principal cosine (max) | `sigma_max(U_a^T U_v)` | Maximum shared direction |
| Principal cosine (mean) | `mean(sigma_i)` | Average subspace alignment |
| Vision-parallel ratio | `||proj_{U_a}(Delta_v)|| / ||Delta_v||` | Fraction of vision in audio subspace |
| Overlap scale m* | Fitted from least-squares | Effective scaling of overlap component |
| Residual improvement | `(||eps_naive|| - ||eps_m*||) / ||eps_naive||` | How much m* correction helps |

### 5.3 Automatic Direction Recommendation

Based on the per-layer `m*` and parallel ratio, the diagnostic classifies the composition regime:

- **m* > 1.15**: Under-coupled. Adapters are too orthogonal; shared directions need amplification.
- **m* < 0.85**: Over-coupled. Adapters interfere in shared directions; need more orthogonality.
- **parallel_ratio < 0.10**: Nearly disjoint. Very little shared structure between adapters.
- **Otherwise**: Balanced. Focus on calibration rather than geometric changes.


## 6. What We Hope to Find

### 6.1 Validating the Theory

1. **Synergy = Hessian coupling (approximately).** The output-level synergy metric should correlate with the sum of per-layer `||epsilon_l||`. If composition fails (negative synergy), we should see large residuals concentrated at specific layers.

2. **Interference is localized.** We expect the non-additivity residual to be unevenly distributed across layers, not uniform. This would indicate that specific layers are "bottleneck" layers for composition, which is actionable.

3. **m* explains the interference.** If the residual improvement from the m* correction is high (say >50%), then most of the non-additivity is captured by a simple scaling of the overlap component. This means the fix is low-dimensional.

### 6.2 Diagnostic-Guided Correction

The diagnostic directly suggests three levels of correction:

**Level 0 -- Free (no retraining):**
Apply per-layer `m*` correction at inference time. During composition, decompose the vision shift into audio-parallel and audio-perpendicular components, and scale the parallel component by `m*`:
```
h' = h + Delta_a + Delta_v_perp + m*_l * Delta_v_parallel
```
This uses values read directly from the diagnostic. Cost: zero additional training.

**Level 1 -- Cheap calibration (~100 paired samples):**
Learn per-layer scalars `alpha_l` (initialized to `m*_l`) on a small paired dataset:
```
alpha_l = nn.Parameter(tensor(m_star_l))    # 1 scalar per fusion layer
```
This is ~6 parameters total for 6 fusion layers. Not joint training -- it's fitting 6 numbers.

**Level 2 -- Informed sequential training (no paired data):**
After training the audio adapter:
1. Extract `U_a^l` per layer (one-time, ~500 samples)
2. Train the vision adapter with an orthogonality-aware regularizer at layers where `m*` indicates interference:
```
L = L_task + sum_l  lambda_l * ||proj_{U_a^l}(Delta_v^l)||^2
```
The regularizer strength `lambda_l` is set per layer based on the diagnostic. This is still independent training -- the vision adapter never sees audio data. It just avoids the audio subspace where doing so would cause interference.

### 6.3 Expected Outcomes Table

| Correction Level | Paired Data Needed | Parameters | Expected Synergy Change |
|------------------|--------------------|------------|------------------------|
| None (naive add) | 0 | 0 | Negative (baseline) |
| m* correction | ~500 (diagnostic) | 0 | Less negative / ~zero |
| Learned alpha_l | ~100 (calibration) | 6 | Near zero / slightly positive |
| Orthogonal reg. | 0 (only U_a extraction) | 0 extra | Reduced interference |
| Full joint training | All paired | All adapter params | Positive (upper bound) |

The key claim: **diagnostic-guided correction can close most of the gap between naive composition and full joint training, at a fraction of the cost.**


## 7. Connection to Prior Work

The synergy metric `y_av - y_a - y_v + y_0` is equivalent to:
- The **interaction effect** in a 2x2 factorial experiment (statistics)
- The **Shapley interaction index** for two players (cooperative game theory)
- The **synergy** term in Partial Information Decomposition (Liang et al., NeurIPS 2023)

The key distinction: PID measures **data-level** synergy (a model-agnostic ceiling on how much the joint signal exceeds the sum of marginals). Our analysis measures **adapter-level** composability (whether a specific model architecture captures that synergy). The data synergy exists; the question is whether the model's adapter composition recovers it. The gap between PID synergy and observed synergy is the **composition deficit**, and the non-additivity residual explains where in the network this deficit arises.
