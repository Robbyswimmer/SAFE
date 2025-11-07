# Goal (one sentence)

Show that **additive, layer-selective fusion** (frozen encoders + small trainable adapters/gates) (a) **matches or exceeds** state-of-the-art audio captioning, (b) **preserves prior VL skills**, and (c) **extends to an additional modality** with the **same recipe**, all while being more **parameter- and compute-efficient** than strong baselines.

---

# Track A — Publishable benchmark core

## A1. Datasets & splits (lock before running anything)

* **Primary**: AudioCaps official splits; metrics: SPIDEr, SPIDEr-max (rerank), CIDEr, METEOR, BLEU-4, SPICE.
* **Secondary**: Clotho v2 (SPIDEr-FL + classic metrics) for generalization.
* **Retention probe** (for Track C): pick **LLaVA-Instruct v1.5 subset** (1000 Q/A) to measure VL retention. Fix this now so results are comparable.

**Protocol**

* Document sampling rate, clip truncation/padding, text normalization (lowercase, punctuation stripping) in the methods appendix.
* Run **3 seeds** for AudioCaps and Clotho; report mean ± stdev.
* Log inference/reference hashes in the run card for reproducibility.

## A2. Systems to compare (keep the table tight)

1. **Additive fusion** (your best XE+rerank model + SCST variant).
2. **Cross-attention-only** (Flamingo-style per-layer cross-attn, matched trainable params).
3. **Concatenation baseline** (feature concat + projector, same layers as additive).
4. **Partial unfreeze**: unfreeze top **4 LLM blocks** with cross-attn (lock this variant up front; no further sweeps).
5. **Data-only baseline**: backbone + XE trained on AudioCaps (no fusion adapters) to show additive gives more than data.

Keep total trainable params within ±10%; if off, include a param-normalized column and mention in the caption. No other variants go in the main table—extra ideas live in the appendix.

## A3. Decoding, reranking, and SCST

* Default decoding: beam=5, max_new_tokens tuned for AudioCaps short captions; compare beam=3 in appendix if needed.
* **Reranking**: CLAP-based reranker with CIDEr/SPIDEr bonus (already implemented). Report SPIDEr vs SPIDEr-max and Δ.
* **SCST**: run on top-performing XE checkpoints only. Report XE baseline and “+SCST” rows in table to keep attribution clear.

## A4. Success criteria

* AudioCaps: SPIDEr ≥ **0.50** (≥0.52 post-SCST ideal), SPIDEr-max ≥0.53, METEOR ≥0.24.
* Clotho: SPIDEr-FL ≥ **0.33**.
* Retention probe: Δ accuracy ≥ −0.5 percentage points after audio tuning.

## A5. Tables & plots (minimal set)

* **T1**: AudioCaps main results (mean ± stdev, XE and XE+SCST versions).
* **T2**: Clotho generalization.
* **T3**: Decoding vs reranking (AudioCaps, additive fusion only).
* **F1**: Training/validation SPIDEr vs epoch for key models.
* **F2**: Bootstrap CI plot for SPIDEr (additive vs best baseline).

---

# Track B — Focused ablations (only what’s needed to isolate additive fusion)

## B1. Layer selection + gate behavior (single sweep)

* Compare fusion sites: {12}, {24}, {12,24}. (Skip {6} to save time unless reviewers ask.)
* For each, log mean |α| per layer + SPIDEr to show where additive helps.

## B2. Token & projector budget (Pareto curve)

* Audio tokens: 8 vs 16 (skip 32 unless 16 underperforms).
* Projector hidden: 512 vs 768.
* Plot SPIDEr vs trainable params and include data-only and cross-attn points for context.

## B3. Loss shaping / decoding ablation

* XE only vs XE+contrastive (CLAP) for additive fusion.
* XE vs XE+contrastive for cross-attn baseline (show additive benefits aren’t only due to the auxiliary loss).
* Include reranker on/off for additive to isolate decoding effects.

These three ablations cover attribution without exploding runtime. Gate init experiments move to appendix if time permits.

---

# Track C — Retention and modality scaling narrative

## C1. Retention on VL task (LLaVA subset)

* Evaluate chosen Q/A subset before audio training, after audio XE, and after SCST.
* Table: Before / After XE / After SCST + Δ.
* Plot: “Stability frontier” — Audio SPIDEr vs VL accuracy across checkpoints.

## C2. Second modality pilot (same recipe)

* Add a minimal video branch (frozen TimeSformer → projector → additive gates at {12,24}).
* Train on a **small** AV-caption or AVQA subset (e.g., 5k samples) with the same training loop.
* Report the video metric (accuracy or CIDEr) + confirm audio/VL retention within tolerance.

This establishes “arbitrary modalities” without chasing SOTA on video.

---

# Track D — Efficiency, significance, polish

## D1. Efficiency accounting

* Record trainable params, FLOPs per forward (approx), GPU-hours to hit SPIDEr 0.50, throughput (samples/sec @ batch size N) for additive and cross-attn baselines.
* Plot SPIDEr vs trainable params (Pareto) with main systems.

## D2. Significance and reproducibility

* Bootstrap (10k resamples) SPIDEr and METEOR for additive vs best baseline; report 95% CIs and paired p-values.
* Provide mean ± stdev over seeds in the main tables.
* Export a JSON run card (dataset hashes, encoder checkpoints, fusion layers, tokens, loss weights, decoding settings, SCST flags) and predictions.json for AudioCaps/Clotho.

---

# Training schedule (prioritized & parallelizable)

**Phase 0 – Sanity (1 day)**
* Verify α=0 reproduces backbone retention results.
* Overfit 1k AudioCaps to ensure loss drops and SPIDEr climbs.

**Phase 1 – Core benchmarks (3–5 days)**
* Train systems in A2 (XE + contrastive) with shared hyperparams.
* For additive fusion, run reranker + SCST once plateau detected (already integrated into training loop).
* Fill T1/T2/T3 with XE and XE+SCST rows. Log retention probe after XE and post-SCST.

**Phase 2 – Key ablations (3 days)**
* Run B1 and B2 sweeps (shortened schedules, early stop on val SPIDEr).
* Run B3 loss/reranker ablations using best layer/token setting.

**Phase 3 – Retention & second modality (3 days)**
* Execute C1 retention measurements (pre, post-XE, post-SCST).
* Train the minimal video branch (C2) and log its metric + retention.

**Phase 4 – Polish & stats (1–2 days)**
* Bootstrap metrics, build Pareto/stability plots, generate run cards/prediction dumps.
* Draft methods/experiments sections with finalized tables/figures.

Note: Begin B1/B2 ablations once the additive XE run is stable—no need to wait for all Phase 1 baselines to finish.

---

# Hyperparameters (baseline defaults)

* Optimizer: AdamW; lr 2e-4 (projector) / 1e-4 (adapters); warmup 3%; cosine decay.
* Batch: target effective 512 sequences via grad accumulation.
* Regularization: weight decay 0.01 on adapters; gate L2 1e-4.
* Gate init α=0.05 scalar per layer.
* Audio tokens=16, projector d=768 for main run.
* Early stop patience 2 evals on val SPIDEr (still keep SCST trigger independent).

---

# Contingency checklist

1. **Additive underperforms cross-attn** → add narrow cross-attn alongside additive (hybrid) or residual alignment penalty; rerun key comparison.
2. **Retention regression** → reduce fusion layers ({24} only) and increase gate regularization; re-measure C1.
3. **Reranker drives all gains** → keep both XE and reranked numbers; report candidate diversity (self-BLEU) to show reranker effect.
4. **Training instability** → halve LR, increase gradient clip, drop audio tokens to 8.

---

# Minimal figure/tables for the paper

* **Fig.1** Architecture schematic.
* **Fig.2** SPIDEr vs trainable params (Pareto).
* **Fig.3** Stability frontier (Audio SPIDEr vs VL retention).
* **Fig.4** Gate magnitude vs layer/epoch (from B1).
* **Tbl.1** AudioCaps results (XE & SCST).
* **Tbl.2** Clotho results.
* **Tbl.3** Ablation summary (layer/token/contrastive).
* **Tbl.4** Decoding vs reranking.

---

# “Are we publishable?” checklist

* [ ] AudioCaps SPIDEr ≥ 0.50 (≥0.52 with SCST) and SPIDEr-max uplift ≥ +0.02.
* [ ] Clotho SPIDEr-FL ≥ 0.33.
* [ ] Retention drop ≤ 0.5 pts on LLaVA probe.
* [ ] Second modality (video) added with same recipe, with positive gain and no regressions.
* [ ] Additive beats cross-attn/partial-unfreeze on SPIDEr-vs-params Pareto curve.
* [ ] Core ablations demonstrate additive contribution beyond data/decoding.
