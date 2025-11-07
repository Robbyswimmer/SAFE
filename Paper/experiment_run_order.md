# Experiment Execution Order

This run order balances publication-critical benchmarks with time-saving parallelism. Finish each block before moving to the next unless otherwise noted.

1. **Phase 0 – Sanity & Infrastructure (Day 0–1)**
   - Verify α=0 run reproduces backbone retention metrics on the fixed LLaVA QA probe.
   - Overfit ~1k AudioCaps samples (XE only) to confirm loss drops and SPIDEr climbs.
   - Smoke-test decoding/reranker + SCST hooks on a tiny subset.

2. **Phase 1 – Core Benchmarks (Day 1–5)**
   - Train AudioCaps XE+contrastive models for:
     1. Additive fusion (primary system).
     2. Cross-attention-only baseline.
     3. Concatenation baseline.
     4. Partial-unfreeze (top 4 LLM blocks) baseline.
     5. Data-only baseline (no fusion adapters).
   - Enable reranker for additive fusion once val SPIDEr stabilizes.
   - Trigger SCST automatically after plateau on the additive model; capture XE vs XE+SCST checkpoints and metrics.
   - Replicate top additive + cross-attention runs on Clotho v2 (same seeds) once AudioCaps training completes.

3. **Phase 2 – Key Ablations (starts Day 3, finish by Day 7)**
   - Layer selection sweep: {12}, {24}, {12,24} for additive fusion (shortened schedules, early stop on val SPIDEr).
   - Token/projector grid: tokens 8 vs 16, projector dim 512 vs 768 (reuse best layer set).
   - Loss/reranker ablation: additive XE-only vs XE+contrastive; reranker ON/OFF. Mirror XE vs XE+contrastive on cross-attention baseline.
   - Log gate magnitudes during the above runs for later plots (no separate jobs).

4. **Phase 3 – Retention & Modality Scaling (Day 6–9)**
   - Measure LLaVA retention: before audio training, after additive XE, after additive SCST.
   - Train minimal video branch (frozen TimeSformer + additive gates at {12,24}) on chosen AV subset; report video metric + post-training retention check.
   - If video run regresses audio/VL metrics beyond tolerance, adjust gate regularization before proceeding.

5. **Phase 4 – Statistical Polishing & Packaging (Day 9–10)**
   - Bootstrap (10k resamples) SPIDEr/METEOR for additive vs best baseline; compute paired p-values.
   - Aggregate seed means/stdevs for all core tables.
   - Generate Pareto and stability plots (SPIDEr vs params, retention frontier) using completed runs.
   - Export run cards + predictions.json for AudioCaps, Clotho, and video pilot; archive SCST finetuned checkpoint in `experiments/full_training/finetuned/`.

6. **Stretch / Appendix (only if time allows)**
   - Additional gate init experiments or robustness curves (SNR) leveraging trained checkpoints.
   - Candidate diversity metrics (self-BLEU) for reranked outputs.

> Tip: Begin Phase 2 layer/parameter sweeps as soon as the additive XE run is stable—no need to wait for every Phase 1 baseline to finish.
