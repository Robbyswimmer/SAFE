# Source Index for NeurIPS

Last updated: 2026-03-17

This directory tracks legacy materials from `paper/eccv2026` and `paper/theory` that are still useful for the NeurIPS paper.

## Archived Result Sources

### Qwen3 MUSIC-AVQA legacy results

Copied here:
- [qwen_legacy_results.md](/Users/robbymoseley/CascadeProjects/SAFE/Neurips/References/qwen_legacy_results.md)

Originals:
- [09_composition_theory.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/reference_notes/09_composition_theory.md)
- [10_composition_experiment.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/reference_notes/10_composition_experiment.md)
- [11_updated_results_and_positioning.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/reference_notes/11_updated_results_and_positioning.md)
- [01_composition_formalism.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/theory/01_composition_formalism.md)
- [same_vs_staggered_slide.py](/Users/robbymoseley/CascadeProjects/SAFE/paper/figures/same_vs_staggered_slide.py)
- [composition_training_comparison.py](/Users/robbymoseley/CascadeProjects/SAFE/paper/figures/composition_training_comparison.py)
- [composition_experiments_slide.py](/Users/robbymoseley/CascadeProjects/SAFE/paper/figures/composition_experiments_slide.py)

Why it matters:
- preserves the negative-composition Qwen baseline
- preserves the stronger archived `73.x` vision-only Qwen figure-script values
- separates adapter numbers from the LoRA `71.4` image baseline

### MUSIC-AVQA scaling summary

Copied here:
- [eccv_music_avqa_scaling_results.md](/Users/robbymoseley/CascadeProjects/SAFE/Neurips/References/eccv_music_avqa_scaling_results.md)

Originals:
- [music_avqa_scaling_results.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/results/music_avqa_scaling_results.md)
- [music_avqa_scaling_results.json](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/results/music_avqa_scaling_results.json)
- [music_avqa_scaling_results.csv](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/results/music_avqa_scaling_results.csv)

Why it matters:
- gives archived benchmark anchors for LLaVA and InternVL on MUSIC-AVQA
- useful as legacy comparison context while Qwen runs are in progress

Key archived values:
- LLaVA 1.5 13B: both `69.75`
- InternVL 3.5 1B: both `77.08`
- InternVL 3.5 4B: both `79.50`
- InternVL 3.5 8B: both `80.35`

## Legacy Paper Notes Worth Reusing

### Composition experiment framing

Original:
- [10_composition_experiment.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/reference_notes/10_composition_experiment.md)

Useful content:
- symmetric composition framing on text-only Qwen
- 4-way evaluation protocol: `text`, `audio`, `vision`, `both`
- definitions for `gain_vs_best_single` and `synergy`

Warning:
- contains older conclusions that assume some configurations are already closed
- treat as archived reasoning, not current source of truth
- use [qwen_legacy_results.md](/Users/robbymoseley/CascadeProjects/SAFE/Neurips/References/qwen_legacy_results.md) as the NeurIPS-facing summary instead of re-reading these raw notes every time

### Updated results and positioning

Original:
- [11_updated_results_and_positioning.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/reference_notes/11_updated_results_and_positioning.md)

Useful content:
- archived InternVL numbers
- competitive positioning notes
- framing ideas for frozen-backbone claims

Warning:
- dated note
- some conclusions are ECCV-specific and should not be copied directly into the NeurIPS draft

## Theory Notes Worth Reusing

### Composition formalism

Original:
- [01_composition_formalism.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/theory/01_composition_formalism.md)

Most reusable parts:
- gate-scaling intuition
- synergy definition
- staggered vs same-layer placement discussion

Warning:
- several statements are stronger than current evidence supports
- for NeurIPS, use this as hypothesis/mechanism language, not theorem language

### Additivity probe

Original:
- [02_additivity_probe.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/theory/02_additivity_probe.md)

Most reusable parts:
- additivity error definition
- layer ranking concept

### Layer placement from gradients

Original:
- [03_layer_placement_from_gradients.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/theory/03_layer_placement_from_gradients.md)

Most reusable parts:
- practical rationale for selecting fusion layers
- combined criterion idea: gradients + additivity

Why it matters now:
- directly supports the staggered-vs-shared Qwen experiment design

## Figure / Artifact References

Potentially useful archived figures:
- [mavqa_modality_composition_v2.png](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/figures/mavqa_modality_composition_v2.png)
- [mavqa_per_question_type.png](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/figures/mavqa_per_question_type.png)
- [scaling_composition_bars.png](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/figures/scaling_composition_bars.png)
- [math_slides.pdf](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/advisor_slides/math_slides.pdf)

Use policy:
- these are references only
- if reused for NeurIPS, re-render or relabel them for the new paper identity
