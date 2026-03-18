# Archived Qwen3 MUSIC-AVQA Results

Last updated: 2026-03-17

This file records legacy Qwen3-8B MUSIC-AVQA numbers that are relevant to the NeurIPS paper. Only locally sourced values are listed here.

## Canonical sourced notes: same-layer independent training

These values are repeated consistently across:
- [paper/eccv2026/reference_notes/09_composition_theory.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/reference_notes/09_composition_theory.md)
- [paper/eccv2026/reference_notes/10_composition_experiment.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/reference_notes/10_composition_experiment.md)
- [paper/eccv2026/reference_notes/11_updated_results_and_positioning.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/reference_notes/11_updated_results_and_positioning.md)
- [paper/theory/01_composition_formalism.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/theory/01_composition_formalism.md)

Pilot / epoch-1 same-layer result:

| Setting | Text | Audio | Vision | Both | Gain vs Best Single |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B, same-layer, independent unimodal training | 27.86 | 64.72 | 65.92 | 53.21 | -12.71 |

Full-data same-layer result:

| Setting | Text | Audio | Vision | Both | Gain vs Best Single |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B, same-layer, independent unimodal training | 22.96 | 58.80 | 64.48 | 59.30 | -5.18 |

Interpretation:
- these are the strongest fully corroborated local Qwen interference numbers
- they establish the core negative-composition result for independently trained adapters on a frozen text-only Qwen hub

## Best archived staggered-layer Qwen result

Best archived figure-script source:
- [paper/figures/same_vs_staggered_slide.py](/Users/robbymoseley/CascadeProjects/SAFE/paper/figures/same_vs_staggered_slide.py)
- [paper/figures/composition_training_comparison.py](/Users/robbymoseley/CascadeProjects/SAFE/paper/figures/composition_training_comparison.py)

Values:
- audio `66.88`
- vision `73.95`
- both `71.12`
- gain vs best single `-2.83`

Additional archived corroboration:
- [paper/figures/composition_experiments_slide.py](/Users/robbymoseley/CascadeProjects/SAFE/paper/figures/composition_experiments_slide.py)
- audio `67.16`
- vision `73.17`
- both `70.51`
- gain vs best single `-2.66`

Recommended NeurIPS usage:
- use `64.48 / 58.80 / 59.30` as the conservative, fully corroborated same-layer interference baseline
- use `vision=73.95` as the best archived unimodal Qwen vision value
- use `both=71.12` as the paired archived staggered composition reference from the same figure-script source

## LoRA comparison note

This is a separate baseline and should not be conflated with the Qwen adapter results.

Source:
- [docs/composition_theory.md](/Users/robbymoseley/CascadeProjects/SAFE/docs/composition_theory.md)

Values:
- image `71.4`
- both `69.7`

Interpretation:
- the `71.4` value is a LoRA image result, not the Qwen adapter vision-only result
