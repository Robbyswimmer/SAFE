# Archived MUSIC-AVQA Scaling Results

Copied from:
- [music_avqa_scaling_results.md](/Users/robbymoseley/CascadeProjects/SAFE/paper/eccv2026/results/music_avqa_scaling_results.md)

Last copied: 2026-03-17

## Canonical MUSIC-AVQA Scaling Results

Source of truth in ECCV workspace:
- `paper/figures/scaling_composition_bars.py`

| Backbone | Audio-only | Image-only | Both | Composition Gain |
|----------|-----------:|-----------:|-----:|-----------------:|
| LLaVA 1.5 13B | 52.55 | 52.56 | 69.75 | +17.19 |
| InternVL 3.5 1B | 56.04 | 55.15 | 77.08 | +21.04 |
| InternVL 3.5 4B | 56.63 | 57.17 | 79.50 | +22.33 |
| InternVL 3.5 8B | 58.17 | 57.08 | 80.35 | +22.18 |

## Notes for NeurIPS

- Treat these as archived reference baselines, not current headline results.
- They are useful for:
  - historical comparison,
  - motivating why composition is plausible,
  - showing that frozen-backbone adapter composition can work strongly on some backbones.
- The NeurIPS primary claim should center on Qwen incremental modality acquisition, not these older backbone results by themselves.
