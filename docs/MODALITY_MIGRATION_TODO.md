# Modality Migration TODO

This checklist tracks outstanding work for expanding SAFE beyond audio.

## Core Architecture
- [x] Generalize `SAFEModel.forward` to iterate over registered modalities rather than hard-coded audio/video branches (`safe/models/safe_model.py`).
- [x] Move modality-specific metadata (batch keys, token ids) into `ModalityComponents.metadata` and register them for audio/video (`safe/models/safe_model.py:120-161`).
- [x] Update fusion adapters to accept modality-agnostic inputs and remove audio naming assumptions (`safe/models/fusion_adapter.py`).
- [x] Extend hook manager to respect registry modality names instead of defaulting to `"audio"` when a list is provided (`safe/models/layer_hooks.py:166-173`).
- [x] Ensure tokenizer/special-token handling works for arbitrary modalities (e.g., temporal tokens for video) (`safe/models/safe_model.py:193-227`).

- [x] Build `StageATrainer.modality_settings` dynamically from the registry, including gate warmup and masks (`safe/training/stage_a.py:136-174`).
- [x] Support per-modality loss factories configured via trainer config with sane defaults for audio/video (`safe/training/stage_a.py:533-591`).
- [x] Allow curriculum/metrics to register modality-specific accuracy + retention targets automatically (`safe/training/stage_a.py:3209-3234`).
- [x] Add evaluator support for multi-modality baseline comparisons (gate=0 vs. original VL, plus modality-off baselines) (`safe/training/stage_a.py:1989-2012`).

- [x] Generalize `_collate_multimodal_batch` to emit `has_<modality>` flags and payloads based on a configurable schema (`safe/data/datasets.py:43-94`).
- [x] Propagate modality schema into dataset validators and curriculum samplers (`safe/data/validation.py:459-520`, `tests/fixtures/mock_datasets.py:424-520`).
- [x] Add loaders/transforms for future modalities (e.g., depth, tactile) with consistent interface (`safe/data/video_sampling.py`, `safe/models/encoders`).

## Configuration
- [x] Replace static audio/video config keys with declarative modality entries (train-time and model-time) (`configs/model_configs.py:55-128`).
- [x] Surface modality registry options via CLI flags (`train_stage_a_curriculum.py:320-368`).

- [x] Implement efficiency/usage metrics per modality (`safe/training/stage_a.py:1827-1840, 3230-3235`).
- [x] Record per-modality gate statistics and usage histograms during training/eval (`safe/training/stage_a.py`, `safe/models/safe_model.py`).
- [x] Add reporting hooks comparing SAFE vs. base VL outputs on modality-disabled runs (`safe/training/stage_a.py:1989-2012`).

## Testing
- [x] Add unit tests covering registry-driven modality configuration (`tests/unit` or new suite).
- [x] Extend integration tests to simulate an additional synthetic modality via mock dataset (`tests/integration/test_modality_flow.py`).
- [x] Add regression tests ensuring gate=0 path matches base VL logits for all registered modalities (`tests/unit/test_stage_a_gate.py`).

---

*Keep this document up to date as migration tasks land.*
