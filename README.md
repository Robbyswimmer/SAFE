# SAFE: Safe Audio Fusion Extension

Adding audio capabilities to frozen vision-language models.

---

## Overview

SAFE adds audio understanding to pre-trained VL models (LLaVA) by training lightweight adapters while keeping the base model frozen.

**Core idea**: Cross-attention fusion between audio tokens and LLM hidden states, trained on audio captioning.

## Architecture

```
Audio (waveform) → CLAP Encoder (frozen) → Audio Projector (trainable)
                                                      ↓
Text + Vision → LLaVA 13B (frozen) ←── Cross-Attention Fusion (trainable)
                                                      ↓
                                              Audio Captions
```

**Trainable components** (~380M parameters):
- Audio projector (CLAP embeddings → LLM tokens)
- LoRA fusion adapter (cross-attention at multiple LLM layers)

**Frozen components** (~13B parameters):
- LLaVA 13B base model
- CLAP audio encoder

## Current Status

**Phase 1: Signal Verification** (In Progress)

Training audio captioning on AudioCaps dataset.

**Goal**: Verify frozen LLM can learn to attend to audio
**Target**: CIDEr > 30 (baseline: 18)
**Config**: 32 audio tokens, LoRA rank 64, multi-layer fusion

## Quick Start

### Training

```bash
# Run Phase 1 training
sbatch scripts/train_phase1.sh

# Or locally
python train_safe.py \
    --model-config phase1 \
    --data-path ./data \
    --output-dir ./checkpoints/phase1 \
    --num-epochs 20 \
    --batch-size 4 \
    --gradient-accumulation-steps 32 \
    --fp16
```

### Evaluation

```bash
# Evaluate checkpoint
python train_safe.py \
    --model-config phase1 \
    --data-path ./data \
    --output-dir ./eval \
    --resume ./checkpoints/phase1/checkpoint_best.pt \
    --eval-only
```

## Project Structure

```
safe/
├── models/
│   ├── safe_model.py       # Main SAFE model
│   ├── audio_encoders.py   # CLAP/Whisper encoders
│   ├── projectors.py       # Audio projectors
│   └── fusion_adapter.py   # Cross-attention fusion
├── data/
│   └── datasets.py         # AudioCaps, WavCaps, AudioSetCaps
└── training/
    ├── losses.py           # Loss functions
    └── stage_a.py          # Training utilities

train_safe.py               # Main training script (937 lines)
scripts/train_phase1.sh     # SLURM launcher
```

## Documentation

- **[QUICK_START.md](QUICK_START.md)** - Run training in 1 command
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete CLI reference
- **[SUMMARY.md](SUMMARY.md)** - Project status and research questions
- **[architecture.md](architecture.md)** - Detailed architecture spec

## Requirements

```bash
# Core dependencies
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0

# Evaluation metrics
pycocoevalcap
bert-score (optional)

# Audio processing
torchaudio
librosa
```

## Training Configuration

Phase 1 defaults (optimized for single GPU):

```python
{
    "model": "phase1",              # LLaVA 13B + CLAP
    "num_audio_tokens": 32,         # Audio sequence length
    "lora_rank": 64,                # Cross-attention rank
    "batch_size": 4,
    "gradient_accumulation": 32,    # Effective batch size: 128
    "lr_projector": 1e-3,           # 5x higher than baseline
    "lr_adapter": 5e-4,             # 5x higher than baseline
    "mixed_precision": True,        # FP16 training
    "epochs": 20
}
```

**Expected**: 12-18 hours, ~22GB VRAM, CIDEr 32-35 after 20 epochs

## Metrics

**Audio Captioning**: CIDEr, BLEU-1/2/3/4, METEOR, ROUGE-L

**Evaluation**: Beam search generation on AudioCaps validation set

## Citation

```bibtex
@software{safe2025,
  title={SAFE: Safe Audio Fusion Extension for Vision-Language Models},
  author={Moseley, Robby},
  year={2025},
  url={https://github.com/yourusername/SAFE}
}
```

---

**Status**: Experimental - Phase 1 training in progress
