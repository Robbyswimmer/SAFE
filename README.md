# SAFE: Simple, Adaptive, Failure-Proof Audio Addition to a VL Model

🎵 **SAFE** is a research implementation that adds audio capabilities to Vision-Language (VL) models while preserving VL performance through careful architectural design and training procedures.

## 🎯 Key Features

- **🔄 Modular Architecture**: Clean separation of audio encoding, projection, and fusion components
- **🛡️ Retention Guarantees**: Preserves base VL performance (≤0.3-0.5% degradation) 
- **🎯 Adaptive Usage**: RL-based policy learns when to consult audio (40-70% efficiency)
- **⚡ Efficient Training**: LoRA-based parameter updates with minimal compute overhead
- **🧠 Smart Gating**: Controllable audio fusion with gate ∈ [0,1] mechanism

## 🏗️ Architecture Overview

```
Input: Text + Vision + Audio
         ↓
    ┌─────────────────┐
    │   Base VL Model │  ← Frozen (CLIP + LLM)
    │   (LLaVA-style) │
    └─────────────────┘
         ↓
    ┌─────────────────┐
    │ Audio Encoder   │  ← Frozen (CLAP/Whisper)
    │ (CLAP/Whisper)  │
    └─────────────────┘
         ↓
    ┌─────────────────┐
    │ Audio Projector │  ← Trainable (2-layer MLP)
    │ (k audio tokens)│
    └─────────────────┘
         ↓
    ┌─────────────────┐
    │ LoRA Fusion     │  ← Trainable (Cross-attention)
    │ Adapter         │
    └─────────────────┘
         ↓
    ┌─────────────────┐
    │ RL Controller   │  ← Learns when to use audio
    │ Policy π_θ      │
    └─────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/safe
cd safe

# Create virtual environment (Python 3.11 recommended)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install SAFE package in development mode
pip install -e .
```

### Basic Usage

```python
import torch
from safe.models.projectors import AudioProjector
from safe.models.fusion_adapter import CrossAttentionBlock
from safe.rl.policy import AudioPolicyNetwork

# Create audio projector
projector = AudioProjector(
    audio_embed_dim=512,      # CLAP embedding dimension
    llm_hidden_size=768,      # LLM hidden dimension  
    num_audio_tokens=8        # Audio tokens to generate
)

# Create fusion adapter
fusion = CrossAttentionBlock(
    hidden_size=768,
    num_attention_heads=12
)

# Create RL policy
policy = AudioPolicyNetwork(
    state_feature_dim=256,
    token_budgets=[0, 4, 8, 12]  # Token budget options
)

# Process audio features
audio_features = torch.randn(2, 512)  # Batch of CLAP features
audio_tokens = projector(audio_features)

# Fuse with LLM hidden states
llm_states = torch.randn(2, 10, 768)
fused_states = fusion(llm_states, audio_tokens)

# Get policy decisions
state_features = torch.randn(2, 256)
actions = policy.get_actions(state_features)
```

### Running Tests

```bash
# Run basic functionality tests
python test_basic_functionality.py

# Run integration tests  
python test_integration.py

# Run usage example
python example_usage.py
```

## 📚 Training Pipeline

SAFE uses a 3-stage training approach:

### Stage A: Supervised Warm-start
- **Objective**: Train projector + LoRA adapter while keeping base models frozen
- **Data**: 50% audio-dependent, 50% VL-only (balanced batches)
- **Loss**: New-task loss + Retention loss (KL distillation + Fisher regularization)

```python
from safe.training.stage_a import StageATrainer

trainer = StageATrainer(
    safe_model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    config={
        "learning_rate_projector": 1e-4,
        "learning_rate_adapter": 5e-5,
        "retention_tolerance": 0.005  # 0.5% VL degradation limit
    }
)
trainer.train()
```

### Stage B: RL Policy Learning
- **Objective**: Learn when and how to use audio efficiently
- **Method**: Contextual bandit with retention constraints
- **Reward**: `r = Score - α·LatencyCost - γ·IrrelevancePenalty`

### Stage C: Policy Distillation (Optional)
- **Objective**: Distill RL policy into lightweight gate for deployment
- **Method**: Supervised learning on (state → action) pairs

## 🎯 Success Criteria

| Metric | Target | Description |
|--------|--------|-------------|
| **Retention** | ΔVQAv2/GQA ≤ 0.3-0.5% | Preserve base VL performance |
| **Audio Gains** | +5-10% | Improvement on audio-dependent tasks |
| **Efficiency** | 40-70% skip rate | Examples that skip audio processing |
| **Robustness** | No hallucinations | Stable on audio-irrelevant queries |

## 📁 Project Structure

```
safe/
├── models/
│   ├── base_vl.py          # LLaVA-style base VL model
│   ├── audio_encoders.py   # CLAP/Whisper audio encoders
│   ├── projectors.py       # Audio-to-LLM token projectors
│   ├── fusion_adapter.py   # LoRA cross-attention fusion
│   └── safe_model.py       # Main SAFE model
├── data/
│   └── datasets.py         # Multi-modal dataset handling
├── rl/
│   ├── state_features.py   # State feature extraction
│   └── policy.py           # RL policy networks
├── training/
│   ├── losses.py           # Loss functions
│   └── stage_a.py          # Stage A trainer
└── README.md
```

## 🔬 Research Paper

> **SAFE: Simple, Adaptive, Failure-Proof Audio Addition to a VL Model**  
> *Authors: [Your Name], et al.*  
> *Conference: [Venue] 2024*  
> *Paper: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)*

## 📊 Datasets

SAFE is designed to work with:

- **Audio-dependent**: AVQA, AudioCaps, Clotho, VGGSound
- **VL retention**: VQAv2, GQA, COCO Captions
- **Mixed training**: Balanced sampling for optimal learning

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LLaVA** team for the base VL architecture inspiration
- **CLAP** and **Whisper** teams for excellent audio foundation models
- **LoRA/PEFT** authors for parameter-efficient fine-tuning methods

## 📞 Contact

For questions about this implementation:
- 📧 Email: [your.email@domain.com]
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/safe/issues)
- 📖 Docs: [Documentation](https://your-repo.github.io/safe)

---

*Built with ❤️ for multimodal AI research*