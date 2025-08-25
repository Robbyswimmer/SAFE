# 🎵 SAFE Stage A Training Guide

This guide shows you how to run Stage A training with different model configurations, from lightweight demo to full production models.

## 🎯 Quick Start

### Demo Training (Recommended for Testing)
```bash
# Lightweight demo - perfect for testing the pipeline
python train_stage_a.py --config demo --num_epochs 2

# Quick demo run
python demo_stage_a.py
```

### Full Production Training
```bash
# Full SAFE model as per paper specification
python train_stage_a.py --config full --num_epochs 5 --batch_size 2
```

### Multimodal Training (CLAP + Whisper)
```bash
# Full multimodal with both audio encoders
python train_stage_a.py --config multimodal --num_epochs 5 --batch_size 1
```

## 📋 Available Configurations

| Config | Description | VRAM | LLM | Vision | Audio | Batch Size |
|--------|-------------|------|-----|---------|-------|------------|
| **demo** | Lightweight testing | 6GB | DialoGPT-small | CLIP-Base | CLAP | 8 |
| **full** | SAFE paper spec | 24GB | LLaVA-7B | CLIP-Large | CLAP | 4 |
| **multimodal** | CLAP + Whisper | 32GB | LLaVA-7B | CLIP-Large | Both | 2 |

```bash
# List all available configurations
python train_stage_a.py --list_configs
```

## 🚀 Training Commands

### 1. Demo Training (Test Setup)
```bash
# Basic demo run
python train_stage_a.py --config demo

# Demo with custom settings
python train_stage_a.py \
    --config demo \
    --batch_size 4 \
    --num_epochs 2 \
    --learning_rate_projector 1e-4 \
    --eval_steps 5
```

### 2. Full Production Training
```bash
# Full SAFE model training
python train_stage_a.py \
    --config full \
    --batch_size 2 \
    --num_epochs 5 \
    --output_dir ./checkpoints/safe_full \
    --save_steps 500 \
    --eval_steps 100

# With real datasets (when available)
python train_stage_a.py \
    --config full \
    --data_path ./datasets \
    --use_dummy_data false \
    --batch_size 4 \
    --num_epochs 10
```

### 3. Multimodal Training
```bash
# CLAP + Whisper combined
python train_stage_a.py \
    --config multimodal \
    --batch_size 1 \
    --num_epochs 3 \
    --learning_rate_projector 5e-5 \
    --learning_rate_adapter 2e-5
```

## 💻 Hardware Requirements

### Demo Configuration
- **VRAM**: 6GB (RTX 3060, RTX 4060, etc.)
- **RAM**: 16GB
- **Training Time**: ~10 minutes (dummy data)

### Full Configuration  
- **VRAM**: 24GB (RTX 3090, RTX 4090, A100, etc.)
- **RAM**: 32GB
- **Training Time**: ~2-8 hours (depending on dataset)

### Multimodal Configuration
- **VRAM**: 32GB (A100, H100)
- **RAM**: 64GB  
- **Training Time**: ~4-12 hours

## 📊 Expected Results

### Demo Training Success Criteria:
```
✓ Model loads without errors
✓ Forward pass completes
✓ Loss decreases over training
✓ Checkpoints save successfully
✓ Audio tokens shape: [batch_size, 8, 768]
```

### Full Training Success Criteria:
```
✓ Retention Score: ≥ 0.995 (≤0.5% VL degradation)
✓ Audio Task Loss: Decreasing trend
✓ Distillation Loss: < 10.0
✓ Trainable Parameters: ~50-100M (projector + LoRA)
✓ Training Speed: ~1-2 samples/second
```

## 🛠️ Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size
python train_stage_a.py --config demo --batch_size 2

# Use gradient accumulation
python train_stage_a.py --config full --batch_size 1

# Switch to demo config
python train_stage_a.py --config demo
```

### Model Download Issues
```bash
# Pre-download models
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')"

# Check internet connection for HuggingFace downloads
```

### Training Crashes
```bash
# Run with minimal setup first
python demo_stage_a.py

# Check CUDA memory
nvidia-smi

# Use CPU for testing
CUDA_VISIBLE_DEVICES="" python train_stage_a.py --config demo
```

## 📁 Output Structure

```
checkpoints/
├── stage_a_demo/           # Demo configuration checkpoints
│   ├── best_checkpoint.pt
│   └── checkpoint_epoch_1_step_100.pt
├── stage_a_full/           # Full configuration checkpoints  
│   ├── best_checkpoint.pt
│   └── checkpoint_epoch_2_step_1000.pt
└── stage_a_multimodal/     # Multimodal configuration checkpoints
    ├── best_checkpoint.pt
    └── checkpoint_epoch_1_step_500.pt
```

## 🔧 Advanced Usage

### Custom Configuration
```python
# Create custom config in configs/model_configs.py
CUSTOM_CONFIG = {
    "name": "custom",
    "llm_model_name": "microsoft/DialoGPT-medium",
    "vision_model_name": "openai/clip-vit-base-patch16", 
    "llm_hidden_size": 1024,
    # ... other settings
}
```

### Resume Training
```bash
# Resume from checkpoint
python train_stage_a.py \
    --config full \
    --resume_from ./checkpoints/stage_a_full/checkpoint_epoch_2_step_1000.pt
```

### Distributed Training
```bash
# Multi-GPU training (when implemented)
torchrun --nproc_per_node=2 train_stage_a.py --config full
```

## 📈 Monitoring

### Weights & Biases Integration
```bash
# Login to wandb
wandb login

# Training will automatically log to wandb
python train_stage_a.py --config full
```

### Key Metrics to Watch
- **Retention Loss**: Should stay low (< 1.0)
- **Audio Task Loss**: Should decrease
- **VL Accuracy**: Should not degrade > 0.5%
- **Training Speed**: samples/second
- **Memory Usage**: VRAM utilization

## 🎯 Next Steps

After successful Stage A training:

1. **Evaluate Results**: Check retention and audio gains
2. **Stage B Training**: Run RL policy optimization
3. **Full Pipeline**: Test complete SAFE system
4. **Real Datasets**: Move from dummy to real data

```bash
# Move to Stage B
python train_stage_b.py --stage_a_checkpoint ./checkpoints/stage_a_full/best_checkpoint.pt
```