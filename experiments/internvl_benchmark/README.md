# InternVL 3.5-8B Benchmark

InternVL 3.5-8B as a stronger frozen LVLM backbone for SAFE audio adapter experiments.

## Architecture

InternVL 3.5-8B consists of:
- **Vision encoder:** InternViT-300M (built-in, no separate CLIP needed)
- **MLP connector:** `mlp1` bridging vision to language
- **LLM backbone:** Qwen3-8B (32 transformer layers, hidden_size=4096)

SAFE hooks into the Qwen3 layers via `LayerHookManager`, injecting audio fusion adapters at layers [10, 19, 29] (early/mid/late). The vision encoder and LLM remain frozen; only SAFE's audio projector and fusion adapters are trained.

## Setup

### 1. Download Model

```bash
bash experiments/internvl_benchmark/scripts/download_internvl.sh
# Small backbones for scaling studies:
bash experiments/internvl_benchmark/scripts/download_internvl_1b.sh
bash experiments/internvl_benchmark/scripts/download_internvl_3b.sh
```

Or directly:
```bash
python scripts/download_internvl.py --model internvl3.5-8b --output-dir models
# 1B
python scripts/download_internvl.py --model internvl3.5-1b --output-dir models
# "3B" alias (maps to 4B; InternVL 3.5 has no official 3B release)
python scripts/download_internvl.py --model internvl3.5-3b --output-dir models
```

### 2. Verify Download

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("models/OpenGVLab_InternVL3_5-8B", trust_remote_code=True)
print(f"Layers: {len(model.language_model.model.layers)}")  # Expected: 32
print(f"Hidden size: {model.config.llm_config.hidden_size}")  # Expected: 4096
```

### 3. Train (Audio-Only)

```bash
# SLURM
sbatch experiments/internvl_benchmark/scripts/train_audio.sh

# Local
bash experiments/internvl_benchmark/scripts/train_audio.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_PATH` | `models/OpenGVLab_InternVL3_5-8B` | Path to InternVL weights |
| `SAFE_QWEN_QUANT` | `none` | Quantization (`none`, `4bit`, `8bit`) |
| `SAFE_GRAD_CKPT` | `0` | Gradient checkpointing |
| `SAFE_PREFER_FLASH2` | `1` | Flash Attention 2 |

## Memory Requirements

- **bf16 (no quant):** ~35GB VRAM
- **4-bit quantized:** ~12GB VRAM
- Batch size 1 recommended with gradient accumulation 16

## Key Differences from Qwen-Only Config

- Loads via `AutoModel` (not `AutoModelForCausalLM`) due to InternVL's multimodal wrapper
- `trust_remote_code=True` required for custom `InternVLChatModel` classes
- Has built-in vision encoder (InternViT-300M) for future audio+vision composition
- Same Qwen3-8B backbone, so fusion layer indices [10, 19, 29] are identical
