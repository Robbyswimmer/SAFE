#!/bin/bash
#SBATCH --job-name=internvl-va
#SBATCH --output=logs/internvl_vision_audio_%j.out
#SBATCH --error=logs/internvl_vision_audio_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# InternVL 3.5-8B Vision+Audio Training with SAFE
#
# Uses InternVL 3.5-8B (InternViT-300M + Qwen3-8B) as frozen backbone,
# with built-in InternVL vision pipeline active alongside SAFE audio adapters.
# This enables full generative MUSIC-AVQA with both video frames and audio.
#
# Usage:
#   sbatch experiments/internvl_benchmark/scripts/train_avqa_vision_audio.sh
#   # Or locally:
#   bash experiments/internvl_benchmark/scripts/train_avqa_vision_audio.sh

set -euo pipefail

# Get SAFE root
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SAFE_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

# Conda environment
CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# InternVL-specific env vars
export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export FP16=0
export LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-8B}

# Derive visible GPU count so sharding config matches Slurm allocation.
GPU_COUNT=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _gpu_arr <<< "${CUDA_VISIBLE_DEVICES}"
    GPU_COUNT=${#_gpu_arr[@]}
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    if [[ "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
        GPU_COUNT=${SLURM_GPUS_ON_NODE}
    else
        GPU_COUNT=$(echo "${SLURM_GPUS_ON_NODE}" | grep -o '[0-9]\+' | head -n1 || echo 0)
    fi
fi
if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" -le 0 ]]; then
    GPU_COUNT=1
fi

if [[ -z "${SAFE_DEVICE_MAP:-}" ]]; then
    if [[ "${GPU_COUNT}" -le 1 ]]; then
        export SAFE_DEVICE_MAP=none
    else
        export SAFE_DEVICE_MAP=auto
    fi
fi

if [[ -z "${SAFE_MAX_MEMORY:-}" ]]; then
    SAFE_PER_GPU_MEMORY=${SAFE_PER_GPU_MEMORY:-46GiB}
    SAFE_CPU_MEMORY=${SAFE_CPU_MEMORY:-160GiB}
    _mem_entries=()
    for ((i=0; i<GPU_COUNT; i++)); do
        _mem_entries+=("${i}=${SAFE_PER_GPU_MEMORY}")
    done
    export SAFE_MAX_MEMORY="$(IFS=,; echo "${_mem_entries[*]}"),cpu=${SAFE_CPU_MEMORY}"
fi

export SAFE_OFFLOAD_FOLDER=${SAFE_OFFLOAD_FOLDER:-$SAFE_ROOT/.hf_offload}

# Model configuration
MODEL_CONFIG=${MODEL_CONFIG:-internvl}

# Modality: train on both vision+audio, evaluate on all combinations
TRAIN_MODALITY=${TRAIN_MODALITY:-both}
EVAL_MODALITIES=${EVAL_MODALITIES:-"both,audio,image"}

# Training configuration
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$SAFE_ROOT/data/music_avqa/manifests/train.jsonl}
VAL_MANIFEST=${VAL_MANIFEST:-$SAFE_ROOT/data/music_avqa/manifests/validation.jsonl}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT/data/music_avqa}
OUTPUT_DIR=${OUTPUT_DIR:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/vision_audio}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-8}
FUSION_LAYERS=${FUSION_LAYERS:-"1,5,9,13,17,21"}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}
FUSION_GATE=${FUSION_GATE:-0.2}
SEED=${SEED:-42}

# Architectural features
LEARNED_GATE=${LEARNED_GATE:-0}          # 1 to enable per-layer learned gating
LEARNED_GATE_INIT=${LEARNED_GATE_INIT:-1.5}
GRAD_ATTRIBUTION=${GRAD_ATTRIBUTION:-1}  # 1 to log per-layer gradient norms
GRAD_LOG_EVERY=${GRAD_LOG_EVERY:-200}
MAX_SAMPLES=${MAX_SAMPLES:-0}             # >0 to limit samples for sanity runs
SLIM_PROJECTOR=${SLIM_PROJECTOR:-1}       # 0 to disable slim projector (default ON, ~80% fewer params)

# W&B settings
WANDB_PROJECT=${WANDB_PROJECT:-"SAFE-InternVL-AVQA"}
WANDB_NAME="internvl_va_${SLURM_JOB_ID:-local}"
WANDB_TAGS=${WANDB_TAGS:-"internvl,vision+audio,avqa"}

echo "========================================"
echo "InternVL 3.5-8B Vision+Audio AVQA Training"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "SAFE root: $SAFE_ROOT"
echo "Model path: $LLM_MODEL_PATH"
echo "Model config: $MODEL_CONFIG"
echo "SAFE_DEVICE_MAP: ${SAFE_DEVICE_MAP}"
echo "SAFE_MAX_MEMORY: ${SAFE_MAX_MEMORY}"
echo "Train modality: $TRAIN_MODALITY"
echo "Eval modalities: $EVAL_MODALITIES"
echo "========================================"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "LR scheduler: $LR_SCHEDULER (warmup_ratio=$WARMUP_RATIO, min_lr_ratio=$MIN_LR_RATIO)"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Fusion layers: $FUSION_LAYERS"
echo "Audio tokens: $NUM_AUDIO_TOKENS"
echo "Fusion gate: $FUSION_GATE"
echo "Seed: $SEED"
echo "========================================"

cd "$SAFE_ROOT"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs
mkdir -p "$SAFE_OFFLOAD_FOLDER"

DIAG_LOGS=${DIAG_LOGS:-1}
if [[ "$DIAG_LOGS" == "1" ]]; then
    echo "[diag] timestamp=$(date -Iseconds)"
    echo "[diag] hostname=$(hostname)"
    echo "[diag] slurm_job_id=${SLURM_JOB_ID:-none}"
    echo "[diag] slurm_node=${SLURMD_NODENAME:-unknown}"
    echo "[diag] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "[diag] python=$(which python3)"
    echo "[diag] conda_env=${CONDA_DEFAULT_ENV:-unset}"
    nvidia-smi -L || true
    nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader || true
    python3 - <<'PY' || true
import os, sys, platform
import torch
print(f"[diag] python_version={sys.version.split()[0]} platform={platform.platform()}")
print(f"[diag] torch={torch.__version__} torch_cuda={torch.version.cuda} cuda_built={torch.backends.cuda.is_built()}")
print(f"[diag] torch_is_available={torch.cuda.is_available()} torch_device_count={torch.cuda.device_count()}")
print(f"[diag] env_CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
PY
fi

# Optional: fail fast on known bad nodes (does not replace scheduler-level --exclude)
BAD_NODES=${BAD_NODES:-}
if [[ -n "${BAD_NODES}" && -n "${SLURMD_NODENAME:-}" ]]; then
    IFS=',' read -r -a _bad_nodes_arr <<< "${BAD_NODES}"
    for _bad in "${_bad_nodes_arr[@]}"; do
        if [[ "${SLURMD_NODENAME}" == "${_bad}" ]]; then
            echo "FATAL: landed on excluded node ${SLURMD_NODENAME} (BAD_NODES=${BAD_NODES})."
            echo "Resubmit with: sbatch --exclude=${BAD_NODES} ..."
            exit 3
        fi
    done
fi

# Build optional flags
REQUIRE_CUDA=${REQUIRE_CUDA:-1}
echo "[launcher] mode=direct-python (nested srun disabled)"
if [[ "$REQUIRE_CUDA" == "1" ]]; then
    nvidia-smi -L || true
    python3 - <<'PY' || { echo "FATAL: No CUDA GPUs available (GPU_COUNT=$GPU_COUNT). Aborting."; exit 2; }
import os
import sys
import torch

err = None
runtime_ok = False
try:
    if torch.cuda.device_count() > 0:
        torch.cuda.init()
        x = torch.tensor([1.0], device="cuda:0")
        runtime_ok = bool(x.is_cuda)
except Exception as e:
    err = repr(e)

print(
    f"[cuda_check] torch={torch.__version__} built_cuda={torch.version.cuda} "
    f"built={torch.backends.cuda.is_built()} available={torch.cuda.is_available()} "
    f"count={torch.cuda.device_count()} visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
    f"runtime_ok={runtime_ok} err={err}",
    flush=True,
)
if runtime_ok:
    try:
        props = torch.cuda.get_device_properties(0)
        print(
            f"[cuda_check] device0 name={props.name} total_mem_gb={props.total_memory / (1024**3):.2f}",
            flush=True,
        )
    except Exception as e:
        print(f"[cuda_check] warning: failed to query device properties: {e!r}", flush=True)
sys.exit(0 if runtime_ok else 2)
PY
fi

# Build optional flags
EXTRA_FLAGS=""
if [ "$LEARNED_GATE" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --learned-gate --learned-gate-init $LEARNED_GATE_INIT"
    echo "Learned gating: ON (init=$LEARNED_GATE_INIT)"
fi
if [ "$GRAD_ATTRIBUTION" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --grad-attribution --grad-log-every $GRAD_LOG_EVERY"
    echo "Gradient attribution: ON (every $GRAD_LOG_EVERY steps)"
fi
if [ "$MAX_SAMPLES" != "0" ] && [ -n "$MAX_SAMPLES" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --max-samples $MAX_SAMPLES"
    echo "Max samples: $MAX_SAMPLES (sanity run)"
fi
if [ "$SLIM_PROJECTOR" = "0" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --no-slim-projector"
    echo "Slim projector: OFF (projector outputs at full llm_hidden_size)"
else
    echo "Slim projector: ON (default, projector outputs at bottleneck_dim)"
fi

python3 experiments/avqa_composition/train_avqa_composition.py \
    --dataset music_avqa \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --media-root "$MEDIA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --model-config "$MODEL_CONFIG" \
    --train-modality "$TRAIN_MODALITY" \
    --eval-modalities "$EVAL_MODALITIES" \
    --fusion-layers "$FUSION_LAYERS" \
    --num-audio-tokens "$NUM_AUDIO_TOKENS" \
    --fusion-gate "$FUSION_GATE" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --lr-scheduler "$LR_SCHEDULER" \
    --warmup-ratio "$WARMUP_RATIO" \
    --min-lr-ratio "$MIN_LR_RATIO" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION" \
    --freeze-audio-encoder \
    --num-workers 2 \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_NAME" \
    --wandb-tags "$WANDB_TAGS" \
    $EXTRA_FLAGS

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================"
