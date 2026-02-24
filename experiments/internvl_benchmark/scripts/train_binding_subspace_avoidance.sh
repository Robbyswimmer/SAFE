#!/bin/bash
#SBATCH --job-name=internvl-bind-sa
#SBATCH --output=logs/internvl_binding_sa_%j.out
#SBATCH --error=logs/internvl_binding_sa_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#
# InternVL 3.5-8B Modality Binding — Reverse Subspace Avoidance
#
# Builds on the audio-only binding experiment by adding a reverse
# compatibility regularizer: collect InternVL's built-in vision
# activation patterns at audio fusion layers, then penalize audio
# adapter shifts that project onto them.
#
# This prevents the audio adapter from interfering with the frozen
# vision pathway, fixing the binding failure observed in the baseline
# audio-only experiment (epoch 9: -13.4pp synergy).
#
# Protocol:
#   - Collect: vision shift subspaces U_v at audio fusion layers
#   - Train:   audio adapter with penalty ||U_v^T delta_a||^2
#   - Eval:    text-only, audio-only, image-only, audio+image
#
# Usage:
#   sbatch experiments/internvl_benchmark/scripts/train_binding_subspace_avoidance.sh

set -euo pipefail

# ---- Cluster paths ----
SAFE_ROOT="${SAFE_ROOT:-/data/SalmanAsif/RobbyMoseley/SAFE/SAFE}"
if [[ ! -d "$SAFE_ROOT" ]]; then
  echo "ERROR: SAFE_ROOT does not exist: $SAFE_ROOT" >&2
  exit 1
fi

# ---- Conda activation ----
CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# ---- InternVL-specific env vars ----
export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export FP16=0
export LLM_MODEL_PATH=${LLM_MODEL_PATH:-models/OpenGVLab_InternVL3_5-8B}

# ---- GPU sharding setup ----
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

# ---- Model configuration ----
MODEL_CONFIG=${MODEL_CONFIG:-internvl_binding}

# ---- KEY: audio-only training, full eval ----
TRAIN_MODALITY=${TRAIN_MODALITY:-audio}
EVAL_MODALITIES=${EVAL_MODALITIES:-"text,audio,image,both"}

# ---- Training configuration ----
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$SAFE_ROOT/data/music_avqa/manifests/train.jsonl}
VAL_MANIFEST=${VAL_MANIFEST:-$SAFE_ROOT/data/music_avqa/manifests/validation.jsonl}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT/data/music_avqa}
OUTPUT_DIR=${OUTPUT_DIR:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/binding_audio_subspace_avoidance}
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

# ---- Architectural features ----
LEARNED_GATE=${LEARNED_GATE:-0}
LEARNED_GATE_INIT=${LEARNED_GATE_INIT:-1.5}
GRAD_ATTRIBUTION=${GRAD_ATTRIBUTION:-1}
GRAD_LOG_EVERY=${GRAD_LOG_EVERY:-200}
MAX_SAMPLES=${MAX_SAMPLES:-0}
SLIM_PROJECTOR=${SLIM_PROJECTOR:-1}

# ---- Reverse subspace avoidance configuration ----
COMPAT_REG_REVERSE=${COMPAT_REG_REVERSE:-1}
COMPAT_REG_ENABLE=${COMPAT_REG_ENABLE:-1}
COMPAT_REG_LAMBDA=${COMPAT_REG_LAMBDA:-0.05}
COMPAT_REG_RANK=${COMPAT_REG_RANK:-8}
COMPAT_REG_AUDIO_SAMPLES=${COMPAT_REG_AUDIO_SAMPLES:-256}
COMPAT_REG_REFRESH_EVERY=${COMPAT_REG_REFRESH_EVERY:-1}

# ---- W&B settings ----
WANDB_PROJECT=${WANDB_PROJECT:-"SAFE-InternVL-Binding"}
WANDB_NAME="internvl_binding_sa_${SLURM_JOB_ID:-local}"
WANDB_TAGS=${WANDB_TAGS:-"internvl,binding,audio-only,reverse-compat,subspace-avoidance"}

echo "========================================"
echo "InternVL 3.5-8B Binding — Subspace Avoidance"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "SAFE root: $SAFE_ROOT"
echo "Model path: $LLM_MODEL_PATH"
echo "Model config: $MODEL_CONFIG"
echo "SAFE_DEVICE_MAP: ${SAFE_DEVICE_MAP}"
echo "SAFE_MAX_MEMORY: ${SAFE_MAX_MEMORY}"
echo "========================================"
echo "TRAIN modality: $TRAIN_MODALITY  (audio only - no images)"
echo "EVAL modalities: $EVAL_MODALITIES  (includes vision for binding test)"
echo "========================================"
echo "Reverse compat reg: ON"
echo "  lambda=$COMPAT_REG_LAMBDA rank=$COMPAT_REG_RANK"
echo "  collection samples=$COMPAT_REG_AUDIO_SAMPLES refresh_every=$COMPAT_REG_REFRESH_EVERY"
echo "========================================"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "LR scheduler: $LR_SCHEDULER (warmup=$WARMUP_RATIO, min_lr=$MIN_LR_RATIO)"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Fusion layers: $FUSION_LAYERS"
echo "Audio tokens: $NUM_AUDIO_TOKENS"
echo "Fusion gate: $FUSION_GATE"
echo "Seed: $SEED"
echo "========================================"

# ---- Create directories and cd ----
cd "$SAFE_ROOT"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs
mkdir -p "$SAFE_OFFLOAD_FOLDER"

# ---- Diagnostics ----
echo "[diag] timestamp=$(date -Iseconds)"
echo "[diag] hostname=$(hostname)"
echo "[diag] slurm_job_id=${SLURM_JOB_ID:-none}"
nvidia-smi -L || true

# ---- CUDA check ----
python3 -c "import torch,sys; ok=torch.cuda.is_available() and torch.cuda.device_count()>0; print(f'[cuda_check] available={torch.cuda.is_available()} count={torch.cuda.device_count()}'); sys.exit(0 if ok else 2)"

# ---- Build optional flags ----
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
    echo "Slim projector: OFF"
else
    echo "Slim projector: ON (projector outputs at bottleneck_dim=756)"
fi

# ---- Reverse subspace avoidance flags ----
if [ "$COMPAT_REG_REVERSE" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --compat-reg-reverse"
    EXTRA_FLAGS="$EXTRA_FLAGS --compat-reg-enable"
    EXTRA_FLAGS="$EXTRA_FLAGS --compat-reg-lambda $COMPAT_REG_LAMBDA"
    EXTRA_FLAGS="$EXTRA_FLAGS --compat-reg-rank $COMPAT_REG_RANK"
    EXTRA_FLAGS="$EXTRA_FLAGS --compat-reg-audio-samples $COMPAT_REG_AUDIO_SAMPLES"
    EXTRA_FLAGS="$EXTRA_FLAGS --compat-reg-refresh-every $COMPAT_REG_REFRESH_EVERY"
    echo "Reverse subspace avoidance: ON"
fi

# ---- Run training ----
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
echo "Subspace Avoidance Binding Experiment Complete!"
echo "========================================"
echo "Output saved to: $OUTPUT_DIR"
echo ""
echo "Key results to check:"
echo "  - text-only:  baseline (should match base InternVL)"
echo "  - audio-only: trained adapter performance"
echo "  - image-only: InternVL built-in vision (should be preserved)"
echo "  - both:       BINDING TEST - audio+vision composition should improve"
echo ""
echo "Compare vs baseline binding (train_binding_audio_only.sh):"
echo "  - both accuracy should be higher (audio no longer corrupts vision)"
echo "  - compat_reg values in logs should be non-zero"
echo "========================================"
