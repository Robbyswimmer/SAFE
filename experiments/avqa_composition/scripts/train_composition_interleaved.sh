#!/bin/bash
#SBATCH --job-name=comp-interleaved
#SBATCH --output=logs/composition_interleaved_%j.out
#SBATCH --error=logs/composition_interleaved_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
# pass --gres=gpu:1 at submit time

# Composition experiment: Interleaved audio + vision training on Qwen3-8B
# Each epoch: train audio adapters → train vision adapters → evaluate 3 conditions:
#   1. text + audio (audio adapter only)
#   2. text + vision (vision adapter only)
#   3. text + audio + vision (composition)
# Tracks composition gain = accuracy(both) - max(audio, vision) per epoch.

set -euo pipefail

if [[ -z "${SAFE_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    SAFE_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SAFE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
  fi
fi

CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

MODEL_CONFIG=${MODEL_CONFIG:-composition_study}
DATA_ROOT=${DATA_ROOT:-data/music_avqa}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/composition_interleaved}
BATCH_SIZE=${BATCH_SIZE:-1}
EPOCHS=${EPOCHS:-10}
LR=${LR:-5e-5}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}
TRAIN_MODALITY=interleaved
EVAL_MODALITIES=${EVAL_MODALITIES:-text,audio,image,both}
FUSION_GATE=${FUSION_GATE:-0.2}
FUSION_LAYERS=${FUSION_LAYERS:-}
AUDIO_FUSION_LAYERS=${AUDIO_FUSION_LAYERS:-}
VISION_FUSION_LAYERS=${VISION_FUSION_LAYERS:-}
SEED=${SEED:-42}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-SAFE-Composition}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-composition_interleaved_${SLURM_JOB_ID:-local}}
WANDB_TAGS=${WANDB_TAGS:-composition,interleaved}
MAX_SAMPLES=${MAX_SAMPLES:-0}
SLIM_PROJECTOR=${SLIM_PROJECTOR:-1}       # 0 to disable slim projector (default ON)
EVAL_DEBUG_SAMPLES=${EVAL_DEBUG_SAMPLES:-0}
MAX_ANSWER_TOKENS=${MAX_ANSWER_TOKENS:-16}
LAYER_ADDITIVITY_PROBE=${LAYER_ADDITIVITY_PROBE:-1}
LAYER_PROBE_SAMPLES=${LAYER_PROBE_SAMPLES:-256}
LAYER_PROBE_EVERY=${LAYER_PROBE_EVERY:-1}
DELTA_NORM_CAP_RATIO=${DELTA_NORM_CAP_RATIO:-0.0}
DELTA_NORM_CAP_EPS=${DELTA_NORM_CAP_EPS:-1e-6}
GATE_DEPTH_DECAY=${GATE_DEPTH_DECAY:-1.0}
AUDIO_GATE_DEPTH_DECAY=${AUDIO_GATE_DEPTH_DECAY:-1.0}
VISION_GATE_DEPTH_DECAY=${VISION_GATE_DEPTH_DECAY:-1.0}

# Optional sequential composition objective (audio->vision compatibility regularizer)
COMPAT_REG_ENABLE=${COMPAT_REG_ENABLE:-0}
COMPAT_REG_LAMBDA=${COMPAT_REG_LAMBDA:-0.05}
COMPAT_REG_RANK=${COMPAT_REG_RANK:-8}
COMPAT_REG_AUDIO_SAMPLES=${COMPAT_REG_AUDIO_SAMPLES:-256}
COMPAT_REG_MIN_SAMPLES=${COMPAT_REG_MIN_SAMPLES:-64}
COMPAT_REG_REFRESH_EVERY=${COMPAT_REG_REFRESH_EVERY:-1}
COMPAT_REG_LAYERS=${COMPAT_REG_LAYERS:-}
COMPAT_REG_WEIGHT_BY_SHIFT_NORM=${COMPAT_REG_WEIGHT_BY_SHIFT_NORM:-1}

# Optional additivity-positive objective (unpaired AV regularizer)
COMPAT_ADD_REG_ENABLE=${COMPAT_ADD_REG_ENABLE:-0}
COMPAT_ADD_REG_LAMBDA=${COMPAT_ADD_REG_LAMBDA:-0.01}
COMPAT_ADD_REG_EVERY=${COMPAT_ADD_REG_EVERY:-200}
COMPAT_ADD_REG_LAYERS=${COMPAT_ADD_REG_LAYERS:-}
COMPAT_ADD_REG_NORMALIZE=${COMPAT_ADD_REG_NORMALIZE:-1}
COMPAT_ADD_BANK_SIZE=${COMPAT_ADD_BANK_SIZE:-64}

# Optional transport objective (cross-layer hidden shift suppression)
COMPAT_TRANSPORT_ENABLE=${COMPAT_TRANSPORT_ENABLE:-0}
COMPAT_TRANSPORT_LAMBDA=${COMPAT_TRANSPORT_LAMBDA:-0.02}
COMPAT_TRANSPORT_CAP=${COMPAT_TRANSPORT_CAP:-0.0}
COMPAT_TRANSPORT_NORMALIZE=${COMPAT_TRANSPORT_NORMALIZE:-1}
COMPAT_TRANSPORT_LAYERS=${COMPAT_TRANSPORT_LAYERS:-}

# Optional interaction-mixer objectives (unpaired, synthetic pairing)
COMPAT_ICM_CANCEL_ENABLE=${COMPAT_ICM_CANCEL_ENABLE:-0}
COMPAT_ICM_CANCEL_LAMBDA=${COMPAT_ICM_CANCEL_LAMBDA:-0.02}
COMPAT_ICM_CANCEL_LOSS_TYPE=${COMPAT_ICM_CANCEL_LOSS_TYPE:-mse}
COMPAT_ICM_CANCEL_LOGIT_TEMP=${COMPAT_ICM_CANCEL_LOGIT_TEMP:-1.0}
COMPAT_ICM_CANCEL_START_STEP=${COMPAT_ICM_CANCEL_START_STEP:-0}
COMPAT_ICM_NOHARM_START_STEP=${COMPAT_ICM_NOHARM_START_STEP:-0}
COMPAT_ICM_UTIL_LAMBDA=${COMPAT_ICM_UTIL_LAMBDA:-0.0}
COMPAT_ICM_UTIL_START_STEP=${COMPAT_ICM_UTIL_START_STEP:-0}
COMPAT_ICM_IDENTITY_LAMBDA=${COMPAT_ICM_IDENTITY_LAMBDA:-0.01}
COMPAT_ICM_SMALL_LAMBDA=${COMPAT_ICM_SMALL_LAMBDA:-0.001}

# Optional gate-product additivity objective (diagnostic-guided)
COMPAT_GATE_ADD_ENABLE=${COMPAT_GATE_ADD_ENABLE:-0}
COMPAT_GATE_ADD_LAMBDA=${COMPAT_GATE_ADD_LAMBDA:-0.02}
COMPAT_GATE_PAIRING=${COMPAT_GATE_PAIRING:-zip}
COMPAT_GATE_TARGET_MODE=${COMPAT_GATE_TARGET_MODE:-inverse_rho}
COMPAT_GATE_PRODUCT_TARGET=${COMPAT_GATE_PRODUCT_TARGET:--1}
COMPAT_GATE_RHO_BETA=${COMPAT_GATE_RHO_BETA:-2.0}
COMPAT_GATE_MIN_EFFECTIVE=${COMPAT_GATE_MIN_EFFECTIVE:-0.0}
COMPAT_GATE_FLOOR_LAMBDA=${COMPAT_GATE_FLOOR_LAMBDA:-0.0}

# Optional no-harm / calibrated fusion / confidence routing objectives
COMPAT_NOHARM_ENABLE=${COMPAT_NOHARM_ENABLE:-0}
COMPAT_NOHARM_LAMBDA=${COMPAT_NOHARM_LAMBDA:-0.02}
COMPAT_NOHARM_MARGIN=${COMPAT_NOHARM_MARGIN:-0.0}
COMPAT_NOHARM_USE_BEST_SINGLE=${COMPAT_NOHARM_USE_BEST_SINGLE:-1}

COMPAT_LOGIT_FUSION_ENABLE=${COMPAT_LOGIT_FUSION_ENABLE:-0}
COMPAT_LOGIT_FUSION_LAMBDA=${COMPAT_LOGIT_FUSION_LAMBDA:-0.02}
COMPAT_LOGIT_FUSION_CONF_TEMP=${COMPAT_LOGIT_FUSION_CONF_TEMP:-0.5}

COMPAT_POE_ENABLE=${COMPAT_POE_ENABLE:-0}
COMPAT_POE_LAMBDA=${COMPAT_POE_LAMBDA:-0.02}
COMPAT_POE_WEIGHT_TEMP=${COMPAT_POE_WEIGHT_TEMP:-0.5}
COMPAT_POE_LOSS_TYPE=${COMPAT_POE_LOSS_TYPE:-kl}
COMPAT_POE_LOGIT_TEMP=${COMPAT_POE_LOGIT_TEMP:-1.0}

COMPAT_ROUTING_ENABLE=${COMPAT_ROUTING_ENABLE:-0}
COMPAT_ROUTING_MIN_SCALE=${COMPAT_ROUTING_MIN_SCALE:-0.25}
COMPAT_ROUTING_MAX_SCALE=${COMPAT_ROUTING_MAX_SCALE:-1.0}

# Optional initialization checkpoints (used for Level-1/2 continuation runs)
INIT_AUDIO_CKPT=${INIT_AUDIO_CKPT:-}
INIT_VISION_CKPT=${INIT_VISION_CKPT:-}

# Optional learned per-layer gating (calibration-friendly)
LEARNED_GATE=${LEARNED_GATE:-0}
LEARNED_GATE_INIT=${LEARNED_GATE_INIT:-0.0}
TRAIN_GATES_ONLY=${TRAIN_GATES_ONLY:-0}
ICM_ENABLE=${ICM_ENABLE:-0}
ICM_DIM=${ICM_DIM:-512}
ICM_HEADS=${ICM_HEADS:-8}
ICM_LAYERS=${ICM_LAYERS:-1}
ICM_DROPOUT=${ICM_DROPOUT:-0.1}
ICM_GATE_INIT=${ICM_GATE_INIT:--2.0}
ICM_MIN_MODALITIES=${ICM_MIN_MODALITIES:-2}
ICM_UTIL_TARGET=${ICM_UTIL_TARGET:-0.7}

# Qwen-specific env vars
export SAFE_QWEN_QUANT=none
export SAFE_GRAD_CKPT=0
export FP16=0

# Derive visible GPU count so sharding config matches the actual Slurm allocation.
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

mkdir -p logs "$OUTPUT_DIR"
mkdir -p "$SAFE_OFFLOAD_FOLDER"

cd "$SAFE_ROOT"

REQUIRE_CUDA=${REQUIRE_CUDA:-1}
if [[ "$REQUIRE_CUDA" == "1" ]]; then
  python3 -c "import torch,sys; ok=torch.cuda.is_available() and torch.cuda.device_count()>0; print(f'[cuda_check] available={torch.cuda.is_available()} count={torch.cuda.device_count()}'); sys.exit(0 if ok else 2)"
fi

WANDB_ARGS=()
if [[ "$WANDB" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME" --wandb-tags "$WANDB_TAGS")
fi

MAX_SAMPLES_ARGS=()
if [[ "$MAX_SAMPLES" != "0" ]]; then
  MAX_SAMPLES_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

EVAL_DEBUG_ARGS=()
if [[ "$EVAL_DEBUG_SAMPLES" != "0" ]]; then
  EVAL_DEBUG_ARGS+=(--eval-debug-samples "$EVAL_DEBUG_SAMPLES")
fi

LAYER_PROBE_ARGS=()
if [[ "$LAYER_ADDITIVITY_PROBE" == "1" ]]; then
  LAYER_PROBE_ARGS+=(--layer-additivity-probe --layer-probe-samples "$LAYER_PROBE_SAMPLES" --layer-probe-every "$LAYER_PROBE_EVERY")
fi

FUSION_LAYER_ARGS=()
if [[ -n "$FUSION_LAYERS" ]]; then
  FUSION_LAYER_ARGS+=(--fusion-layers "$FUSION_LAYERS")
fi
if [[ -n "$AUDIO_FUSION_LAYERS" ]]; then
  FUSION_LAYER_ARGS+=(--audio-fusion-layers "$AUDIO_FUSION_LAYERS")
fi
if [[ -n "$VISION_FUSION_LAYERS" ]]; then
  FUSION_LAYER_ARGS+=(--vision-fusion-layers "$VISION_FUSION_LAYERS")
fi

TRANSPORT_ARCH_ARGS=()
if [[ "$DELTA_NORM_CAP_RATIO" != "0" && "$DELTA_NORM_CAP_RATIO" != "0.0" ]]; then
  TRANSPORT_ARCH_ARGS+=(--delta-norm-cap-ratio "$DELTA_NORM_CAP_RATIO" --delta-norm-cap-eps "$DELTA_NORM_CAP_EPS")
fi
if [[ "$GATE_DEPTH_DECAY" != "1" && "$GATE_DEPTH_DECAY" != "1.0" ]]; then
  TRANSPORT_ARCH_ARGS+=(--gate-depth-decay "$GATE_DEPTH_DECAY")
fi
if [[ "$AUDIO_GATE_DEPTH_DECAY" != "1" && "$AUDIO_GATE_DEPTH_DECAY" != "1.0" ]]; then
  TRANSPORT_ARCH_ARGS+=(--audio-gate-depth-decay "$AUDIO_GATE_DEPTH_DECAY")
fi
if [[ "$VISION_GATE_DEPTH_DECAY" != "1" && "$VISION_GATE_DEPTH_DECAY" != "1.0" ]]; then
  TRANSPORT_ARCH_ARGS+=(--vision-gate-depth-decay "$VISION_GATE_DEPTH_DECAY")
fi

COMPAT_REG_ARGS=()
if [[ "$COMPAT_REG_ENABLE" == "1" ]]; then
  COMPAT_REG_ARGS+=(
    --compat-reg-enable
    --compat-reg-lambda "$COMPAT_REG_LAMBDA"
    --compat-reg-rank "$COMPAT_REG_RANK"
    --compat-reg-audio-samples "$COMPAT_REG_AUDIO_SAMPLES"
    --compat-reg-min-samples "$COMPAT_REG_MIN_SAMPLES"
    --compat-reg-refresh-every "$COMPAT_REG_REFRESH_EVERY"
  )
  if [[ "$COMPAT_REG_WEIGHT_BY_SHIFT_NORM" == "1" ]]; then
    COMPAT_REG_ARGS+=(--compat-reg-weight-by-shift-norm)
  else
    COMPAT_REG_ARGS+=(--no-compat-reg-weight-by-shift-norm)
  fi
  if [[ -n "$COMPAT_REG_LAYERS" ]]; then
    COMPAT_REG_ARGS+=(--compat-reg-layers "$COMPAT_REG_LAYERS")
  fi
fi

COMPAT_ADD_ARGS=()
if [[ "$COMPAT_ADD_REG_ENABLE" == "1" ]]; then
  COMPAT_ADD_ARGS+=(
    --compat-add-reg-enable
    --compat-add-reg-lambda "$COMPAT_ADD_REG_LAMBDA"
    --compat-add-reg-every "$COMPAT_ADD_REG_EVERY"
    --compat-add-bank-size "$COMPAT_ADD_BANK_SIZE"
  )
  if [[ "$COMPAT_ADD_REG_NORMALIZE" == "1" ]]; then
    COMPAT_ADD_ARGS+=(--compat-add-reg-normalize)
  else
    COMPAT_ADD_ARGS+=(--no-compat-add-reg-normalize)
  fi
  if [[ -n "$COMPAT_ADD_REG_LAYERS" ]]; then
    COMPAT_ADD_ARGS+=(--compat-add-reg-layers "$COMPAT_ADD_REG_LAYERS")
  fi
fi

COMPAT_TRANSPORT_ARGS=()
if [[ "$COMPAT_TRANSPORT_ENABLE" == "1" ]]; then
  COMPAT_TRANSPORT_ARGS+=(
    --compat-transport-enable
    --compat-transport-lambda "$COMPAT_TRANSPORT_LAMBDA"
    --compat-transport-cap "$COMPAT_TRANSPORT_CAP"
  )
  if [[ "$COMPAT_TRANSPORT_NORMALIZE" == "1" ]]; then
    COMPAT_TRANSPORT_ARGS+=(--compat-transport-normalize)
  else
    COMPAT_TRANSPORT_ARGS+=(--no-compat-transport-normalize)
  fi
  if [[ -n "$COMPAT_TRANSPORT_LAYERS" ]]; then
    COMPAT_TRANSPORT_ARGS+=(--compat-transport-layers "$COMPAT_TRANSPORT_LAYERS")
  fi
fi

COMPAT_GATE_ARGS=()
if [[ "$COMPAT_GATE_ADD_ENABLE" == "1" ]]; then
  COMPAT_GATE_ARGS+=(
    --compat-gate-add-enable
    --compat-gate-add-lambda "$COMPAT_GATE_ADD_LAMBDA"
    --compat-gate-pairing "$COMPAT_GATE_PAIRING"
    --compat-gate-target-mode "$COMPAT_GATE_TARGET_MODE"
    --compat-gate-product-target "$COMPAT_GATE_PRODUCT_TARGET"
    --compat-gate-rho-beta "$COMPAT_GATE_RHO_BETA"
    --compat-gate-min-effective "$COMPAT_GATE_MIN_EFFECTIVE"
    --compat-gate-floor-lambda "$COMPAT_GATE_FLOOR_LAMBDA"
  )
fi

COMPAT_EXTRA_ARGS=()
if [[ "$COMPAT_NOHARM_ENABLE" == "1" ]]; then
  COMPAT_EXTRA_ARGS+=(
    --compat-noharm-enable
    --compat-noharm-lambda "$COMPAT_NOHARM_LAMBDA"
    --compat-noharm-margin "$COMPAT_NOHARM_MARGIN"
  )
  if [[ "$COMPAT_NOHARM_USE_BEST_SINGLE" == "1" ]]; then
    COMPAT_EXTRA_ARGS+=(--compat-noharm-use-best-single)
  fi
fi

if [[ "$COMPAT_LOGIT_FUSION_ENABLE" == "1" ]]; then
  COMPAT_EXTRA_ARGS+=(
    --compat-logit-fusion-enable
    --compat-logit-fusion-lambda "$COMPAT_LOGIT_FUSION_LAMBDA"
    --compat-logit-fusion-conf-temp "$COMPAT_LOGIT_FUSION_CONF_TEMP"
  )
fi

if [[ "$COMPAT_POE_ENABLE" == "1" ]]; then
  COMPAT_EXTRA_ARGS+=(
    --compat-poe-enable
    --compat-poe-lambda "$COMPAT_POE_LAMBDA"
    --compat-poe-weight-temp "$COMPAT_POE_WEIGHT_TEMP"
    --compat-poe-loss-type "$COMPAT_POE_LOSS_TYPE"
    --compat-poe-logit-temp "$COMPAT_POE_LOGIT_TEMP"
  )
fi

if [[ "$COMPAT_ROUTING_ENABLE" == "1" ]]; then
  COMPAT_EXTRA_ARGS+=(
    --compat-routing-enable
    --compat-routing-min-scale "$COMPAT_ROUTING_MIN_SCALE"
    --compat-routing-max-scale "$COMPAT_ROUTING_MAX_SCALE"
  )
fi

SLIM_ARGS=()
if [ "$SLIM_PROJECTOR" = "0" ]; then
  SLIM_ARGS+=(--no-slim-projector)
fi

INIT_ARGS=()
if [[ -n "$INIT_AUDIO_CKPT" ]]; then
  INIT_ARGS+=(--init-audio-ckpt "$INIT_AUDIO_CKPT")
fi
if [[ -n "$INIT_VISION_CKPT" ]]; then
  INIT_ARGS+=(--init-vision-ckpt "$INIT_VISION_CKPT")
fi

GATE_ARGS=()
if [[ "$LEARNED_GATE" == "1" ]]; then
  GATE_ARGS+=(--learned-gate --learned-gate-init "$LEARNED_GATE_INIT")
fi
if [[ "$TRAIN_GATES_ONLY" == "1" ]]; then
  GATE_ARGS+=(--train-gates-only)
fi

ICM_ARGS=()
if [[ "$ICM_ENABLE" == "1" ]]; then
  ICM_ARGS+=(
    --icm-enable
    --icm-dim "$ICM_DIM"
    --icm-heads "$ICM_HEADS"
    --icm-layers "$ICM_LAYERS"
    --icm-dropout "$ICM_DROPOUT"
    --icm-gate-init "$ICM_GATE_INIT"
    --icm-min-modalities "$ICM_MIN_MODALITIES"
    --icm-util-target "$ICM_UTIL_TARGET"
  )
fi

COMPAT_ICM_ARGS=(
  --compat-icm-noharm-start-step "$COMPAT_ICM_NOHARM_START_STEP"
  --compat-icm-util-lambda "$COMPAT_ICM_UTIL_LAMBDA"
  --compat-icm-util-start-step "$COMPAT_ICM_UTIL_START_STEP"
  --compat-icm-identity-lambda "$COMPAT_ICM_IDENTITY_LAMBDA"
  --compat-icm-small-lambda "$COMPAT_ICM_SMALL_LAMBDA"
)
if [[ "$COMPAT_ICM_CANCEL_ENABLE" == "1" ]]; then
  COMPAT_ICM_ARGS+=(
    --compat-icm-cancel-enable
    --compat-icm-cancel-lambda "$COMPAT_ICM_CANCEL_LAMBDA"
    --compat-icm-cancel-loss-type "$COMPAT_ICM_CANCEL_LOSS_TYPE"
    --compat-icm-cancel-logit-temp "$COMPAT_ICM_CANCEL_LOGIT_TEMP"
    --compat-icm-cancel-start-step "$COMPAT_ICM_CANCEL_START_STEP"
  )
fi

python3 "$SAFE_ROOT/experiments/avqa_composition/train_avqa_composition.py" \
  --dataset music_avqa \
  --model-config "$MODEL_CONFIG" \
  --train-manifest "$DATA_ROOT/manifests/train.jsonl" \
  --val-manifest "$DATA_ROOT/manifests/validation.jsonl" \
  --media-root "$MEDIA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs "$EPOCHS" \
  --learning-rate "$LR" \
  --num-audio-tokens "$NUM_AUDIO_TOKENS" \
  --fusion-gate "$FUSION_GATE" \
  --seed "$SEED" \
  --train-modality "$TRAIN_MODALITY" \
  --eval-modalities "$EVAL_MODALITIES" \
  --max-answer-tokens "$MAX_ANSWER_TOKENS" \
  --freeze-audio-encoder \
  "${SLIM_ARGS[@]}" \
  "${INIT_ARGS[@]}" \
  "${GATE_ARGS[@]}" \
  "${ICM_ARGS[@]}" \
  "${MAX_SAMPLES_ARGS[@]}" \
  "${EVAL_DEBUG_ARGS[@]}" \
  "${LAYER_PROBE_ARGS[@]}" \
  "${FUSION_LAYER_ARGS[@]}" \
  "${TRANSPORT_ARCH_ARGS[@]}" \
  "${COMPAT_REG_ARGS[@]}" \
  "${COMPAT_ADD_ARGS[@]}" \
  "${COMPAT_TRANSPORT_ARGS[@]}" \
  "${COMPAT_ICM_ARGS[@]}" \
  "${COMPAT_GATE_ARGS[@]}" \
  "${COMPAT_EXTRA_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
