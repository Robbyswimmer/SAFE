#!/bin/bash
#SBATCH --job-name=ave-llm-probe
#SBATCH --output=logs/ave_llm_probe_%j.log
#SBATCH --error=logs/ave_llm_probe_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

set -e

# Configuration (override via env)
DATA_PATH=${DATA_PATH:-"/data/SalmanAsif/AVE_Dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/ave_llm_probe"}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_EPOCHS=${NUM_EPOCHS:-20}
LEARNING_RATE=${LEARNING_RATE:-6e-5}
SAFE_LR=${SAFE_LR:-$LEARNING_RATE}
HEAD_LR=${HEAD_LR:-1e-3}
MODEL_CONFIG=${MODEL_CONFIG:-"phase1"}
FUSION_LAYER_INDICES=${FUSION_LAYER_INDICES:-"1"}  # requested: layer 1 to start
FUSION_INJECTION_POINT=${FUSION_INJECTION_POINT:-""}  # optional: pre_ffn or post_layer
FUSION_MODE=${FUSION_MODE:-""}  # optional: residual or film
POOLING=${POOLING:-"last"}
FP16=${FP16:-1}
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"SAFE"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"ave-llm-probe-${SLURM_JOB_ID:-local}"}
LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-""}
HEAD_ONLY=${HEAD_ONLY:-0}
FORCE_GATE=${FORCE_GATE:-""}
HEAD_WARMUP_STEPS=${HEAD_WARMUP_STEPS:-0}
BYPASS_LLM=${BYPASS_LLM:-0}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-""}
RUN_PROJECTOR_ABLATION=${RUN_PROJECTOR_ABLATION:-0}
ABLATION_BATCH_SIZE=${ABLATION_BATCH_SIZE:-128}
HEAD_TYPE=${HEAD_TYPE:-"linear"}
PROJECTOR_OUTPUT_DIM=${PROJECTOR_OUTPUT_DIM:-""}
DISABLE_OUTPUT_NORM=${DISABLE_OUTPUT_NORM:-0}
DISABLE_INPUT_NORM=${DISABLE_INPUT_NORM:-0}
DISABLE_SCALE=${DISABLE_SCALE:-0}
IDENTITY_PROJECTOR=${IDENTITY_PROJECTOR:-0}

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "AVE LLM-Probe (SAFE fusion -> frozen LLM)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Safe LR: $SAFE_LR"
echo "Head LR: $HEAD_LR"
echo "Model config: $MODEL_CONFIG"
echo "Fusion layers: $FUSION_LAYER_INDICES"
echo "Fusion injection point: ${FUSION_INJECTION_POINT:-'(default)'}"
echo "Fusion mode: ${FUSION_MODE:-'(default)'}"
echo "Pooling: $POOLING"
echo "FP16: $FP16"
echo "Load checkpoint: ${LOAD_CHECKPOINT:-'(none)'}"
echo "Head only: $HEAD_ONLY"
echo "Force gate: ${FORCE_GATE:-'(unset)'}"
echo "Head warmup steps: $HEAD_WARMUP_STEPS"
echo "========================================"

# Activate conda if available
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate safe-env
elif [ -f ~/.bashrc ]; then
  source ~/.bashrc
  conda activate safe-env 2>/dev/null || true
fi

echo "Python: $(which python)"

WANDB_ARGS=""
if [ "$USE_WANDB" = "1" ]; then
  WANDB_ARGS="--wandb --wandb-project $WANDB_PROJECT --wandb-run-name $WANDB_RUN_NAME"
fi

FP16_ARG=""
if [ "$FP16" = "1" ]; then
  FP16_ARG="--fp16"
fi

CKPT_ARGS=""
if [ -n "$LOAD_CHECKPOINT" ]; then
  CKPT_ARGS="--load-checkpoint $LOAD_CHECKPOINT"
fi

HEAD_ONLY_ARG=""
if [ "$HEAD_ONLY" = "1" ]; then
  HEAD_ONLY_ARG="--head-only"
fi

BYPASS_LLM_ARG=""
if [ "$BYPASS_LLM" = "1" ]; then
  BYPASS_LLM_ARG="--bypass-llm"
fi

NUM_AUDIO_TOKENS_ARG=""
if [ -n "$NUM_AUDIO_TOKENS" ]; then
  NUM_AUDIO_TOKENS_ARG="--num-audio-tokens $NUM_AUDIO_TOKENS"
fi

PROJECTOR_ABLATION_ARG=""
if [ "$RUN_PROJECTOR_ABLATION" = "1" ]; then
  PROJECTOR_ABLATION_ARG="--run-projector-ablation --ablation-batch-size $ABLATION_BATCH_SIZE"
fi

HEAD_TYPE_ARG=""
if [ -n "$HEAD_TYPE" ]; then
  HEAD_TYPE_ARG="--head-type $HEAD_TYPE"
fi

PROJECTOR_OUTPUT_DIM_ARG=""
if [ -n "$PROJECTOR_OUTPUT_DIM" ]; then
  PROJECTOR_OUTPUT_DIM_ARG="--projector-output-dim $PROJECTOR_OUTPUT_DIM"
fi

DISABLE_OUTPUT_NORM_ARG=""
if [ "$DISABLE_OUTPUT_NORM" = "1" ]; then
  DISABLE_OUTPUT_NORM_ARG="--disable-output-norm"
fi

DISABLE_INPUT_NORM_ARG=""
if [ "$DISABLE_INPUT_NORM" = "1" ]; then
  DISABLE_INPUT_NORM_ARG="--disable-input-norm"
fi

DISABLE_SCALE_ARG=""
if [ "$DISABLE_SCALE" = "1" ]; then
  DISABLE_SCALE_ARG="--disable-scale"
fi

IDENTITY_PROJECTOR_ARG=""
if [ "$IDENTITY_PROJECTOR" = "1" ]; then
  IDENTITY_PROJECTOR_ARG="--identity-projector"
fi

FORCE_GATE_ARG=""
if [ -n "$FORCE_GATE" ]; then
  FORCE_GATE_ARG="--force-gate $FORCE_GATE"
fi

FUSION_INJECTION_ARG=""
if [ -n "$FUSION_INJECTION_POINT" ]; then
  FUSION_INJECTION_ARG="--fusion-injection-point $FUSION_INJECTION_POINT"
fi

FUSION_MODE_ARG=""
if [ -n "$FUSION_MODE" ]; then
  FUSION_MODE_ARG="--fusion-mode $FUSION_MODE"
fi

echo "Resolved CLI extras: ${FUSION_INJECTION_ARG} ${FUSION_MODE_ARG} ${CKPT_ARGS} ${HEAD_ONLY_ARG} ${BYPASS_LLM_ARG} ${FORCE_GATE_ARG} ${FP16_ARG} ${WANDB_ARGS}"

python train_audio_llm_probe.py \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs "$NUM_EPOCHS" \
  --safe-learning-rate "$SAFE_LR" \
  --head-learning-rate "$HEAD_LR" \
  --model-config "$MODEL_CONFIG" \
  --fusion-layer-indices "$FUSION_LAYER_INDICES" \
  $NUM_AUDIO_TOKENS_ARG \
  $HEAD_TYPE_ARG \
  $PROJECTOR_OUTPUT_DIM_ARG \
  $DISABLE_OUTPUT_NORM_ARG \
  $DISABLE_INPUT_NORM_ARG \
  $DISABLE_SCALE_ARG \
  $IDENTITY_PROJECTOR_ARG \
  $FUSION_INJECTION_ARG \
  $FUSION_MODE_ARG \
  --pooling "$POOLING" \
  --num-workers 4 \
  --log-interval 10 \
  $FP16_ARG \
  $CKPT_ARGS \
  $HEAD_ONLY_ARG \
  $BYPASS_LLM_ARG \
  $PROJECTOR_ABLATION_ARG \
  $FORCE_GATE_ARG \
  --head-warmup-steps "$HEAD_WARMUP_STEPS" \
  $WANDB_ARGS

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
