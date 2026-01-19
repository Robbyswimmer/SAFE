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
MODEL_CONFIG=${MODEL_CONFIG:-"phase1"}
FUSION_LAYER_INDICES=${FUSION_LAYER_INDICES:-"1"}  # requested: layer 1 to start
FUSION_INJECTION_POINT=${FUSION_INJECTION_POINT:-""}  # optional: pre_ffn or post_layer
POOLING=${POOLING:-"last"}
FP16=${FP16:-1}
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"SAFE"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"ave-llm-probe-${SLURM_JOB_ID:-local}"}
LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-""}
HEAD_ONLY=${HEAD_ONLY:-0}
FORCE_GATE=${FORCE_GATE:-""}

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
echo "Model config: $MODEL_CONFIG"
echo "Fusion layers: $FUSION_LAYER_INDICES"
echo "Fusion injection point: ${FUSION_INJECTION_POINT:-'(default)'}"
echo "Pooling: $POOLING"
echo "FP16: $FP16"
echo "Load checkpoint: ${LOAD_CHECKPOINT:-'(none)'}"
echo "Head only: $HEAD_ONLY"
echo "Force gate: ${FORCE_GATE:-'(unset)'}"
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

FORCE_GATE_ARG=""
if [ -n "$FORCE_GATE" ]; then
  FORCE_GATE_ARG="--force-gate $FORCE_GATE"
fi

FUSION_INJECTION_ARG=""
if [ -n "$FUSION_INJECTION_POINT" ]; then
  FUSION_INJECTION_ARG="--fusion-injection-point $FUSION_INJECTION_POINT"
fi

python train_audio_llm_probe.py \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs "$NUM_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --model-config "$MODEL_CONFIG" \
  --fusion-layer-indices "$FUSION_LAYER_INDICES" \
  $FUSION_INJECTION_ARG \
  --pooling "$POOLING" \
  --num-workers 4 \
  --log-interval 10 \
  $FP16_ARG \
  $CKPT_ARGS \
  $HEAD_ONLY_ARG \
  $FORCE_GATE_ARG \
  $WANDB_ARGS

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
