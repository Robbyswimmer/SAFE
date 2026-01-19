#!/bin/bash
#SBATCH --job-name=ave-llm-like
#SBATCH --output=logs/ave_llm_likelihood_%j.log
#SBATCH --error=logs/ave_llm_likelihood_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

set -e

DATA_PATH=${DATA_PATH:-"/data/SalmanAsif/AVE_Dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/ave_llm_likelihood"}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
MODEL_CONFIG=${MODEL_CONFIG:-"phase1"}
FUSION_LAYER_INDICES=${FUSION_LAYER_INDICES:-"12"}
FUSION_INJECTION_POINT=${FUSION_INJECTION_POINT:-""}
FUSION_MODE=${FUSION_MODE:-""}  # optional: residual or film
USE_BOTTLENECK=${USE_BOTTLENECK:-""}  # 1/0 to override
BOTTLENECK_DIM=${BOTTLENECK_DIM:-""}
LORA_RANK=${LORA_RANK:-""}
NUM_NEGATIVES=${NUM_NEGATIVES:-7}
TEMPLATE=${TEMPLATE:-"The sound is: {label}."}
FP16=${FP16:-1}
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"SAFE"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"ave-llm-like-${SLURM_JOB_ID:-local}"}
LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-""}
FORCE_GATE=${FORCE_GATE:-""}

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "AVE Closed-Set Likelihood (SAFE -> frozen LLM)"
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
echo "Fusion mode: ${FUSION_MODE:-'(default)'}"
echo "Use bottleneck: ${USE_BOTTLENECK:-'(default)'}"
echo "Bottleneck dim: ${BOTTLENECK_DIM:-'(default)'}"
echo "LoRA rank: ${LORA_RANK:-'(default)'}"
echo "Negatives per sample: $NUM_NEGATIVES"
echo "Template: $TEMPLATE"
echo "FP16: $FP16"
echo "Load checkpoint: ${LOAD_CHECKPOINT:-'(none)'}"
echo "Force gate: ${FORCE_GATE:-'(unset)'}"
echo "========================================"

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate safe-env
elif [ -f ~/.bashrc ]; then
  source ~/.bashrc
  conda activate safe-env 2>/dev/null || true
fi

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

BOTTLENECK_ARG=""
if [ "$USE_BOTTLENECK" = "1" ]; then
  BOTTLENECK_ARG="--use-bottleneck"
elif [ "$USE_BOTTLENECK" = "0" ]; then
  BOTTLENECK_ARG="--no-bottleneck"
fi

BOTTLENECK_DIM_ARG=""
if [ -n "$BOTTLENECK_DIM" ]; then
  BOTTLENECK_DIM_ARG="--bottleneck-dim $BOTTLENECK_DIM"
fi

LORA_RANK_ARG=""
if [ -n "$LORA_RANK" ]; then
  LORA_RANK_ARG="--lora-rank $LORA_RANK"
fi

python train_audio_llm_likelihood.py \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs "$NUM_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --model-config "$MODEL_CONFIG" \
  --fusion-layer-indices "$FUSION_LAYER_INDICES" \
  $FUSION_INJECTION_ARG \
  $FUSION_MODE_ARG \
  $BOTTLENECK_ARG \
  $BOTTLENECK_DIM_ARG \
  $LORA_RANK_ARG \
  --num-negatives "$NUM_NEGATIVES" \
  --template "$TEMPLATE" \
  --num-workers 4 \
  --log-interval 10 \
  $FP16_ARG \
  $CKPT_ARGS \
  $FORCE_GATE_ARG \
  $WANDB_ARGS

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
