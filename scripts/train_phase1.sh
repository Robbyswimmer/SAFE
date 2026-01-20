#!/bin/bash
#
# Phase 1 Training Script - Clean SAFE Training
#
# Usage:
#   bash scripts/train_phase1.sh                    # Single GPU
#   NUM_GPUS=3 bash scripts/train_phase1.sh         # Multi-GPU
#   sbatch scripts/train_phase1.sh                  # SLURM single GPU
#   NUM_GPUS=3 sbatch scripts/train_phase1.sh       # SLURM multi-GPU

#SBATCH --job-name=SAFE-Train
#SBATCH --output=logs/train_%j.txt
#SBATCH --error=logs/train_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH -p gpu
# NOTE: --gres=gpu:N must be passed on sbatch command line, e.g.:
#   sbatch --gres=gpu:1 scripts/train_phase1.sh

set -euo pipefail

# Memory optimization - prevent CUDA OOM from fragmentation during long training runs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Environment setup - activate conda environment
CONDA_ENV=${CONDA_ENV:-"safe-env"}

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

echo "Activating conda environment '${CONDA_ENV}'"
conda activate "${CONDA_ENV}"

# Verify environment
python --version
which python

# Configuration
MODEL_CONFIG=${MODEL_CONFIG:-"phase1"}
DATA_PATH=${DATA_PATH:-"$PWD/experiments/full_training/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/phase1_clean"}
NUM_EPOCHS=${NUM_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-2}                 # Smaller microbatches reduce peak VRAM/CPU usage
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-64}  # Maintain effective batch size via accumulation
LR_PROJECTOR=${LR_PROJECTOR:-1e-3}
LR_ADAPTER=${LR_ADAPTER:-5e-4}
WARMUP_STEPS=${WARMUP_STEPS:-2000}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}         # LR floor at 10% of base (prevents plateau)
EVAL_FREQUENCY=${EVAL_FREQUENCY:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-20}
NUM_BEAMS=${NUM_BEAMS:-5}
FP16=${FP16:-"--fp16"}
SEED=${SEED:-42}
USE_WAVCAPS=${USE_WAVCAPS:-1}
WAVCAPS_RATIO=${WAVCAPS_RATIO:-0.8}
USE_CLOTHO=${USE_CLOTHO:-1}
USE_MACS=${USE_MACS:-1}
AUDIO_CONTRASTIVE_WEIGHT=${AUDIO_CONTRASTIVE_WEIGHT:-0.1}
AUDIO_CONTRASTIVE_TEMPERATURE=${AUDIO_CONTRASTIVE_TEMPERATURE:-0.07}
AUDIO_CONTRASTIVE_MAX_LENGTH=${AUDIO_CONTRASTIVE_MAX_LENGTH:-48}
GATE_WARMUP_STEPS=${GATE_WARMUP_STEPS:-0}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-""}
FUSION_LAYER_INDICES=${FUSION_LAYER_INDICES:-""}  # e.g., "8,16,24" - overrides config default
LORA_RANK=${LORA_RANK:-""}                        # e.g., "8" - overrides config default
LABEL_SMOOTHING=${LABEL_SMOOTHING:-""}            # e.g., "0.1" - overrides config default
TRAIN_EVAL_STEPS=${TRAIN_EVAL_STEPS:-30}          # Compute train CIDEr/METEOR every N steps
TRAIN_EVAL_SAMPLES=${TRAIN_EVAL_SAMPLES:-300}     # Number of train samples for accuracy eval
TRAIN_EVAL_SPLIT=${TRAIN_EVAL_SPLIT:-"val"}       # Use val by default (multi-ref AudioCaps)
TRAIN_EVAL_ABLATE_AUDIO=${TRAIN_EVAL_ABLATE_AUDIO:-0}
TRAIN_EVAL_ABLATE_MAX_BATCHES=${TRAIN_EVAL_ABLATE_MAX_BATCHES:-10}
EVAL_ABLATE_AUDIO=${EVAL_ABLATE_AUDIO:-0}
SUPPRESS_EOS_FOR_AUDIO_EARLY_STEPS=${SUPPRESS_EOS_FOR_AUDIO_EARLY_STEPS:-0}
EVAL_REPETITION_PENALTY=${EVAL_REPETITION_PENALTY:-1.1}
EVAL_NO_REPEAT_NGRAM_SIZE=${EVAL_NO_REPEAT_NGRAM_SIZE:-3}
AUDIO_AUGMENT=${AUDIO_AUGMENT:-0}                 # Audio augmentation (SpecAugment + waveform), off by default
AUDIO_AUGMENT_PROB=${AUDIO_AUGMENT_PROB:-0.5}     # Probability of applying augmentation per sample
EXTRA_ARGS=${EXTRA_ARGS:-""}

# Memory optimization - enable by default for 48GB GPUs with large datasets
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
NUM_WORKERS=${NUM_WORKERS:-0}              # Disable workers to avoid per-worker RAM overhead
MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-10}   # Keep eval lightweight to avoid memory buildup

# Multi-GPU configuration
NUM_GPUS=${NUM_GPUS:-1}                    # Number of GPUs to use (default: 1)

# Weights & Biases logging (optional)
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-""}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_NAME=${WANDB_NAME:-""}
WANDB_GROUP=${WANDB_GROUP:-""}
WANDB_TAGS=${WANDB_TAGS:-""}
WANDB_MODE=${WANDB_MODE:-""}
WANDB_DIR=${WANDB_DIR:-""}
WANDB_NOTES=${WANDB_NOTES:-""}
WANDB_LOG_CODE=${WANDB_LOG_CODE:-0}
WANDB_LOG_CHECKPOINTS=${WANDB_LOG_CHECKPOINTS:-0}
WANDB_SAMPLE_COUNT=${WANDB_SAMPLE_COUNT:-30}
WANDB_WATCH=${WANDB_WATCH:-"false"}
WANDB_WATCH_LOG_FREQ=${WANDB_WATCH_LOG_FREQ:-500}

WANDB_ARGS=()
if [[ "${USE_WANDB}" != "0" ]]; then
  WANDB_ARGS+=(--wandb)
  [[ -n "${WANDB_PROJECT}" ]] && WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT}")
  [[ -n "${WANDB_ENTITY}" ]] && WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
  [[ -n "${WANDB_NAME}" ]] && WANDB_ARGS+=(--wandb-name "${WANDB_NAME}")
  [[ -n "${WANDB_GROUP}" ]] && WANDB_ARGS+=(--wandb-group "${WANDB_GROUP}")
  [[ -n "${WANDB_TAGS}" ]] && WANDB_ARGS+=(--wandb-tags "${WANDB_TAGS}")
  [[ -n "${WANDB_MODE}" ]] && WANDB_ARGS+=(--wandb-mode "${WANDB_MODE}")
  [[ -n "${WANDB_DIR}" ]] && WANDB_ARGS+=(--wandb-dir "${WANDB_DIR}")
  [[ -n "${WANDB_NOTES}" ]] && WANDB_ARGS+=(--wandb-notes "${WANDB_NOTES}")
  [[ "${WANDB_LOG_CODE}" != "0" ]] && WANDB_ARGS+=(--wandb-log-code)
  [[ "${WANDB_LOG_CHECKPOINTS}" != "0" ]] && WANDB_ARGS+=(--wandb-log-checkpoints)
  [[ -n "${WANDB_SAMPLE_COUNT}" ]] && WANDB_ARGS+=(--wandb-sample-count "${WANDB_SAMPLE_COUNT}")
  [[ -n "${WANDB_WATCH}" ]] && WANDB_ARGS+=(--wandb-watch "${WANDB_WATCH}")
  [[ -n "${WANDB_WATCH_LOG_FREQ}" ]] && WANDB_ARGS+=(--wandb-watch-log-freq "${WANDB_WATCH_LOG_FREQ}")
fi

MAX_TRAIN_ARGS=()
if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  MAX_TRAIN_ARGS+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
fi

FUSION_LAYER_ARGS=()
if [[ -n "${FUSION_LAYER_INDICES}" ]]; then
  FUSION_LAYER_ARGS+=(--fusion-layer-indices "${FUSION_LAYER_INDICES}")
fi

LORA_RANK_ARGS=()
if [[ -n "${LORA_RANK}" ]]; then
  LORA_RANK_ARGS+=(--lora-rank "${LORA_RANK}")
fi

LABEL_SMOOTHING_ARGS=()
if [[ -n "${LABEL_SMOOTHING}" ]]; then
  LABEL_SMOOTHING_ARGS+=(--label-smoothing "${LABEL_SMOOTHING}")
fi

AUDIO_AUGMENT_ARGS=()
if [[ "${AUDIO_AUGMENT}" != "0" ]]; then
  AUDIO_AUGMENT_ARGS+=(--audio-augment)
  AUDIO_AUGMENT_ARGS+=(--audio-augment-prob "${AUDIO_AUGMENT_PROB}")
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# Verify data path exists
if [[ ! -d "${DATA_PATH}" ]]; then
  echo "ERROR: Data path not found: ${DATA_PATH}" >&2
  echo "Please set DATA_PATH environment variable to your data directory" >&2
  exit 1
fi

# Log configuration
echo "========================================"
echo "SAFE Training Configuration"
echo "========================================"
echo "Model config: ${MODEL_CONFIG}"
echo "Data path: ${DATA_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Gradient accumulation: ${GRADIENT_ACCUMULATION}"
echo "Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "LR projector: ${LR_PROJECTOR}"
echo "LR adapter: ${LR_ADAPTER}"
echo "Warmup steps: ${WARMUP_STEPS}"
echo "Mixed precision: ${FP16}"
echo "Seed: ${SEED}"
echo "Use WavCaps: ${USE_WAVCAPS} (ratio=${WAVCAPS_RATIO})"
echo "Use Clotho: ${USE_CLOTHO}"
echo "Use MACS: ${USE_MACS}"
echo "Audio contrastive weight: ${AUDIO_CONTRASTIVE_WEIGHT}"
echo "Gate warmup steps: ${GATE_WARMUP_STEPS}"
echo "Gradient checkpointing: ${GRADIENT_CHECKPOINTING}"
echo "Num workers: ${NUM_WORKERS}"
echo "Max eval batches: ${MAX_EVAL_BATCHES}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Train eval split: ${TRAIN_EVAL_SPLIT}"
echo "Eval ablate audio: ${EVAL_ABLATE_AUDIO}"
echo "Train eval ablate audio: ${TRAIN_EVAL_ABLATE_AUDIO} (max_batches=${TRAIN_EVAL_ABLATE_MAX_BATCHES})"
echo "Suppress EOS for audio early steps: ${SUPPRESS_EOS_FOR_AUDIO_EARLY_STEPS}"
echo "Eval repetition penalty: ${EVAL_REPETITION_PENALTY}"
echo "Eval no-repeat ngram size: ${EVAL_NO_REPEAT_NGRAM_SIZE}"
if [[ -n "${FUSION_LAYER_INDICES}" ]]; then
  echo "Fusion layer indices: ${FUSION_LAYER_INDICES}"
fi
if [[ -n "${LORA_RANK}" ]]; then
  echo "LoRA rank: ${LORA_RANK}"
fi
if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  echo "Max train samples: ${MAX_TRAIN_SAMPLES}"
fi
echo "W&B enabled: ${USE_WANDB}"
if [[ "${USE_WANDB}" != "0" ]]; then
  echo "W&B mode: ${WANDB_MODE:-(auto)}"
  echo "W&B project: ${WANDB_PROJECT:-(default)}"
  echo "W&B dir: ${WANDB_DIR:-(default)}"
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  echo "Extra args: ${EXTRA_ARGS}"
fi
echo "========================================"
echo ""

# Run training with torchrun for multi-GPU support
# torchrun handles setting RANK, WORLD_SIZE, LOCAL_RANK environment variables
if [[ "${NUM_GPUS}" -gt 1 ]]; then
    echo "🚀 Launching distributed training with ${NUM_GPUS} GPUs..."
    LAUNCHER="torchrun --nproc_per_node=${NUM_GPUS} --standalone"
else
    echo "🚀 Launching single-GPU training..."
    LAUNCHER="python"
fi

${LAUNCHER} train_safe.py \
    --model-config "${MODEL_CONFIG}" \
    --data-path "${DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-epochs "${NUM_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION}" \
    --learning-rate-projector "${LR_PROJECTOR}" \
    --learning-rate-adapter "${LR_ADAPTER}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --min-lr-ratio "${MIN_LR_RATIO}" \
    --eval-frequency "${EVAL_FREQUENCY}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --num-beams "${NUM_BEAMS}" \
    --seed "${SEED}" \
    $( [[ "${USE_WAVCAPS}" != "0" ]] && echo --use-wavcaps ) \
    --wavcaps-ratio "${WAVCAPS_RATIO}" \
    $( [[ "${USE_CLOTHO}" != "0" ]] && echo --use-clotho ) \
    $( [[ "${USE_MACS}" != "0" ]] && echo --use-macs ) \
    --audio-contrastive-weight "${AUDIO_CONTRASTIVE_WEIGHT}" \
    --audio-contrastive-temperature "${AUDIO_CONTRASTIVE_TEMPERATURE}" \
    --audio-contrastive-max-length "${AUDIO_CONTRASTIVE_MAX_LENGTH}" \
    --gate-warmup-steps "${GATE_WARMUP_STEPS}" \
    --num-workers "${NUM_WORKERS}" \
    --max-eval-batches "${MAX_EVAL_BATCHES}" \
    --suppress-eos-for-audio-early-steps "${SUPPRESS_EOS_FOR_AUDIO_EARLY_STEPS}" \
    --train-eval-steps "${TRAIN_EVAL_STEPS}" \
    --train-eval-samples "${TRAIN_EVAL_SAMPLES}" \
    --train-eval-split "${TRAIN_EVAL_SPLIT}" \
    --eval-repetition-penalty "${EVAL_REPETITION_PENALTY}" \
    --eval-no-repeat-ngram-size "${EVAL_NO_REPEAT_NGRAM_SIZE}" \
    $( [[ "${EVAL_ABLATE_AUDIO}" != "0" ]] && echo --eval-ablate-audio ) \
    $( [[ "${TRAIN_EVAL_ABLATE_AUDIO}" != "0" ]] && echo --train-eval-ablate-audio ) \
    --train-eval-ablate-max-batches "${TRAIN_EVAL_ABLATE_MAX_BATCHES}" \
    $( [[ "${GRADIENT_CHECKPOINTING}" != "0" ]] && echo --gradient-checkpointing ) \
    ${FP16} \
    "${MAX_TRAIN_ARGS[@]}" \
    "${FUSION_LAYER_ARGS[@]}" \
    "${LORA_RANK_ARGS[@]}" \
    "${LABEL_SMOOTHING_ARGS[@]}" \
    "${AUDIO_AUGMENT_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    ${EXTRA_ARGS}

echo ""
echo "========================================"
echo "Training complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"
