#!/bin/bash -l

# Phase 1-Core: Supervised-only training for the high-capacity SAFE architecture.
# - No SCST / RL
# - No CLAP-based reranking
# - Conservative learning rates
# - Aimed at honest evaluation of the Phase-1 architecture with AudioCaps/WavCaps.

#SBATCH --job-name="SAFE-Phase1Core"
#SBATCH --output=logs/phase1_core_%j.txt
#SBATCH --error=logs/phase1_core_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH -p gpu

set -euo pipefail
export PYTHONUNBUFFERED=1

# Activate environment
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

CONDA_ENV=${CONDA_ENV:-safe-env}
echo "Activating conda environment '${CONDA_ENV}'"
conda activate "${CONDA_ENV}"

echo "=========================================="
echo "PHASE 1-CORE: Supervised Audio Training"
echo "=========================================="
echo "Goal: Evaluate Phase-1 SAFE architecture under plain cross-entropy training"
echo "Started: $(date)"
echo ""

# Data / paths
DATA_ROOT=${DATA_ROOT:-"$PWD/experiments/full_training/data"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PWD/experiments/phase1/runs/${SLURM_JOB_ID}_core"}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
VAL_AUDIO_SPLIT=${VAL_AUDIO_SPLIT:-val}
VAL_VQA_SPLIT=${VAL_VQA_SPLIT:-val}

# Phase 1-Core training params (more conservative than phase1_training.sh)
NUM_EPOCHS=${NUM_EPOCHS:-10}
TRAIN_BS=${TRAIN_BS:-4}
VAL_BS=${VAL_BS:-8}
# Use fewer DataLoader workers to reduce RAM footprint
NUM_WORKERS=${NUM_WORKERS:-2}
SEED=${SEED:-42}
MODEL_CONFIG="phase1"  # Use Phase-1 architecture

# Learning rates: match "full" config scale, not aggressive Phase-1 LR
LR_PROJECTOR=${LR_PROJECTOR:-1e-4}
LR_ADAPTER=${LR_ADAPTER:-5e-5}

# Gradient accumulation: moderate effective batch size (default 4 * 8 = 32)
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}

# Evaluation caps for speed / stability
MAX_AUDIO_EVAL_SAMPLES=${MAX_AUDIO_EVAL_SAMPLES:-600}
MAX_VL_EVAL_SAMPLES=${MAX_VL_EVAL_SAMPLES:-600}
MAX_AUDIO_VAL_SAMPLES=${MAX_AUDIO_VAL_SAMPLES:-4096}
MAX_VQA_VAL_SAMPLES=${MAX_VQA_VAL_SAMPLES:-4096}

# Retention / RL settings: explicitly disabled
VARIANT_ORDER="no_retention"
DISABLE_BERTSCORE=${DISABLE_BERTSCORE:-1}

DEBUG_LOGGING=${DEBUG_LOGGING:-0}
DISABLE_EVAL_AUDIO_GATE=${DISABLE_EVAL_AUDIO_GATE:-0}
EVAL_AUDIO_GATE_COMPARISON=${EVAL_AUDIO_GATE_COMPARISON:-1}
DISABLE_TRAIN_SHUFFLE=${DISABLE_TRAIN_SHUFFLE:-0}
DISABLE_VAL_SHUFFLE=${DISABLE_VAL_SHUFFLE:-1}
USE_WAVCAPS=${USE_WAVCAPS:-0}
WAVCAPS_RATIO=${WAVCAPS_RATIO:-0.5}
SAVE_AUDIO_CSV=${SAVE_AUDIO_CSV:-0}
EVAL_AUDIO_TEST=${EVAL_AUDIO_TEST:-1}
MAX_AUDIO_TEST_SAMPLES=${MAX_AUDIO_TEST_SAMPLES:-600}

mkdir -p logs
mkdir -p "$OUTPUT_ROOT"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] Expected data root at '$DATA_ROOT' but it was not found." >&2
  exit 1
fi

echo "Phase 1-Core Configuration Summary:"
echo "==================================="
echo "Model Config:     $MODEL_CONFIG (Phase-1 architecture)"
echo "Variant:          no_retention (no KL/Fisher/nullspace)"
echo "LR Projector:     $LR_PROJECTOR"
echo "LR Adapter:       $LR_ADAPTER"
echo "Epochs:           $NUM_EPOCHS"
echo "Train BS:         $TRAIN_BS"
echo "Grad Accum:       $GRADIENT_ACCUMULATION_STEPS"
echo "Effective BS:     $((TRAIN_BS * GRADIENT_ACCUMULATION_STEPS))"
echo "Use WavCaps:      $USE_WAVCAPS (ratio=$WAVCAPS_RATIO)"
echo "Disable BERTScore:$DISABLE_BERTSCORE"
echo ""
echo "Starting training at $(date)"
echo "============================"
echo ""

args=(
    --model-config "$MODEL_CONFIG"
    --variant no_retention
    --data-root "$DATA_ROOT"
    --output-root "$OUTPUT_ROOT"
    --seed "$SEED"
    --train-split "$TRAIN_SPLIT"
    --val-audio-split "$VAL_AUDIO_SPLIT"
    --val-vqa-split "$VAL_VQA_SPLIT"
    --train-batch-size "$TRAIN_BS"
    --val-batch-size "$VAL_BS"
    --num-workers "$NUM_WORKERS"
    --num-epochs "$NUM_EPOCHS"
    --lr-projector "$LR_PROJECTOR"
    --lr-adapter "$LR_ADAPTER"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --max-audio-eval-samples "$MAX_AUDIO_EVAL_SAMPLES"
    --max-vl-eval-samples "$MAX_VL_EVAL_SAMPLES"
    --max-audio-val-samples "$MAX_AUDIO_VAL_SAMPLES"
    --max-vqa-val-samples "$MAX_VQA_VAL_SAMPLES"
)

[[ "$DEBUG_LOGGING" != "0" ]] && args+=(--debug-logging)
[[ "$DISABLE_EVAL_AUDIO_GATE" != "0" ]] && args+=(--disable-eval-audio-gate)
[[ "$EVAL_AUDIO_GATE_COMPARISON" != "0" ]] && args+=(--eval-audio-gate-comparison)
[[ "$DISABLE_TRAIN_SHUFFLE" != "0" ]] && args+=(--disable-train-shuffle)
[[ "$DISABLE_VAL_SHUFFLE" != "0" ]] && args+=(--disable-val-shuffle)
[[ "$DISABLE_BERTSCORE" != "0" ]] && args+=(--disable-bertscore)
[[ "$SAVE_AUDIO_CSV" != "0" ]] && args+=(--save-audio-csv)
[[ "$USE_WAVCAPS" != "0" ]] && args+=(--use-wavcaps --wavcaps-ratio "$WAVCAPS_RATIO")
[[ "$EVAL_AUDIO_TEST" != "0" ]] && args+=(--eval-audio-test --max-audio-test-samples "$MAX_AUDIO_TEST_SAMPLES")

python -u experiments/full_training/run_full_training.py "${args[@]}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Phase 1-Core Complete: $(date)"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo ""
