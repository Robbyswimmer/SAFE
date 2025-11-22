#!/bin/bash -l

#SBATCH --job-name="SAFE-Phase1"
#SBATCH --output=logs/phase1_%j.txt
#SBATCH --error=logs/phase1_%j.err
#SBATCH --time=24:00:00
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
echo "PHASE 1: Signal Verification Experiment"
echo "=========================================="
echo "Goal: Prove frozen LLM can learn to attend to audio"
echo "Method: Remove all capacity bottlenecks"
echo "Started: $(date)"
echo ""

# Phase 1 Configuration
DATA_ROOT=${DATA_ROOT:-"$PWD/experiments/full_training/data"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PWD/experiments/phase1/runs/${SLURM_JOB_ID}"}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
VAL_AUDIO_SPLIT=${VAL_AUDIO_SPLIT:-val}
VAL_VQA_SPLIT=${VAL_VQA_SPLIT:-val}

# Phase 1 Training Parameters
NUM_EPOCHS=20  # Short run - if it doesn't learn by epoch 5, it won't
TRAIN_BS=4     # Reduce due to 32 tokens (from 8)
VAL_BS=8       # Reduce due to 32 tokens (from 16)
NUM_WORKERS=8
SEED=42
MODEL_CONFIG="phase1"  # Use new Phase 1 config

# Phase 1 Learning Rates (AGGRESSIVE - 5x higher than baseline)
LR_PROJECTOR=1e-3      # Was 2e-4
LR_ADAPTER=5e-4        # Was 1e-4

# Gradient Accumulation (target effective BS = 128)
# 4 * 32 = 128 effective batch size
GRADIENT_ACCUMULATION_STEPS=32

# Evaluation limits (keep for speed)
MAX_AUDIO_EVAL_SAMPLES=600
MAX_VL_EVAL_SAMPLES=600
MAX_AUDIO_VAL_SAMPLES=4096
MAX_VQA_VAL_SAMPLES=4096

# PHASE 1 CRITICAL: Disable all retention mechanisms
# Only run no_retention variant to isolate connectivity testing
VARIANT_ORDER="no_retention"

# Other settings
EVAL_LOGGING_STEPS=10
TRAIN_ACCURACY_INTERVAL=0
TRAIN_ACCURACY_WARMUP=5
TRAIN_EVAL_BATCHES=0
GEN_MAX_NEW_TOKENS=32
DEBUG_LOGGING=0
DISABLE_EVAL_AUDIO_GATE=0
EVAL_AUDIO_GATE_COMPARISON=0
DISABLE_TRAIN_SHUFFLE=0
DISABLE_VAL_SHUFFLE=1
USE_WAVCAPS=0
USE_AUDIOSETCAPS=0
DISABLE_BERTSCORE=0
PROGRESS_LOG_TIMEOUT=600
SAVE_AUDIO_CSV=0

mkdir -p logs
mkdir -p "$OUTPUT_ROOT"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] Expected data root at '$DATA_ROOT' but it was not found." >&2
  exit 1
fi

echo "Phase 1 Configuration Summary:"
echo "=============================="
echo "Model Changes:"
echo "  Model Config:     $MODEL_CONFIG"
echo "  LoRA Rank:        64 (was 8) - removes cross-modal compression"
echo "  Fusion Layers:    [8, 16, 24, 32] (was [6, 12, 24])"
echo "  Audio Tokens:     32 (was 16) - more granular representation"
echo "  Projector Scale:  10.0 (was 2.0) - match LLaVA embedding magnitude"
echo "  Residual Init:    0.3 (was 0.05) - force immediate gradient flow"
echo "  Residual Max:     1.0 (was 0.3) - allow stronger audio influence"
echo ""
echo "Training Changes:"
echo "  Projector LR:     $LR_PROJECTOR (was 2e-4) - 5x higher"
echo "  Adapter LR:       $LR_ADAPTER (was 1e-4) - 5x higher"
echo "  Epochs:           $NUM_EPOCHS (short experiment)"
echo "  Train BS:         $TRAIN_BS (reduced due to more tokens)"
echo "  Grad Accum:       $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective BS:     $((TRAIN_BS * GRADIENT_ACCUMULATION_STEPS))"
echo "  Retention:        DISABLED (no retention loss)"
echo ""
echo "Expected Outcomes:"
echo "  Trainable params: ~380M (was 189M) - 2x increase"
echo "  Target CIDEr:     30-35 (baseline was 18)"
echo "  Loss convergence: Epoch 2-3 (should be fast)"
echo ""
echo "Success Criteria:"
echo "  ✅ CIDEr > 30     = Connectivity VERIFIED"
echo "  ⚠️  CIDEr 25-30   = Partial success"
echo "  ❌ CIDEr < 25     = Still blocked"
echo ""
echo "Starting training at $(date)"
echo "=============================="
echo ""

# Build argument array
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
    --eval-logging-steps "$EVAL_LOGGING_STEPS"
    --train-accuracy-interval "$TRAIN_ACCURACY_INTERVAL"
    --train-accuracy-warmup "$TRAIN_ACCURACY_WARMUP"
    --train-eval-batches "$TRAIN_EVAL_BATCHES"
    --generation-max-new-tokens "$GEN_MAX_NEW_TOKENS"
    --progress-log-timeout "$PROGRESS_LOG_TIMEOUT"
)

# Add flags
[[ "$DEBUG_LOGGING" != "0" ]] && args+=(--debug-logging)
[[ "$DISABLE_EVAL_AUDIO_GATE" != "0" ]] && args+=(--disable-eval-audio-gate)
[[ "$EVAL_AUDIO_GATE_COMPARISON" != "0" ]] && args+=(--eval-audio-gate-comparison)
[[ "$DISABLE_TRAIN_SHUFFLE" != "0" ]] && args+=(--disable-train-shuffle)
[[ "$DISABLE_VAL_SHUFFLE" != "0" ]] && args+=(--disable-val-shuffle)
[[ "$DISABLE_BERTSCORE" != "0" ]] && args+=(--disable-bertscore)
[[ "$SAVE_AUDIO_CSV" != "0" ]] && args+=(--save-audio-csv)

# Run training
python -u experiments/full_training/run_full_training.py "${args[@]}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Phase 1 Complete: $(date)"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo ""

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Next Steps:"
    echo "  1. Check validation logs for CIDEr score"
    echo "  2. Look for: 'audio_cider' in evaluation output"
    echo "  3. If CIDEr > 30: CONNECTIVITY VERIFIED ✅"
    echo "  4. If CIDEr 25-30: Partial success, may need Phase 2 ⚠️"
    echo "  5. If CIDEr < 25: Check loss convergence for issues ❌"
    echo ""
    echo "To proceed to Phase 2 (if successful):"
    echo "  - Remove projector bottleneck (1024 → None or 4096)"
    echo "  - Train for 50-100 epochs"
    echo "  - Target CIDEr: 35-42"
else
    echo "Training failed with exit code $EXIT_CODE"
    echo "Check logs at:"
    echo "  stdout: logs/phase1_${SLURM_JOB_ID}.txt"
    echo "  stderr: logs/phase1_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
