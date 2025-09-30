#!/bin/bash -l

#SBATCH --job-name="SAFE-FullTrain"
#SBATCH --output=logs/full_training_%j.txt
#SBATCH --error=logs/full_training_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH -p gpu

set -euo pipefail

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

CONDA_ENV=${CONDA_ENV:-safe-env}
echo "Activating conda environment '${CONDA_ENV}'"
conda activate "${CONDA_ENV}"

echo "Starting SAFE full-scale training at $(date)"

DATA_ROOT=${DATA_ROOT:-"$PWD/experiments/full_training/data"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PWD/experiments/full_training/runs/${SLURM_JOB_ID}"}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
VAL_AUDIO_SPLIT=${VAL_AUDIO_SPLIT:-val}
VAL_VQA_SPLIT=${VAL_VQA_SPLIT:-val}
NUM_EPOCHS=${NUM_EPOCHS:-10}
TRAIN_BS=${TRAIN_BS:-8}
VAL_BS=${VAL_BS:-16}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
MODEL_CONFIG=${MODEL_CONFIG:-full}
LR_PROJECTOR=${LR_PROJECTOR:-2e-4}
LR_ADAPTER=${LR_ADAPTER:-1e-4}
NULL_SPACE_RANK=${NULL_SPACE_RANK:-8}
NULL_SPACE_MIN_SAMPLES=${NULL_SPACE_MIN_SAMPLES:-128}
NULL_SPACE_REFRESH=${NULL_SPACE_REFRESH:-4000}
MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:--1}
MAX_AUDIO_EVAL_BATCHES=${MAX_AUDIO_EVAL_BATCHES:-32}
MAX_VL_EVAL_BATCHES=${MAX_VL_EVAL_BATCHES:-64}
MAX_AUDIO_VAL_SAMPLES=${MAX_AUDIO_VAL_SAMPLES:-4096}
MAX_VQA_VAL_SAMPLES=${MAX_VQA_VAL_SAMPLES:-4096}
EVAL_LOGGING_STEPS=${EVAL_LOGGING_STEPS:-10}
TRAIN_ACCURACY_INTERVAL=${TRAIN_ACCURACY_INTERVAL:-0}
TRAIN_ACCURACY_WARMUP=${TRAIN_ACCURACY_WARMUP:-5}
TRAIN_EVAL_BATCHES=${TRAIN_EVAL_BATCHES:-0}
GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS:-32}
DEBUG_LOGGING=${DEBUG_LOGGING:-0}
DISABLE_EVAL_AUDIO_GATE=${DISABLE_EVAL_AUDIO_GATE:-0}
EVAL_AUDIO_GATE_COMPARISON=${EVAL_AUDIO_GATE_COMPARISON:-0}
DISABLE_TRAIN_SHUFFLE=${DISABLE_TRAIN_SHUFFLE:-0}
DISABLE_VAL_SHUFFLE=${DISABLE_VAL_SHUFFLE:-0}
VARIANT_ORDER=${VARIANT_ORDER:-"no_retention soft_retention fisher_retention nullspace_retention full_retention"}

mkdir -p logs
mkdir -p "$OUTPUT_ROOT"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] Expected data root at '$DATA_ROOT' but it was not found." >&2
  exit 1
fi

variants=($VARIANT_ORDER)

debug_flag=()
if [[ "$DEBUG_LOGGING" != "0" ]]; then
  debug_flag+=(--debug-logging)
fi

if [[ "$DISABLE_EVAL_AUDIO_GATE" != "0" ]]; then
  eval_gate_flag=(--disable-eval-audio-gate)
else
  eval_gate_flag=()
fi

if [[ "$DISABLE_TRAIN_SHUFFLE" != "0" ]]; then
  train_shuffle_flag=(--disable-train-shuffle)
else
  train_shuffle_flag=()
fi

if [[ "$DISABLE_VAL_SHUFFLE" != "0" ]]; then
  val_shuffle_flag=(--disable-val-shuffle)
else
  val_shuffle_flag=()
fi

eval_comparison_flag=()
if [[ "$EVAL_AUDIO_GATE_COMPARISON" != "0" ]]; then
  eval_comparison_flag=(--eval-audio-gate-comparison)
fi

for variant in "${variants[@]}"; do
  echo "\n=== Running variant: ${variant} ==="
  python -m experiments.full_training.run_full_training \
    --variant "${variant}" \
    --seed "${SEED}" \
    --data-root "${DATA_ROOT}" \
    --train-split "${TRAIN_SPLIT}" \
    --val-audio-split "${VAL_AUDIO_SPLIT}" \
    --val-vqa-split "${VAL_VQA_SPLIT}" \
    --train-batch-size "${TRAIN_BS}" \
    --val-batch-size "${VAL_BS}" \
    --num-workers "${NUM_WORKERS}" \
    --num-epochs "${NUM_EPOCHS}" \
    --lr-projector "${LR_PROJECTOR}" \
    --lr-adapter "${LR_ADAPTER}" \
    --null-space-rank "${NULL_SPACE_RANK}" \
    --null-space-min-samples "${NULL_SPACE_MIN_SAMPLES}" \
    --null-space-refresh "${NULL_SPACE_REFRESH}" \
    --max-eval-batches "${MAX_EVAL_BATCHES}" \
    --max-audio-eval-batches "${MAX_AUDIO_EVAL_BATCHES}" \
    --max-vl-eval-batches "${MAX_VL_EVAL_BATCHES}" \
    --max-audio-val-samples "${MAX_AUDIO_VAL_SAMPLES}" \
    --max-vqa-val-samples "${MAX_VQA_VAL_SAMPLES}" \
    --eval-logging-steps "${EVAL_LOGGING_STEPS}" \
    --train-accuracy-interval "${TRAIN_ACCURACY_INTERVAL}" \
    --train-accuracy-warmup "${TRAIN_ACCURACY_WARMUP}" \
    --train-eval-batches "${TRAIN_EVAL_BATCHES}" \
    --generation-max-new-tokens "${GEN_MAX_NEW_TOKENS}" \
    --output-root "${OUTPUT_ROOT}" \
    --model-config "${MODEL_CONFIG}" \
    "${debug_flag[@]}" \
    "${eval_gate_flag[@]}" \
    "${eval_comparison_flag[@]}" \
    "${train_shuffle_flag[@]}" \
    "${val_shuffle_flag[@]}"
  echo "=== Completed variant: ${variant} ===\n"
done

echo "SAFE full-scale training sweep complete at $(date)"
