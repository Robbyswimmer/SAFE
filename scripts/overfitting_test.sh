#!/bin/bash -l

#SBATCH --job-name="SAFE-Overfit"
#SBATCH --output=logs/overfitting_%j.txt
#SBATCH --error=logs/overfitting_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH -p gpu

set -euo pipefail

# Optional: load/activate your environment
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

CONDA_ENV=${CONDA_ENV:-safe-env}
echo "Activating conda environment '${CONDA_ENV}'"
conda activate "${CONDA_ENV}"

echo "Starting SAFE overfitting ablation at $(date)"

DATA_ROOT=${DATA_ROOT:-"$PWD/data_pack"}
PACK_ROOT=${PACK_ROOT:-"$DATA_ROOT"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PWD/experiments/overfitting/runs/${SLURM_JOB_ID}"}
SUBSET_SIZE=${SUBSET_SIZE:-400}
TRAIN_SOURCE=${TRAIN_SOURCE:-"pack"}
VAL_SOURCE=${VAL_SOURCE:-"pack_vl"}
VAL_SIZE=${VAL_SIZE:-500}
NUM_EPOCHS=${NUM_EPOCHS:-100}
TRAIN_BS=${TRAIN_BS:-8}
VAL_BS=${VAL_BS:-16}
NUM_WORKERS=${NUM_WORKERS:-0}
SEED=${SEED:-42}
MODEL_CONFIG=${MODEL_CONFIG:-full}
MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-8}
EVAL_LOGGING_STEPS=${EVAL_LOGGING_STEPS:-1}
DEBUG_LOGGING=${DEBUG_LOGGING:-0}

mkdir -p logs
mkdir -p "$OUTPUT_ROOT"
 
# Sanity check for packaged data
if [[ ! -d "$PACK_ROOT" ]]; then
  echo "[ERROR] Expected data pack at '$PACK_ROOT' but it was not found." >&2
  exit 1
fi

variants=(no_retention soft_retention full_safe)

debug_flag=()
if [[ "${DEBUG_LOGGING}" != "0" ]]; then
  debug_flag+=(--debug-logging)
fi

for variant in "${variants[@]}"; do
  echo "\n=== Running variant: ${variant} ==="
  python -m experiments.overfitting.run_overfitting \
    --variant "${variant}" \
    --subset-size "${SUBSET_SIZE}" \
    --seed "${SEED}" \
    --train-source "${TRAIN_SOURCE}" \
    --val-source "${VAL_SOURCE}" \
    --val-size "${VAL_SIZE}" \
    --train-batch-size "${TRAIN_BS}" \
    --val-batch-size "${VAL_BS}" \
    --num-epochs "${NUM_EPOCHS}" \
    --num-workers "${NUM_WORKERS}" \
    --data-path "${DATA_ROOT}" \
    --pack-root "${PACK_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --null-space-min-samples 64 \
    --null-space-rank 8 \
    --null-space-refresh 2000 \
    --model-config "${MODEL_CONFIG}" \
    --max-eval-batches "${MAX_EVAL_BATCHES}" \
    --eval-logging-steps "${EVAL_LOGGING_STEPS}" \
    "${debug_flag[@]}"
  echo "=== Completed variant: ${variant} ===\n"
done

echo "SAFE overfitting ablation complete at $(date)"
