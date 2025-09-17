#!/bin/bash -l

#SBATCH --job-name="SAFE-Overfit"
#SBATCH --output=logs/overfitting_%j.txt
#SBATCH --error=logs/overfitting_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=you@example.com   # <-- update to your email or comment out if not needed
#SBATCH -p batch

set -euo pipefail

# Optional: load/activate your environment
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
# TODO: replace 'safe-env' with the correct environment name for your cluster setup
echo "Activating conda environment 'safe-env'"
conda activate safe-env

echo "Starting SAFE overfitting ablation at $(date)"

DATA_ROOT=${DATA_ROOT:-"./data"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"experiments/overfitting/runs/${SLURM_JOB_ID}"}
SUBSET_SIZE=${SUBSET_SIZE:-400}
TRAIN_SOURCE=${TRAIN_SOURCE:-"pack"}
VAL_SOURCE=${VAL_SOURCE:-"pack_vl"}
VAL_SIZE=${VAL_SIZE:-500}
NUM_EPOCHS=${NUM_EPOCHS:-30}
TRAIN_BS=${TRAIN_BS:-8}
VAL_BS=${VAL_BS:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}
PACK_ROOT=${PACK_ROOT:-"experiments/overfitting/data_pack"}

mkdir -p logs
mkdir -p "$OUTPUT_ROOT"

variants=(no_retention soft_retention full_safe)

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
    --null-space-refresh 2000
  echo "=== Completed variant: ${variant} ===\n"
done

echo "SAFE overfitting ablation complete at $(date)"
