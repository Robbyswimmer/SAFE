#!/bin/bash
#SBATCH --job-name=compose-calib-sweep
#SBATCH --output=logs/compose_calib_sweep_%j.out
#SBATCH --error=logs/compose_calib_sweep_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1

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
export SAFE_GRAD_CKPT=0

SWEEP_ROOT="${SWEEP_ROOT:-checkpoints/composition_calibration_sweep}"
CALIBRATION_BUDGETS="${CALIBRATION_BUDGETS:-1 5 10 50 100 500}"
CALIBRATION_TRAINABLE="${CALIBRATION_TRAINABLE:-fusion}"

mkdir -p "$SAFE_ROOT/logs" "$SWEEP_ROOT"
cd "$SAFE_ROOT"

for N in $CALIBRATION_BUDGETS; do
  OUTPUT_DIR="$SWEEP_ROOT/n${N}_${CALIBRATION_TRAINABLE}" \
  TRAIN_MAX_SAMPLES="$N" \
  CALIBRATION_TRAINABLE="$CALIBRATION_TRAINABLE" \
    bash "$SAFE_ROOT/experiments/avqa_composition/scripts/calibrate_composition_paired.sh"
done

python3 "$SAFE_ROOT/experiments/avqa_composition/scripts/summarize_calibration_sweep.py" \
  --sweep-root "$SWEEP_ROOT" \
  --output "$SWEEP_ROOT/summary.csv"
