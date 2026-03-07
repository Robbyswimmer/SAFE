#!/bin/bash
#SBATCH --job-name=preprocess_scannet
#SBATCH --output=logs/preprocess_scannet_%j.out
#SBATCH --error=logs/preprocess_scannet_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Preprocess ScanNet data: extract point clouds from meshes + images from .sens files,
# then validate ScanQA annotations against processed scenes.
# Target cluster: UCR HPCC (cluster.hpcc.ucr.edu)

set -euo pipefail

# ─── HPCC paths ──────────────────────────────────────────────────────
if [[ -z "${SAFE_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    SAFE_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SAFE_ROOT="/bigdata/asiflab/rmose009/SAFE/SAFE"
  fi
fi

DATA_ROOT=${DATA_ROOT:-$SAFE_ROOT/data}

# ─── HPCC module + conda setup ──────────────────────────────────────
CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/.conda/etc/profile.d/conda.sh" ]]; then
  source "$HOME/.conda/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
else
  module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
  eval "$(conda shell.bash hook 2>/dev/null)" || true
fi
conda activate "$CONDA_ENV"

mkdir -p logs

cd "$SAFE_ROOT"

echo "============================================"
echo "  ScanNet Preprocessing — ${SLURM_JOB_ID:-local}"
echo "============================================"
echo "SAFE_ROOT: $SAFE_ROOT"
echo "DATA_ROOT: $DATA_ROOT"
echo "============================================"
echo ""

# Step 1: Extract point clouds from meshes + images from .sens files
echo "[Step 1] Extracting point clouds and images from ScanNet..."
python "$SAFE_ROOT/experiments/scannet_composition/scripts/preprocess_scannet.py" \
    --scannet-root "$DATA_ROOT/scannet/scans" \
    --output-dir "$DATA_ROOT" \
    --num-points 8192

echo ""
echo "[Step 1] Done. Checking outputs..."
PC_COUNT=$(find "$DATA_ROOT/scannet/pointclouds" -name "*.npy" 2>/dev/null | wc -l)
IMG_COUNT=$(find "$DATA_ROOT/scannet/images" -name "*.jpg" 2>/dev/null | wc -l)
echo "  Point clouds: $PC_COUNT .npy files"
echo "  Images: $IMG_COUNT .jpg files"

# Step 2: Validate ScanQA annotations against processed scenes
echo ""
echo "[Step 2] Validating ScanQA annotations..."
python "$SAFE_ROOT/experiments/scanqa_composition/scripts/preprocess_scanqa.py" \
    --scanqa-root "$DATA_ROOT/scanqa" \
    --output-dir "$DATA_ROOT" \
    --validate-scenes

echo ""
echo "Preprocessing complete!"
