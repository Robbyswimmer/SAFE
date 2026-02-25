#!/bin/bash
# Setup script for HPCC cluster: download ScanNet + ScanQA data
# Run from HPCC login node (bluejay)
#
# Usage:
#   bash scripts/setup_hpcc_data.sh
#
# Prerequisites:
#   - HPCC account with access to /bigdata/asiflab/rmose009
#   - ScanNet download permission (already obtained)
#   - conda environment with python3

set -euo pipefail

SAFE_ROOT="/bigdata/asiflab/rmose009/SAFE/SAFE"
SCANNET_DIR="${SAFE_ROOT}/data/scannet"
SCANQA_DIR="${SAFE_ROOT}/data/scanqa"

echo "============================================"
echo "  HPCC Data Setup"
echo "============================================"
echo "SAFE_ROOT: ${SAFE_ROOT}"
echo ""

# ─────────────────────────────────────────────
# 1. ScanQA QA pairs (no registration needed)
# ─────────────────────────────────────────────
echo "[1/4] Downloading ScanQA QA pairs..."
mkdir -p "${SCANQA_DIR}"

if [[ ! -d "${SCANQA_DIR}/repo" ]]; then
  git clone https://github.com/ATR-DBI/ScanQA.git "${SCANQA_DIR}/repo"
  echo "  Cloned ScanQA repo"
else
  echo "  ScanQA repo already exists, skipping"
fi

# Copy QA data to a clean location
if [[ -d "${SCANQA_DIR}/repo/data/qa" ]]; then
  cp -r "${SCANQA_DIR}/repo/data/qa" "${SCANQA_DIR}/qa" 2>/dev/null || true
  echo "  QA pairs copied to ${SCANQA_DIR}/qa"
fi

echo "[1/4] Done."
echo ""

# ─────────────────────────────────────────────
# 2. ScanNet download script
# ─────────────────────────────────────────────
echo "[2/4] Getting ScanNet download script..."
mkdir -p "${SCANNET_DIR}"

DOWNLOAD_SCRIPT="${SCANNET_DIR}/download-scannet.py"
if [[ ! -f "${DOWNLOAD_SCRIPT}" ]]; then
  wget -q -O "${DOWNLOAD_SCRIPT}" http://kaldir.vc.cit.tum.de/scannet/download-scannet.py
  echo "  Downloaded download-scannet.py"
else
  echo "  download-scannet.py already exists"
fi

echo "[2/4] Done."
echo ""

# ─────────────────────────────────────────────
# 3. Download ScanNet meshes + annotations
#    (~30GB for meshes, small for annotations)
# ─────────────────────────────────────────────
echo "[3/4] Downloading ScanNet data (meshes + annotations)..."
echo "  This will take a while (~30GB for meshes)..."
echo ""

# Cleaned triangle meshes (point clouds can be sampled from these)
echo "  [3a] Downloading cleaned meshes (_vh_clean_2.ply)..."
python3 "${DOWNLOAD_SCRIPT}" -o "${SCANNET_DIR}/scans" --type _vh_clean_2.ply || {
  echo "  WARNING: Mesh download may have failed. Check output above."
}

# Object aggregation labels (which objects are in each scene)
echo "  [3b] Downloading aggregation annotations..."
python3 "${DOWNLOAD_SCRIPT}" -o "${SCANNET_DIR}/scans" --type .aggregation.json || {
  echo "  WARNING: Aggregation download may have failed."
}

# Segmentation indices (per-vertex segment IDs)
echo "  [3c] Downloading segmentation indices..."
python3 "${DOWNLOAD_SCRIPT}" -o "${SCANNET_DIR}/scans" --type _vh_clean_2.0.010000.segs.json || {
  echo "  WARNING: Segmentation download may have failed."
}

# Axis alignment matrices (needed for bounding boxes)
echo "  [3d] Downloading axis alignment metadata..."
python3 "${DOWNLOAD_SCRIPT}" -o "${SCANNET_DIR}/scans" --type .txt || {
  echo "  WARNING: Metadata download may have failed."
}

echo "[3/4] Done."
echo ""

# ─────────────────────────────────────────────
# 4. Download train/test splits
# ─────────────────────────────────────────────
echo "[4/4] Downloading ScanNet train/val/test splits..."
SPLITS_DIR="${SCANNET_DIR}/splits"
mkdir -p "${SPLITS_DIR}"

SPLITS_BASE="https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark"
for split in scannetv2_train.txt scannetv2_val.txt scannetv2_test.txt; do
  if [[ ! -f "${SPLITS_DIR}/${split}" ]]; then
    wget -q -O "${SPLITS_DIR}/${split}" "${SPLITS_BASE}/${split}" || echo "  WARNING: Failed to download ${split}"
    echo "  Downloaded ${split}"
  else
    echo "  ${split} already exists"
  fi
done

echo "[4/4] Done."
echo ""

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
echo "============================================"
echo "  Setup Complete"
echo "============================================"
echo ""
echo "ScanQA QA pairs:  ${SCANQA_DIR}/qa/"
echo "ScanNet scans:    ${SCANNET_DIR}/scans/"
echo "ScanNet splits:   ${SCANNET_DIR}/splits/"
echo ""
echo "Verify:"
echo "  ls ${SCANQA_DIR}/qa/"
echo "  ls ${SCANNET_DIR}/scans/ | head -20"
echo "  wc -l ${SCANNET_DIR}/splits/*.txt"
echo ""
echo "Data sizes:"
du -sh "${SCANQA_DIR}" 2>/dev/null || true
du -sh "${SCANNET_DIR}" 2>/dev/null || true
echo ""
echo "Next steps:"
echo "  1. Transfer models from BCC (from BCC terminal):"
echo "     rsync -avP /data/SalmanAsif/RobbyMoseley/SAFE/SAFE/models/ \\"
echo "       rmose009@cluster.hpcc.ucr.edu:${SAFE_ROOT}/models/"
echo "  2. Preprocess ScanNet point clouds (sample points from meshes)"
echo "  3. Extract RGB frames from ScanNet .sens files (if needed for composition)"
