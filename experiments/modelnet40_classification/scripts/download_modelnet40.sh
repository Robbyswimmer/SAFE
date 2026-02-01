#!/bin/bash
# Download ModelNet40 dataset in HDF5 format
#
# Source: https://github.com/antao97/PointCloudDatasets
# This provides pre-processed point clouds (2048 points per model, with normals)
#
# Usage:
#   bash experiments/modelnet40_classification/scripts/download_modelnet40.sh

set -e

# Get SAFE root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Data directory (consistent with other datasets)
DATA_DIR="${DATA_PATH:-$SAFE_ROOT/experiments/full_training/data}/modelnet40"

echo "========================================"
echo "ModelNet40 Dataset Download"
echo "========================================"
echo "SAFE root: $SAFE_ROOT"
echo "Data directory: $DATA_DIR"
echo "========================================"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Download HDF5 files from antao97's preprocessed dataset
# These contain 2048 points per model with normals (x,y,z,nx,ny,nz)
BASE_URL="https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"

if [ -f "modelnet40_train.h5" ] && [ -f "modelnet40_test.h5" ]; then
    echo "Dataset already exists. Skipping download."
    echo ""
    echo "Files:"
    ls -lh *.h5
else
    echo "Downloading ModelNet40 HDF5 dataset..."

    # Try the Stanford ShapeNet URL first
    if ! wget -q --show-progress "$BASE_URL" -O modelnet40_hdf5.zip 2>/dev/null; then
        # Fallback: try alternative sources
        ALT_URL="https://raw.githubusercontent.com/antao97/PointCloudDatasets/master/modelnet40_hdf5_2048"
        echo "Primary download failed. Trying alternative source..."

        # Download individual h5 files
        for i in 0 1 2 3 4; do
            wget -q --show-progress "${ALT_URL}/ply_data_train${i}.h5" -O "train_${i}.h5" || true
        done
        for i in 0 1; do
            wget -q --show-progress "${ALT_URL}/ply_data_test${i}.h5" -O "test_${i}.h5" || true
        done

        # Merge h5 files if downloaded individually
        if [ -f "train_0.h5" ]; then
            echo "Merging HDF5 files..."
            python3 << 'EOF'
import h5py
import numpy as np
import glob

def merge_h5_files(pattern, output_file):
    files = sorted(glob.glob(pattern))
    if not files:
        return False

    all_data = []
    all_labels = []

    for f in files:
        with h5py.File(f, 'r') as h5f:
            all_data.append(h5f['data'][:])
            all_labels.append(h5f['label'][:])

    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    with h5py.File(output_file, 'w') as h5f:
        h5f.create_dataset('data', data=data)
        h5f.create_dataset('label', data=labels)

    print(f"Created {output_file}: {data.shape[0]} samples")
    return True

merge_h5_files('train_*.h5', 'modelnet40_train.h5')
merge_h5_files('test_*.h5', 'modelnet40_test.h5')
EOF
            # Clean up individual files
            rm -f train_*.h5 test_*.h5
        fi
    else
        echo "Extracting..."
        unzip -q modelnet40_hdf5.zip

        # Reorganize files
        if [ -d "modelnet40_ply_hdf5_2048" ]; then
            # Merge the split h5 files into single train/test files
            echo "Merging HDF5 files..."
            python3 << 'EOF'
import h5py
import numpy as np
import glob

def merge_h5_files(pattern, output_file):
    files = sorted(glob.glob(pattern))
    if not files:
        return False

    all_data = []
    all_labels = []

    for f in files:
        with h5py.File(f, 'r') as h5f:
            all_data.append(h5f['data'][:])
            all_labels.append(h5f['label'][:])

    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    with h5py.File(output_file, 'w') as h5f:
        h5f.create_dataset('data', data=data)
        h5f.create_dataset('label', data=labels)

    print(f"Created {output_file}: {data.shape[0]} samples")
    return True

merge_h5_files('modelnet40_ply_hdf5_2048/ply_data_train*.h5', 'modelnet40_train.h5')
merge_h5_files('modelnet40_ply_hdf5_2048/ply_data_test*.h5', 'modelnet40_test.h5')
EOF
            # Clean up
            rm -rf modelnet40_ply_hdf5_2048
        fi
        rm -f modelnet40_hdf5.zip
    fi
fi

# Verify download
echo ""
echo "========================================"
echo "Verifying dataset..."
echo "========================================"

python3 << 'EOF'
import h5py
import numpy as np

for split in ['train', 'test']:
    path = f'modelnet40_{split}.h5'
    try:
        with h5py.File(path, 'r') as f:
            data = f['data'][:]
            labels = f['label'][:]
            print(f"{split}: {data.shape[0]} samples, {data.shape[1]} points, {data.shape[2]} dims")
            print(f"  Labels: {labels.min()} - {labels.max()} (should be 0-39)")
            print(f"  Point range: [{data.min():.3f}, {data.max():.3f}]")
    except Exception as e:
        print(f"Error loading {path}: {e}")

# Show class distribution
with h5py.File('modelnet40_train.h5', 'r') as f:
    labels = f['label'][:].squeeze()
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nClass distribution (train): {len(unique)} classes")
    print(f"  Min samples/class: {counts.min()}")
    print(f"  Max samples/class: {counts.max()}")
    print(f"  Mean samples/class: {counts.mean():.1f}")
EOF

echo ""
echo "========================================"
echo "Download complete!"
echo "========================================"
echo "Data saved to: $DATA_DIR"
echo ""
echo "Next steps:"
echo "  sbatch experiments/modelnet40_classification/scripts/train_baseline.sh"
echo "========================================"
