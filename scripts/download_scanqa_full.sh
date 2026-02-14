#!/usr/bin/env bash
# ============================================================
# Download & preprocess ScanNet data for ScanQA experiments.
#
# What this does:
#   1. Clones ScanQA repo to get annotation JSONs
#   2. Downloads ONLY _vh_clean_2.ply from ScanNet (meshes)
#      for the ~800 scenes referenced in ScanQA (~24GB)
#   3. Downloads scannet_frames_25k.zip (5.6GB preprocessed frames)
#   4. Preprocesses: PLY → NPY point clouds, picks one frame per scene
#   5. Outputs in format expected by ScanQADataset
#
# Usage:
#   bash scripts/download_scanqa_full.sh /path/to/data
#
# Output structure:
#   <DATA_DIR>/
#     scanqa/
#       ScanQA_v1.0_train.json
#       ScanQA_v1.0_val.json
#       ScanQA_v1.0_test.json  (if available)
#     scannet/
#       pointclouds/{scene_id}.npy     (N x 6, xyz + rgb)
#       images/{scene_id}.jpg          (representative frame)
#       scans/{scene_id}/...           (raw, can delete after)
#       frames_25k/...                 (extracted frames)
# ============================================================
set -euo pipefail

DATA_DIR="${1:?Usage: $0 <data_dir>}"
SCANNET_DOWNLOAD_SCRIPT="${2:-download-scannet.py}"

echo "=== ScanQA Data Pipeline ==="
echo "Data directory: $DATA_DIR"
echo ""

mkdir -p "$DATA_DIR"

# ============================================================
# Step 1: Get ScanQA annotations
# ============================================================
echo "=== Step 1: ScanQA annotations ==="
SCANQA_DIR="$DATA_DIR/scanqa"
mkdir -p "$SCANQA_DIR"

if [ -f "$SCANQA_DIR/ScanQA_v1.0_train.json" ]; then
    echo "ScanQA annotations already exist, skipping."
else
    echo "Cloning ScanQA repo..."
    TMPDIR_QA=$(mktemp -d)
    git clone --depth 1 https://github.com/ATR-DBI/ScanQA.git "$TMPDIR_QA/ScanQA" 2>/dev/null || {
        echo "ERROR: Could not clone ScanQA repo."
        echo "Download annotations manually from https://github.com/ATR-DBI/ScanQA"
        exit 1
    }

    # ScanQA annotations are typically in data/qa/ in the repo
    # Try multiple possible locations
    for candidate in \
        "$TMPDIR_QA/ScanQA/data/qa" \
        "$TMPDIR_QA/ScanQA/data" \
        "$TMPDIR_QA/ScanQA"; do
        if ls "$candidate"/ScanQA_v1.0_*.json 2>/dev/null | head -1 > /dev/null; then
            cp "$candidate"/ScanQA_v1.0_*.json "$SCANQA_DIR/"
            echo "Copied annotations from $candidate"
            break
        fi
    done

    # Check if we got them
    if [ ! -f "$SCANQA_DIR/ScanQA_v1.0_train.json" ]; then
        echo "WARNING: Could not find ScanQA_v1.0_*.json in cloned repo."
        echo "The annotations may need to be downloaded from Google Drive."
        echo "Check: https://github.com/ATR-DBI/ScanQA/blob/main/docs/dataset.md"
        echo ""
        echo "Looking for any JSON files..."
        find "$TMPDIR_QA/ScanQA" -name "*.json" -path "*/qa/*" | head -10
        find "$TMPDIR_QA/ScanQA" -name "*.json" -path "*/data/*" | head -10
    fi

    rm -rf "$TMPDIR_QA"
fi

echo ""

# ============================================================
# Step 2: Extract unique scene IDs from ScanQA
# ============================================================
echo "=== Step 2: Extract scene IDs ==="
SCENE_LIST="$DATA_DIR/scanqa_scene_ids.txt"

python3 - "$SCANQA_DIR" "$SCENE_LIST" << 'PYEOF'
import json, sys, glob
from pathlib import Path

qa_dir = Path(sys.argv[1])
out_file = sys.argv[2]

scene_ids = set()
for jf in sorted(qa_dir.glob("*.json")):
    try:
        data = json.loads(jf.read_text())
        for item in data:
            sid = item.get("scene_id", item.get("scan_id", ""))
            if sid:
                scene_ids.add(sid)
    except Exception as e:
        print(f"  Warning: could not parse {jf.name}: {e}")

scene_ids = sorted(scene_ids)
with open(out_file, "w") as f:
    for sid in scene_ids:
        f.write(sid + "\n")

print(f"  Found {len(scene_ids)} unique scenes across ScanQA splits")
PYEOF

NUM_SCENES=$(wc -l < "$SCENE_LIST" | tr -d ' ')
echo "  Scene list saved to $SCENE_LIST ($NUM_SCENES scenes)"
echo ""

# ============================================================
# Step 3: Download ScanNet meshes (only _vh_clean_2.ply)
# ============================================================
echo "=== Step 3: Download ScanNet meshes ==="
echo "  Downloading _vh_clean_2.ply for $NUM_SCENES scenes (~24GB)"
echo ""

SCANS_DIR="$DATA_DIR/scannet/scans"
mkdir -p "$SCANS_DIR"

if [ ! -f "$SCANNET_DOWNLOAD_SCRIPT" ]; then
    echo "ERROR: ScanNet download script not found at: $SCANNET_DOWNLOAD_SCRIPT"
    echo "Provide path as second argument: $0 <data_dir> <path_to_download-scannet.py>"
    exit 1
fi

# Download each scene's mesh file individually
DOWNLOADED=0
SKIPPED=0
while IFS= read -r scene_id; do
    out_file="$SCANS_DIR/$scene_id/${scene_id}_vh_clean_2.ply"
    if [ -f "$out_file" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    # Pipe newline to bypass TOS prompt + .sens prompt
    echo -e "\n" | python3 "$SCANNET_DOWNLOAD_SCRIPT" \
        -o "$DATA_DIR/scannet" \
        --id "$scene_id" \
        --type _vh_clean_2.ply \
        --skip_existing 2>&1 | tail -1
    DOWNLOADED=$((DOWNLOADED + 1))

    # Progress
    TOTAL=$((DOWNLOADED + SKIPPED))
    if [ $((TOTAL % 50)) -eq 0 ]; then
        echo "  Progress: $TOTAL / $NUM_SCENES"
    fi
done < "$SCENE_LIST"

echo "  Done: downloaded $DOWNLOADED, skipped $SKIPPED (already existed)"
echo ""

# ============================================================
# Step 4: Download preprocessed frames (for images)
# ============================================================
echo "=== Step 4: Download preprocessed frames (5.6GB) ==="
FRAMES_ZIP="$DATA_DIR/scannet/tasks/scannet_frames_25k.zip"
FRAMES_DIR="$DATA_DIR/scannet/frames_25k"

if [ -d "$FRAMES_DIR" ] && [ "$(ls -A "$FRAMES_DIR" 2>/dev/null)" ]; then
    echo "  Frames already extracted, skipping."
else
    if [ ! -f "$FRAMES_ZIP" ]; then
        echo "  Downloading scannet_frames_25k.zip..."
        echo -e "\n" | python3 "$SCANNET_DOWNLOAD_SCRIPT" \
            -o "$DATA_DIR/scannet" \
            --preprocessed_frames 2>&1 | tail -3
    fi

    if [ -f "$FRAMES_ZIP" ]; then
        echo "  Extracting frames..."
        mkdir -p "$FRAMES_DIR"
        unzip -q "$FRAMES_ZIP" -d "$FRAMES_DIR"
        echo "  Extracted."
    else
        echo "  WARNING: Could not download frames. Will try to extract from .sens files later."
    fi
fi
echo ""

# ============================================================
# Step 5: Preprocess — PLY → NPY point clouds, pick images
# ============================================================
echo "=== Step 5: Preprocess for ScanQADataset ==="

python3 - "$DATA_DIR" "$SCENE_LIST" << 'PYEOF'
"""
Convert ScanNet meshes to point cloud .npy files and
pick one representative image per scene.
"""
import sys
import numpy as np
from pathlib import Path

DATA_DIR = Path(sys.argv[1])
SCENE_LIST = Path(sys.argv[2])

SCANS_DIR = DATA_DIR / "scannet" / "scans"
FRAMES_DIR = DATA_DIR / "scannet" / "frames_25k"
PC_OUT = DATA_DIR / "scannet" / "pointclouds"
IMG_OUT = DATA_DIR / "scannet" / "images"
PC_OUT.mkdir(parents=True, exist_ok=True)
IMG_OUT.mkdir(parents=True, exist_ok=True)

scene_ids = [s.strip() for s in SCENE_LIST.read_text().splitlines() if s.strip()]

# --- Point cloud extraction ---
print(f"  Extracting point clouds from {len(scene_ids)} scenes...")

try:
    import trimesh
    USE_TRIMESH = True
except ImportError:
    try:
        from plyfile import PlyData
        USE_TRIMESH = False
    except ImportError:
        print("  ERROR: Need either 'trimesh' or 'plyfile' package.")
        print("  Install: pip install trimesh  OR  pip install plyfile")
        sys.exit(1)

pc_done = 0
pc_skip = 0
pc_miss = 0

for scene_id in scene_ids:
    out_file = PC_OUT / f"{scene_id}.npy"
    if out_file.exists():
        pc_skip += 1
        continue

    ply_file = SCANS_DIR / scene_id / f"{scene_id}_vh_clean_2.ply"
    if not ply_file.exists():
        pc_miss += 1
        continue

    try:
        if USE_TRIMESH:
            mesh = trimesh.load(str(ply_file), process=False)
            vertices = np.array(mesh.vertices, dtype=np.float32)  # (N, 3)
            # Try to get vertex colors
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                colors = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
                points = np.concatenate([vertices, colors], axis=1)  # (N, 6)
            else:
                points = vertices  # (N, 3)
        else:
            plydata = PlyData.read(str(ply_file))
            verts = plydata['vertex']
            x = np.array(verts['x'], dtype=np.float32)
            y = np.array(verts['y'], dtype=np.float32)
            z = np.array(verts['z'], dtype=np.float32)
            try:
                r = np.array(verts['red'], dtype=np.float32) / 255.0
                g = np.array(verts['green'], dtype=np.float32) / 255.0
                b = np.array(verts['blue'], dtype=np.float32) / 255.0
                points = np.stack([x, y, z, r, g, b], axis=1)
            except ValueError:
                points = np.stack([x, y, z], axis=1)

        np.save(str(out_file), points)
        pc_done += 1
    except Exception as e:
        print(f"    Warning: failed {scene_id}: {e}")
        pc_miss += 1

    if (pc_done + pc_skip) % 100 == 0:
        print(f"    Progress: {pc_done + pc_skip + pc_miss}/{len(scene_ids)}")

print(f"  Point clouds: {pc_done} created, {pc_skip} existed, {pc_miss} missing/failed")

# --- Image extraction ---
print(f"  Picking representative images...")

img_done = 0
img_skip = 0
img_miss = 0

# Possible frame directory structures after unzipping
frame_roots = [
    FRAMES_DIR,
    FRAMES_DIR / "scannet_frames_25k",
    FRAMES_DIR / "frames_25k",
]

for scene_id in scene_ids:
    out_file = IMG_OUT / f"{scene_id}.jpg"
    if out_file.exists():
        img_skip += 1
        continue

    # Search for a color image for this scene
    found = False
    for froot in frame_roots:
        scene_frame_dir = froot / scene_id / "color"
        if not scene_frame_dir.exists():
            scene_frame_dir = froot / scene_id
        if not scene_frame_dir.exists():
            continue

        # Pick middle frame (usually most representative)
        frames = sorted(scene_frame_dir.glob("*.jpg")) + sorted(scene_frame_dir.glob("*.png"))
        if frames:
            mid = len(frames) // 2
            import shutil
            shutil.copy2(str(frames[mid]), str(out_file))
            img_done += 1
            found = True
            break

    if not found:
        img_miss += 1

print(f"  Images: {img_done} created, {img_skip} existed, {img_miss} missing")

# --- Summary ---
print()
print("=" * 50)
print("  SUMMARY")
print("=" * 50)
print(f"  Scenes requested:  {len(scene_ids)}")
print(f"  Point clouds ready: {pc_done + pc_skip}")
print(f"  Images ready:       {img_done + img_skip}")
print(f"  Point clouds dir:   {PC_OUT}")
print(f"  Images dir:         {IMG_OUT}")
print(f"  QA annotations:     {DATA_DIR / 'scanqa'}")
print()

if pc_miss > 0 or img_miss > 0:
    print(f"  WARNING: {pc_miss} scenes missing point clouds, {img_miss} missing images")
    print(f"  ScanQADataset will filter these out automatically.")
print()
print("  To use with SAFE:")
print(f"    --data-path {DATA_DIR}")
PYEOF

echo ""
echo "=== Done ==="
echo "Data ready at: $DATA_DIR"
