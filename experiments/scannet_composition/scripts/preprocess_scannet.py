#!/usr/bin/env python3
"""
Preprocess ScanNet dataset for scene classification.

Extracts:
- Point clouds from mesh files
- Representative RGB frames from .sens files
- Scene type labels

Usage:
    python experiments/scannet_composition/scripts/preprocess_scannet.py \
        --scannet-root /path/to/scannet \
        --output-dir experiments/full_training/data
"""

import argparse
import json
import os
import struct
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

try:
    import plyfile
except ImportError:
    print("Please install plyfile: pip install plyfile")
    exit(1)


def load_ply_mesh(ply_path: Path) -> np.ndarray:
    """Load point cloud from PLY mesh file."""
    plydata = plyfile.PlyData.read(str(ply_path))
    vertices = plydata["vertex"]

    x = np.array(vertices["x"])
    y = np.array(vertices["y"])
    z = np.array(vertices["z"])

    points = np.stack([x, y, z], axis=-1)
    return points.astype(np.float32)


def load_scene_type(scene_dir: Path) -> str:
    """Load scene type from scene metadata."""
    # Try to load from .txt file
    txt_file = scene_dir / f"{scene_dir.name}.txt"
    if txt_file.exists():
        with open(txt_file) as f:
            for line in f:
                if line.startswith("sceneType"):
                    return line.split("=")[1].strip().lower().replace(" ", "_")

    return "unknown"


class SensReader:
    """Reader for ScanNet .sens files to extract RGB frames."""

    def __init__(self, sens_path: str):
        self.sens_path = sens_path
        self.file = open(sens_path, "rb")
        self._read_header()

    def _read_header(self):
        """Read .sens file header."""
        # Version
        self.version = struct.unpack("I", self.file.read(4))[0]

        # String length + sensor name
        strlen = struct.unpack("Q", self.file.read(8))[0]
        self.sensor_name = self.file.read(strlen).decode("utf-8")

        # Intrinsics
        self.intrinsic_color = np.frombuffer(self.file.read(64), dtype=np.float64).reshape(4, 4)
        self.extrinsic_color = np.frombuffer(self.file.read(64), dtype=np.float64).reshape(4, 4)
        self.intrinsic_depth = np.frombuffer(self.file.read(64), dtype=np.float64).reshape(4, 4)
        self.extrinsic_depth = np.frombuffer(self.file.read(64), dtype=np.float64).reshape(4, 4)

        # Color compression type
        self.color_compression = struct.unpack("i", self.file.read(4))[0]

        # Depth compression type
        self.depth_compression = struct.unpack("i", self.file.read(4))[0]

        # Image dimensions
        self.color_width = struct.unpack("I", self.file.read(4))[0]
        self.color_height = struct.unpack("I", self.file.read(4))[0]
        self.depth_width = struct.unpack("I", self.file.read(4))[0]
        self.depth_height = struct.unpack("I", self.file.read(4))[0]

        # Depth shift
        self.depth_shift = struct.unpack("I", self.file.read(4))[0]

        # Number of frames
        self.num_frames = struct.unpack("Q", self.file.read(8))[0]

    def read_frame(self, frame_idx: int) -> tuple:
        """Read a specific frame (simplified - reads sequentially)."""
        # This is a simplified reader - for production use ScanNet's official reader
        # For now, we'll extract a middle frame

        # Skip to frame data
        for _ in range(frame_idx):
            # Camera to world matrix
            self.file.read(64)
            # Timestamp
            self.file.read(8)
            # Color size + data
            color_size = struct.unpack("Q", self.file.read(8))[0]
            self.file.read(color_size)
            # Depth size + data
            depth_size = struct.unpack("Q", self.file.read(8))[0]
            self.file.read(depth_size)

        # Read target frame
        cam_to_world = np.frombuffer(self.file.read(64), dtype=np.float64).reshape(4, 4)
        timestamp = struct.unpack("Q", self.file.read(8))[0]

        color_size = struct.unpack("Q", self.file.read(8))[0]
        color_data = self.file.read(color_size)

        depth_size = struct.unpack("Q", self.file.read(8))[0]
        depth_data = self.file.read(depth_size)

        return color_data, depth_data, cam_to_world

    def close(self):
        self.file.close()


def extract_middle_frame(sens_path: Path, output_path: Path) -> bool:
    """Extract middle frame from .sens file."""
    try:
        reader = SensReader(str(sens_path))

        # Get middle frame
        mid_frame = reader.num_frames // 2

        # Reset file position and read frame
        reader.file.seek(0)
        reader._read_header()
        color_data, _, _ = reader.read_frame(mid_frame)
        reader.close()

        # Decode JPEG
        from io import BytesIO
        img = Image.open(BytesIO(color_data))
        img.save(str(output_path))

        return True

    except Exception as e:
        print(f"Error extracting frame from {sens_path}: {e}")
        return False


def preprocess_scene(
    scene_dir: Path,
    output_dir: Path,
    num_points: int = 8192,
) -> dict:
    """Preprocess a single scene."""
    scene_id = scene_dir.name

    # Output paths
    pc_output = output_dir / "pointclouds" / f"{scene_id}.npy"
    img_output = output_dir / "images" / f"{scene_id}.jpg"

    # Load mesh and extract point cloud
    mesh_file = scene_dir / f"{scene_id}_vh_clean_2.ply"
    if not mesh_file.exists():
        mesh_file = scene_dir / f"{scene_id}_vh_clean.ply"
    if not mesh_file.exists():
        return None

    points = load_ply_mesh(mesh_file)

    # Subsample
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]

    # Save point cloud
    pc_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(pc_output), points)

    # Extract RGB frame from .sens file
    sens_file = scene_dir / f"{scene_id}.sens"
    if sens_file.exists():
        img_output.parent.mkdir(parents=True, exist_ok=True)
        if not extract_middle_frame(sens_file, img_output):
            # Try to use existing color frames if .sens extraction fails
            color_dir = scene_dir / "color"
            if color_dir.exists():
                frames = sorted(color_dir.glob("*.jpg"))
                if frames:
                    mid_frame = frames[len(frames) // 2]
                    Image.open(mid_frame).save(str(img_output))

    # Get scene type
    scene_type = load_scene_type(scene_dir)

    return {
        "scene_id": scene_id,
        "scene_type": scene_type,
        "pointcloud_path": f"pointclouds/{scene_id}.npy",
        "image_path": f"images/{scene_id}.jpg",
        "num_points": len(points),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess ScanNet for scene classification")
    parser.add_argument("--scannet-root", type=str, required=True, help="Path to ScanNet download")
    parser.add_argument("--output-dir", type=str, default="experiments/full_training/data", help="Output directory")
    parser.add_argument("--num-points", type=int, default=8192, help="Points per scene")
    args = parser.parse_args()

    scannet_root = Path(args.scannet_root)
    output_dir = Path(args.output_dir) / "scannet"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process train and val splits
    for split, folder in [("train", "scans"), ("val", "scans")]:
        scans_dir = scannet_root / folder

        if not scans_dir.exists():
            print(f"Warning: {scans_dir} not found")
            continue

        # Load split file
        split_file = scannet_root / f"scannetv2_{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                scene_ids = [line.strip() for line in f if line.strip()]
        else:
            # Use all scenes in directory
            scene_ids = [d.name for d in scans_dir.iterdir() if d.is_dir()]

        print(f"\nProcessing {split} split ({len(scene_ids)} scenes)...")

        samples = []
        for scene_id in tqdm(scene_ids):
            scene_dir = scans_dir / scene_id
            if not scene_dir.exists():
                continue

            sample = preprocess_scene(scene_dir, output_dir, args.num_points)
            if sample:
                samples.append(sample)

        # Save metadata
        metadata_file = output_dir / f"{split}_samples.json"
        with open(metadata_file, "w") as f:
            json.dump(samples, f, indent=2)

        print(f"Saved {len(samples)} samples to {metadata_file}")

        # Print scene type distribution
        scene_types = {}
        for s in samples:
            st = s["scene_type"]
            scene_types[st] = scene_types.get(st, 0) + 1

        print(f"\nScene type distribution ({split}):")
        for st, count in sorted(scene_types.items(), key=lambda x: -x[1]):
            print(f"  {st}: {count}")

    print("\nPreprocessing complete!")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
