#!/usr/bin/env python3
"""
Extract one representative RGB frame per ScanNet scene from preprocessed frames.

Usage:
    python scripts/extract_scannet_images.py \
        --frames-dir experiments/full_training/data/scannet/scans/tasks/scannet_frames_25k \
        --output-dir experiments/full_training/data/scannet/images
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extract one image per ScanNet scene")
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="experiments/full_training/data/scannet/scans/tasks/scannet_frames_25k",
        help="Path to extracted scannet_frames_25k directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/full_training/data/scannet/images",
        help="Output directory for per-scene images",
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not frames_dir.exists():
        print(f"Error: frames directory not found: {frames_dir}")
        return

    scenes = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    print(f"Found {len(scenes)} scenes in {frames_dir}")

    count = 0
    skipped = 0
    for scene in scenes:
        color_dir = scene / "color"
        if not color_dir.exists():
            skipped += 1
            continue

        frames = sorted(color_dir.glob("*.jpg"))
        if not frames:
            skipped += 1
            continue

        # Pick the middle frame as representative
        mid = frames[len(frames) // 2]
        dst = output_dir / f"{scene.name}.jpg"
        shutil.copy2(mid, dst)
        count += 1

    print(f"Copied {count} images to {output_dir}")
    if skipped:
        print(f"Skipped {skipped} scenes (no color frames)")


if __name__ == "__main__":
    main()
