#!/usr/bin/env python3
"""
Download Point-BERT pre-trained weights.

Downloads from Tsinghua Cloud (official source).
Source: https://github.com/Julie-tang00/Point-BERT

Usage:
    python scripts/download_pointbert.py
    python scripts/download_pointbert.py --model all
    python scripts/download_pointbert.py --model modelnet40
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# Official download links from Point-BERT GitHub
# Using Tsinghua Cloud links (direct download with ?dl=1)
POINTBERT_MODELS = {
    # Core pre-trained models
    "dvae": {
        "url": "https://cloud.tsinghua.edu.cn/f/c76274f9afb34cdbb57e/?dl=1",
        "filename": "dvae_shapenet.pt",
        "description": "dVAE model pre-trained on ShapeNet (required for Point-BERT)",
    },
    "pointbert": {
        "url": "https://cloud.tsinghua.edu.cn/f/202b29805eea45d7be92/?dl=1",
        "filename": "pointbert_shapenet.pt",
        "description": "Point-BERT model pre-trained on ShapeNet",
    },

    # Fine-tuned on ModelNet40
    "modelnet40_1k": {
        "url": "https://cloud.tsinghua.edu.cn/f/9be5d9dcbaeb48adb360/?dl=1",
        "filename": "pointbert_modelnet40_1024.pt",
        "description": "Fine-tuned on ModelNet40 (1024 points) - 92.7% acc",
    },
    "modelnet40_4k": {
        "url": "https://cloud.tsinghua.edu.cn/f/121b2651374e4ab1ade6/?dl=1",
        "filename": "pointbert_modelnet40_4096.pt",
        "description": "Fine-tuned on ModelNet40 (4096 points) - 93.4% acc",
    },
    "modelnet40_8k": {
        "url": "https://cloud.tsinghua.edu.cn/f/3ee8e437e07f4dc49738/?dl=1",
        "filename": "pointbert_modelnet40_8192.pt",
        "description": "Fine-tuned on ModelNet40 (8192 points) - 93.8% acc",
    },

    # Fine-tuned on ScanObjectNN
    "scanobjectnn_obj": {
        "url": "https://cloud.tsinghua.edu.cn/f/60260a3cbd8940f5bf0d/?dl=1",
        "filename": "pointbert_scanobjectnn_obj.pt",
        "description": "Fine-tuned on ScanObjectNN (object only)",
    },
    "scanobjectnn_bg": {
        "url": "https://cloud.tsinghua.edu.cn/f/c66c28c771e24cd588ad/?dl=1",
        "filename": "pointbert_scanobjectnn_bg.pt",
        "description": "Fine-tuned on ScanObjectNN (object + background)",
    },
    "scanobjectnn_hard": {
        "url": "https://cloud.tsinghua.edu.cn/f/2edb5b2810dc4bd9b796/?dl=1",
        "filename": "pointbert_scanobjectnn_hardest.pt",
        "description": "Fine-tuned on ScanObjectNN (hardest setting) - 83.1% acc",
    },
}

# Model groups for convenience
MODEL_GROUPS = {
    "core": ["dvae", "pointbert"],
    "modelnet40": ["modelnet40_1k", "modelnet40_4k", "modelnet40_8k"],
    "scanobjectnn": ["scanobjectnn_obj", "scanobjectnn_bg", "scanobjectnn_hard"],
    "all": list(POINTBERT_MODELS.keys()),
    # Recommended for SAFE point cloud training
    "recommended": ["pointbert", "modelnet40_1k"],
}


def download_file(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """Download file with progress bar and verification."""
    try:
        # Use stream and longer timeout for large files
        response = requests.get(
            url,
            stream=True,
            timeout=120,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            # Might be a redirect page or error
            content_length = int(response.headers.get("content-length", 0))
            if content_length < 1_000_000:  # Less than 1MB is suspicious for model
                print(f"  Warning: Got HTML response, might need manual download")
                return False

        total_size = int(response.headers.get("content-length", 0))

        # Download with progress bar
        with open(dest, "wb") as f:
            with tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc=desc,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

        # Verify file was downloaded
        if dest.stat().st_size < 1_000_000:  # Less than 1MB
            print(f"  Warning: File seems too small ({dest.stat().st_size} bytes)")
            return False

        return True

    except Exception as e:
        print(f"  Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Point-BERT pre-trained weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model options:
  core         - dVAE + Point-BERT pre-trained (ShapeNet)
  modelnet40   - All ModelNet40 fine-tuned variants
  scanobjectnn - All ScanObjectNN fine-tuned variants
  recommended  - Point-BERT + ModelNet40-1k (best for SAFE)
  all          - Download everything

Individual models:
  dvae, pointbert, modelnet40_1k, modelnet40_4k, modelnet40_8k,
  scanobjectnn_obj, scanobjectnn_bg, scanobjectnn_hard
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="recommended",
        help="Model(s) to download (default: recommended)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/pointbert",
        help="Output directory for weights",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    # List models
    if args.list:
        print("\nAvailable Point-BERT models:\n")
        for name, info in POINTBERT_MODELS.items():
            print(f"  {name:20} - {info['description']}")
        print("\nModel groups:")
        for group, models in MODEL_GROUPS.items():
            print(f"  {group:20} - {', '.join(models)}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Point-BERT Pre-trained Weights Download")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Source: https://github.com/Julie-tang00/Point-BERT")

    # Determine which models to download
    if args.model in MODEL_GROUPS:
        models_to_download = MODEL_GROUPS[args.model]
    elif args.model in POINTBERT_MODELS:
        models_to_download = [args.model]
    else:
        print(f"\nError: Unknown model '{args.model}'")
        print("Use --list to see available options")
        sys.exit(1)

    print(f"\nModels to download: {', '.join(models_to_download)}")
    print("=" * 60)

    # Download each model
    success_count = 0
    fail_count = 0

    for model_name in models_to_download:
        info = POINTBERT_MODELS[model_name]
        dest_path = output_dir / info["filename"]

        print(f"\n[{model_name}] {info['description']}")

        if dest_path.exists() and not args.force:
            print(f"  Already exists: {dest_path}")
            success_count += 1
            continue

        print(f"  Downloading to: {dest_path}")

        if download_file(info["url"], dest_path, f"  {model_name}"):
            print(f"  ✓ Success: {dest_path.stat().st_size / 1e6:.1f} MB")
            success_count += 1
        else:
            print(f"  ✗ Failed")
            fail_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Download complete: {success_count} succeeded, {fail_count} failed")

    if fail_count > 0:
        print("\nFor failed downloads, try manual download from:")
        print("  https://github.com/Julie-tang00/Point-BERT#pretrained-models")

    # Print usage instructions
    print("\nTo use in SAFE point cloud training:")
    print(f'  POINTBERT_CHECKPOINT="{output_dir}/pointbert_shapenet.pt" \\')
    print("  sbatch --gres=gpu:1 scripts/train_pointcloud.sh")

    print("=" * 60)


if __name__ == "__main__":
    main()
