#!/usr/bin/env python3
"""
Download Qwen model weights from HuggingFace.

Downloads Qwen3-14B (or other Qwen variants) for use with SAFE.

Usage:
    # Download Qwen3-14B (default)
    python scripts/download_qwen.py

    # Download to specific directory
    python scripts/download_qwen.py --output-dir /path/to/models

    # Download different variant
    python scripts/download_qwen.py --model qwen3-8b
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


QWEN_MODELS = {
    # Qwen3 (latest)
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    # Qwen2.5 (fallback)
    "qwen2.5-14b": "Qwen/Qwen2.5-14B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
}


def download_model(
    model_name: str,
    output_dir: Path,
    use_auth_token: bool = False,
) -> Path:
    """Download Qwen model from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Please install: pip install huggingface-hub")
        sys.exit(1)

    # Resolve model name
    if model_name.lower() in QWEN_MODELS:
        repo_id = QWEN_MODELS[model_name.lower()]
    else:
        repo_id = model_name  # Assume full HF repo name

    model_dir = output_dir / repo_id.replace("/", "_")

    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"Model already exists at {model_dir}")
        print("Delete the directory to re-download.")
        return model_dir

    print(f"Downloading {repo_id} from HuggingFace...")
    print(f"Output directory: {model_dir}")
    print()
    print("NOTE: Qwen3-14B is ~28GB. This may take a while.")
    print()

    try:
        # Download all model files
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN") if use_auth_token else None,
        )
        print(f"\nDownloaded to: {local_path}")
        return Path(local_path)

    except Exception as e:
        print(f"Error downloading model: {e}")
        print()
        print("If you need authentication, set HF_TOKEN environment variable:")
        print("  export HF_TOKEN=your_token_here")
        print("  python scripts/download_qwen.py --use-auth-token")
        sys.exit(1)


def verify_download(model_dir: Path) -> bool:
    """Verify the download contains required files."""
    required_files = [
        "config.json",
        "tokenizer.json",
    ]

    # Check for model weights (either safetensors or bin)
    has_weights = (
        list(model_dir.glob("*.safetensors")) or
        list(model_dir.glob("model*.bin")) or
        list(model_dir.glob("pytorch_model*.bin"))
    )

    missing = []
    for f in required_files:
        if not (model_dir / f).exists():
            missing.append(f)

    if missing:
        print(f"Warning: Missing files: {missing}")
        return False

    if not has_weights:
        print("Warning: No model weight files found")
        return False

    # Count total size
    total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    size_gb = total_size / (1024**3)
    print(f"Download verified: {size_gb:.1f}GB total")

    return True


def create_symlink(model_dir: Path, link_name: str = "qwen") -> None:
    """Create a convenient symlink to the model."""
    link_path = model_dir.parent / link_name

    if link_path.exists():
        if link_path.is_symlink():
            link_path.unlink()
        else:
            print(f"Cannot create symlink: {link_path} already exists")
            return

    link_path.symlink_to(model_dir.name)
    print(f"Created symlink: {link_path} -> {model_dir.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen model weights from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Qwen3-14B to default location
  python scripts/download_qwen.py

  # Download to specific directory
  python scripts/download_qwen.py --output-dir /scratch/models

  # Download smaller variant
  python scripts/download_qwen.py --model qwen3-8b

  # Download with authentication (for gated models)
  export HF_TOKEN=your_token
  python scripts/download_qwen.py --use-auth-token

Available model shortcuts:
  qwen3-14b    -> Qwen/Qwen3-14B (28GB, recommended)
  qwen3-8b     -> Qwen/Qwen3-8B (16GB)
  qwen3-32b    -> Qwen/Qwen3-32B (65GB)
  qwen2.5-14b  -> Qwen/Qwen2.5-14B (28GB, fallback)
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-14b",
        help="Model to download (shortcut or full HF repo name)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for downloaded models (default: models/)",
    )
    parser.add_argument(
        "--use-auth-token",
        action="store_true",
        help="Use HF_TOKEN environment variable for authentication",
    )
    parser.add_argument(
        "--no-symlink",
        action="store_true",
        help="Don't create convenience symlink",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Qwen Model Download")
    print("=" * 60)

    # Resolve model name for display
    if args.model.lower() in QWEN_MODELS:
        display_name = QWEN_MODELS[args.model.lower()]
    else:
        display_name = args.model

    print(f"Model: {display_name}")
    print(f"Output: {output_dir.absolute()}")
    print()

    # Download
    model_dir = download_model(
        args.model,
        output_dir,
        use_auth_token=args.use_auth_token,
    )

    # Verify
    print()
    if verify_download(model_dir):
        print("Download complete and verified!")
    else:
        print("Download may be incomplete. Check the output directory.")

    # Create symlink
    if not args.no_symlink:
        create_symlink(model_dir)

    # Print usage instructions
    print()
    print("=" * 60)
    print("Usage with SAFE:")
    print("=" * 60)
    print()
    print("# Option 1: Use the config (recommended)")
    print("python train_safe.py --model-config qwen3_14b")
    print()
    print("# Option 2: Override LLM path directly")
    print(f"python train_safe.py --model-config kv_augment \\")
    print(f"    --llm-model {model_dir}")
    print()


if __name__ == "__main__":
    main()
