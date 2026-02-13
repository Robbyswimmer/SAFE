#!/usr/bin/env python3
"""
Download InternVL model weights from HuggingFace.

Downloads InternVL 3.5-8B (or other InternVL variants) for use with SAFE.
InternVL uses custom model code (trust_remote_code=True is required).

Usage:
    # Download InternVL 3.5-8B (default)
    python scripts/download_internvl.py

    # Download to specific directory
    python scripts/download_internvl.py --output-dir /path/to/models

    # Download different variant
    python scripts/download_internvl.py --model internvl3.5-4b
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


INTERNVL_MODELS = {
    # InternVL 3.5 (latest)
    "internvl3.5-1b": "OpenGVLab/InternVL3_5-1B",
    "internvl3.5-2b": "OpenGVLab/InternVL3_5-2B",
    "internvl3.5-4b": "OpenGVLab/InternVL3_5-4B",
    "internvl3.5-8b": "OpenGVLab/InternVL3_5-8B",
    "internvl3.5-14b": "OpenGVLab/InternVL3_5-14B",
    "internvl3.5-38b": "OpenGVLab/InternVL3_5-38B",
    # InternVL 2.5
    "internvl2.5-8b": "OpenGVLab/InternVL2_5-8B",
    "internvl2.5-26b": "OpenGVLab/InternVL2_5-26B",
    "internvl2.5-38b": "OpenGVLab/InternVL2_5-38B",
}

# Helpful aliases for common shorthand names.
# Note: InternVL 3.5 does not publish a native 3B checkpoint.
INTERNVL_MODEL_ALIASES = {
    "internvl3.5-3b": "internvl3.5-4b",
}


def download_model(
    model_name: str,
    output_dir: Path,
    use_auth_token: bool = False,
) -> Path:
    """Download InternVL model from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Please install: pip install huggingface-hub")
        sys.exit(1)

    # Resolve model name
    normalized = model_name.lower()
    if normalized in INTERNVL_MODEL_ALIASES:
        alias = INTERNVL_MODEL_ALIASES[normalized]
        print(
            f"Warning: '{model_name}' is an alias. "
            f"Using '{alias}' ({INTERNVL_MODELS[alias]}).",
            flush=True,
        )
        normalized = alias

    if normalized in INTERNVL_MODELS:
        repo_id = INTERNVL_MODELS[normalized]
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
    print("NOTE: InternVL 3.5-8B is ~17GB. This may take a while.")
    print("NOTE: InternVL uses custom model code (trust_remote_code=True).")
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
        print("  python scripts/download_internvl.py --use-auth-token")
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

    # Check for custom model code (critical for InternVL)
    has_custom_code = (
        list(model_dir.glob("modeling_*.py")) or
        list(model_dir.glob("configuration_*.py"))
    )
    if has_custom_code:
        print("Custom model code found (required for InternVL)")
    else:
        print("Warning: No custom model code found - trust_remote_code may fail")

    # Count total size
    total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    size_gb = total_size / (1024**3)
    print(f"Download verified: {size_gb:.1f}GB total")

    return True


def create_symlink(model_dir: Path, link_name: str = "internvl") -> None:
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
        description="Download InternVL model weights from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download InternVL 3.5-8B to default location
  python scripts/download_internvl.py

  # Download to specific directory
  python scripts/download_internvl.py --output-dir /scratch/models

  # Download smaller variant
  python scripts/download_internvl.py --model internvl3.5-4b

  # Download with authentication (for gated models)
  export HF_TOKEN=your_token
  python scripts/download_internvl.py --use-auth-token

Available model shortcuts:
  internvl3.5-1b   -> OpenGVLab/InternVL3_5-1B   (~2-3GB)
  internvl3.5-2b   -> OpenGVLab/InternVL3_5-2B   (~4GB)
  internvl3.5-4b   -> OpenGVLab/InternVL3_5-4B   (~8GB)
  internvl3.5-3b   -> alias to internvl3.5-4b (no official 3B release)
  internvl3.5-8b   -> OpenGVLab/InternVL3_5-8B   (~17GB, default)
  internvl3.5-14b  -> OpenGVLab/InternVL3_5-14B  (~30GB)
  internvl3.5-38b  -> OpenGVLab/InternVL3_5-38B  (~76GB)
  internvl2.5-8b   -> OpenGVLab/InternVL2_5-8B   (~17GB)
  internvl2.5-26b  -> OpenGVLab/InternVL2_5-26B  (~52GB)
  internvl2.5-38b  -> OpenGVLab/InternVL2_5-38B  (~76GB)
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="internvl3.5-8b",
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
    print("InternVL Model Download")
    print("=" * 60)

    # Resolve model name for display (including aliases)
    display_key = args.model.lower()
    if display_key in INTERNVL_MODEL_ALIASES:
        display_key = INTERNVL_MODEL_ALIASES[display_key]
    if display_key in INTERNVL_MODELS:
        display_name = INTERNVL_MODELS[display_key]
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
    print("python train_safe.py --model-config internvl")
    print()
    print("# Option 2: Override LLM path directly")
    print(f"python train_safe.py --model-config internvl \\")
    print(f"    --llm-model {model_dir}")
    print()


if __name__ == "__main__":
    main()
