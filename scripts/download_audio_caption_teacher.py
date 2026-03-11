#!/usr/bin/env python3
"""
Download an external audio captioning teacher model from Hugging Face.

Practical default:
  - MU-NLPC/whisper-small-audio-captioning

Heavier alternatives are included as aliases for convenience.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


AUDIO_CAPTION_TEACHERS = {
    # Practical default: lightweight and easy to run.
    "whisper-small-audio-captioning": "MU-NLPC/whisper-small-audio-captioning",
    # Stronger classical system; may need extra repo-specific integration later.
    "conette-base": "Labbeti/conette-base",
    # Heavier Omni captioner kept as alias for completeness, but not the recommended default.
    "qwen-omni-captioner": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
}


def download_model(model_name: str, output_dir: Path, use_auth_token: bool = False) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Please install: pip install huggingface-hub")
        sys.exit(1)

    normalized = model_name.lower()
    repo_id = AUDIO_CAPTION_TEACHERS.get(normalized, model_name)
    model_dir = output_dir / repo_id.replace("/", "_")

    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"Model already exists at {model_dir}")
        print("Delete the directory to re-download.")
        return model_dir

    print(f"Downloading {repo_id} from HuggingFace...")
    print(f"Output directory: {model_dir}")
    print()
    print("Recommended default for this project: whisper-small-audio-captioning")
    print()

    try:
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
        print("If authentication is required, set HF_TOKEN and rerun with --use-auth-token")
        sys.exit(1)


def verify_download(model_dir: Path) -> bool:
    required_any = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    has_any_required = any((model_dir / name).exists() for name in required_any)
    has_weights = (
        list(model_dir.glob("*.safetensors"))
        or list(model_dir.glob("model*.bin"))
        or list(model_dir.glob("pytorch_model*.bin"))
    )

    if not has_any_required:
        print("Warning: Could not find standard config/tokenizer files")
        return False
    if not has_weights:
        print("Warning: No model weight files found")
        return False

    total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    size_gb = total_size / (1024 ** 3)
    print(f"Download verified: {size_gb:.1f}GB total")
    return True


def create_symlink(model_dir: Path, link_name: str = "audio_caption_teacher") -> None:
    link_path = model_dir.parent / link_name
    if link_path.exists():
        if link_path.is_symlink():
            link_path.unlink()
        else:
            print(f"Cannot create symlink: {link_path} already exists")
            return
    link_path.symlink_to(model_dir.name)
    print(f"Created symlink: {link_path} -> {model_dir.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an external audio captioning teacher model from Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_audio_caption_teacher.py
  python scripts/download_audio_caption_teacher.py --model whisper-small-audio-captioning
  python scripts/download_audio_caption_teacher.py --model MU-NLPC/whisper-small-audio-captioning

Available shortcuts:
  whisper-small-audio-captioning -> MU-NLPC/whisper-small-audio-captioning
  conette-base                   -> Labbeti/conette-base
  qwen-omni-captioner            -> Qwen/Qwen3-Omni-30B-A3B-Captioner
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="whisper-small-audio-captioning",
        help="Model shortcut or full HF repo id",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for downloaded models",
    )
    parser.add_argument(
        "--use-auth-token",
        action="store_true",
        help="Use HF_TOKEN for authenticated downloads",
    )
    parser.add_argument(
        "--no-symlink",
        action="store_true",
        help="Do not create a convenience symlink",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Audio Caption Teacher Download")
    print("=" * 60)
    print(f"Model: {AUDIO_CAPTION_TEACHERS.get(args.model.lower(), args.model)}")
    print(f"Output: {output_dir.absolute()}")
    print()

    model_dir = download_model(args.model, output_dir, use_auth_token=args.use_auth_token)

    print()
    if verify_download(model_dir):
        print("Download complete and verified!")
    else:
        print("Download may be incomplete. Check the output directory.")

    if not args.no_symlink:
        print()
        create_symlink(model_dir)


if __name__ == "__main__":
    main()
