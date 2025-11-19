#!/usr/bin/env python3
"""Regenerate AudioCaps metadata from existing audio files."""

import json
from pathlib import Path
import argparse


def regenerate_metadata(audio_dir: Path, output_file: Path, split: str = "train"):
    """Generate metadata entries for all WAV files in audio directory."""

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    wav_files = sorted(audio_dir.glob("*.wav"))

    if not wav_files:
        raise ValueError(f"No WAV files found in {audio_dir}")

    print(f"Found {len(wav_files)} WAV files in {audio_dir}")

    entries = []
    for wav_file in wav_files:
        filename = wav_file.name
        rel_path = f"audio/{split}/{filename}"

        # Extract youtube ID from filename if present (format: YTID_NNNNNN.wav)
        ytid = filename.split("_")[0] if "_" in filename else filename.replace(".wav", "")

        entry = {
            "split": split,
            "sound_name": filename,
            "file_path": rel_path,
            "captions": [""],  # Empty caption - will need to be filled if available
            "ytid": ytid,
        }
        entries.append(entry)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"✓ Wrote {len(entries)} entries to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate AudioCaps metadata from audio files")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing WAV files (e.g., experiments/full_training/data/audiocaps/audio/train)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file (e.g., experiments/full_training/data/audiocaps/train_regenerated.jsonl)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name (train/val/test)",
    )

    args = parser.parse_args()
    regenerate_metadata(args.audio_dir, args.output, args.split)


if __name__ == "__main__":
    main()
