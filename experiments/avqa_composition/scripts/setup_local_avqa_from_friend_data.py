#!/usr/bin/env python3
"""
One-shot local setup for AVQA + MUSIC-AVQA using read-only source files from a
friend directory. All generated files are written under LOCAL_ROOT.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


def split_json(src: Path, out_dir: Path, prefix: str, seed: int, train_ratio: float) -> tuple[Path, Path]:
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list) or not data:
        raise ValueError(f"Unexpected/empty data in {src}")

    random.seed(seed)
    random.shuffle(data)
    k = max(1, int(train_ratio * len(data)))
    train = data[:k]
    val = data[k:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{prefix}_train.json"
    val_path = out_dir / f"{prefix}_val.json"
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train, f, indent=2)
    with val_path.open("w", encoding="utf-8") as f:
        json.dump(val, f, indent=2)

    print(f"[split] {prefix}: total={len(data)} train={len(train)} val={len(val)}")
    print(f"[split] wrote: {train_path}")
    print(f"[split] wrote: {val_path}")
    return train_path, val_path


def run_prepare(
    prep_script: Path,
    dataset: str,
    train_json: Path,
    val_json: Path,
    output_root: Path,
    media_root: Path,
    audio_root: Path,
    image_root: Path,
) -> None:
    cmd = [
        sys.executable,
        str(prep_script),
        "--dataset",
        dataset,
        "--train-json",
        str(train_json),
        "--val-json",
        str(val_json),
        "--output-root",
        str(output_root),
        "--media-root",
        str(media_root),
        "--audio-root",
        str(audio_root),
        "--image-root",
        str(image_root),
        "--require-both",
    ]
    print("[run] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Setup local AVQA manifests from friend data")
    p.add_argument("--local-root", type=Path, default=Path("/data/SalmanAsif/RobbyMoseley/SAFE/SAFE"))
    p.add_argument("--friend-root", type=Path, default=Path("/data/SalmanAsif/Kaykobad-Reza/Model-Merging"))
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.9)
    args = p.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError(f"--train-ratio must be between 0 and 1, got {args.train_ratio}")

    local_root = args.local_root.resolve()
    friend_root = args.friend_root.resolve()
    prep_script = local_root / "experiments/avqa_composition/scripts/prepare_avqa_manifests.py"
    avqa_src = friend_root / "data/test/avqa-test_mm_video+image+audio.json"
    music_src = friend_root / "data/test/music-avqa-test_mm_video+image+audio.json"

    for req in [prep_script, avqa_src, music_src]:
        if not req.exists():
            raise FileNotFoundError(f"Missing required file: {req}")

    avqa_local_root = local_root / "data/avqa_local"
    music_local_root = local_root / "data/music_avqa_local"
    avqa_train, avqa_val = split_json(avqa_src, avqa_local_root, "avqa_pseudo", args.split_seed, args.train_ratio)
    music_train, music_val = split_json(music_src, music_local_root, "music_avqa_pseudo", args.split_seed, args.train_ratio)

    run_prepare(
        prep_script,
        "avqa",
        avqa_train,
        avqa_val,
        avqa_local_root,
        friend_root / "data/evaluation_datasets/AVQA",
        friend_root / "data/evaluation_datasets/AVQA/audio",
        friend_root / "data/evaluation_datasets/AVQA/frames",
    )
    run_prepare(
        prep_script,
        "music_avqa",
        music_train,
        music_val,
        music_local_root,
        friend_root / "data/evaluation_datasets/MUSIC-AVQA",
        friend_root / "data/evaluation_datasets/MUSIC-AVQA/audio",
        friend_root / "data/evaluation_datasets/MUSIC-AVQA/frames",
    )

    print("\n[done] Local setup complete.")
    print(f"[verify] ls -lh {avqa_local_root / 'manifests'}")
    print(f"[verify] ls -lh {music_local_root / 'manifests'}")
    print("[submit AVQA]")
    print(
        "sbatch --gres=gpu:1 "
        f"--export=ALL,DATA_ROOT={avqa_local_root},MEDIA_ROOT=/,OUTPUT_DIR={local_root}/checkpoints/avqa_composition/preffn_local "
        f"{local_root}/experiments/avqa_composition/scripts/train_preffn_avqa.sh"
    )
    print("[submit MUSIC-AVQA]")
    print(
        "sbatch --gres=gpu:1 "
        f"--export=ALL,DATA_ROOT={music_local_root},MEDIA_ROOT=/,OUTPUT_DIR={local_root}/checkpoints/avqa_composition/music_preffn_local "
        f"{local_root}/experiments/avqa_composition/scripts/train_preffn_music_avqa.sh"
    )


if __name__ == "__main__":
    main()

