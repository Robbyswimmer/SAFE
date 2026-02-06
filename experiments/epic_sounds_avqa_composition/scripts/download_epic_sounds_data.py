#!/usr/bin/env python3
"""
Download helper for EPIC-SOUNDS AV-QA experiment.

This script does three things:
1. Downloads EPIC-SOUNDS and EPIC-KITCHENS annotation CSV files from official GitHub repos.
2. Produces `required_videos.txt` from EPIC-SOUNDS annotations.
3. Optionally invokes `epic-kitchens-download-scripts/epic_downloader.py` to fetch videos.

Notes:
- Actual video downloads may require accepted terms and credentials depending on source host.
- This helper keeps all downloaded artifacts under a single experiment data root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Set
from urllib.request import urlretrieve

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

EPIC_SOUNDS_URLS = {
    "train": "https://raw.githubusercontent.com/epic-kitchens/epic-sounds-annotations/master/EPIC_Sounds_train.csv",
    "validation": "https://raw.githubusercontent.com/epic-kitchens/epic-sounds-annotations/master/EPIC_Sounds_validation.csv",
    "detection": "https://raw.githubusercontent.com/epic-kitchens/epic-sounds-annotations/master/EPIC_Sounds_detection_test_videos.csv",
}

EPIC_100_URLS = {
    "train": "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_train.csv",
    "validation": "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_validation.csv",
    "verb_classes": "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_verb_classes.csv",
    "noun_classes": "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_noun_classes.csv",
}


def _download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {output_path}")
    urlretrieve(url, output_path)


def _read_video_ids(csv_path: Path) -> Set[str]:
    video_ids: Set[str] = set()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = (row.get("video_id") or "").strip()
            if video_id:
                video_ids.add(video_id)
    return video_ids


def _write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(f"{value}\n")


def _read_lines(path: Path) -> list[str]:
    values: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                values.append(value)
    return values


def _validate_video_ids(video_ids: Iterable[str]) -> tuple[list[str], list[str]]:
    """
    EPIC downloader expects IDs like PXX_YY or PXX_YYY.
    Returns (valid_ids, invalid_ids).
    """
    pattern = re.compile(r"^P\d{2}_\d{2,3}$")
    valid: list[str] = []
    invalid: list[str] = []
    for raw in video_ids:
        vid = raw.strip().upper()
        if pattern.match(vid):
            valid.append(vid)
        else:
            invalid.append(raw)
    return valid, invalid


def _run(cmd: List[str], cwd: Path | None = None) -> None:
    print("[run] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _progress(iterable, total: int, desc: str):
    """Use tqdm when available, otherwise return iterable unchanged."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def maybe_clone_download_repo(repo_dir: Path) -> None:
    if repo_dir.exists():
        print(f"[info] downloader repo already exists: {repo_dir}")
        return
    _run([
        "git",
        "clone",
        "https://github.com/epic-kitchens/epic-kitchens-download-scripts.git",
        str(repo_dir),
    ])


def download_videos(
    downloader_repo: Path,
    videos_file: Path,
    output_dir: Path,
    num_workers: int,
    chunksize: int,
    dry_run: bool,
) -> None:
    script = downloader_repo / "epic_downloader.py"
    if not script.exists():
        raise FileNotFoundError(f"Downloader script not found: {script}")

    # The downloader CLI differs across versions. Probe --help and use supported flags.
    help_text = subprocess.check_output(
        [sys.executable, str(script), "--help"],
        text=True,
    )

    # When running with cwd=downloader_repo, call script by filename
    # so we don't accidentally duplicate the repo path.
    base_cmd = [
        sys.executable,
        script.name,
        "--videos",
    ]

    if "--download-path" in help_text:
        base_cmd.extend(["--download-path", str(output_dir)])
    elif "--output-path" in help_text:
        base_cmd.extend(["--output-path", str(output_dir)])
    else:
        raise RuntimeError(
            "Could not find a supported output directory flag in epic_downloader.py help "
            "(expected --download-path or --output-path)."
        )

    # Optional perf flags only if supported by installed downloader version
    if "--num-workers" in help_text:
        base_cmd.extend(["--num-workers", str(num_workers)])
    if "--chunksize" in help_text:
        base_cmd.extend(["--chunksize", str(chunksize)])

    # IMPORTANT: this downloader version expects IDs directly after --specific-videos,
    # not a file path. So we read the file and submit in batches.
    video_ids = _read_lines(videos_file)
    if not video_ids:
        raise RuntimeError(f"No video IDs found in {videos_file}")

    batch_size = max(1, int(chunksize))
    total_batches = (len(video_ids) + batch_size - 1) // batch_size

    for batch_idx in _progress(range(total_batches), total_batches, desc="download-batches"):
        start = batch_idx * batch_size
        end = min(len(video_ids), start + batch_size)
        ids_batch = video_ids[start:end]
        batch_arg = ",".join(ids_batch)
        cmd = list(base_cmd) + ["--specific-videos", batch_arg]

        print(
            f"[download-batch] {batch_idx + 1}/{total_batches} "
            f"videos={len(ids_batch)} first={ids_batch[0]} last={ids_batch[-1]}",
            flush=True,
        )

        if dry_run:
            print("[dry-run] " + " ".join(cmd))
            continue

        # Run from downloader repo so its relative data paths resolve
        # (e.g., data/epic_55_splits.csv, data/epic_100_splits.csv).
        _run(cmd, cwd=downloader_repo)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download EPIC-SOUNDS data and metadata")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("experiments/epic_sounds_avqa_composition/data"),
        help="Experiment data root",
    )
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Invoke epic_downloader.py after preparing required_videos.txt",
    )
    parser.add_argument(
        "--downloader-repo",
        type=Path,
        default=Path("experiments/epic_sounds_avqa_composition/third_party/epic-kitchens-download-scripts"),
        help="Path to epic-kitchens-download-scripts checkout",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ann_dir = args.data_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # 1) Download official metadata CSVs
    for key, url in EPIC_SOUNDS_URLS.items():
        _download(url, ann_dir / f"EPIC_Sounds_{key}.csv")

    for key, url in EPIC_100_URLS.items():
        _download(url, ann_dir / f"EPIC_100_{key}.csv")

    # 2) Build required video list from EPIC-SOUNDS train+val
    train_ids = _read_video_ids(ann_dir / "EPIC_Sounds_train.csv")
    val_ids = _read_video_ids(ann_dir / "EPIC_Sounds_validation.csv")
    all_ids = sorted(train_ids | val_ids)

    videos_file = args.data_root / "required_videos.txt"
    _write_lines(videos_file, all_ids)

    valid_ids, invalid_ids = _validate_video_ids(all_ids)
    filtered_videos_file = args.data_root / "required_videos_filtered.txt"
    _write_lines(filtered_videos_file, sorted(set(valid_ids)))
    invalid_file = args.data_root / "invalid_video_ids.txt"
    _write_lines(invalid_file, invalid_ids)

    summary = {
        "num_train_video_ids": len(train_ids),
        "num_val_video_ids": len(val_ids),
        "num_unique_video_ids": len(all_ids),
        "required_videos_file": str(videos_file),
        "required_videos_filtered_file": str(filtered_videos_file),
        "num_valid_video_ids": len(set(valid_ids)),
        "num_invalid_video_ids": len(invalid_ids),
        "invalid_video_ids_file": str(invalid_file),
    }
    with (args.data_root / "download_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[summary] " + json.dumps(summary, indent=2))

    # 3) Optional video download
    if args.download_videos:
        maybe_clone_download_repo(args.downloader_repo)
        output_dir = args.data_root / "raw_videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        if invalid_ids:
            print(
                "[warn] Some video IDs are not in downloader format and will be skipped. "
                f"See: {invalid_file}",
                flush=True,
            )
        download_videos(
            downloader_repo=args.downloader_repo,
            videos_file=filtered_videos_file,
            output_dir=output_dir,
            num_workers=args.num_workers,
            chunksize=args.chunksize,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
