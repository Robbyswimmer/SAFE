#!/usr/bin/env python3
"""
Download helper for EPIC-SOUNDS AV-QA experiment.

This script does three things:
1. Downloads EPIC-SOUNDS and EPIC-KITCHENS annotation CSV files from official GitHub repos.
2. Produces `required_videos.txt` from EPIC-SOUNDS annotations.
3. Optionally invokes `epic-kitchens-download-scripts/epic_downloader.py` to fetch videos.

Features:
- Resume support: automatically skips already-downloaded videos
- Parallel batch downloads via concurrent.futures
- Real-time subprocess output streaming
- Frequent progress updates with disk usage, speed, and ETA
- Per-video file tracking in output directory

Notes:
- Actual video downloads may require accepted terms and credentials depending on source host.
- This helper keeps all downloaded artifacts under a single experiment data root.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, List, Set
from urllib.request import urlretrieve


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


def _format_seconds(total_seconds: float) -> str:
    total = int(max(0, total_seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def _dir_size(path: Path) -> tuple[int, int]:
    """Return (total_bytes, file_count) for a directory tree."""
    total = 0
    count = 0
    if not path.exists():
        return 0, 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
            count += 1
    return total, count


def _find_downloaded_videos(output_dir: Path) -> set[str]:
    """Scan output dir for already-downloaded video files and return their IDs."""
    downloaded: set[str] = set()
    if not output_dir.exists():
        return downloaded
    video_exts = {".mp4", ".avi", ".mkv", ".webm", ".mov"}
    for f in output_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in video_exts and f.stat().st_size > 0:
            # Extract video ID from filename (e.g., P01_01.MP4 -> P01_01)
            downloaded.add(f.stem.upper())
    return downloaded


def _run_streaming(
    cmd: List[str],
    cwd: Path | None = None,
    *,
    status_every_seconds: int = 15,
    status_label: str | None = None,
    output_dir: Path | None = None,
) -> None:
    """Run a subprocess with streamed output and periodic disk-usage status."""
    label = status_label or cmd[0]
    print(f"[run] {label}: " + " ".join(cmd), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    start = time.time()
    last_status = start
    initial_bytes, initial_files = _dir_size(output_dir) if output_dir else (0, 0)

    # Stream subprocess output in a thread so we can also print status
    output_lines: list[str] = []
    lock = threading.Lock()

    def _reader():
        for line in process.stdout:
            stripped = line.rstrip()
            with lock:
                output_lines.append(stripped)
            # Print significant lines from the downloader
            lower = stripped.lower()
            if any(kw in lower for kw in ("download", "error", "fail", "complet", "skip", "start")):
                print(f"  [{label}] {stripped}", flush=True)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    while True:
        returncode = process.poll()
        now = time.time()

        if returncode is not None:
            reader_thread.join(timeout=5)
            elapsed = now - start
            if output_dir:
                final_bytes, final_files = _dir_size(output_dir)
                new_bytes = final_bytes - initial_bytes
                new_files = final_files - initial_files
                speed = new_bytes / max(1, elapsed)
                print(
                    f"[done] {label} elapsed={_format_seconds(elapsed)} "
                    f"new_files={new_files} new_data={_format_bytes(new_bytes)} "
                    f"avg_speed={_format_bytes(speed)}/s",
                    flush=True,
                )
            else:
                print(f"[done] {label} elapsed={_format_seconds(elapsed)}", flush=True)
            if returncode != 0:
                with lock:
                    last_lines = output_lines[-20:]
                print(f"[error] {label} exited with code {returncode}. Last output:", flush=True)
                for line in last_lines:
                    print(f"  {line}", flush=True)
                raise subprocess.CalledProcessError(returncode, cmd)
            return

        if (now - last_status) >= status_every_seconds:
            elapsed = now - start
            status_parts = [f"elapsed={_format_seconds(elapsed)}"]
            if output_dir:
                cur_bytes, cur_files = _dir_size(output_dir)
                new_bytes = cur_bytes - initial_bytes
                new_files = cur_files - initial_files
                speed = new_bytes / max(1, elapsed)
                status_parts.extend([
                    f"new_files={new_files}",
                    f"new_data={_format_bytes(new_bytes)}",
                    f"speed={_format_bytes(speed)}/s",
                    f"total_disk={_format_bytes(cur_bytes)}",
                ])
            print(f"[status] {label} {' '.join(status_parts)}", flush=True)
            last_status = now
        time.sleep(1.0)


def maybe_clone_download_repo(repo_dir: Path) -> None:
    if repo_dir.exists():
        print(f"[info] downloader repo already exists: {repo_dir}")
        return
    print(f"[clone] epic-kitchens-download-scripts -> {repo_dir}", flush=True)
    subprocess.check_call([
        "git",
        "clone",
        "https://github.com/epic-kitchens/epic-kitchens-download-scripts.git",
        str(repo_dir),
    ])


def _run_batch(
    batch_idx: int,
    total_batches: int,
    ids_batch: list[str],
    base_cmd: list[str],
    downloader_repo: Path,
    output_dir: Path,
    dry_run: bool,
) -> tuple[int, float, int]:
    """Download a single batch. Returns (batch_idx, elapsed, num_videos)."""
    batch_arg = ",".join(ids_batch)
    cmd = list(base_cmd) + ["--specific-videos", batch_arg]

    print(
        f"[batch {batch_idx + 1}/{total_batches}] "
        f"videos={len(ids_batch)} range={ids_batch[0]}..{ids_batch[-1]}",
        flush=True,
    )

    if dry_run:
        print("[dry-run] " + " ".join(cmd), flush=True)
        return batch_idx, 0.0, len(ids_batch)

    batch_start = time.time()
    _run_streaming(
        cmd,
        cwd=downloader_repo,
        status_every_seconds=15,
        status_label=f"batch {batch_idx + 1}/{total_batches}",
        output_dir=output_dir,
    )
    return batch_idx, time.time() - batch_start, len(ids_batch)


def download_videos(
    downloader_repo: Path,
    videos_file: Path,
    output_dir: Path,
    num_workers: int,
    chunksize: int,
    dry_run: bool,
    parallel_batches: int = 1,
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
    all_video_ids = _read_lines(videos_file)
    if not all_video_ids:
        raise RuntimeError(f"No video IDs found in {videos_file}")

    # Resume: skip already-downloaded videos
    already_downloaded = _find_downloaded_videos(output_dir)
    video_ids = [vid for vid in all_video_ids if vid.upper() not in already_downloaded]
    skipped = len(all_video_ids) - len(video_ids)

    if skipped > 0:
        print(
            f"[resume] skipping {skipped} already-downloaded videos, "
            f"{len(video_ids)} remaining out of {len(all_video_ids)} total",
            flush=True,
        )

    if not video_ids:
        print("[done] all videos already downloaded!", flush=True)
        return

    batch_size = max(1, int(chunksize))
    batches: list[list[str]] = []
    for i in range(0, len(video_ids), batch_size):
        batches.append(video_ids[i : i + batch_size])
    total_batches = len(batches)

    overall_start = time.time()
    completed_videos = 0
    initial_bytes, initial_files = _dir_size(output_dir)

    print(
        f"[download-plan] total_videos={len(video_ids)} skipped={skipped} "
        f"batch_size={batch_size} total_batches={total_batches} "
        f"parallel_batches={parallel_batches} "
        f"existing_disk={_format_bytes(initial_bytes)} existing_files={initial_files}",
        flush=True,
    )

    if parallel_batches <= 1:
        # Serial execution (original behavior, but with better progress)
        for batch_idx, ids_batch in enumerate(batches):
            _, batch_elapsed, n = _run_batch(
                batch_idx, total_batches, ids_batch, base_cmd,
                downloader_repo, output_dir, dry_run,
            )
            completed_videos += n
            overall_elapsed = time.time() - overall_start
            cur_bytes, cur_files = _dir_size(output_dir)
            new_bytes = cur_bytes - initial_bytes
            speed = new_bytes / max(1, overall_elapsed)
            avg_sec_per_batch = overall_elapsed / max(1, batch_idx + 1)
            remaining_batches = total_batches - (batch_idx + 1)
            eta_seconds = remaining_batches * avg_sec_per_batch
            print(
                f"[progress] batches={batch_idx + 1}/{total_batches} "
                f"videos={completed_videos}/{len(video_ids)} ({skipped} skipped) "
                f"downloaded={_format_bytes(new_bytes)} speed={_format_bytes(speed)}/s "
                f"batch_time={_format_seconds(batch_elapsed)} "
                f"elapsed={_format_seconds(overall_elapsed)} "
                f"eta={_format_seconds(eta_seconds)}",
                flush=True,
            )
    else:
        # Parallel batch execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_batches) as executor:
            futures = {
                executor.submit(
                    _run_batch, i, total_batches, batch, base_cmd,
                    downloader_repo, output_dir, dry_run,
                ): i
                for i, batch in enumerate(batches)
            }
            completed_batches = 0
            for future in concurrent.futures.as_completed(futures):
                batch_idx, batch_elapsed, n = future.result()
                completed_batches += 1
                completed_videos += n
                overall_elapsed = time.time() - overall_start
                cur_bytes, cur_files = _dir_size(output_dir)
                new_bytes = cur_bytes - initial_bytes
                speed = new_bytes / max(1, overall_elapsed)
                avg_sec_per_batch = overall_elapsed / max(1, completed_batches)
                remaining_batches = total_batches - completed_batches
                # ETA based on throughput with parallelism
                eta_seconds = (remaining_batches / parallel_batches) * avg_sec_per_batch
                print(
                    f"[progress] batches={completed_batches}/{total_batches} "
                    f"videos={completed_videos}/{len(video_ids)} ({skipped} skipped) "
                    f"downloaded={_format_bytes(new_bytes)} speed={_format_bytes(speed)}/s "
                    f"elapsed={_format_seconds(overall_elapsed)} "
                    f"eta={_format_seconds(eta_seconds)}",
                    flush=True,
                )

    # Final summary
    final_bytes, final_files = _dir_size(output_dir)
    total_elapsed = time.time() - overall_start
    total_new = final_bytes - initial_bytes
    print(
        f"\n[download-complete] "
        f"videos={completed_videos} skipped={skipped} "
        f"downloaded={_format_bytes(total_new)} total_disk={_format_bytes(final_bytes)} "
        f"files={final_files} elapsed={_format_seconds(total_elapsed)} "
        f"avg_speed={_format_bytes(total_new / max(1, total_elapsed))}/s",
        flush=True,
    )


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
    parser.add_argument(
        "--parallel-batches",
        type=int,
        default=2,
        help="Number of download batches to run concurrently (default: 2)",
    )
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
            parallel_batches=args.parallel_batches,
        )


if __name__ == "__main__":
    main()
