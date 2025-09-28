"""Download helper for the full-scale SAFE Stage A experiment.

The new experiment consumes substantially more data than the overfitting ablation,
so this script streamlines fetching the pre-packed AudioCaps (training) and VQA
(validation) assets onto a cluster node.
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import requests

CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _detect_archive_type(path: Path) -> str:
    suffixes = ''.join(path.suffixes).lower()
    if suffixes.endswith('.tar.gz') or suffixes.endswith('.tgz'):
        return 'tar.gz'
    if suffixes.endswith('.tar'):
        return 'tar'
    if suffixes.endswith('.zip'):
        return 'zip'
    raise ValueError(f"Unsupported archive format for '{path}'")


def download_file(url: str, destination: Path, *, timeout: int = 30, resume: bool = False) -> Path:
    """Stream a file from ``url`` into ``destination`` with a progress message."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    mode = 'wb'
    if resume and destination.exists():
        offset = destination.stat().st_size
        headers['Range'] = f'bytes={offset}-'
        mode = 'ab'
    else:
        offset = 0

    with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
        response.raise_for_status()
        total = response.headers.get('Content-Length')
        if total is not None:
            total = int(total) + offset
        downloaded = offset
        print(f"Downloading {url} → {destination} ({'unknown size' if total is None else f'{total/1e6:.1f} MB'})")
        with open(destination, mode) as fh:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total is not None:
                        pct = downloaded / total * 100
                        sys.stdout.write(f"\r  {downloaded/1e6:8.1f} / {total/1e6:8.1f} MB ({pct:5.1f}%)")
                    else:
                        sys.stdout.write(f"\r  {downloaded/1e6:8.1f} MB")
                    sys.stdout.flush()
        sys.stdout.write("\n")
    return destination


def extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive_type = _detect_archive_type(archive_path)
    print(f"Extracting {archive_path} into {destination} ({archive_type})")
    if archive_type in {'tar.gz', 'tar'}:
        mode = 'r:gz' if archive_type == 'tar.gz' else 'r:'
        with tarfile.open(archive_path, mode) as tar:
            tar.extractall(destination)
    elif archive_type == 'zip':
        with zipfile.ZipFile(archive_path) as zip_fh:
            zip_fh.extractall(destination)
    print("Extraction complete")


def validate_dataset_root(root: Path) -> None:
    audiocaps_dir = root / 'audiocaps'
    vqa_dir = root / 'vqa'
    missing = [p for p in (audiocaps_dir, vqa_dir) if not p.exists()]
    if missing:
        formatted = ', '.join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Expected experiment data to contain both 'audiocaps' and 'vqa' directories. Missing: {formatted}"
        )


def resolve_url(arg_value: Optional[str], env_var: str, description: str) -> str:
    value = arg_value or os.environ.get(env_var)
    if not value:
        raise SystemExit(
            f"{description} URL not provided. Pass --{description.lower().replace(' ', '-')} or set {env_var}."
        )
    return value


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download full SAFE training data pack")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("experiments/full_training/data"),
        help="Where to extract the datasets (default: experiments/full_training/data)",
    )
    parser.add_argument(
        "--audiocaps-url",
        type=str,
        help="HTTP(S) URL pointing to the AudioCaps archive",
    )
    parser.add_argument(
        "--vqa-url",
        type=str,
        help="HTTP(S) URL pointing to the VQA archive",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for each request (default: 60)",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Do not delete the downloaded archive files after extraction",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume partially downloaded archives if present",
    )

    args = parser.parse_args(argv)

    audiocaps_url = resolve_url(args.audiocaps_url, "SAFE_FULL_AUDIOCAPS_URL", "AudioCaps")
    vqa_url = resolve_url(args.vqa_url, "SAFE_FULL_VQA_URL", "VQA")

    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    archives_dir = destination / "archives"
    archives_dir.mkdir(exist_ok=True)

    try:
        audiocaps_archive = download_file(
            audiocaps_url,
            archives_dir / Path(audiocaps_url).name,
            timeout=args.timeout,
            resume=args.resume,
        )
        extract_archive(audiocaps_archive, destination)
        if not args.keep_archives:
            audiocaps_archive.unlink(missing_ok=True)

        vqa_archive = download_file(
            vqa_url,
            archives_dir / Path(vqa_url).name,
            timeout=args.timeout,
            resume=args.resume,
        )
        extract_archive(vqa_archive, destination)
        if not args.keep_archives:
            vqa_archive.unlink(missing_ok=True)

        validate_dataset_root(destination)
        print(f"Dataset ready at {destination}")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
