"""Download helper for the full-scale SAFE Stage A experiment.

The new experiment consumes substantially more data than the overfitting ablation,
so this script streamlines fetching the pre-packed AudioCaps (training) and VQA
(validation) assets onto a cluster node.

Supports:
- Multiple URLs per dataset (comma-separated)
- Mixed archive and plain file downloads
- Safe extraction with path traversal protection
- Robust resume logic with proper HTTP handling
- Timeout and retry mechanisms
- Dataset-specific validation
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import os
import sys
import tarfile
import zipfile
import time
from contextlib import closing
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

CHUNK_SIZE = 1024 * 1024  # 1 MiB
DEFAULT_RETRIES = 3
DEFAULT_CONNECT_TIMEOUT = 10
DEFAULT_READ_TIMEOUT = 60


def is_archive(path: Path) -> bool:
    """Check if a file is an archive that needs extraction."""
    suffixes = ''.join(path.suffixes).lower()
    return any(suffixes.endswith(ext) for ext in ['.tar.gz', '.tgz', '.tar', '.zip'])



def _detect_archive_type(path: Path) -> str:
    """Detect archive type for extraction."""
    suffixes = ''.join(path.suffixes).lower()
    if suffixes.endswith('.tar.gz') or suffixes.endswith('.tgz'):
        return 'tar.gz'
    if suffixes.endswith('.tar'):
        return 'tar'
    if suffixes.endswith('.zip'):
        return 'zip'
    raise ValueError(f"Unsupported archive format for '{path}'")


def _stream_to_file(response, destination: Path, mode: str, offset: int, total: Optional[int]) -> None:
    """Stream response content to file with progress reporting."""
    downloaded = offset
    size_info = f'{total/1e6:.1f} MB' if total else 'unknown size'
    print(f"  Starting download ({size_info})")
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
    sys.stdout.write(" ✓\n")


def download_file(url: str, destination: Path, *, 
                 timeout: Tuple[int, int] = (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
                 resume: bool = False, retries: int = DEFAULT_RETRIES) -> Path:
    """Download a file with robust error handling and resume support."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries + 1):
        try:
            headers = {'User-Agent': 'SAFE-Downloader/1.0 (+https://github.com/safe-project/safe)'}
            mode = 'wb'
            offset = 0

            if resume and destination.exists():
                offset = destination.stat().st_size
                headers['Range'] = f'bytes={offset}-'
                mode = 'ab'
                print(f"Resuming download from byte {offset}")

            print(f"Downloading {url} → {destination} (attempt {attempt + 1}/{retries + 1})")

            def open_request(hdrs):
                return closing(requests.get(url, stream=True, timeout=timeout, headers=hdrs, allow_redirects=True))

            with open_request(headers) as response:
                # Handle resume semantics
                if resume and offset > 0:
                    if response.status_code == 200:
                        print("  Server doesn't support resume, restarting download")
                        mode, offset = 'wb', 0
                        headers.pop('Range', None)
                        with open_request(headers) as response:
                            response.raise_for_status()
                            total = None
                            cr = response.headers.get('Content-Range')
                            if cr and '/' in cr:
                                try:
                                    total = int(cr.split('/')[-1])
                                except ValueError:
                                    total = None
                            if total is None:
                                cl = response.headers.get('Content-Length')
                                if cl:
                                    total = int(cl)
                            _stream_to_file(response, destination, mode, offset, total)
                            return destination

                    elif response.status_code == 416 and destination.exists():
                        print("  Range not satisfiable, file may be complete")
                        return destination
                    elif response.status_code == 206:
                        print("  Resume successful")

                response.raise_for_status()

                # compute total size
                total = None
                cr = response.headers.get('Content-Range')
                if cr and '/' in cr:
                    try:
                        total = int(cr.split('/')[-1])
                    except ValueError:
                        total = None
                if total is None:
                    cl = response.headers.get('Content-Length')
                    if cl:
                        total = int(cl) + offset

                # guard against oversized partial
                if resume and destination.exists() and total is not None:
                    if destination.stat().st_size > total:
                        print("  Local file larger than expected; restarting clean")
                        destination.unlink(missing_ok=True)
                        mode, offset = 'wb', 0

                _stream_to_file(response, destination, mode, offset, total)
                return destination

        except (requests.exceptions.RequestException, OSError) as e:
            if attempt < retries:
                wait = 2 ** attempt
                print(f"  Download failed: {e}\n  Retrying in {wait} s...")
                time.sleep(wait)
            else:
                print(f"  Download failed after {retries + 1} attempts: {e}")
                raise


def extract_archive(archive_path: Path, dataset_destination: Path) -> None:
    """Safely extract an archive to a dataset-specific directory."""
    dataset_destination.mkdir(parents=True, exist_ok=True)
    archive_type = _detect_archive_type(archive_path)
    print(f"Extracting {archive_path} into {dataset_destination} ({archive_type})")

    def _is_safe(name: str) -> bool:
        if name.startswith('/') or '..' in Path(name).parts:
            print(f"  Skipping potentially unsafe path: {name}")
            return False
        # Final canonical check
        try:
            (dataset_destination / name).resolve().relative_to(dataset_destination.resolve())
            return True
        except Exception:
            print(f"  Skipping path that escapes destination: {name}")
            return False

    if archive_type in {'tar.gz', 'tar'}:
        mode = 'r:gz' if archive_type == 'tar.gz' else 'r:'
        with tarfile.open(archive_path, mode) as tar:
            extracted = 0
            for member in tar.getmembers():
                # Skip links and special files
                if member.islnk() or member.issym() or member.ischr() or member.isblk() or member.isfifo():
                    print(f"  Skipping non-regular file: {member.name}")
                    continue
                if not _is_safe(member.name):
                    continue
                try:
                    tar.extract(member, dataset_destination)
                    extracted += 1
                except Exception as e:
                    print(f"  Failed to extract {member.name}: {e}")
            print(f"  Extracted {extracted} files")

    elif archive_type == 'zip':
        with zipfile.ZipFile(archive_path) as zf:
            extracted = 0
            for name in zf.namelist():
                if not _is_safe(name):
                    continue
                try:
                    zf.extract(name, dataset_destination)
                    extracted += 1
                except Exception as e:
                    print(f"  Failed to extract {name}: {e}")
            print(f"  Extracted {extracted} files")

    print("  Extraction complete")


def place_plain_file(file_path: Path, dataset_destination: Path) -> None:
    """Place a non-archive file in the appropriate dataset directory."""
    dataset_destination.mkdir(parents=True, exist_ok=True)
    target_path = dataset_destination / file_path.name
    
    print(f"Placing {file_path} into {target_path}")
    
    # Move or copy the file
    if file_path.parent == dataset_destination:
        # Already in the right place
        return
    
    import shutil
    shutil.move(str(file_path), str(target_path))
    print(f"  Moved to {target_path}")


def validate_audiocaps(audiocaps_dir: Path) -> List[str]:
    """Validate AudioCaps dataset structure."""
    issues = []
    expected_files = ['train.csv', 'val.csv', 'test.csv']
    
    if not audiocaps_dir.exists():
        issues.append(f"AudioCaps directory not found: {audiocaps_dir}")
        return issues
    
    for expected_file in expected_files:
        file_path = audiocaps_dir / expected_file
        if not file_path.exists():
            # Try alternative naming patterns
            alternatives = [
                audiocaps_dir / f"audiocaps_{expected_file}",
                audiocaps_dir / f"AudioCaps_{expected_file}",
            ]
            if not any(alt.exists() for alt in alternatives):
                issues.append(f"Missing AudioCaps file: {expected_file}")
    
    # Warn if no audio media present
    if not list(audiocaps_dir.rglob('*.wav')) and not list(audiocaps_dir.rglob('*.flac')):
        issues.append("No audio files found. AudioCaps GitHub provides CSV metadata only; "
                      "you must fetch audio from YouTube IDs or use a pre-extracted pack.")
    
    return issues


def validate_vqa(vqa_dir: Path) -> List[str]:
    """Validate VQA dataset structure."""
    issues = []

    if not vqa_dir.exists():
        issues.append(f"VQA directory not found: {vqa_dir}")
        return issues

    found_files = list(vqa_dir.rglob('*.json'))
    if not found_files:
        issues.append("No JSON files found in VQA directory")
    else:
        # Check for essential files
        if not any('questions' in f.name.lower() for f in found_files):
            issues.append("No question files found (did the zip extract correctly?)")
        if not any('annotation' in f.name.lower() for f in found_files):
            issues.append("No annotation files found")

    return issues


def validate_coco(coco_dir: Path) -> List[str]:
    """Validate MSCOCO dataset structure."""
    issues = []

    if not coco_dir.exists():
        issues.append("MSCOCO directory not found (use --coco-url to download)")
        return issues

    # Check for train2014 and val2014 directories
    train_dir = None
    val_dir = None

    for candidate in [coco_dir / "train2014", coco_dir / "images" / "train2014"]:
        if candidate.exists() and candidate.is_dir():
            train_dir = candidate
            break

    for candidate in [coco_dir / "val2014", coco_dir / "images" / "val2014"]:
        if candidate.exists() and candidate.is_dir():
            val_dir = candidate
            break

    if not train_dir:
        issues.append("train2014 directory not found")
    else:
        train_images = list(train_dir.glob("*.jpg"))
        if not train_images:
            issues.append("No images found in train2014 directory")

    if not val_dir:
        issues.append("val2014 directory not found")
    else:
        val_images = list(val_dir.glob("*.jpg"))
        if not val_images:
            issues.append("No images found in val2014 directory")

    return issues


def validate_dataset_root(root: Path) -> None:
    """Validate the complete dataset structure."""
    audiocaps_dir = root / 'audiocaps'
    vqa_dir = root / 'vqa'
    coco_dir = root / 'coco'

    issues = []
    issues.extend(validate_audiocaps(audiocaps_dir))
    issues.extend(validate_vqa(vqa_dir))
    issues.extend(validate_coco(coco_dir))

    if issues:
        print("\n⚠️  Dataset validation warnings:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nNote: Some files might be organized differently than expected.")
        print("Check the dataset directories manually if needed.")
    else:
        print("\n✓ Dataset validation passed")


def parse_urls(url_string: str) -> List[str]:
    """Parse comma-separated URLs."""
    if not url_string:
        return []
    return [url.strip() for url in url_string.split(',') if url.strip()]


def resolve_url(arg_value: Optional[str], env_var: str, description: str) -> List[str]:
    """Resolve URLs from arguments or environment variables."""
    value = arg_value or os.environ.get(env_var)
    if not value:
        raise SystemExit(
            f"{description} URL(s) not provided. Pass --{description.lower().replace(' ', '-')} "
            f"or set {env_var}. Multiple URLs can be comma-separated."
        )
    return parse_urls(value)


@functools.lru_cache(maxsize=1)
def _load_audiocaps_audio_module() -> ModuleType:
    """Dynamically load the AudioCaps YouTube downloader helper."""
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    module_path = scripts_dir / "download_audiocaps_audio.py"
    if not module_path.exists():
        raise FileNotFoundError(f"AudioCaps downloader script not found at {module_path}")

    spec = importlib.util.spec_from_file_location(
        "experiments.full_training._download_audiocaps_audio",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _find_audiocaps_csv(audiocaps_dir: Path, split: str) -> Optional[Path]:
    """Locate the metadata CSV for a given AudioCaps split."""
    candidate_names = [
        f"{split}.csv",
        f"audiocaps_{split}.csv",
        f"AudioCaps_{split}.csv",
    ]

    for name in candidate_names:
        candidate = audiocaps_dir / name
        if candidate.exists():
            return candidate

    matches = []
    for name in candidate_names:
        matches.extend(audiocaps_dir.rglob(name))

    if not matches:
        return None

    def _priority(path: Path) -> Tuple[int, int]:
        try:
            relative = path.relative_to(audiocaps_dir)
            parts = relative.parts
        except ValueError:
            parts = path.parts
        metadata_score = 0 if any(part.lower() == "metadata" for part in parts) else 1
        return metadata_score, len(parts)

    matches.sort(key=_priority)
    return matches[0]


def download_audiocaps_audio_from_youtube(
    audiocaps_dir: Path,
    *,
    splits: List[str],
    max_downloads: Optional[int],
    num_workers: int = 8,
    cookies_file: Optional[str] = None,
) -> None:
    """Fetch AudioCaps waveforms from YouTube using the project helper script."""
    try:
        downloader = _load_audiocaps_audio_module()
    except (FileNotFoundError, ImportError) as exc:
        print(f"  Skipping AudioCaps audio download: {exc}")
        return

    required_attrs = ("check_ytdlp", "process_audiocaps_csv")
    missing_attrs = [attr for attr in required_attrs if not hasattr(downloader, attr)]
    if missing_attrs:
        print(
            "  Skipping AudioCaps audio download: downloader helper is missing required"
            f" attributes {missing_attrs}"
        )
        return

    limit = None if max_downloads is None or max_downloads <= 0 else max_downloads

    metadata_by_split: Dict[str, Path] = {}
    for split in splits:
        csv_path = _find_audiocaps_csv(audiocaps_dir, split)
        if not csv_path:
            print(f"  Metadata for AudioCaps split '{split}' not found; skipping this split")
            continue
        metadata_by_split[split] = csv_path

    if not metadata_by_split:
        print("  No AudioCaps CSV metadata located; skipping YouTube audio download")
        return

    print("\n=== Downloading AudioCaps audio from YouTube ===")

    if not downloader.check_ytdlp():
        print(
            "  yt-dlp not available. Install it with 'pip install yt-dlp' to enable"
            " AudioCaps audio downloads."
        )
        return

    audio_root = audiocaps_dir / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    for split, csv_path in metadata_by_split.items():
        destination = audio_root / split
        destination.mkdir(parents=True, exist_ok=True)
        print(f"\n>>> Fetching AudioCaps audio for split '{split}'")
        print(f"    Metadata: {csv_path}")
        try:
            downloader.process_audiocaps_csv(csv_path, destination, limit, num_workers, cookies_file)
        except Exception as exc:  # noqa: BLE001 - keep script resilient
            print(f"  Failed to download audio for split '{split}': {exc}")

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download full SAFE training data pack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with COCO images and parallel downloads
  python -m experiments.full_training.download_data \\
    --audiocaps-url "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/train.csv,..." \\
    --vqa-url "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip,..." \\
    --coco-url "http://images.cocodataset.org/zips/train2014.zip,http://images.cocodataset.org/zips/val2014.zip" \\
    --audiocaps-workers 16 \\
    --audiocaps-max-downloads 5000 \\
    --resume

  # Skip COCO if already downloaded
  python -m experiments.full_training.download_data \\
    --audiocaps-url "..." \\
    --vqa-url "..." \\
    --skip-coco
        """
    )
    
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("experiments/full_training/data"),
        help="Where to extract the datasets (default: experiments/full_training/data)",
    )
    parser.add_argument(
        "--audiocaps-url",
        type=str,
        help="HTTP(S) URL(s) for AudioCaps data (comma-separated for multiple files)",
    )
    parser.add_argument(
        "--vqa-url",
        type=str,
        help="HTTP(S) URL(s) for VQA data (comma-separated for multiple files)",
    )
    parser.add_argument(
        "--coco-url",
        type=str,
        help="HTTP(S) URL(s) for MSCOCO images (comma-separated for multiple files)",
    )
    parser.add_argument(
        "--skip-coco",
        action="store_true",
        help="Skip downloading MSCOCO images",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=DEFAULT_CONNECT_TIMEOUT,
        help=f"HTTP connect timeout in seconds (default: {DEFAULT_CONNECT_TIMEOUT})",
    )
    parser.add_argument(
        "--read-timeout",
        type=int,
        default=DEFAULT_READ_TIMEOUT,
        help=f"HTTP read timeout in seconds (default: {DEFAULT_READ_TIMEOUT})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Number of retry attempts for failed downloads (default: {DEFAULT_RETRIES})",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Do not delete downloaded archive files after extraction",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume partially downloaded files if present",
    )
    parser.add_argument(
        "--skip-audiocaps-audio",
        action="store_true",
        help="Do not fetch AudioCaps audio from YouTube after downloading metadata",
    )
    parser.add_argument(
        "--audiocaps-max-downloads",
        type=int,
        default=None,
        help="Limit the number of AudioCaps clips to fetch per split (default: all)",
    )
    parser.add_argument(
        "--audiocaps-splits",
        type=str,
        default="train,val,test",
        help="Comma-separated AudioCaps splits to download audio for (default: train,val,test)",
    )
    parser.add_argument(
        "--audiocaps-workers",
        type=int,
        default=8,
        help="Number of parallel workers for AudioCaps YouTube downloads (default: 8)",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        help="Path to cookies.txt file for YouTube authentication",
    )

    args = parser.parse_args(argv)

    audio_splits = [split.strip() for split in args.audiocaps_splits.split(',') if split.strip()]
    if not audio_splits:
        audio_splits = ["train", "val", "test"]

    # Resolve URLs
    try:
        audiocaps_urls = resolve_url(args.audiocaps_url, "SAFE_FULL_AUDIOCAPS_URL", "AudioCaps")
        vqa_urls = resolve_url(args.vqa_url, "SAFE_FULL_VQA_URL", "VQA")
    except SystemExit as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # COCO images are optional
    coco_urls = []
    if not args.skip_coco:
        coco_arg = args.coco_url or os.environ.get("SAFE_FULL_COCO_URL")
        if coco_arg:
            coco_urls = parse_urls(coco_arg)
        else:
            print("\nNote: MSCOCO images not specified. Pass --coco-url or set SAFE_FULL_COCO_URL")
            print("      to download train2014/val2014 images, or use --skip-coco to suppress this message.")

    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    # Create dataset-specific directories
    audiocaps_dir = destination / "audiocaps"
    vqa_dir = destination / "vqa"
    coco_dir = destination / "coco"
    archives_dir = destination / "archives"
    archives_dir.mkdir(exist_ok=True)

    timeout = (args.connect_timeout, args.read_timeout)

    print(f"Downloading to: {destination}")
    print(f"AudioCaps URLs: {len(audiocaps_urls)} files")
    print(f"VQA URLs: {len(vqa_urls)} files")
    print(f"COCO URLs: {len(coco_urls)} files")
    print(f"Timeout: {timeout}, Retries: {args.retries}")

    try:
        # Process AudioCaps URLs
        print(f"\n=== Downloading AudioCaps ({len(audiocaps_urls)} files) ===")
        for i, url in enumerate(audiocaps_urls, 1):
            print(f"\n[{i}/{len(audiocaps_urls)}] Processing: {url}")
            
            filename = Path(urlparse(url).path).name
            if not filename:
                filename = f"audiocaps_file_{i}"
            
            download_path = archives_dir / filename
            downloaded_file = download_file(
                url, download_path,
                timeout=timeout,
                resume=args.resume,
                retries=args.retries
            )
            
            if is_archive(downloaded_file):
                extract_archive(downloaded_file, audiocaps_dir)
                if not args.keep_archives:
                    downloaded_file.unlink(missing_ok=True)
            else:
                place_plain_file(downloaded_file, audiocaps_dir)

        if args.skip_audiocaps_audio:
            print("\nSkipping AudioCaps YouTube audio download (--skip-audiocaps-audio)")
        else:
            download_audiocaps_audio_from_youtube(
                audiocaps_dir,
                splits=audio_splits,
                max_downloads=args.audiocaps_max_downloads,
                num_workers=args.audiocaps_workers,
                cookies_file=args.cookies,
            )

        # Process VQA URLs
        print(f"\n=== Downloading VQA ({len(vqa_urls)} files) ===")
        for i, url in enumerate(vqa_urls, 1):
            print(f"\n[{i}/{len(vqa_urls)}] Processing: {url}")

            filename = Path(urlparse(url).path).name
            if not filename:
                filename = f"vqa_file_{i}"

            download_path = archives_dir / filename
            downloaded_file = download_file(
                url, download_path,
                timeout=timeout,
                resume=args.resume,
                retries=args.retries
            )

            if is_archive(downloaded_file):
                extract_archive(downloaded_file, vqa_dir)
                if not args.keep_archives:
                    downloaded_file.unlink(missing_ok=True)
            else:
                place_plain_file(downloaded_file, vqa_dir)

        # Process COCO URLs
        if coco_urls:
            print(f"\n=== Downloading MSCOCO images ({len(coco_urls)} files) ===")
            for i, url in enumerate(coco_urls, 1):
                print(f"\n[{i}/{len(coco_urls)}] Processing: {url}")

                filename = Path(urlparse(url).path).name
                if not filename:
                    filename = f"coco_file_{i}"

                download_path = archives_dir / filename
                downloaded_file = download_file(
                    url, download_path,
                    timeout=timeout,
                    resume=args.resume,
                    retries=args.retries
                )

                if is_archive(downloaded_file):
                    extract_archive(downloaded_file, coco_dir)
                    if not args.keep_archives:
                        downloaded_file.unlink(missing_ok=True)
                else:
                    place_plain_file(downloaded_file, coco_dir)

        # Clean up empty archives directory
        if not args.keep_archives and archives_dir.exists():
            try:
                archives_dir.rmdir()  # Only works if empty
            except OSError:
                pass  # Directory not empty, leave it

        # Validate the final dataset structure
        validate_dataset_root(destination)
        
        print(f"\n✓ Dataset download complete!")
        print(f"  Location: {destination}")
        print(f"  AudioCaps: {audiocaps_dir}")
        print(f"  VQA: {vqa_dir}")
        if coco_urls:
            print(f"  COCO: {coco_dir}")
        
        return 0
        
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
