#!/usr/bin/env python3
"""
Download, process, and auto-cleanup AudioCaps audio segments with parallel processing.

This script implements ethical data practices by:
1. Downloading audio segments to temporary storage only (parallel downloads)
2. Extracting CLAP embeddings immediately
3. Deleting raw audio before next download
4. Storing only derived features (embeddings)
5. Logging all source metadata

Usage:
    # With cookies (recommended to avoid bot detection)
    python scripts/download_and_process_audiocaps.py --split train --cookies cookies.txt

    # Default: 4 parallel workers
    python scripts/download_and_process_audiocaps.py --split train --max-downloads 100 --cookies cookies.txt

    # Custom parallelism
    python scripts/download_and_process_audiocaps.py --split train --num-workers 8 --cookies cookies.txt

    # Append to existing file
    python scripts/download_and_process_audiocaps.py --split val --append --cookies cookies.txt

    # Resume from specific index
    python scripts/download_and_process_audiocaps.py --split train --resume-from 500 --cookies cookies.txt

Ethical Note:
    This implements fair use by never storing raw audio permanently.
    Only derived features (embeddings) are retained for research.

Notes:
    - The `--cookies` file must be in Netscape format. Use `scripts/extract_cookies.sh`
      to generate one directly from your browser session.
    - The downloader retries transient YouTube errors and aborts early on
      configuration issues so you can fix them without wasting time.
    - If YouTube rate-limits your session, the script exits quickly so you can
      wait before resuming. Lower `--num-workers` and raise `--sleep-min` /
      `--sleep-max` if this happens frequently.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

DEFAULT_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
)

NON_RETRY_ERROR_PATTERNS = (
    'video unavailable',
    'private video',
    'removed for violating youtube\'s terms of service',
    'account associated with this video has been terminated',
    'this video is no longer available',
)

COOKIE_FORMAT_ERROR = 'does not look like a netscape format cookies file'
RATE_LIMIT_PATTERN = 'current session has been rate-limited'


class DownloadFatalError(RuntimeError):
    """Raised for configuration problems that require aborting the run."""
    pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from safe.models.audio_encoders import CLAPAudioEncoder


def compute_checksum(data: np.ndarray) -> str:
    """Compute SHA256 checksum of embedding data."""
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


def load_metadata(csv_path: Path) -> List[Dict]:
    """Load AudioCaps metadata CSV."""
    metadata = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append({
                'audiocap_id': row['audiocap_id'],
                'youtube_id': row['youtube_id'],
                'start_time': int(row['start_time']),
                'caption': row['caption']
            })
    return metadata


def download_audio_segment(
    youtube_id: str,
    start_time: int,
    temp_dir: Path,
    duration: int = 10,
    cookies_file: Optional[str] = None,
    max_retries: int = 3,
    sleep_min: float = 0.5,
    sleep_max: float = 2.0,
) -> Optional[Path]:
    """
    Download 10s audio segment using yt-dlp to temporary directory.

    Returns:
        Path to downloaded WAV file, or None if download failed
    """
    # Include timestamp in filename to avoid collisions when the same video is requested multiple times
    output_stem = f"{youtube_id}_{int(start_time):06d}"
    output_path = temp_dir / f"{output_stem}.wav"

    base_cmd = [
        'yt-dlp',
        '--quiet',
        '--no-progress',
        '--no-warnings',
        '--ignore-config',
        '--no-playlist',
        '--force-overwrites',
        '--extract-audio',
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '--user-agent', DEFAULT_USER_AGENT,
        '--geo-bypass',
        '--sleep-requests', '1',
        '--sleep-interval', f'{sleep_min}',
        '--max-sleep-interval', f'{sleep_max}',
        '--retries', '3',
        '--fragment-retries', '3',
        '--output', str((temp_dir / output_stem)),
        '--postprocessor-args', f'ffmpeg:-ss {start_time} -t {duration}',
    ]

    cookie_args: List[str] = []
    if cookies_file:
        if cookies_file.startswith('browser:'):
            browser = cookies_file.split(':', 1)[1]
            cookie_args.extend(['--cookies-from-browser', browser])
        else:
            cookie_path = Path(cookies_file)
            if cookie_path.exists():
                cookie_args.extend(['--cookies', str(cookie_path)])

    url = f'https://www.youtube.com/watch?v={youtube_id}'

    # Strategies for retry attempts
    # Note: Android client bypasses most bot detection without cookies
    retry_configs = [
        {'force_android_client': True},  # Try Android client first (no auth needed)
        {'force_ipv4': True, 'force_android_client': True},
        {},  # Fallback to default
        {'force_ipv4': True},
    ]

    def build_cmd(force_ipv4: bool = False, force_android_client: bool = False) -> List[str]:
        cmd = list(base_cmd)
        if force_ipv4:
            cmd.append('--force-ipv4')
        if force_android_client:
            cmd.extend(['--extractor-args', 'youtube:player_client=android'])
        cmd.extend(cookie_args)
        cmd.append(url)
        return cmd

    attempts = min(max_retries, len(retry_configs))
    last_error = ''

    for attempt_idx in range(attempts):
        retry_cfg = retry_configs[attempt_idx]
        cmd = build_cmd(**retry_cfg)

        if output_path.exists():
            output_path.unlink()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=90,
            )

            if result.returncode == 0 and output_path.exists():
                return output_path

            stderr = (result.stderr or '')
            stdout = (result.stdout or '')
            combined_msg = (stderr + '\n' + stdout).strip()
            last_error = combined_msg or 'Unknown error'
            error_lower = last_error.lower()

            if COOKIE_FORMAT_ERROR in error_lower:
                raise DownloadFatalError(
                    "Cookies file is not in Netscape format. Regenerate it using yt-dlp "
                    "(see scripts/extract_cookies.sh)."
                )

            if RATE_LIMIT_PATTERN in error_lower:
                raise DownloadFatalError(
                    "YouTube rate-limited this session. Pause downloads for ~60 minutes, "
                    "reduce parallelism (e.g., --num-workers 1), increase --sleep-min/--sleep-max "
                    "and consider regenerating cookies."
                )

            if any(pattern in error_lower for pattern in NON_RETRY_ERROR_PATTERNS):
                break

        except subprocess.TimeoutExpired:
            last_error = 'Download timed out'
        except Exception as exc:  # pragma: no cover - defensive
            last_error = f'Unexpected error: {exc}'
            break

        # Backoff before retrying (skip after final attempt)
        if attempt_idx < attempts - 1:
            time.sleep(2 ** attempt_idx)

    if last_error:
        print(f"  ⚠️  Download failed: {youtube_id} - {last_error}")
    else:
        print(f"  ⚠️  Download failed: {youtube_id} - Unknown error")

    return None


def extract_clap_embedding(
    audio_path: Path,
    encoder: CLAPAudioEncoder,
    device: str = 'cuda'
) -> Tuple[np.ndarray, bool]:
    """
    Extract CLAP embedding from audio file.

    Returns:
        (embedding, success): embedding as numpy array, success flag
    """
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Move to device
        waveform = waveform.to(device)

        # Extract embedding
        with torch.no_grad():
            embedding = encoder(waveform)  # (1, embed_dim)

        # Convert to numpy
        embedding_np = embedding.cpu().numpy().squeeze()  # (embed_dim,)

        return embedding_np, True

    except Exception as e:
        print(f"  ⚠️  Failed to extract embedding: {e}")
        return np.zeros(512, dtype=np.float32), False


def append_to_hdf5(
    h5_path: Path,
    youtube_id: str,
    embedding: np.ndarray,
    metadata: Dict,
    lock: threading.Lock
) -> bool:
    """
    Append new embedding to existing HDF5 file (thread-safe).

    Returns:
        True if successful, False otherwise
    """
    try:
        with lock:  # Ensure thread-safe HDF5 access
            with h5py.File(h5_path, 'a') as hf:
                # Get existing data
                embeddings = hf['embeddings'][:]
                youtube_ids = hf['youtube_ids'][:].astype(str)

                # Check if already exists
                if youtube_id in youtube_ids:
                    return True

                # Append new data
                new_embeddings = np.vstack([embeddings, embedding[np.newaxis, :]])
                new_ids = np.append(youtube_ids, youtube_id)

                # Delete old datasets
                del hf['embeddings']
                del hf['youtube_ids']

                # Create new datasets with appended data
                hf.create_dataset('embeddings', data=new_embeddings, compression='gzip')
                hf.create_dataset('youtube_ids', data=np.array(new_ids, dtype='S11'))

                # Update count
                hf.attrs['num_samples'] = len(new_ids)

        return True

    except Exception as e:
        print(f"  ⚠️  Failed to append to HDF5: {e}")
        return False


def create_hdf5(
    h5_path: Path,
    split: str,
    embedding_dim: int = 512
) -> None:
    """Create new HDF5 file with proper structure."""
    with h5py.File(h5_path, 'w') as hf:
        # Create empty datasets
        hf.create_dataset('embeddings', shape=(0, embedding_dim),
                         maxshape=(None, embedding_dim), compression='gzip')
        hf.create_dataset('youtube_ids', shape=(0,), maxshape=(None,), dtype='S11')

        # Store metadata
        hf.attrs['split'] = split
        hf.attrs['num_samples'] = 0
        hf.attrs['embedding_dim'] = embedding_dim
        hf.attrs['encoder'] = 'CLAP'
        hf.attrs['encoder_model'] = 'laion/clap-htsat-unfused'


def process_single_video(
    meta: Dict,
    temp_dir: Path,
    encoder: CLAPAudioEncoder,
    h5_path: Path,
    sources_dict: Dict,
    h5_lock: threading.Lock,
    sources_lock: threading.Lock,
    device: str,
    cookies_file: Optional[str] = None,
    existing_ids: set = None,
    sleep_min: float = 0.5,
    sleep_max: float = 2.0,
) -> Tuple[str, str]:
    """
    Process a single video: download, extract, save, delete.

    Returns:
        (youtube_id, status): status is one of 'success', 'skip', 'download_fail', 'extract_fail', 'save_fail'
    """
    youtube_id = meta['youtube_id']

    # Check if already in HDF5 file (skip expensive operations)
    if existing_ids and youtube_id in existing_ids:
        return (youtube_id, 'skip_hdf5')

    # Check if already processed in this session (thread-safe read)
    with sources_lock:
        if youtube_id in sources_dict:
            return (youtube_id, 'skip_session')

    # Download to temporary directory
    audio_path = download_audio_segment(
        youtube_id,
        meta['start_time'],
        temp_dir,
        cookies_file=cookies_file,
        sleep_min=sleep_min,
        sleep_max=sleep_max,
    )

    if audio_path is None:
        return (youtube_id, 'download_fail')

    # Download succeeded (return success indicator)
    # Note: Print happens in main loop to avoid tqdm conflicts

    try:
        # Extract embedding
        embedding, extract_success = extract_clap_embedding(
            audio_path, encoder, device
        )

        if not extract_success:
            audio_path.unlink()
            return (youtube_id, 'extract_fail')

        # Append to HDF5 (thread-safe)
        if not append_to_hdf5(h5_path, youtube_id, embedding, meta, h5_lock):
            audio_path.unlink()
            return (youtube_id, 'save_fail')

        # Store source info (thread-safe)
        with sources_lock:
            sources_dict[youtube_id] = {
                'youtube_id': youtube_id,
                'audiocap_id': meta['audiocap_id'],
                'start_time': meta['start_time'],
                'caption': meta['caption'],
                'checksum': compute_checksum(embedding)
            }

        # Delete temporary WAV file
        audio_path.unlink()

        return (youtube_id, 'success')

    except Exception as e:
        # Cleanup on any error
        if audio_path.exists():
            audio_path.unlink()
        return (youtube_id, f'error: {e}')


def process_split(split: str, args) -> int:
    """Process a single split."""
    data_root = Path(args.data_root).expanduser().resolve()
    metadata_path = data_root / f'{split}.csv'
    features_dir = data_root / 'features'
    h5_path = features_dir / f'{split}_clap_embeddings.h5'
    sources_path = features_dir / f'{split}_sources.json'

    # Validate metadata
    if not metadata_path.exists():
        print(f"❌ Metadata not found: {metadata_path}")
        return 1

    # Check HDF5 existence
    if h5_path.exists() and not args.append:
        print(f"❌ HDF5 file already exists: {h5_path}")
        print(f"   Use --append to add to existing file")
        return 1

    # Load metadata
    print(f"📋 Loading metadata from {metadata_path}")
    metadata = load_metadata(metadata_path)
    print(f"   Found {len(metadata)} entries in metadata")

    # Apply resume/max filters
    start_idx = args.resume_from
    end_idx = len(metadata)
    if args.max_downloads:
        end_idx = min(start_idx + args.max_downloads, end_idx)

    metadata_subset = metadata[start_idx:end_idx]
    print(f"   Processing entries {start_idx} to {end_idx} ({len(metadata_subset)} files)")

    # Initialize CLAP encoder
    print(f"🔧 Initializing CLAP encoder on {args.device}...")
    encoder = CLAPAudioEncoder(freeze=True)
    encoder = encoder.to(args.device)
    encoder.eval()
    print(f"   CLAP embedding dimension: {encoder.audio_embed_dim}")

    # Create or load HDF5
    existing_ids = set()
    if not h5_path.exists():
        print(f"📦 Creating new HDF5 file: {h5_path}")
        create_hdf5(h5_path, split, encoder.audio_embed_dim)
    else:
        print(f"📦 Appending to existing HDF5 file: {h5_path}")
        # Load existing IDs to skip re-processing
        with h5py.File(h5_path, 'r') as hf:
            if 'youtube_ids' in hf:
                existing_ids = set(hf['youtube_ids'][:].astype(str))
                print(f"   Found {len(existing_ids)} existing embeddings - will skip these")

    # Load existing sources or create new
    if sources_path.exists():
        with open(sources_path, 'r') as f:
            sources_dict = json.load(f)
        print(f"   Loaded {len(sources_dict)} existing source entries")
    else:
        sources_dict = {}

    # Create temporary directory for downloads
    temp_dir = Path(tempfile.mkdtemp(prefix=f'audiocaps_{split}_'))
    print(f"📁 Using temporary directory: {temp_dir}")

    # Thread locks for thread-safe operations
    h5_lock = threading.Lock()
    sources_lock = threading.Lock()

    # Process files
    success_count = 0
    download_fail_count = 0
    extraction_fail_count = 0
    save_fail_count = 0
    skip_count = 0

    print(f"\n🔄 Processing {len(metadata_subset)} files with {args.num_workers} parallel workers...\n")

    try:
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            future_to_meta = {
                executor.submit(
                    process_single_video,
                    meta,
                    temp_dir,
                    encoder,
                    h5_path,
                    sources_dict,
                    h5_lock,
                    sources_lock,
                    args.device,
                    args.cookies,
                    existing_ids,
                    args.sleep_min,
                    args.sleep_max,
                ): meta
                for meta in metadata_subset
            }

            # Process results with progress bar
            fatal_error: Optional[Exception] = None
            with tqdm(total=len(metadata_subset), desc=f"[{split}] Downloading & extracting") as pbar:
                processed_count = 0
                for future in as_completed(future_to_meta):
                    try:
                        youtube_id, status = future.result()
                    except DownloadFatalError as err:
                        fatal_error = err
                        pbar.write(f"  ❌ {err}")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    if status == 'success':
                        success_count += 1
                        pbar.write(f"  ✅ {youtube_id}")
                    elif status in ('skip_hdf5', 'skip_session'):
                        skip_count += 1
                    elif status == 'download_fail':
                        download_fail_count += 1
                    elif status == 'extract_fail':
                        extraction_fail_count += 1
                    elif status == 'save_fail':
                        save_fail_count += 1
                    else:
                        # Handle unexpected errors
                        extraction_fail_count += 1

                    pbar.update(1)
                    processed_count += 1

                    # Periodic save of sources (thread-safe)
                    if processed_count % args.batch_log_interval == 0:
                        with sources_lock:
                            with open(sources_path, 'w') as f:
                                json.dump(sources_dict, f, indent=2)

            if fatal_error:
                return 1

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up temporary directory")

    # Final save of sources
    print(f"\n💾 Saving final source log to {sources_path}")
    with sources_lock:
        with open(sources_path, 'w') as f:
            json.dump(sources_dict, f, indent=2)
    print(f"   Logged {len(sources_dict)} total sources")

    # Summary
    print(f"\n" + "="*60)
    print(f"SUMMARY - {split.upper()} split")
    print(f"="*60)
    print(f"Requested:              {len(metadata_subset)}")
    print(f"Already processed:      {skip_count}")
    print(f"Successfully added:     {success_count}")
    print(f"Download failures:      {download_fail_count}")
    print(f"Extraction failures:    {extraction_fail_count}")
    print(f"Save failures:          {save_fail_count}")
    print(f"Parallel workers:       {args.num_workers}")
    print(f"HDF5 file:              {h5_path}")
    print(f"Sources log:            {sources_path}")
    print(f"="*60)

    # Get HDF5 stats
    with h5py.File(h5_path, 'r') as hf:
        total_samples = hf.attrs['num_samples']
        file_size_mb = h5_path.stat().st_size / (1024 * 1024)
        print(f"\n📊 HDF5 Statistics:")
        print(f"   Total embeddings: {total_samples}")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Avg per sample: {file_size_mb / max(total_samples, 1) * 1024:.1f} KB")

    print(f"\n✅ COMPLETE - No raw audio files retained")
    print(f"   ✓  All temporary files deleted")
    print(f"   ✓  Only derived features stored")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Download and process AudioCaps with auto-cleanup'
    )
    parser.add_argument('--split', choices=['train', 'val', 'test'], default=None,
                        help='Dataset split to process')
    parser.add_argument('--all-splits', action='store_true',
                        help='Process all splits (train, val, test)')
    parser.add_argument('--data-root', type=str, default='experiments/full_training/data/audiocaps',
                        help='Root directory for AudioCaps data')
    parser.add_argument('--max-downloads', type=int, default=None,
                        help='Maximum number of files to download (default: all)')
    parser.add_argument('--resume-from', type=int, default=0,
                        help='Resume from this index in metadata (skip first N)')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing HDF5 file (default: error if exists)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device for CLAP encoder')
    parser.add_argument('--batch-log-interval', type=int, default=50,
                        help='Save sources log every N files')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel download workers')
    parser.add_argument('--cookies', type=str, default=None,
                        help='Path to cookies.txt file OR "browser:chrome" to extract from browser')
    parser.add_argument('--sleep-min', type=float, default=0.5,
                        help='Minimum seconds to sleep between successive requests (default: 0.5)')
    parser.add_argument('--sleep-max', type=float, default=2.0,
                        help='Maximum seconds to sleep between successive requests (default: 2.0)')

    args = parser.parse_args()

    if args.sleep_min < 0 or args.sleep_max < 0:
        parser.error('Sleep intervals must be non-negative values')
    if args.sleep_max < args.sleep_min:
        parser.error('--sleep-max must be greater than or equal to --sleep-min')

    # Validate split arguments
    if not args.split and not args.all_splits:
        parser.error("Must specify --split or --all-splits")
    if args.split and args.all_splits:
        parser.error("Cannot specify both --split and --all-splits")

    # Determine which splits to process
    splits_to_process = ['train', 'val', 'test'] if args.all_splits else [args.split]

    # Setup paths
    data_root = Path(args.data_root).expanduser().resolve()
    features_dir = data_root / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)

    # Check cookies file
    if args.cookies:
        cookies_path = Path(args.cookies)
        if not cookies_path.exists():
            print(f"❌ Cookies file not found: {cookies_path}")
            print(f"   Export cookies from browser using 'Get cookies.txt' extension")
            return 1
        print(f"✓ Using cookies from: {cookies_path}")

    # Check for yt-dlp
    try:
        subprocess.run(['yt-dlp', '--version'],
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ yt-dlp not found. Install with: pip install yt-dlp")
        return 1

    # Process each split
    print(f"\n{'='*60}")
    print(f"Processing {len(splits_to_process)} split(s): {', '.join(splits_to_process)}")
    print(f"{'='*60}\n")

    for split in splits_to_process:
        print(f"\n{'#'*60}")
        print(f"# SPLIT: {split.upper()}")
        print(f"{'#'*60}\n")

        result = process_split(split, args)
        if result != 0:
            print(f"⚠️  Failed to process {split} split")

        # Small delay between splits
        if split != splits_to_process[-1]:
            print(f"\n⏸️  Pausing 5 seconds before next split...\n")
            import time
            time.sleep(5)

    # Check HDF5 existence
    if h5_path.exists() and not args.append:
        print(f"❌ HDF5 file already exists: {h5_path}")
        print(f"   Use --append to add to existing file")
        return 1

    # Check for yt-dlp
    try:
        subprocess.run(['yt-dlp', '--version'],
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ yt-dlp not found. Install with: pip install yt-dlp")
        return 1

    # Load metadata
    print(f"📋 Loading metadata from {metadata_path}")
    metadata = load_metadata(metadata_path)
    print(f"   Found {len(metadata)} entries in metadata")

    # Apply resume/max filters
    start_idx = args.resume_from
    end_idx = len(metadata)
    if args.max_downloads:
        end_idx = min(start_idx + args.max_downloads, end_idx)

    metadata_subset = metadata[start_idx:end_idx]
    print(f"   Processing entries {start_idx} to {end_idx} ({len(metadata_subset)} files)")

    # Initialize CLAP encoder
    print(f"🔧 Initializing CLAP encoder on {args.device}...")
    encoder = CLAPAudioEncoder(freeze=True)
    encoder = encoder.to(args.device)
    encoder.eval()
    print(f"   CLAP embedding dimension: {encoder.audio_embed_dim}")

    # Create or load HDF5
    if not h5_path.exists():
        print(f"📦 Creating new HDF5 file: {h5_path}")
        create_hdf5(h5_path, args.split, encoder.audio_embed_dim)
    else:
        print(f"📦 Appending to existing HDF5 file: {h5_path}")

    # Load existing sources or create new
    if sources_path.exists():
        with open(sources_path, 'r') as f:
            sources_dict = json.load(f)
        print(f"   Loaded {len(sources_dict)} existing source entries")
    else:
        sources_dict = {}

    # Create temporary directory for downloads
    temp_dir = Path(tempfile.mkdtemp(prefix='audiocaps_'))
    print(f"📁 Using temporary directory: {temp_dir}")

    # Thread locks for thread-safe operations
    h5_lock = threading.Lock()
    sources_lock = threading.Lock()

    # Process files
    success_count = 0
    download_fail_count = 0
    extraction_fail_count = 0
    save_fail_count = 0
    skip_count = 0

    print(f"\n🔄 Processing {len(metadata_subset)} files with {args.num_workers} parallel workers...\n")

    try:
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            future_to_meta = {
                executor.submit(
                    process_single_video,
                    meta,
                    temp_dir,
                    encoder,
                    h5_path,
                    sources_dict,
                    h5_lock,
                    sources_lock,
                    args.device,
                    args.cookies,
                    existing_ids
                ): meta
                for meta in metadata_subset
            }

            # Process results with progress bar
            with tqdm(total=len(metadata_subset), desc="Downloading & extracting") as pbar:
                processed_count = 0
                for future in as_completed(future_to_meta):
                    youtube_id, status = future.result()

                    if status == 'success':
                        success_count += 1
                    elif status in ('skip_hdf5', 'skip_session'):
                        skip_count += 1
                    elif status == 'download_fail':
                        download_fail_count += 1
                    elif status == 'extract_fail':
                        extraction_fail_count += 1
                    elif status == 'save_fail':
                        save_fail_count += 1
                    else:
                        # Handle unexpected errors
                        extraction_fail_count += 1

                    pbar.update(1)
                    processed_count += 1

                    # Periodic save of sources (thread-safe)
                    if processed_count % args.batch_log_interval == 0:
                        with sources_lock:
                            with open(sources_path, 'w') as f:
                                json.dump(sources_dict, f, indent=2)

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up temporary directory")

    # Final save of sources
    print(f"\n💾 Saving final source log to {sources_path}")
    with sources_lock:
        with open(sources_path, 'w') as f:
            json.dump(sources_dict, f, indent=2)
    print(f"   Logged {len(sources_dict)} total sources")

    # Summary
    print(f"\n" + "="*60)
    print(f"SUMMARY - {args.split.upper()} split")
    print(f"="*60)
    print(f"Requested:              {len(metadata_subset)}")
    print(f"Already processed:      {skip_count}")
    print(f"Successfully added:     {success_count}")
    print(f"Download failures:      {download_fail_count}")
    print(f"Extraction failures:    {extraction_fail_count}")
    print(f"Save failures:          {save_fail_count}")
    print(f"Parallel workers:       {args.num_workers}")
    print(f"HDF5 file:              {h5_path}")
    print(f"Sources log:            {sources_path}")
    print(f"="*60)

    # Get HDF5 stats
    with h5py.File(h5_path, 'r') as hf:
        total_samples = hf.attrs['num_samples']
        file_size_mb = h5_path.stat().st_size / (1024 * 1024)
        print(f"\n📊 HDF5 Statistics:")
        print(f"   Total embeddings: {total_samples}")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Avg per sample: {file_size_mb / max(total_samples, 1) * 1024:.1f} KB")

    print(f"\n✅ COMPLETE - No raw audio files retained")
    print(f"   ✓  All temporary files deleted")
    print(f"   ✓  Only derived features stored")

    return 0


if __name__ == '__main__':
    sys.exit(main())
