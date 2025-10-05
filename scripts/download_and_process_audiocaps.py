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
    # Default: 4 parallel workers
    python scripts/download_and_process_audiocaps.py --split train --max-downloads 100

    # Custom parallelism
    python scripts/download_and_process_audiocaps.py --split train --num-workers 8

    # Append to existing file
    python scripts/download_and_process_audiocaps.py --split val --append

    # Resume from specific index
    python scripts/download_and_process_audiocaps.py --split train --resume-from 500

Ethical Note:
    This implements fair use by never storing raw audio permanently.
    Only derived features (embeddings) are retained for research.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

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
    duration: int = 10
) -> Optional[Path]:
    """
    Download 10s audio segment using yt-dlp to temporary directory.

    Returns:
        Path to downloaded WAV file, or None if download failed
    """
    output_path = temp_dir / f"{youtube_id}.wav"

    # yt-dlp command to extract audio segment
    cmd = [
        'yt-dlp',
        '-x',  # Extract audio
        '--audio-format', 'wav',
        '--audio-quality', '0',  # Best quality
        '-o', str(output_path.with_suffix('')),  # Output template (without extension)
        '--postprocessor-args', f'ffmpeg:-ss {start_time} -t {duration}',  # Extract segment
        '--no-playlist',
        '--quiet',
        '--no-warnings',
        f'https://www.youtube.com/watch?v={youtube_id}'
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)

        # yt-dlp adds .wav extension
        if output_path.exists():
            return output_path
        else:
            print(f"  ⚠️  Download succeeded but file not found: {youtube_id}")
            return None

    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Download timeout: {youtube_id}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  Download failed: {youtube_id} - {e.stderr.decode()[:100]}")
        return None
    except Exception as e:
        print(f"  ⚠️  Unexpected error: {youtube_id} - {e}")
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
    device: str
) -> Tuple[str, str]:
    """
    Process a single video: download, extract, save, delete.

    Returns:
        (youtube_id, status): status is one of 'success', 'skip', 'download_fail', 'extract_fail', 'save_fail'
    """
    youtube_id = meta['youtube_id']

    # Check if already processed (thread-safe read)
    with sources_lock:
        if youtube_id in sources_dict:
            return (youtube_id, 'skip')

    # Download to temporary directory
    audio_path = download_audio_segment(
        youtube_id,
        meta['start_time'],
        temp_dir
    )

    if audio_path is None:
        return (youtube_id, 'download_fail')

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


def main():
    parser = argparse.ArgumentParser(
        description='Download and process AudioCaps with auto-cleanup'
    )
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train',
                        help='Dataset split to process')
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

    args = parser.parse_args()

    # Setup paths
    data_root = Path(args.data_root).expanduser().resolve()
    metadata_path = data_root / 'metadata' / f'{args.split}.csv'
    features_dir = data_root / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)

    h5_path = features_dir / f'{args.split}_clap_embeddings.h5'
    sources_path = features_dir / f'{args.split}_sources.json'

    # Validate paths
    if not metadata_path.exists():
        print(f"❌ Metadata not found: {metadata_path}")
        return 1

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
                    args.device
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
                    elif status == 'skip':
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
