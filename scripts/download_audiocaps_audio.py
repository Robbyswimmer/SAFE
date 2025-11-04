#!/usr/bin/env python3
"""
Download AudioCaps audio files from YouTube.
Requires yt-dlp: pip install yt-dlp
"""

import os
import sys
import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_ytdlp():
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: yt-dlp not found.")
        print("Install with: pip install yt-dlp")
        return False

def download_audio(youtube_id, output_path, max_retries=3, cookies_file=None):
    """Download audio from YouTube using yt-dlp."""
    url = f"https://www.youtube.com/watch?v={youtube_id}"

    # Base yt-dlp command for audio-only download
    base_cmd = [
        'yt-dlp',
        '--extract-audio',
        '--audio-format', 'wav',
        '--audio-quality', '0',  # Best quality
        '--output', str(output_path / f'{youtube_id}.%(ext)s'),
        '--no-playlist',
        '--ignore-errors',
        '--sleep-interval', '1',  # Sleep 1-2 seconds between downloads
        '--max-sleep-interval', '2',
        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    ]

    # Add cookies if provided
    if cookies_file and os.path.exists(cookies_file):
        base_cmd.extend(['--cookies', cookies_file])

    # Retry strategies: Android client first (no auth needed), then fallback
    retry_strategies = [
        ['--extractor-args', 'youtube:player_client=android'],  # Android client (bypasses bot detection)
        ['--force-ipv4', '--extractor-args', 'youtube:player_client=android'],
        [],  # Default (no extra args)
    ]

    for attempt in range(max_retries):
        # Use different strategy for each attempt
        strategy = retry_strategies[min(attempt, len(retry_strategies) - 1)]
        cmd = base_cmd + strategy + [url]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if result.returncode == 0:
                return True
            else:
                # Check for specific errors
                stderr_lower = result.stderr.lower()
                if 'sign in' in stderr_lower or 'bot' in stderr_lower:
                    if attempt == 0:
                        print(f"⚠ {youtube_id}: YouTube bot detection - trying Android client...")
                elif 'private' in stderr_lower:
                    # Don't retry private videos
                    return False

                if attempt == max_retries - 1:
                    # Only print error on final attempt
                    error_msg = result.stderr.split('\n')[0] if result.stderr else 'Unknown error'
                    print(f"✗ {youtube_id}: {error_msg}")
        except subprocess.TimeoutExpired:
            print(f"⏱ Timeout downloading {youtube_id} (attempt {attempt + 1})")
        except Exception as e:
            print(f"✗ Error downloading {youtube_id}: {e}")

        # Exponential backoff between retries
        time.sleep(2 ** attempt)

    return False

def download_single_clip(youtube_id, output_dir, cookies_file=None):
    """Download a single audio clip. Returns (youtube_id, success)."""
    # Skip if already downloaded
    potential_files = list(output_dir.glob(f"{youtube_id}.*"))
    if potential_files:
        return (youtube_id, True, "already_exists")

    # Download
    success = download_audio(youtube_id, output_dir, cookies_file=cookies_file)
    return (youtube_id, success, "success" if success else "failed")


def process_audiocaps_csv(csv_path, output_dir, max_downloads=None, num_workers=4, cookies_file=None):
    """Process AudioCaps CSV and download audio files in parallel."""
    print(f"Processing {csv_path}...")

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return

    # Read CSV
    df = pd.read_csv(csv_path)

    # Limit downloads for testing
    if max_downloads:
        df = df.head(max_downloads)

    print(f"Found {len(df)} audio clips to download")
    print(f"Using {num_workers} parallel workers")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track statistics
    successful = 0
    failed = 0
    already_exists = 0

    # Download files in parallel (reduced workers to avoid rate limiting)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all download tasks
        future_to_id = {
            executor.submit(download_single_clip, row['youtube_id'], output_dir, cookies_file): row['youtube_id']
            for idx, row in df.iterrows()
        }

        # Process completed downloads with progress bar
        with tqdm(total=len(df), desc="Downloading") as pbar:
            for future in as_completed(future_to_id):
                youtube_id, success, status = future.result()

                if status == "already_exists":
                    already_exists += 1
                    successful += 1
                elif success:
                    successful += 1
                else:
                    failed += 1

                pbar.update(1)

    print(f"Download complete: {successful} successful ({already_exists} already existed), {failed} failed")

def download_metadata_csv(split, metadata_dir):
    """Download AudioCaps metadata CSV from GitHub if not present."""
    csv_path = metadata_dir / f"{split}.csv"
    if csv_path.exists():
        return csv_path

    # Download from official AudioCaps GitHub
    base_url = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset"
    url = f"{base_url}/{split}.csv"

    print(f"Downloading metadata for {split} split from {url}...")
    try:
        import requests
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        metadata_dir.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(response.text)
        print(f"✓ Downloaded metadata: {csv_path}")
        return csv_path
    except Exception as e:
        print(f"✗ Failed to download metadata: {e}")
        return None


def main():
    """Main download function."""
    import argparse

    parser = argparse.ArgumentParser(description="Download AudioCaps audio from YouTube")
    parser.add_argument("--cookies", type=str, help="Path to cookies.txt file for YouTube authentication")
    parser.add_argument("--max-downloads", type=int, default=None, help="Maximum downloads per split (default: all)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (reduce if rate limited)")
    parser.add_argument("--data-root", type=str, default="experiments/full_training/data/audiocaps",
                        help="Root directory for AudioCaps data")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"],
                        help="Specific split to download (default: all splits)")
    args = parser.parse_args()

    print("AudioCaps Audio Downloader")
    print("="*50)

    if args.cookies:
        if not os.path.exists(args.cookies):
            print(f"⚠ Warning: Cookies file not found: {args.cookies}")
            print("You can export cookies using a browser extension like 'Get cookies.txt'")
        else:
            print(f"✓ Using cookies from: {args.cookies}")
    else:
        print("ℹ No cookies specified - using Android client to bypass bot detection")

    # Check dependencies
    if not check_ytdlp():
        return

    # Set up paths
    data_dir = Path(args.data_root).expanduser().resolve()
    audio_dir = data_dir / "audio"

    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Downloading up to {args.max_downloads or 'all'} AudioCaps samples per split...")
    print(f"Using {args.workers} parallel workers")

    print(f"\nStarting download...")

    # Determine which splits to download
    splits_to_download = [args.split] if args.split else ["train", "val", "test"]

    # Download each split
    for split in splits_to_download:
        # Download metadata CSV if needed
        csv_path = download_metadata_csv(split, data_dir)
        if not csv_path:
            print(f"Skipping {split}: metadata not available")
            continue

        split_audio_dir = audio_dir / split

        if csv_path.exists():
            print(f"\n--- Processing {split} split ---")
            process_audiocaps_csv(csv_path, split_audio_dir, args.max_downloads, args.workers, args.cookies)
        else:
            print(f"Skipping {split}: {csv_path} not found")
    
    # Summary
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    
    for split in ["train", "val", "test"]:
        split_dir = audio_dir / split
        if split_dir.exists():
            file_count = len(list(split_dir.glob("*.wav")))
            print(f"{split.capitalize():<10}: {file_count} audio files")
        else:
            print(f"{split.capitalize():<10}: 0 audio files")
    
    total_files = len(list(audio_dir.rglob("*.wav")))
    print(f"{'Total':<10}: {total_files} audio files")
    print(f"\n📁 Audio files saved to: {audio_dir.absolute()}")
    
    if total_files > 0:
        print("\n✓ AudioCaps audio download complete!")
        print("You can now test with real data:")
        print("python train_stage_a_curriculum.py --config demo --use-real-data")
    else:
        print("\n⚠ No audio files were downloaded successfully.")
        print("Check your internet connection and try again.")

if __name__ == "__main__":
    main()