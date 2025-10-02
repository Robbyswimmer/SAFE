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

    # yt-dlp command for audio-only download
    cmd = [
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
        url
    ]

    # Add cookies if provided
    if cookies_file and os.path.exists(cookies_file):
        cmd.insert(-1, '--cookies')
        cmd.insert(-1, cookies_file)

    for attempt in range(max_retries):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if result.returncode == 0:
                return True
            else:
                # Check for specific errors
                stderr_lower = result.stderr.lower()
                if 'sign in' in stderr_lower or 'bot' in stderr_lower:
                    if attempt == 0:
                        print(f"⚠ {youtube_id}: YouTube bot detection - consider using --cookies")
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

def main():
    """Main download function."""
    import argparse

    parser = argparse.ArgumentParser(description="Download AudioCaps audio from YouTube")
    parser.add_argument("--cookies", type=str, help="Path to cookies.txt file for YouTube authentication")
    parser.add_argument("--max-downloads", type=int, default=2000, help="Maximum downloads per split")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (reduce if rate limited)")
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
        print("ℹ No cookies file specified. If you encounter bot detection errors:")
        print("  1. Install browser extension 'Get cookies.txt' or 'cookies.txt'")
        print("  2. Export YouTube cookies to cookies.txt")
        print("  3. Run with: python download_audiocaps_audio.py --cookies cookies.txt")

    # Check dependencies
    if not check_ytdlp():
        return

    # Set up paths
    data_dir = Path("./data/audiocaps")
    metadata_dir = data_dir / "metadata"
    audio_dir = data_dir / "audio"

    if not metadata_dir.exists():
        print(f"Metadata directory not found: {metadata_dir}")
        print("Please run the dataset setup script first:")
        print("bash scripts/setup_datasets.sh")
        return

    print(f"Downloading up to {args.max_downloads} AudioCaps samples per split...")
    print(f"Using {args.workers} parallel workers")

    print(f"\nStarting download (max: {args.max_downloads or 'unlimited'} per split)...")

    # Download each split
    for split in ["train", "val", "test"]:
        csv_path = metadata_dir / f"{split}.csv"
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