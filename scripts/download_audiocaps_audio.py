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
import random
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


def get_random_user_agent():
    """Get random user agent to avoid bot detection."""
    user_agents = [
        # Android YouTube app (most effective for bypassing restrictions)
        'com.google.android.youtube/17.36.4 (Linux; U; Android 11) gzip',
        'com.google.android.youtube/18.11.34 (Linux; U; Android 13) gzip',
        'com.google.android.youtube/19.09.36 (Linux; U; Android 14) gzip',
        # iOS
        'com.google.ios.youtube/19.09.3 (iPhone14,3; U; CPU iOS 16_0 like Mac OS X)',
        # Desktop browsers
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]
    return random.choice(user_agents)

def download_audio(youtube_id, output_path, start_time=0, max_retries=3, cookies_file=None, adaptive_rate_limit=True):
    """
    Download audio from YouTube using yt-dlp with advanced anti-bot detection.

    Args:
        youtube_id: YouTube video ID
        output_path: Directory to save audio file
        start_time: Start time in seconds for 10-second clip (default: 0)
        max_retries: Number of download attempts
        cookies_file: Path to cookies.txt file
        adaptive_rate_limit: Enable human-like timing and rate limiting
    """
    url = f"https://www.youtube.com/watch?v={youtube_id}"

    # Human-like random delay before starting download (2-8 seconds)
    if adaptive_rate_limit:
        human_delay = random.uniform(2.0, 8.0)
        time.sleep(human_delay)

    # Calculate end time for 10-second clip
    end_time = start_time + 10

    # Base yt-dlp command for audio-only download with precise clipping
    base_cmd = [
        'yt-dlp',
        '--extract-audio',
        '--audio-format', 'wav',
        '--audio-quality', '0',  # Best quality
        '--output', str(output_path / f'{youtube_id}.%(ext)s'),
        '--no-playlist',
        '--ignore-errors',
        '--no-warnings',
        '--quiet',  # Suppress most output
        '--user-agent', get_random_user_agent(),  # Rotate user agents
        # Download and clip to exactly 10 seconds using ffmpeg
        '--postprocessor-args', f'ffmpeg:-ss {start_time} -t 10',
    ]

    # Add human-like random sleep intervals
    if adaptive_rate_limit:
        sleep_min = random.uniform(1.0, 2.0)
        sleep_max = random.uniform(3.0, 5.0)
        base_cmd.extend([
            '--sleep-interval', str(sleep_min),
            '--max-sleep-interval', str(sleep_max),
        ])

    # Add cookies if provided
    if cookies_file and os.path.exists(cookies_file):
        base_cmd.extend(['--cookies', cookies_file])

    # Advanced retry strategies with anti-bot measures
    retry_strategies = [
        # Strategy 1: Android client (most effective)
        ['--extractor-args', 'youtube:player_client=android'],
        # Strategy 2: Android with additional headers
        [
            '--extractor-args', 'youtube:player_client=android',
            '--add-header', 'X-YouTube-Client-Name:3',
            '--add-header', 'X-YouTube-Client-Version:17.36.4',
        ],
        # Strategy 3: iOS client
        ['--extractor-args', 'youtube:player_client=ios'],
        # Strategy 4: Force IPv4 with Android
        ['--force-ipv4', '--extractor-args', 'youtube:player_client=android'],
        # Strategy 5: Web client with custom headers
        [
            '--add-header', 'Accept-Language:en-US,en;q=0.9',
            '--add-header', 'Sec-Fetch-Dest:document',
            '--add-header', 'Sec-Fetch-Mode:navigate',
        ],
    ]

    bot_detected = False
    rate_limited = False

    for attempt in range(max_retries):
        # Use different strategy for each attempt
        strategy = retry_strategies[min(attempt, len(retry_strategies) - 1)]
        cmd = base_cmd + strategy + [url]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                return True
            else:
                # Check for specific errors
                stderr_lower = result.stderr.lower()

                # Detect bot/rate limit patterns (silently)
                if any(indicator in stderr_lower for indicator in [
                    'sign in', 'bot', 'captcha', 'too many requests',
                    'rate limit', 'unusual traffic', '403', '429'
                ]):
                    bot_detected = True

                if any(indicator in stderr_lower for indicator in ['rate', 'quota', 'limit']):
                    if not rate_limited:
                        rate_limited = True
                        # Adaptive backoff for rate limiting
                        backoff = random.uniform(10.0, 20.0) * (attempt + 1)
                        time.sleep(backoff)

                # Don't retry unavailable/private videos
                if any(indicator in stderr_lower for indicator in [
                    'private', 'unavailable', 'not available', 'removed'
                ]):
                    return False

        except subprocess.TimeoutExpired:
            pass  # Silent - will be counted in stats
        except Exception as e:
            pass  # Silent - will be counted in stats

        # Exponential backoff with jitter between retries
        if attempt < max_retries - 1:
            backoff = (2 ** attempt) + random.uniform(0, 2)
            time.sleep(backoff)

    return False

def download_single_clip(youtube_id, output_dir, start_time=0, cookies_file=None):
    """
    Download a single 10-second audio clip.

    Args:
        youtube_id: YouTube video ID
        output_dir: Directory to save audio
        start_time: Start time in seconds for the 10-second clip
        cookies_file: Path to cookies.txt file

    Returns:
        Tuple of (youtube_id, success, status_message)
    """
    # Skip if already downloaded
    potential_files = list(output_dir.glob(f"{youtube_id}.*"))
    if potential_files:
        return (youtube_id, True, "already_exists")

    # Download 10-second clip starting at start_time
    success = download_audio(youtube_id, output_dir, start_time=start_time, cookies_file=cookies_file)
    return (youtube_id, success, "success" if success else "failed")


def process_audiocaps_csv(csv_path, output_dir, max_downloads=None, num_workers=4, cookies_file=None):
    """Process AudioCaps CSV and download 10-second audio clips in parallel."""
    print(f"Processing {csv_path}...")

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return

    # Read CSV
    df = pd.read_csv(csv_path)

    # Check if start_time column exists
    has_start_time = 'start_time' in df.columns
    if not has_start_time:
        print("⚠️  Warning: CSV does not have 'start_time' column - downloading from beginning of videos")
        print("   For accurate 10-second clips, ensure CSV has 'start_time' column")
        df['start_time'] = 0  # Default to start

    # Limit downloads for testing
    if max_downloads:
        df = df.head(max_downloads)

    print(f"Found {len(df)} audio clips to download")
    print(f"Downloading 10-second clips" + (f" starting at specified times" if has_start_time else " from video start"))
    print(f"Using {num_workers} parallel workers")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track statistics
    successful = 0
    failed = 0
    already_exists = 0
    processed = 0

    # Download files in parallel (reduced workers to avoid rate limiting)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all download tasks with start_time
        future_to_id = {
            executor.submit(
                download_single_clip,
                row['youtube_id'],
                output_dir,
                start_time=int(row.get('start_time', 0)),  # Get start_time from CSV
                cookies_file=cookies_file
            ): row['youtube_id']
            for idx, row in df.iterrows()
        }

        # Process completed downloads with periodic updates (every 50 files)
        print(f"Downloading {len(df)} clips...")
        for future in as_completed(future_to_id):
            youtube_id, success, status = future.result()

            if status == "already_exists":
                already_exists += 1
                successful += 1
            elif success:
                successful += 1
            else:
                failed += 1

            processed += 1

            # Print progress every 50 files
            if processed % 50 == 0 or processed == len(df):
                downloaded = successful - already_exists
                print(f"Progress: {processed}/{len(df)} | Downloaded: {downloaded} | Skipped: {already_exists} | Failed: {failed}")

    print(f"\n✓ Download complete: {successful} successful ({already_exists} already existed), {failed} failed")

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