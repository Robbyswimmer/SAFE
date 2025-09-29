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

def download_audio(youtube_id, output_path, max_retries=3):
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
        url
    ]
    
    for attempt in range(max_retries):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return True
            else:
                if attempt == max_retries - 1:
                    print(f"Failed to download {youtube_id}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"Timeout downloading {youtube_id} (attempt {attempt + 1})")
        except Exception as e:
            print(f"Error downloading {youtube_id}: {e}")
        
        time.sleep(1)  # Brief pause between retries
    
    return False

def download_single_clip(youtube_id, output_dir):
    """Download a single audio clip. Returns (youtube_id, success)."""
    # Skip if already downloaded
    potential_files = list(output_dir.glob(f"{youtube_id}.*"))
    if potential_files:
        return (youtube_id, True, "already_exists")

    # Download
    success = download_audio(youtube_id, output_dir)
    return (youtube_id, success, "success" if success else "failed")


def process_audiocaps_csv(csv_path, output_dir, max_downloads=None, num_workers=8):
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

    # Download files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all download tasks
        future_to_id = {
            executor.submit(download_single_clip, row['youtube_id'], output_dir): row['youtube_id']
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
    print("AudioCaps Audio Downloader")
    print("="*50)
    
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
    
    # Default to 2000 samples for Stage 1 testing
    max_downloads = 2000
    print(f"Downloading {max_downloads} AudioCaps samples for Stage 1 validation...")
    
    print(f"\nStarting download (max: {max_downloads or 'unlimited'} per split)...")
    
    # Download each split
    for split in ["train", "val", "test"]:
        csv_path = metadata_dir / f"{split}.csv"
        split_audio_dir = audio_dir / split
        
        if csv_path.exists():
            print(f"\n--- Processing {split} split ---")
            process_audiocaps_csv(csv_path, split_audio_dir, max_downloads)
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