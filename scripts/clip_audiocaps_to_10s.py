#!/usr/bin/env python3
"""
Clip existing AudioCaps audio files to exactly 10 seconds using metadata.

This script fixes audio files that were downloaded as full videos instead of
10-second clips, using the start_time from the AudioCaps metadata CSV.
"""

import argparse
import subprocess
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def clip_audio_file(input_path, output_path, start_time, duration=10):
    """
    Clip audio file to specified duration using ffmpeg.

    Args:
        input_path: Path to input audio file
        output_path: Path to output clipped audio file
        start_time: Start time in seconds
        duration: Duration in seconds (default: 10)

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-ss', str(start_time),
        '-t', str(duration),
        '-acodec', 'pcm_s16le',  # WAV format
        '-ar', '48000',  # 48kHz sample rate
        '-ac', '1',  # Mono
        '-y',  # Overwrite output file
        '-loglevel', 'error',  # Only show errors
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"⏱ Timeout clipping {input_path.name}")
        return False
    except Exception as e:
        print(f"✗ Error clipping {input_path.name}: {e}")
        return False


def get_audio_duration(audio_path):
    """Get duration of audio file in seconds using ffprobe."""
    cmd = [
        'ffprobe',
        '-i', str(audio_path),
        '-show_entries', 'format=duration',
        '-v', 'quiet',
        '-of', 'csv=p=0'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    return None


def process_audiocaps_directory(csv_path, audio_dir, output_dir=None, min_duration=15.0):
    """
    Process directory of AudioCaps audio files and clip to 10 seconds.

    Args:
        csv_path: Path to AudioCaps CSV with youtube_id and start_time columns
        audio_dir: Directory containing audio files
        output_dir: Output directory (default: same as audio_dir with _10s suffix)
        min_duration: Only clip files longer than this duration (default: 15 seconds)
    """
    print(f"Processing AudioCaps audio files...")
    print(f"CSV: {csv_path}")
    print(f"Audio directory: {audio_dir}")

    if not csv_path.exists():
        print(f"✗ CSV file not found: {csv_path}")
        return

    if not audio_dir.exists():
        print(f"✗ Audio directory not found: {audio_dir}")
        return

    # Set output directory
    if output_dir is None:
        output_dir = audio_dir.parent / f"{audio_dir.name}_10s"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Check for required columns
    if 'youtube_id' not in df.columns:
        print(f"✗ CSV missing 'youtube_id' column")
        return

    has_start_time = 'start_time' in df.columns
    if not has_start_time:
        print("⚠️  Warning: CSV does not have 'start_time' column - clipping from beginning")
        df['start_time'] = 0

    # Create mapping from youtube_id to start_time
    # Handle both naming schemes:
    # 1. youtube_id.wav (from YouTube downloader)
    # 2. youtube_id_starttime.wav (from HuggingFace downloader)
    id_to_start_time = {}
    for _, row in df.iterrows():
        youtube_id = row['youtube_id']
        start_time = row['start_time']

        # Map plain youtube_id
        id_to_start_time[youtube_id] = start_time

        # Map youtube_id_starttime format (start_time in milliseconds or seconds)
        # Try both milliseconds and seconds
        start_ms = int(start_time * 1000)
        start_s = int(start_time)
        id_to_start_time[f"{youtube_id}_{start_ms:06d}"] = start_time
        id_to_start_time[f"{youtube_id}_{start_s:06d}"] = start_time

    # Find all audio files
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
    print(f"Found {len(audio_files)} audio files")

    # Stats
    clipped = 0
    skipped_already_10s = 0
    skipped_no_metadata = 0
    failed = 0
    already_exists = 0

    # Process each file
    for audio_file in tqdm(audio_files, desc="Clipping audio"):
        # Extract filename stem (handles both youtube_id.wav and youtube_id_starttime.wav)
        file_stem = audio_file.stem

        # Check if output already exists (preserve original filename)
        output_file = output_dir / f"{file_stem}.wav"
        if output_file.exists():
            already_exists += 1
            continue

        # Check if we have metadata for this file
        if file_stem not in id_to_start_time:
            skipped_no_metadata += 1
            continue

        # Get duration
        duration = get_audio_duration(audio_file)
        if duration is None:
            failed += 1
            continue

        # Skip files that are already ~10 seconds
        if duration < min_duration:
            skipped_already_10s += 1
            # Copy short files as-is
            import shutil
            shutil.copy2(audio_file, output_file)
            continue

        # Clip to 10 seconds using start_time from metadata
        start_time = int(id_to_start_time[file_stem])
        success = clip_audio_file(audio_file, output_file, start_time, duration=10)

        if success:
            clipped += 1
        else:
            failed += 1

    # Summary
    print("\n" + "="*70)
    print("CLIPPING SUMMARY")
    print("="*70)
    print(f"Clipped to 10s:       {clipped}")
    print(f"Already ~10s:         {skipped_already_10s}")
    print(f"Already processed:    {already_exists}")
    print(f"No metadata:          {skipped_no_metadata}")
    print(f"Failed:               {failed}")
    print(f"Total files:          {len(audio_files)}")
    print(f"\n📁 Clipped files saved to: {output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Clip existing AudioCaps audio files to 10 seconds"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to AudioCaps CSV with youtube_id and start_time columns"
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing audio files to clip"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for clipped files (default: {audio_dir}_10s)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=15.0,
        help="Only clip files longer than this duration in seconds (default: 15)"
    )

    args = parser.parse_args()

    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Error: ffmpeg not found. Please install ffmpeg first.")
        return

    process_audiocaps_directory(
        args.csv,
        args.audio_dir,
        args.output_dir,
        args.min_duration
    )


if __name__ == "__main__":
    main()
