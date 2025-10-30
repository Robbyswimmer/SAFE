#!/usr/bin/env python3
"""
Download MUSIC-AVQA dataset with YouTube throttling protection.

This script downloads:
1. MUSIC-AVQA metadata (questions, answers, annotations)
2. Videos from YouTube with rate limiting and retry logic
3. Audio tracks extracted separately

Features:
- Intelligent rate limiting to avoid YouTube blocks
- Resume capability for interrupted downloads
- Parallel downloads with throttling
- Progress tracking and logging
- Cluster-friendly (can run in background)

Usage:
    python download_avqa.py --data-dir ./data/avqa --workers 4 --rate-limit 50K
"""

import os
import sys
import json
import time
import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import requests
from tqdm import tqdm
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('avqa_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AVQADownloader:
    """Download MUSIC-AVQA dataset with YouTube throttling protection."""

    # MUSIC-AVQA GitHub URLs
    METADATA_URLS = {
        "avqa-train": "https://raw.githubusercontent.com/gewu-lab/MUSIC-AVQA/main/data/avqa-train.json",
        "avqa-val": "https://raw.githubusercontent.com/gewu-lab/MUSIC-AVQA/main/data/avqa-val.json",
        "avqa-test": "https://raw.githubusercontent.com/gewu-lab/MUSIC-AVQA/main/data/avqa-test.json",
    }

    def __init__(
        self,
        data_dir: str,
        workers: int = 4,
        rate_limit: str = "50K",
        retry_attempts: int = 5,
        sleep_between_downloads: float = 3.0,
        video_format: str = "mp4",
        audio_format: str = "wav"
    ):
        """
        Initialize AVQA downloader.

        Args:
            data_dir: Root directory for AVQA data
            workers: Number of parallel download workers
            rate_limit: YouTube download rate limit (e.g., "50K", "100K")
            retry_attempts: Number of retry attempts for failed downloads
            sleep_between_downloads: Base sleep time between downloads (seconds)
            video_format: Video format to download
            audio_format: Audio format to extract
        """
        self.data_dir = Path(data_dir)
        self.workers = workers
        self.rate_limit = rate_limit
        self.retry_attempts = retry_attempts
        self.sleep_between_downloads = sleep_between_downloads
        self.video_format = video_format
        self.audio_format = audio_format

        # Create directories
        self.videos_dir = self.data_dir / "videos"
        self.audio_dir = self.data_dir / "audio"
        self.metadata_dir = self.data_dir / "metadata"

        for dir_path in [self.videos_dir, self.audio_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Track progress
        self.progress_file = self.data_dir / "download_progress.json"
        self.failed_file = self.data_dir / "failed_downloads.json"
        self.completed_videos = self._load_progress()
        self.failed_videos = self._load_failed()

        logger.info(f"Initialized AVQA downloader")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Workers: {workers}, Rate limit: {rate_limit}")

    def _load_progress(self) -> Set[str]:
        """Load previously completed downloads."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    data = json.load(f)
                    completed = set(data.get("completed", []))
                    logger.info(f"Loaded {len(completed)} completed downloads")
                    return completed
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")
        return set()

    def _load_failed(self) -> Dict[str, int]:
        """Load failed download attempts."""
        if self.failed_file.exists():
            try:
                with open(self.failed_file) as f:
                    failed = json.load(f)
                    logger.info(f"Loaded {len(failed)} failed downloads")
                    return failed
            except Exception as e:
                logger.warning(f"Could not load failed downloads: {e}")
        return {}

    def _save_progress(self):
        """Save download progress."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    "completed": list(self.completed_videos),
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save progress: {e}")

    def _save_failed(self):
        """Save failed downloads."""
        try:
            with open(self.failed_file, 'w') as f:
                json.dump(self.failed_videos, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save failed downloads: {e}")

    def check_yt_dlp(self) -> bool:
        """Check if yt-dlp is installed."""
        try:
            result = subprocess.run(
                ["yt-dlp", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"yt-dlp version: {version}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        logger.error("yt-dlp not found. Install with: pip install yt-dlp")
        return False

    def download_metadata(self) -> Dict[str, List[Dict]]:
        """Download MUSIC-AVQA metadata files."""
        logger.info("="*60)
        logger.info("DOWNLOADING MUSIC-AVQA METADATA")
        logger.info("="*60)

        all_metadata = {}

        for split_name, url in self.METADATA_URLS.items():
            logger.info(f"Downloading {split_name}...")

            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                # Save raw JSON
                metadata_file = self.metadata_dir / f"{split_name}.json"
                with open(metadata_file, 'w') as f:
                    json.dump(response.json(), f, indent=2)

                all_metadata[split_name] = response.json()
                logger.info(f"✓ Saved {split_name} metadata: {len(response.json())} samples")

            except Exception as e:
                logger.error(f"✗ Failed to download {split_name}: {e}")
                continue

        return all_metadata

    def extract_video_ids(self, metadata: Dict[str, List[Dict]]) -> Dict[str, Set[str]]:
        """Extract unique YouTube video IDs from metadata."""
        logger.info("Extracting YouTube video IDs...")

        video_ids_by_split = {}
        all_video_ids = set()

        for split_name, samples in metadata.items():
            video_ids = set()

            for sample in samples:
                # MUSIC-AVQA stores video IDs in different possible fields
                video_id = None

                if isinstance(sample, dict):
                    video_id = (
                        sample.get("video_id") or
                        sample.get("video") or
                        sample.get("youtube_id")
                    )

                if video_id:
                    video_ids.add(video_id)
                    all_video_ids.add(video_id)

            video_ids_by_split[split_name] = video_ids
            logger.info(f"  {split_name}: {len(video_ids)} unique videos")

        logger.info(f"Total unique videos: {len(all_video_ids)}")

        # Save video ID list
        id_list_file = self.metadata_dir / "video_ids.json"
        with open(id_list_file, 'w') as f:
            json.dump({
                "by_split": {k: list(v) for k, v in video_ids_by_split.items()},
                "all_ids": list(all_video_ids)
            }, f, indent=2)

        return video_ids_by_split

    def download_video(self, video_id: str, retry_count: int = 0) -> bool:
        """
        Download a single video from YouTube with throttling.

        Args:
            video_id: YouTube video ID
            retry_count: Current retry attempt

        Returns:
            True if successful, False otherwise
        """
        # Check if already completed
        if video_id in self.completed_videos:
            return True

        # Check if permanently failed
        if self.failed_videos.get(video_id, 0) >= self.retry_attempts:
            logger.debug(f"Skipping {video_id} (permanently failed)")
            return False

        video_path = self.videos_dir / f"{video_id}.{self.video_format}"
        audio_path = self.audio_dir / f"{video_id}.{self.audio_format}"

        # Skip if both files exist
        if video_path.exists() and audio_path.exists():
            self.completed_videos.add(video_id)
            self._save_progress()
            return True

        # Add random jitter to sleep time to avoid synchronized requests
        sleep_time = self.sleep_between_downloads + random.uniform(0, 2.0)
        time.sleep(sleep_time)

        youtube_url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            # Download video with yt-dlp
            cmd = [
                "yt-dlp",
                "--quiet",
                "--no-warnings",
                f"--rate-limit={self.rate_limit}",
                "--format=best[height<=480]",  # Limit to 480p to save bandwidth
                f"--output={video_path}",
                "--no-playlist",
                "--no-overwrites",
                youtube_url
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"yt-dlp failed: {result.stderr}")

            # Extract audio separately
            audio_cmd = [
                "yt-dlp",
                "--quiet",
                "--no-warnings",
                f"--rate-limit={self.rate_limit}",
                "--extract-audio",
                f"--audio-format={self.audio_format}",
                f"--output={audio_path}",
                "--no-playlist",
                "--no-overwrites",
                youtube_url
            ]

            result = subprocess.run(
                audio_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                logger.warning(f"Audio extraction failed for {video_id}: {result.stderr}")

            # Verify files exist
            if video_path.exists():
                self.completed_videos.add(video_id)
                self._save_progress()

                # Clear from failed list if it was there
                if video_id in self.failed_videos:
                    del self.failed_videos[video_id]
                    self._save_failed()

                logger.info(f"✓ Downloaded: {video_id}")
                return True
            else:
                raise RuntimeError("Video file not created")

        except Exception as e:
            logger.warning(f"✗ Failed to download {video_id} (attempt {retry_count + 1}): {e}")

            # Track failed attempt
            self.failed_videos[video_id] = self.failed_videos.get(video_id, 0) + 1
            self._save_failed()

            # Retry if attempts remaining
            if retry_count < self.retry_attempts - 1:
                backoff_time = (2 ** retry_count) * 5  # Exponential backoff
                logger.info(f"Retrying {video_id} in {backoff_time}s...")
                time.sleep(backoff_time)
                return self.download_video(video_id, retry_count + 1)

            return False

    def download_videos_parallel(self, video_ids: Set[str]):
        """Download videos in parallel with throttling."""
        logger.info("="*60)
        logger.info(f"DOWNLOADING {len(video_ids)} VIDEOS")
        logger.info("="*60)
        logger.info(f"Workers: {self.workers}")
        logger.info(f"Rate limit: {self.rate_limit}")
        logger.info(f"Sleep between downloads: {self.sleep_between_downloads}s")
        logger.info("="*60)

        # Filter out already completed
        remaining_ids = [vid for vid in video_ids if vid not in self.completed_videos]

        logger.info(f"Already completed: {len(video_ids) - len(remaining_ids)}")
        logger.info(f"Remaining to download: {len(remaining_ids)}")

        if not remaining_ids:
            logger.info("All videos already downloaded!")
            return

        # Download with progress bar
        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Submit all download tasks
            futures = {
                executor.submit(self.download_video, video_id): video_id
                for video_id in remaining_ids
            }

            # Track progress
            with tqdm(total=len(remaining_ids), desc="Downloading videos") as pbar:
                for future in as_completed(futures):
                    video_id = futures[future]
                    try:
                        success = future.result()
                        if success:
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"Unexpected error for {video_id}: {e}")
                        failed += 1

                    pbar.update(1)
                    pbar.set_postfix({
                        "success": successful,
                        "failed": failed
                    })

        logger.info("="*60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("="*60)
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total completed: {len(self.completed_videos)}")
        logger.info(f"Success rate: {successful / len(remaining_ids) * 100:.1f}%")

    def verify_downloads(self) -> Dict[str, int]:
        """Verify downloaded files."""
        logger.info("="*60)
        logger.info("VERIFYING DOWNLOADS")
        logger.info("="*60)

        video_files = list(self.videos_dir.glob(f"*.{self.video_format}"))
        audio_files = list(self.audio_dir.glob(f"*.{self.audio_format}"))

        stats = {
            "videos": len(video_files),
            "audio": len(audio_files),
            "metadata_files": len(list(self.metadata_dir.glob("*.json")))
        }

        logger.info(f"Video files: {stats['videos']}")
        logger.info(f"Audio files: {stats['audio']}")
        logger.info(f"Metadata files: {stats['metadata_files']}")

        # Check for orphaned files
        video_ids = {f.stem for f in video_files}
        audio_ids = {f.stem for f in audio_files}

        only_video = video_ids - audio_ids
        only_audio = audio_ids - video_ids

        if only_video:
            logger.warning(f"Files with video only: {len(only_video)}")
        if only_audio:
            logger.warning(f"Files with audio only: {len(only_audio)}")

        return stats

    def run(self):
        """Run complete download pipeline."""
        logger.info("="*60)
        logger.info("MUSIC-AVQA DATASET DOWNLOADER")
        logger.info("="*60)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Check dependencies
        if not self.check_yt_dlp():
            return False

        # Step 1: Download metadata
        metadata = self.download_metadata()
        if not metadata:
            logger.error("Failed to download metadata")
            return False

        # Step 2: Extract video IDs
        video_ids_by_split = self.extract_video_ids(metadata)
        all_video_ids = set()
        for video_ids in video_ids_by_split.values():
            all_video_ids.update(video_ids)

        # Step 3: Download videos
        self.download_videos_parallel(all_video_ids)

        # Step 4: Verify
        stats = self.verify_downloads()

        # Final summary
        logger.info("="*60)
        logger.info("FINAL SUMMARY")
        logger.info("="*60)
        logger.info(f"Total videos to download: {len(all_video_ids)}")
        logger.info(f"Successfully downloaded: {len(self.completed_videos)}")
        logger.info(f"Failed downloads: {len(self.failed_videos)}")
        logger.info(f"Video files: {stats['videos']}")
        logger.info(f"Audio files: {stats['audio']}")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Download MUSIC-AVQA dataset with YouTube throttling protection"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/avqa",
        help="Root directory for AVQA data (default: ./data/avqa)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4, max recommended: 8)"
    )
    parser.add_argument(
        "--rate-limit",
        type=str,
        default="50K",
        help="YouTube download rate limit, e.g., '50K', '100K' (default: 50K)"
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=5,
        help="Number of retry attempts for failed downloads (default: 5)"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="Base sleep time between downloads in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--video-format",
        type=str,
        default="mp4",
        help="Video format (default: mp4)"
    )
    parser.add_argument(
        "--audio-format",
        type=str,
        default="wav",
        help="Audio format (default: wav)"
    )

    args = parser.parse_args()

    # Create downloader
    downloader = AVQADownloader(
        data_dir=args.data_dir,
        workers=args.workers,
        rate_limit=args.rate_limit,
        retry_attempts=args.retry_attempts,
        sleep_between_downloads=args.sleep,
        video_format=args.video_format,
        audio_format=args.audio_format
    )

    # Run download
    try:
        success = downloader.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\nDownload interrupted by user.")
        logger.info("Progress saved. Run again to resume.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
