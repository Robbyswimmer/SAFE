#!/usr/bin/env python3
"""
Download General AVQA dataset (Yang et al., 2022) with YouTube throttling protection.

This dataset uses VGG-Sound videos for audio-visual question answering on daily activities.

Dataset: 57,015 videos, 57,335 QA pairs
Source: https://mn.cs.tsinghua.edu.cn/avqa/
Paper: AVQA: A Dataset for Audio-Visual Question Answering on Videos (ACM MM 2022)

Usage:
    python download_general_avqa.py --data-dir ./data/avqa --workers 4 --rate-limit 50K
"""

import os
import sys
import json
import time
import argparse
import logging
import subprocess
import csv
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
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
        logging.FileHandler('avqa_general_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GeneralAVQADownloader:
    """Download General AVQA dataset with YouTube throttling protection."""

    # OneDrive share link from official website
    ONEDRIVE_SHARE_LINK = "https://tsinghuaeducn-my.sharepoint.com/:u:/g/personal/xin_wang_tsinghua_edu_cn/EQ_7OroeDPZFjajxJXsCh34BtYs-6GDdb-KFPfqgsu50cw?e=qeCgOO"

    # Alternative: Direct file download attempts
    # These might work if OneDrive link is publicly accessible
    METADATA_URLS = {
        "train_qa.json": "https://tsinghuaeducn-my.sharepoint.com/:u:/g/personal/xin_wang_tsinghua_edu_cn/EQ_7OroeDPZFjajxJXsCh34BtYs-6GDdb-KFPfqgsu50cw?download=1",
        "val_qa.json": "https://tsinghuaeducn-my.sharepoint.com/:u:/g/personal/xin_wang_tsinghua_edu_cn/EQ_7OroeDPZFjajxJXsCh34BtYs-6GDdb-KFPfqgsu50cw?download=1"
    }

    # VGG-Sound CSV URL
    VGGSOUND_CSV_URL = "https://www.robots.ox.ac.uk/~vgg/data/vggsound/vggsound.csv"

    def __init__(
        self,
        data_dir: str,
        workers: int = 4,
        rate_limit: str = "50K",
        retry_attempts: int = 5,
        sleep_between_downloads: float = 3.0,
        video_format: str = "mp4",
        audio_format: str = "wav",
        quality: str = "360",  # 360p or 480p
        max_videos: Optional[int] = None
    ):
        """
        Initialize General AVQA downloader.

        Args:
            data_dir: Root directory for AVQA data
            workers: Number of parallel download workers
            rate_limit: YouTube download rate limit (e.g., "50K", "100K")
            retry_attempts: Number of retry attempts for failed downloads
            sleep_between_downloads: Base sleep time between downloads (seconds)
            video_format: Video format to download
            audio_format: Audio format to extract
            quality: Video quality (360 or 480)
            max_videos: Maximum number of videos to download (None for all)
        """
        self.data_dir = Path(data_dir)
        self.workers = workers
        self.rate_limit = rate_limit
        self.retry_attempts = retry_attempts
        self.sleep_between_downloads = sleep_between_downloads
        self.video_format = video_format
        self.audio_format = audio_format
        self.quality = quality
        self.max_videos = max_videos

        # Create directories
        self.videos_dir = self.data_dir / "videos"
        self.audio_dir = self.data_dir / "audio"
        self.metadata_dir = self.data_dir / "metadata"
        self.features_dir = self.data_dir / "features"

        for dir_path in [self.videos_dir, self.audio_dir, self.metadata_dir, self.features_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Track progress
        self.progress_file = self.data_dir / "download_progress.json"
        self.failed_file = self.data_dir / "failed_downloads.json"
        self.completed_videos = self._load_progress()
        self.failed_videos = self._load_failed()

        logger.info(f"Initialized General AVQA downloader")
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

    def _download_from_onedrive(self, url: str, output_path: Path, description: str) -> bool:
        """
        Attempt to download a file from OneDrive share link.
        Tries multiple methods to handle OneDrive redirects and authentication.
        """
        try:
            logger.info(f"Attempting to download {description}...")

            # Method 1: Try direct download with redirects
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

            response = session.get(url, allow_redirects=True, timeout=120, stream=True)

            # Check if we got actual content
            content_type = response.headers.get('content-type', '').lower()

            if response.status_code == 200:
                # Check if it's JSON or a download page
                if 'application/json' in content_type or 'text/plain' in content_type:
                    # Likely got JSON directly
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    # Verify it's valid JSON
                    try:
                        with open(output_path) as f:
                            json.load(f)
                        logger.info(f"✓ Successfully downloaded {description}")
                        return True
                    except json.JSONDecodeError:
                        logger.warning(f"Downloaded file is not valid JSON")
                        output_path.unlink(missing_ok=True)
                        return False

                elif 'application/octet-stream' in content_type or 'application/zip' in content_type:
                    # Might be a download
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    # Try to parse as JSON
                    try:
                        with open(output_path) as f:
                            json.load(f)
                        logger.info(f"✓ Successfully downloaded {description}")
                        return True
                    except:
                        output_path.unlink(missing_ok=True)
                        return False
                else:
                    logger.warning(f"Got unexpected content type: {content_type}")
                    return False
            else:
                logger.warning(f"HTTP {response.status_code} for {description}")
                return False

        except Exception as e:
            logger.warning(f"Could not download {description}: {e}")
            return False

    def download_metadata(self) -> bool:
        """Download AVQA metadata files (train_qa.json, val_qa.json)."""
        logger.info("="*60)
        logger.info("DOWNLOADING GENERAL AVQA METADATA")
        logger.info("="*60)

        required_files = {
            "train_qa.json": "Training QA pairs",
            "val_qa.json": "Validation QA pairs"
        }

        # Check if files already exist
        all_exist = all((self.metadata_dir / f).exists() for f in required_files.keys())

        if all_exist:
            logger.info("✓ All metadata files already exist!")
            return True

        # Try to download from OneDrive automatically
        logger.info("Attempting automatic download from OneDrive...")
        logger.info("Note: This requires the OneDrive link to be publicly accessible")

        success_count = 0

        # Try downloading the shared folder/file
        # OneDrive shared links often point to a folder or ZIP
        # We'll try to download and extract if it's a ZIP

        try:
            logger.info(f"Accessing OneDrive share link...")
            response = requests.get(self.ONEDRIVE_SHARE_LINK, allow_redirects=True, timeout=120)

            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()

                # Check if it's a ZIP file
                if 'zip' in content_type or 'application/octet-stream' in content_type:
                    logger.info("Detected ZIP archive, downloading and extracting...")

                    zip_path = self.metadata_dir / "avqa_data.zip"
                    with open(zip_path, 'wb') as f:
                        f.write(response.content)

                    # Extract JSON files
                    import zipfile
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            # Extract only JSON files
                            for member in zip_ref.namelist():
                                if member.endswith('.json') and ('train_qa' in member or 'val_qa' in member):
                                    # Extract to metadata dir
                                    zip_ref.extract(member, self.metadata_dir)

                                    # Move to correct location if in subdirectory
                                    extracted_path = self.metadata_dir / member
                                    if '/' in member:
                                        filename = Path(member).name
                                        target_path = self.metadata_dir / filename
                                        extracted_path.rename(target_path)

                                    logger.info(f"✓ Extracted {Path(member).name}")
                                    success_count += 1

                        zip_path.unlink()  # Clean up ZIP

                    except zipfile.BadZipFile:
                        logger.warning("Downloaded file is not a valid ZIP archive")
                        zip_path.unlink(missing_ok=True)
                else:
                    logger.warning(f"OneDrive returned unexpected content type: {content_type}")
            else:
                logger.warning(f"OneDrive returned status code: {response.status_code}")

        except Exception as e:
            logger.warning(f"Could not access OneDrive share link: {e}")

        # Check what we got
        for filename, description in required_files.items():
            file_path = self.metadata_dir / filename
            if file_path.exists():
                logger.info(f"✓ Found {filename}")
                success_count += 1

        # If we have all files, success
        if success_count == len(required_files):
            logger.info("✓ All metadata files downloaded successfully!")
            return True

        # If automatic download failed, provide instructions
        logger.error("=" * 60)
        logger.error("AUTOMATIC DOWNLOAD FAILED")
        logger.error("=" * 60)
        logger.error("Could not automatically download metadata from OneDrive.")
        logger.error("This usually happens because:")
        logger.error("1. The OneDrive link requires authentication")
        logger.error("2. The link structure has changed")
        logger.error("3. Network/firewall restrictions")
        logger.error("")
        logger.error("MANUAL STEPS REQUIRED:")
        logger.error("1. Visit: https://mn.cs.tsinghua.edu.cn/avqa/")
        logger.error("2. Click the OneDrive link")
        logger.error("3. Download the files (likely a ZIP archive)")
        logger.error("4. Extract train_qa.json and val_qa.json")
        logger.error(f"5. Place them in: {self.metadata_dir}")
        logger.error("")
        logger.error("Required files:")
        for filename in required_files.keys():
            file_path = self.metadata_dir / filename
            status = "✓ EXISTS" if file_path.exists() else "✗ MISSING"
            logger.error(f"  {status}: {filename}")
        logger.error("")
        logger.error("After placing files, run this script again to continue.")
        logger.error("=" * 60)

        return False

    def download_vggsound_csv(self) -> Optional[Path]:
        """Download VGG-Sound CSV with video URLs."""
        logger.info("Downloading VGG-Sound CSV with video URLs...")

        csv_path = self.metadata_dir / "vggsound.csv"

        if csv_path.exists():
            logger.info(f"✓ VGG-Sound CSV already exists")
            return csv_path

        try:
            response = requests.get(self.VGGSOUND_CSV_URL, timeout=60)
            response.raise_for_status()

            with open(csv_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"✓ Downloaded VGG-Sound CSV")
            return csv_path

        except Exception as e:
            logger.error(f"✗ Failed to download VGG-Sound CSV: {e}")
            logger.info("You may need to download manually from:")
            logger.info("https://www.robots.ox.ac.uk/~vgg/data/vggsound/")
            return None

    def extract_video_info_from_metadata(self) -> Set[str]:
        """Extract video IDs from train_qa.json and val_qa.json."""
        logger.info("Extracting video information from metadata...")

        video_ids = set()

        for filename in ["train_qa.json", "val_qa.json"]:
            metadata_path = self.metadata_dir / filename

            if not metadata_path.exists():
                logger.warning(f"Metadata file not found: {filename}")
                continue

            try:
                with open(metadata_path) as f:
                    data = json.load(f)

                for item in data:
                    # Extract video ID from various possible fields
                    video_id = item.get("video_id") or item.get("video") or item.get("id")

                    if video_id:
                        # VGG-Sound format: video IDs are typically like "---1_cCGK4M_000001"
                        video_ids.add(video_id)

                logger.info(f"  {filename}: {len(data)} QA pairs")

            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")

        logger.info(f"Total unique video IDs: {len(video_ids)}")

        # Save video ID list
        id_list_file = self.metadata_dir / "required_video_ids.json"
        with open(id_list_file, 'w') as f:
            json.dump({"video_ids": list(video_ids)}, f, indent=2)

        return video_ids

    def parse_vggsound_csv(self, required_ids: Set[str]) -> Dict[str, str]:
        """
        Parse VGG-Sound CSV to get YouTube URLs for required videos.

        Returns:
            Dict mapping video_id -> youtube_url
        """
        logger.info("Parsing VGG-Sound CSV for YouTube URLs...")

        csv_path = self.metadata_dir / "vggsound.csv"
        if not csv_path.exists():
            logger.error("VGG-Sound CSV not found!")
            return {}

        video_url_map = {}

        try:
            with open(csv_path, 'r') as f:
                # VGG-Sound CSV format: youtube_id, start_time, label, split
                reader = csv.reader(f)

                for row in reader:
                    if len(row) < 4:
                        continue

                    youtube_id, start_time, label, split = row[0], row[1], row[2], row[3]

                    # VGG-Sound video ID format: {youtube_id}_{start_time:06d}
                    # Timestamps are zero-padded to 6 digits (e.g., 000001, 000030)
                    video_id = f"{youtube_id}_{int(start_time):06d}"

                    if video_id in required_ids:
                        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
                        video_url_map[video_id] = youtube_url

            logger.info(f"Found YouTube URLs for {len(video_url_map)} / {len(required_ids)} videos")

            # Save mapping
            mapping_file = self.metadata_dir / "video_url_mapping.json"
            with open(mapping_file, 'w') as f:
                json.dump(video_url_map, f, indent=2)

        except Exception as e:
            logger.error(f"Error parsing VGG-Sound CSV: {e}")

        return video_url_map

    def download_video(self, video_id: str, youtube_url: str, retry_count: int = 0) -> bool:
        """
        Download a single video from YouTube with throttling.

        Args:
            video_id: VGG-Sound video ID (e.g., "---1_cCGK4M_000001")
            youtube_url: YouTube URL
            retry_count: Current retry attempt

        Returns:
            True if successful, False otherwise
        """
        # Check if already completed
        if video_id in self.completed_videos:
            return True

        # Check if permanently failed
        if self.failed_videos.get(video_id, 0) >= self.retry_attempts:
            return False

        video_path = self.videos_dir / f"{video_id}.{self.video_format}"
        audio_path = self.audio_dir / f"{video_id}.{self.audio_format}"

        # Skip if both files exist
        if video_path.exists() and audio_path.exists():
            self.completed_videos.add(video_id)
            self._save_progress()
            return True

        # Add random jitter to sleep time
        sleep_time = self.sleep_between_downloads + random.uniform(0, 2.0)
        time.sleep(sleep_time)

        try:
            # Download video with yt-dlp
            cmd = [
                "yt-dlp",
                "--quiet",
                "--no-warnings",
                f"--rate-limit={self.rate_limit}",
                f"--format=best[height<={self.quality}]",
                f"--output={video_path}",
                "--no-playlist",
                "--no-overwrites",
                youtube_url
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                raise RuntimeError(f"yt-dlp failed: {result.stderr}")

            # Extract audio
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

            # Verify
            if video_path.exists():
                self.completed_videos.add(video_id)
                self._save_progress()

                if video_id in self.failed_videos:
                    del self.failed_videos[video_id]
                    self._save_failed()

                logger.info(f"✓ Downloaded: {video_id}")
                return True
            else:
                raise RuntimeError("Video file not created")

        except Exception as e:
            logger.warning(f"✗ Failed {video_id} (attempt {retry_count + 1}): {e}")

            self.failed_videos[video_id] = self.failed_videos.get(video_id, 0) + 1
            self._save_failed()

            if retry_count < self.retry_attempts - 1:
                backoff_time = (2 ** retry_count) * 5
                logger.info(f"Retrying in {backoff_time}s...")
                time.sleep(backoff_time)
                return self.download_video(video_id, youtube_url, retry_count + 1)

            return False

    def download_videos_parallel(self, video_url_map: Dict[str, str]):
        """Download videos in parallel with throttling."""
        logger.info("="*60)
        logger.info(f"DOWNLOADING {len(video_url_map)} VIDEOS")
        logger.info("="*60)

        remaining_items = [
            (vid, url) for vid, url in video_url_map.items()
            if vid not in self.completed_videos
        ]

        # Limit to max_videos if specified
        if self.max_videos is not None:
            total_to_download = self.max_videos - len(self.completed_videos)
            if total_to_download > 0:
                remaining_items = remaining_items[:total_to_download]
                logger.info(f"Limiting download to {self.max_videos} total videos")
            else:
                logger.info(f"Already reached max_videos limit ({self.max_videos})")
                return

        logger.info(f"Already completed: {len(self.completed_videos)}")
        logger.info(f"Remaining: {len(remaining_items)}")

        if not remaining_items:
            logger.info("All videos already downloaded!")
            return

        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self.download_video, video_id, url): video_id
                for video_id, url in remaining_items
            }

            with tqdm(total=len(remaining_items), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    video_id = futures[future]
                    try:
                        if future.result():
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"Unexpected error for {video_id}: {e}")
                        failed += 1

                    pbar.update(1)
                    pbar.set_postfix({"success": successful, "failed": failed})

        logger.info("="*60)
        logger.info(f"Successful: {successful}, Failed: {failed}")
        logger.info(f"Total completed: {len(self.completed_videos)}")

    def verify_downloads(self):
        """Verify downloaded files."""
        logger.info("="*60)
        logger.info("VERIFYING DOWNLOADS")
        logger.info("="*60)

        video_files = list(self.videos_dir.glob(f"*.{self.video_format}"))
        audio_files = list(self.audio_dir.glob(f"*.{self.audio_format}"))

        logger.info(f"Video files: {len(video_files)}")
        logger.info(f"Audio files: {len(audio_files)}")

    def run(self):
        """Run complete download pipeline."""
        logger.info("="*60)
        logger.info("GENERAL AVQA DATASET DOWNLOADER")
        logger.info("="*60)
        logger.info(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Check dependencies
        if not self.check_yt_dlp():
            return False

        # Step 1: Download metadata (train_qa.json, val_qa.json)
        if not self.download_metadata():
            logger.error("Failed to download metadata")
            return False

        # Step 2: Extract required video IDs from metadata
        required_video_ids = self.extract_video_info_from_metadata()
        if not required_video_ids:
            logger.error("No video IDs found in metadata")
            return False

        # Step 3: Download VGG-Sound CSV
        csv_path = self.download_vggsound_csv()
        if not csv_path:
            logger.error("Failed to download VGG-Sound CSV")
            logger.info("\nManual steps:")
            logger.info("1. Download vggsound.csv from: https://www.robots.ox.ac.uk/~vgg/data/vggsound/")
            logger.info(f"2. Place it in: {self.metadata_dir}")
            logger.info("3. Run this script again")
            return False

        # Step 4: Parse CSV to get YouTube URLs
        video_url_map = self.parse_vggsound_csv(required_video_ids)
        if not video_url_map:
            logger.error("No video URLs found")
            return False

        # Step 5: Download videos
        self.download_videos_parallel(video_url_map)

        # Step 6: Verify
        self.verify_downloads()

        logger.info("="*60)
        logger.info(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Download General AVQA dataset with YouTube throttling"
    )
    parser.add_argument("--data-dir", type=str, default="./data/avqa",
                        help="Root directory for AVQA data")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--rate-limit", type=str, default="50K",
                        help="YouTube rate limit (default: 50K)")
    parser.add_argument("--retry-attempts", type=int, default=5,
                        help="Retry attempts (default: 5)")
    parser.add_argument("--sleep", type=float, default=3.0,
                        help="Sleep between downloads (default: 3.0)")
    parser.add_argument("--quality", type=str, default="360",
                        choices=["360", "480", "720"],
                        help="Video quality (default: 360)")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Maximum number of videos to download (default: all)")

    args = parser.parse_args()

    downloader = GeneralAVQADownloader(
        data_dir=args.data_dir,
        workers=args.workers,
        rate_limit=args.rate_limit,
        retry_attempts=args.retry_attempts,
        sleep_between_downloads=args.sleep,
        quality=args.quality,
        max_videos=args.max_videos
    )

    try:
        success = downloader.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nInterrupted. Progress saved.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
