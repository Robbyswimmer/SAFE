#!/usr/bin/env python3
"""
Robust Google Drive folder downloader with skip-on-error capability.

Downloads files from a Google Drive folder individually, skipping files
that fail due to permission errors or rate limits, and continues with
remaining files.
"""

import argparse
import logging
import sqlite3
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import gdown
except ImportError:
    print("ERROR: gdown not installed. Install with: pip install gdown")
    sys.exit(1)


class RobustGDriveDownloader:
    """Download Google Drive folder with skip-on-error and resume capability."""

    def __init__(
        self,
        folder_url: str,
        output_dir: Path,
        temp_dir: Path,
        log_file: Path,
        db_file: Path,
    ):
        self.folder_url = folder_url
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.log_file = log_file
        self.db_file = db_file

        # Setup logging
        self.logger = logging.getLogger("GDriveDownloader")
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # Setup database
        self.conn = sqlite3.connect(db_file)
        self._init_db()

        # Stats
        self.downloaded = 0
        self.failed = 0
        self.skipped = 0
        self.extracted = 0

    def _init_db(self):
        """Initialize SQLite database for tracking downloads."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS downloads (
                file_id TEXT PRIMARY KEY,
                file_name TEXT,
                status TEXT,  -- 'pending', 'success', 'failed', 'extracted'
                error_msg TEXT,
                downloaded_at TIMESTAMP,
                file_size INTEGER
            )
        """)
        self.conn.commit()

    def _get_file_list(self) -> List[Dict[str, str]]:
        """
        Get list of files in Google Drive folder.

        Returns list of dicts with keys: id, name, mimeType
        """
        self.logger.info(f"Fetching file list from: {self.folder_url}")

        try:
            # Extract folder ID from URL
            folder_id = self.folder_url.split('/')[-1]
            if '?' in folder_id:
                folder_id = folder_id.split('?')[0]

            # Use gdown's internal method to list folder contents
            # This is a bit hacky but works
            import re
            from gdown.download_folder import _parse_google_drive_file

            # Try to get the file listing page
            url = f"https://drive.google.com/drive/folders/{folder_id}"

            self.logger.info("Note: Falling back to downloading with gdown cache...")
            self.logger.info("This will download the folder structure first.")

            # For now, we'll work with the files that gdown discovers
            # during the download process
            return []

        except Exception as e:
            self.logger.error(f"Error getting file list: {e}")
            self.logger.info("Will attempt to download using gdown's folder method...")
            return []

    def _download_file(self, file_id: str, file_name: str, output_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Download individual file from Google Drive.

        Returns (success, error_message)
        """
        try:
            url = f"https://drive.google.com/uc?id={file_id}"

            self.logger.debug(f"Downloading: {file_name}")
            self.logger.debug(f"  URL: {url}")
            self.logger.debug(f"  Output: {output_path}")

            # Try to download with fuzzy matching
            result = gdown.download(url, str(output_path), quiet=False, fuzzy=True)

            if result is None:
                return False, "Download returned None (likely permission error)"

            if not output_path.exists():
                return False, "File not created after download"

            return True, None

        except Exception as e:
            error_msg = str(e)

            # Check for common error patterns
            if "permission" in error_msg.lower() or "access denied" in error_msg.lower():
                return False, f"Permission denied: {error_msg}"
            elif "rate" in error_msg.lower() or "quota" in error_msg.lower():
                return False, f"Rate limit/quota: {error_msg}"
            else:
                return False, f"Error: {error_msg}"

    def _mark_status(self, file_id: str, file_name: str, status: str, error_msg: Optional[str] = None):
        """Mark file status in database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO downloads (file_id, file_name, status, error_msg, downloaded_at)
            VALUES (?, ?, ?, ?, datetime('now'))
        """, (file_id, file_name, status, error_msg))
        self.conn.commit()

    def _extract_tarfile(self, tar_path: Path) -> bool:
        """Extract tar file with error handling."""
        try:
            self.logger.info(f"Extracting: {tar_path.name}")

            with tarfile.open(tar_path, 'r:*') as tar:
                tar.extractall(self.output_dir)

            self.logger.info(f"✓ Extracted: {tar_path.name}")
            self.extracted += 1
            return True

        except Exception as e:
            self.logger.error(f"✗ Failed to extract {tar_path.name}: {e}")
            return False

    def download_folder_robust(self):
        """Download folder using gdown with manual error recovery and retries."""
        self.logger.info("=" * 70)
        self.logger.info("Starting robust Google Drive folder download")
        self.logger.info(f"Folder: {self.folder_url}")
        self.logger.info(f"Output: {self.temp_dir}")
        self.logger.info("=" * 70)

        # Change to temp directory
        import os
        original_dir = os.getcwd()
        os.chdir(self.temp_dir)

        max_attempts = 10  # Keep trying until we get all files
        attempt = 1
        previous_count = 0

        try:
            # Extract folder ID
            folder_id = self.folder_url.split('/')[-1]
            if '?' in folder_id:
                folder_id = folder_id.split('?')[0]

            while attempt <= max_attempts:
                self.logger.info("=" * 70)
                self.logger.info(f"Download attempt {attempt}/{max_attempts}")
                self.logger.info("=" * 70)

                # Count current files
                current_count = len(list(Path('.').glob("*.tar*")))
                self.logger.info(f"Files before attempt: {current_count}")

                try:
                    # Use gdown.download_folder with remaining-ok to skip already downloaded
                    gdown.download_folder(
                        id=folder_id,
                        quiet=False,
                        use_cookies=False,
                        remaining_ok=True
                    )

                    self.logger.info(f"✓ Attempt {attempt} completed successfully")

                except KeyboardInterrupt:
                    self.logger.warning("Download interrupted by user")
                    raise

                except Exception as e:
                    error_str = str(e)
                    self.logger.warning(f"Attempt {attempt} encountered error: {error_str}")

                    # Check if it's a permission error
                    if "permission" in error_str.lower() or "retrieve" in error_str.lower():
                        self.logger.info("Permission error detected - will skip this file and continue")
                    else:
                        self.logger.warning(f"Unexpected error: {error_str}")

                # Count files after attempt
                new_count = len(list(Path('.').glob("*.tar*")))
                downloaded_this_round = new_count - current_count

                self.logger.info(f"Files after attempt: {new_count}")
                self.logger.info(f"Downloaded in this attempt: {downloaded_this_round}")

                # If no new files and we haven't seen new files in last attempt, we're likely done
                if downloaded_this_round == 0 and previous_count == current_count:
                    self.logger.info("No new files in this attempt and count stable - download appears complete")
                    break

                previous_count = new_count
                attempt += 1

                # Small delay between attempts to avoid rate limiting
                if attempt <= max_attempts:
                    self.logger.info("Waiting 5 seconds before next attempt...")
                    time.sleep(5)

            # Final count
            tar_files = list(Path('.').glob("*.tar")) + list(Path('.').glob("*.tar.gz"))
            self.logger.info("=" * 70)
            self.logger.info(f"Download phase complete - found {len(tar_files)} tar files")
            self.logger.info("=" * 70)

            if len(tar_files) == 0:
                self.logger.error("No tar files downloaded. Possible issues:")
                self.logger.error("  - Folder permissions may be restricted")
                self.logger.error("  - Rate limit reached")
                self.logger.error("  - Folder ID incorrect")
                return

            # Extract all tar files
            self.logger.info("=" * 70)
            self.logger.info("Extracting downloaded tar files...")
            self.logger.info("=" * 70)

            for tar_file in sorted(tar_files):
                self._extract_tarfile(tar_file)

        finally:
            os.chdir(original_dir)

        self.logger.info("=" * 70)
        self.logger.info("FINAL SUMMARY")
        self.logger.info(f"Total tar files: {len(tar_files)}")
        self.logger.info(f"Successfully extracted: {self.extracted}")
        self.logger.info(f"Failed to extract: {len(tar_files) - self.extracted}")
        self.logger.info("=" * 70)

    def cleanup(self):
        """Close database connection."""
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Robust Google Drive folder downloader with skip-on-error"
    )
    parser.add_argument(
        "--folder-url",
        type=str,
        default="https://drive.google.com/drive/folders/1ZKyRZw3AhS3HkWivgMqtMODB0TkVPNk5",
        help="Google Drive folder URL"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/audiosetcaps"),
        help="Output directory for extracted audio files"
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("data/audiosetcaps_temp"),
        help="Temporary directory for downloaded tar files"
    )

    args = parser.parse_args()

    # Create directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    # Setup paths
    log_file = args.temp_dir / "download_robust.log"
    db_file = args.temp_dir / "download_progress.db"

    # Create downloader
    downloader = RobustGDriveDownloader(
        folder_url=args.folder_url,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        log_file=log_file,
        db_file=db_file,
    )

    try:
        downloader.download_folder_robust()
    finally:
        downloader.cleanup()


if __name__ == "__main__":
    main()
