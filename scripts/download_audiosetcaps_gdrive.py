#!/usr/bin/env python3
"""
Download AudioSetCaps audio files from Google Drive.

The audio files are provided as .tar archives on Google Drive.
This script downloads and extracts them.
"""

import argparse
import subprocess
import tarfile
from pathlib import Path
import logging

# Google Drive folder: https://drive.google.com/drive/folders/1ZKyRZw3AhS3HkWivgMqtMODB0TkVPNk5
# The folder contains .tar files with pre-downloaded audio

# File IDs can be extracted from Google Drive links
# Format: https://drive.google.com/file/d/{FILE_ID}/view
GDRIVE_FILES = {
    # Add file IDs here as we identify them
    # Example: "audiocaps_part1.tar": "FILE_ID_HERE",
}

def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("AudioSetCapsGDrive")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def download_from_gdrive(file_id: str, output_path: Path, logger: logging.Logger) -> bool:
    """Download file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        logger.error("gdown not installed. Install with: pip install gdown")
        return False

    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info(f"Downloading {output_path.name} from Google Drive...")

    try:
        gdown.download(url, str(output_path), quiet=False)
        return True
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        return False


def extract_tar(tar_path: Path, extract_to: Path, logger: logging.Logger) -> bool:
    """Extract tar archive."""
    logger.info(f"Extracting {tar_path.name}...")

    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_to)
        logger.info(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download AudioSetCaps audio from Google Drive"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/audiosetcaps"),
        help="Output directory for audio files"
    )
    parser.add_argument(
        "--tar-dir",
        type=Path,
        default=Path("data/audiosetcaps_tars"),
        help="Directory to store downloaded tar files"
    )
    parser.add_argument(
        "--keep-tars",
        action="store_true",
        help="Keep tar files after extraction"
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tar_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(args.output_dir / ".download_gdrive.log")

    if not GDRIVE_FILES:
        logger.error(
            "No Google Drive file IDs configured. You need to:\n"
            "1. Go to https://drive.google.com/drive/folders/1ZKyRZw3AhS3HkWivgMqtMODB0TkVPNk5\n"
            "2. Identify the tar files and their IDs\n"
            "3. Add them to GDRIVE_FILES in this script\n\n"
            "Or manually download the tar files and run:\n"
            "  tar -xf audiosetcaps.tar -C data/audiosetcaps/"
        )
        return

    for filename, file_id in GDRIVE_FILES.items():
        tar_path = args.tar_dir / filename

        # Download if not already present
        if not tar_path.exists():
            success = download_from_gdrive(file_id, tar_path, logger)
            if not success:
                continue
        else:
            logger.info(f"Using existing tar: {tar_path}")

        # Extract
        success = extract_tar(tar_path, args.output_dir, logger)
        if success and not args.keep_tars:
            logger.info(f"Removing tar file: {tar_path}")
            tar_path.unlink()

    logger.info("✓ Download complete!")


if __name__ == "__main__":
    main()
