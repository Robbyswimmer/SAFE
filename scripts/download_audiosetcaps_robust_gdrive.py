#!/usr/bin/env python3
"""
Robust Google Drive folder downloader with skip-on-error capability.

Downloads files from a Google Drive folder individually, skipping files
that fail due to permission errors or rate limits, and continues with
remaining files.
"""

import argparse
import logging
import os
import re
import sqlite3
import subprocess
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

_GDOWN_FOLDER_IMPORT_ERROR: Optional[str] = None
try:
    import gdown.download_folder as _gdown_download_folder  # type: ignore
except Exception as exc:  # pragma: no cover - best-effort import for runtime env
    _get_files_by_folder_id = None
    _GDOWN_FOLDER_IMPORT_ERROR = str(exc)
else:
    _get_files_by_folder_id = getattr(_gdown_download_folder, "_get_files_by_folder_id", None)
    if _get_files_by_folder_id is None:
        _GDOWN_FOLDER_IMPORT_ERROR = (
            "gdown.download_folder does not expose _get_files_by_folder_id. "
            f"Available names: {', '.join(dir(_gdown_download_folder))}"
        )


class RobustGDriveDownloader:
    """Download Google Drive folder with skip-on-error and resume capability."""

    def __init__(
        self,
        folder_url: str,
        output_dir: Path,
        temp_dir: Path,
        log_file: Path,
        db_file: Path,
        skip_ids: Optional[List[str]] = None,
    ):
        self.folder_url = folder_url
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.log_file = log_file
        self.db_file = db_file
        self.skip_ids = set(skip_ids or [])

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
        self.max_retries = 3

    def _init_db(self):
        """Initialize SQLite database for tracking downloads."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS downloads (
                file_id TEXT PRIMARY KEY,
                file_name TEXT,
                rel_path TEXT,
                status TEXT,  -- 'pending', 'success', 'failed', 'extracted'
                error_msg TEXT,
                downloaded_at TIMESTAMP,
                file_size INTEGER
            )
        """)
        # Add missing columns when upgrading existing DBs.
        cursor.execute("PRAGMA table_info(downloads)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if "rel_path" not in existing_columns:
            cursor.execute("ALTER TABLE downloads ADD COLUMN rel_path TEXT")
        if "file_size" not in existing_columns:
            cursor.execute("ALTER TABLE downloads ADD COLUMN file_size INTEGER")
        self.conn.commit()

    def _get_file_list(self) -> List[Dict[str, str]]:
        """Fetch manifest of Drive files if possible, else fall back to CLI probe."""
        if _get_files_by_folder_id is None:
            if _GDOWN_FOLDER_IMPORT_ERROR:
                self.logger.info(
                    "gdown manifest helper unavailable (%s) - probing via CLI",
                    _GDOWN_FOLDER_IMPORT_ERROR,
                )
            else:
                self.logger.info("gdown manifest helper unavailable - probing via CLI")
            return self._fetch_manifest_via_cli()

        self.logger.info(f"Fetching file list from: {self.folder_url}")

        folder_id = self.folder_url.split('/')[-1]
        if '?' in folder_id:
            folder_id = folder_id.split('?')[0]

        try:
            raw_entries = list(
                _get_files_by_folder_id(  # type: ignore[misc]
                    folder_id,
                    use_cookies=True,
                    remaining_ok=True,
                )
            )
        except Exception as exc:  # pragma: no cover - network-driven
            self.logger.warning(
                f"Unable to fetch manifest via gdown helper ({exc}) - probing via CLI"
            )
            return self._fetch_manifest_via_cli()

        manifest: List[Dict[str, str]] = []
        for entry in raw_entries:
            file_id = entry.get("id")
            if not file_id or file_id in self.skip_ids:
                continue

            mime = entry.get("mimeType") or entry.get("mime_type")
            entry_type = entry.get("type")
            if mime == "application/vnd.google-apps.folder" or entry_type == "folder":
                continue

            rel_path = entry.get("path") or entry.get("name") or entry.get("title")
            name = entry.get("name") or entry.get("title") or rel_path

            if not rel_path or not name:
                continue

            manifest.append(
                {
                    "id": file_id,
                    "name": name,
                    "rel_path": rel_path,
                    "size": entry.get("sizeBytes") or entry.get("size"),
                }
            )

        if not manifest:
            self.logger.warning("Manifest helper returned no files - probing via CLI fallback")
            return self._fetch_manifest_via_cli()

        self.logger.info(f"Discovered {len(manifest)} files via manifest helper")
        return manifest

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

    def _get_status(self, file_id: str) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT status FROM downloads WHERE file_id = ?", (file_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    def _mark_status(
        self,
        file_id: str,
        file_name: str,
        status: str,
        error_msg: Optional[str] = None,
        rel_path: Optional[str] = None,
        file_size: Optional[int] = None,
    ):
        """Mark file status in database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO downloads (file_id, file_name, rel_path, status, error_msg, downloaded_at, file_size)
            VALUES (?, ?, ?, ?, ?, datetime('now'), ?)
        """, (file_id, file_name, rel_path, status, error_msg, file_size))
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

    def _fetch_manifest_via_cli(self) -> List[Dict[str, str]]:
        """Use `gdown --folder` output to capture the manifest before download starts."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "gdown",
            "--folder",
            self.folder_url,
            "--remaining-ok",
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        self.logger.info("Collecting manifest via gdown CLI probe (will stop before downloads)...")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(self.temp_dir),
            env=env,
        )

        manifest: Dict[str, Dict[str, str]] = {}
        processing_re = re.compile(r"Processing file\\s+([^\\s]+)\\s+(.+)")
        finished_listing = False

        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue

                match = processing_re.match(line)
                if match:
                    file_id, name = match.groups()
                    if file_id in self.skip_ids:
                        continue
                    manifest[file_id] = {
                        "id": file_id,
                        "name": name,
                        "rel_path": name,
                    }
                    continue

                if "Building directory structure completed" in line:
                    finished_listing = True
                    break
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        if not finished_listing:
            self.logger.warning("Could not capture manifest via CLI probe")
            return []

        if proc.stdout is not None:
            proc.stdout.close()

        manifest_list = list(manifest.values())
        self.logger.info(f"Captured {len(manifest_list)} files via CLI manifest probe")
        return manifest_list

    def _download_with_manifest(self, manifest: List[Dict[str, str]]) -> bool:
        """Download files one-by-one using the manifest returned by gdown."""
        if not manifest:
            return False

        tar_entries = [m for m in manifest if m["rel_path"].endswith((".tar", ".tar.gz"))]
        if not tar_entries:
            self.logger.warning("Manifest does not contain tar files - falling back to legacy mode")
            return False

        skipped_permission = []
        total = len(tar_entries)
        self.logger.info(f"Starting manifest-driven download of {total} files")

        for idx, entry in enumerate(tar_entries, start=1):
            rel_path = Path(entry["rel_path"])  # type: ignore[arg-type]
            safe_rel = Path(*[part for part in rel_path.parts if part not in ("..", ".")])
            if not safe_rel.parts:
                safe_rel = Path(entry["name"])
            output_path = self.temp_dir / safe_rel
            output_path.parent.mkdir(parents=True, exist_ok=True)

            status = self._get_status(entry["id"])
            if status == "success" and output_path.exists():
                self.logger.info(f"[{idx}/{total}] Already have {safe_rel}, skipping")
                continue

            self.logger.info(f"[{idx}/{total}] Downloading {safe_rel}")

            attempt = 1
            success = False
            last_error: Optional[str] = None
            while attempt <= self.max_retries:
                success, error_msg = self._download_file(entry["id"], entry["name"], output_path)
                if success:
                    size = output_path.stat().st_size if output_path.exists() else None
                    self._mark_status(
                        entry["id"],
                        entry["name"],
                        "success",
                        rel_path=str(safe_rel),
                        file_size=size,
                    )
                    self.downloaded += 1
                    break

                last_error = error_msg or "unknown error"
                if last_error and "permission" in last_error.lower():
                    self.logger.warning(f"Permission denied for {entry['name']} ({entry['id']}), skipping")
                    skipped_permission.append(entry)
                    self._mark_status(
                        entry["id"],
                        entry["name"],
                        "permission_denied",
                        error_msg=last_error,
                        rel_path=str(safe_rel),
                    )
                    self.skipped += 1
                    break

                self.logger.warning(
                    f"Failed to download {entry['name']} (attempt {attempt}/{self.max_retries}): {last_error}"
                )
                attempt += 1
                if attempt <= self.max_retries:
                    sleep_time = min(5 * attempt, 30)
                    self.logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)

            if not success and (not last_error or "permission" not in (last_error or "").lower()):
                self.failed += 1
                self._mark_status(
                    entry["id"],
                    entry["name"],
                    "failed",
                    error_msg=last_error,
                    rel_path=str(safe_rel),
                )

        if skipped_permission:
            self.logger.warning("The following files were skipped due to permission errors:")
            for entry in skipped_permission:
                self.logger.warning(f"  - {entry['name']} ({entry['id']})")

        return True

    def _legacy_gdown_download(self, folder_id: str):
        """Fallback to gdown's folder downloader when manifest is unavailable."""
        max_attempts = 10
        attempt = 1
        previous_count = 0

        while attempt <= max_attempts:
            self.logger.info("=" * 70)
            self.logger.info(f"Download attempt {attempt}/{max_attempts}")
            self.logger.info("=" * 70)

            current_count = len(list(Path('.').glob("**/*.tar*")))
            self.logger.info(f"Files before attempt: {current_count}")

            try:
                gdown.download_folder(
                    id=folder_id,
                    quiet=False,
                    use_cookies=False,
                    remaining_ok=True,
                )
                self.logger.info(f"✓ Attempt {attempt} completed successfully")
            except KeyboardInterrupt:
                self.logger.warning("Download interrupted by user")
                raise
            except Exception as e:
                error_str = str(e)
                self.logger.warning(f"Attempt {attempt} encountered error: {error_str}")
                if "permission" in error_str.lower() or "retrieve" in error_str.lower():
                    self.logger.info("Permission error detected - will skip this file and continue")
                else:
                    self.logger.warning(f"Unexpected error: {error_str}")

            new_count = len(list(Path('.').glob("**/*.tar*")))
            downloaded_this_round = new_count - current_count

            self.logger.info(f"Files after attempt: {new_count}")
            self.logger.info(f"Downloaded in this attempt: {downloaded_this_round}")

            if downloaded_this_round == 0 and previous_count == current_count:
                self.logger.info("No new files in this attempt and count stable - download appears complete")
                break

            previous_count = new_count
            attempt += 1

            if attempt <= max_attempts:
                self.logger.info("Waiting 5 seconds before next attempt...")
                time.sleep(5)

    def download_folder_robust(self):
        """Download folder using manifest when possible, fallback to gdown loop."""
        self.logger.info("=" * 70)
        self.logger.info("Starting robust Google Drive folder download")
        self.logger.info(f"Folder: {self.folder_url}")
        self.logger.info(f"Output: {self.temp_dir}")
        self.logger.info("=" * 70)

        import os
        original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        tar_files: List[Path] = []

        try:
            folder_id = self.folder_url.split('/')[-1]
            if '?' in folder_id:
                folder_id = folder_id.split('?')[0]

            manifest = self._get_file_list()
            manifest_used = self._download_with_manifest(manifest) if manifest else False

            if not manifest_used:
                self.logger.info("Manifest unavailable or download failed - falling back to legacy gdown mode")
                self._legacy_gdown_download(folder_id)

            tar_files = list(Path('.').glob("**/*.tar")) + list(Path('.').glob("**/*.tar.gz"))
            self.logger.info("=" * 70)
            self.logger.info(f"Download phase complete - found {len(tar_files)} tar files")
            self.logger.info("=" * 70)

            if len(tar_files) == 0:
                self.logger.error("No tar files downloaded. Possible issues:")
                self.logger.error("  - Folder permissions may be restricted")
                self.logger.error("  - Rate limit reached")
                self.logger.error("  - Folder ID incorrect")
                return

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
    parser.add_argument(
        "--skip-file-id",
        action="append",
        default=[],
        help="Google Drive file IDs to skip (can be provided multiple times)",
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
        skip_ids=args.skip_file_id,
    )

    try:
        downloader.download_folder_robust()
    finally:
        downloader.cleanup()


if __name__ == "__main__":
    main()
