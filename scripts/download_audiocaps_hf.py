#!/usr/bin/env python3
"""Download the jp1924/AudioCaps dataset (all parquet shards) from Hugging Face.

The script enumerates parquet files in the dataset repo and downloads them to a
local directory, skipping any files that already exist unless --force is set.

Features:
- Automatic resume: skips completed files
- Partial download detection: re-downloads corrupted/incomplete files
- Retry logic: retries failed downloads with exponential backoff
- Progress tracking: saves state to allow resuming after script restart
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, hf_hub_download


def list_parquet_files(api: HfApi, repo_id: str) -> List[str]:
    files = [
        f
        for f in api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        if f.endswith(".parquet")
    ]
    if not files:
        raise RuntimeError(f"No parquet files found in repo {repo_id}")
    return files


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_remote_file_info(api: HfApi, repo_id: str, filename: str, token: Optional[str]) -> dict:
    """Get remote file metadata (size, etag) for verification."""
    try:
        # Get file info from HF Hub
        file_info = api.hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            token=token,
        )
        # Note: hf_hub_download returns local path, we need to get metadata differently
        # For now, just check if we can access the file
        return {"accessible": True}
    except Exception:
        return {"accessible": False}


def load_progress_state(state_file: Path) -> dict:
    """Load download progress state."""
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load progress state: {e}")
    return {"completed_files": [], "failed_files": {}}


def save_progress_state(state_file: Path, state: dict) -> None:
    """Save download progress state."""
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save progress state: {e}")


def download_with_retry(
    repo_id: str,
    filename: str,
    token: str,
    local_dir: Path,
    max_retries: int = 5,
    initial_delay: float = 5.0,
) -> bool:
    """Download a file with exponential backoff retry logic."""
    delay = initial_delay

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [Attempt {attempt}/{max_retries}] Downloading...", flush=True)
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                token=token,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,  # Enable resume for partial downloads
            )
            return True
        except KeyboardInterrupt:
            print("\n⚠️  Download interrupted by user")
            raise
        except Exception as e:
            if attempt < max_retries:
                print(f"  ❌ Download failed: {e}")
                print(f"  ⏳ Retrying in {delay:.1f}s...", flush=True)
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"  ❌ Download failed after {max_retries} attempts: {e}")
                return False

    return False


def verify_parquet_file(file_path: Path) -> bool:
    """Basic verification that a parquet file is valid."""
    if not file_path.exists():
        return False

    # Check file size (parquet files should be > 1KB)
    if file_path.stat().st_size < 1024:
        print(f"  ⚠️  File too small ({file_path.stat().st_size} bytes), likely incomplete")
        return False

    # Try to read parquet header (first 4 bytes should be "PAR1")
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
            if header != b"PAR1":
                print(f"  ⚠️  Invalid parquet header, file may be corrupted")
                return False
    except Exception as e:
        print(f"  ⚠️  Could not read file: {e}")
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AudioCaps parquet shards with robust resume capability"
    )
    parser.add_argument(
        "--dataset",
        default="jp1924/AudioCaps",
        help="Hugging Face dataset repo id (default: jp1924/AudioCaps)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/full_training/data/audiocaps_hf"),
        help="Directory to store downloaded shards",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts per file (default: 5)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify parquet file integrity (checks header and size)",
    )
    args = parser.parse_args()

    if not args.token:
        raise SystemExit(
            "Missing Hugging Face token. Run `huggingface-cli login` or provide --token."
        )

    api = HfApi(token=args.token)
    files = list_parquet_files(api, args.dataset)

    # Load progress state
    state_file = args.output_dir / ".download_progress.json"
    state = load_progress_state(state_file)
    completed_files = set(state.get("completed_files", []))
    failed_files = state.get("failed_files", {})

    print(f"📦 Found {len(files)} parquet files in {args.dataset}")
    print(f"✅ Already completed: {len(completed_files)} files")
    print(f"❌ Previously failed: {len(failed_files)} files")
    print(f"📥 Remaining to download: {len(files) - len(completed_files)} files\n")

    successful = 0
    skipped = 0
    failed = 0

    for idx, remote_path in enumerate(files, 1):
        local_path = args.output_dir / remote_path

        # Check if already completed successfully
        if remote_path in completed_files and not args.force:
            # Verify the file still exists and is valid
            if args.verify:
                if verify_parquet_file(local_path):
                    print(f"[{idx}/{len(files)}] ✓ Verified: {remote_path}")
                    skipped += 1
                    continue
                else:
                    print(f"[{idx}/{len(files)}] ⚠️  Verification failed, re-downloading: {remote_path}")
                    completed_files.discard(remote_path)
            else:
                if local_path.exists():
                    print(f"[{idx}/{len(files)}] ⏭️  Skipping: {remote_path}")
                    skipped += 1
                    continue
                else:
                    print(f"[{idx}/{len(files)}] ⚠️  File missing, re-downloading: {remote_path}")
                    completed_files.discard(remote_path)

        # Check for existing file (without --force)
        if local_path.exists() and not args.force:
            # Verify integrity
            if args.verify and verify_parquet_file(local_path):
                print(f"[{idx}/{len(files)}] ✓ Exists and verified: {remote_path}")
                completed_files.add(remote_path)
                skipped += 1
                save_progress_state(state_file, {
                    "completed_files": list(completed_files),
                    "failed_files": failed_files,
                })
                continue
            elif not args.verify:
                print(f"[{idx}/{len(files)}] ⏭️  File exists: {remote_path}")
                completed_files.add(remote_path)
                skipped += 1
                save_progress_state(state_file, {
                    "completed_files": list(completed_files),
                    "failed_files": failed_files,
                })
                continue
            else:
                print(f"[{idx}/{len(files)}] ⚠️  File exists but verification failed: {remote_path}")

        # Create parent directory
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with retry
        print(f"[{idx}/{len(files)}] 📥 Downloading: {remote_path}")
        size_str = f"{local_path.stat().st_size / (1024**3):.2f} GB" if local_path.exists() else "unknown size"
        print(f"  File: {local_path.name} ({size_str})")

        success = download_with_retry(
            repo_id=args.dataset,
            filename=remote_path,
            token=args.token,
            local_dir=args.output_dir,
            max_retries=args.max_retries,
        )

        if success:
            # Verify downloaded file
            if args.verify:
                if verify_parquet_file(local_path):
                    print(f"  ✅ Download successful and verified")
                    completed_files.add(remote_path)
                    failed_files.pop(remote_path, None)
                    successful += 1
                else:
                    print(f"  ❌ Download completed but verification failed")
                    failed_files[remote_path] = "Verification failed"
                    failed += 1
            else:
                print(f"  ✅ Download successful")
                completed_files.add(remote_path)
                failed_files.pop(remote_path, None)
                successful += 1
        else:
            print(f"  ❌ Download failed")
            failed_files[remote_path] = "Max retries exceeded"
            failed += 1

        # Save progress after each file
        save_progress_state(state_file, {
            "completed_files": list(completed_files),
            "failed_files": failed_files,
        })

    print("\n" + "=" * 60)
    print("📊 Download Summary:")
    print(f"  ✅ Successful: {successful} files")
    print(f"  ⏭️  Skipped: {skipped} files")
    print(f"  ❌ Failed: {failed} files")
    print(f"  📁 Total: {len(files)} files")
    print("=" * 60)

    if failed > 0:
        print("\n⚠️  Some files failed to download. You can re-run this script to retry.")
        print(f"Failed files are tracked in: {state_file}")
        raise SystemExit(1)
    else:
        print("\n🎉 All files downloaded successfully!")
        # Optionally clean up progress file on complete success
        # state_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
