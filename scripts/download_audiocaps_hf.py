#!/usr/bin/env python3
"""Download the jp1924/AudioCaps dataset (all parquet shards) from Hugging Face.

The script enumerates parquet files in the dataset repo and downloads them to a
local directory, skipping any files that already exist unless --force is set.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AudioCaps parquet shards")
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
    args = parser.parse_args()

    if not args.token:
        raise SystemExit(
            "Missing Hugging Face token. Run `huggingface-cli login` or provide --token."
        )

    api = HfApi(token=args.token)
    files = list_parquet_files(api, args.dataset)

    print(f"Found {len(files)} parquet files in {args.dataset}")

    for idx, remote_path in enumerate(files, 1):
        local_path = args.output_dir / remote_path
        if local_path.exists() and not args.force:
            print(f"[{idx}/{len(files)}] Skipping existing file: {remote_path}")
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(files)}] Downloading {remote_path}")
        hf_hub_download(
            repo_id=args.dataset,
            repo_type="dataset",
            filename=remote_path,
            token=args.token,
            local_dir=str(args.output_dir),
            local_dir_use_symlinks=False,
        )

    print("All requested files downloaded.")


if __name__ == "__main__":
    main()
