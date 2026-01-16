#!/usr/bin/env python3
"""
Download AudioCaps audio files from YouTube with IP rotation and cluster file skipping.

Features:
- IP/proxy rotation for avoiding rate limits
- Pre-scan cluster directories to skip already downloaded files
- Multiple client strategies (Android, iOS, web)
- Secure proxy handling via environment variables

Requires: pip install yt-dlp requests

Environment Variables for Proxies (optional):
- YTDLP_PROXY_LIST: Path to file containing proxy URLs (one per line)
- YTDLP_PROXY: Single proxy URL (fallback if no list)
- HTTP_PROXY / HTTPS_PROXY: Standard proxy environment variables

Proxy file format (one per line):
    socks5://user:pass@host:port
    http://host:port
    socks5h://host:port
"""

import os
import sys
import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm
import time
import random
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Set, Tuple
from dataclasses import dataclass, field

@dataclass
class ProxyRotator:
    """
    Thread-safe proxy rotation for distributed downloading.

    Loads proxies from:
    1. YTDLP_PROXY_LIST env var (path to file with proxy list)
    2. YTDLP_PROXY env var (single proxy)
    3. Standard HTTP_PROXY/HTTPS_PROXY env vars

    Security: Proxies are loaded from env vars or files, never hardcoded.
    """
    proxies: List[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _index: int = field(default=0, repr=False)
    _failed_proxies: Set[str] = field(default_factory=set, repr=False)
    _proxy_failures: dict = field(default_factory=dict, repr=False)
    max_failures: int = 5  # Max failures before temporarily removing proxy

    def __post_init__(self):
        """Load proxies from environment on initialization."""
        if not self.proxies:
            self._load_from_environment()

    def _load_from_environment(self):
        """Load proxy configuration from environment variables securely."""
        # Priority 1: Proxy list file
        proxy_list_path = os.environ.get('YTDLP_PROXY_LIST')
        if proxy_list_path and os.path.exists(proxy_list_path):
            try:
                with open(proxy_list_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if line and not line.startswith('#'):
                            # Validate proxy format
                            if self._validate_proxy_format(line):
                                self.proxies.append(line)
                if self.proxies:
                    print(f"Loaded {len(self.proxies)} proxies from {proxy_list_path}")
                    return
            except (IOError, OSError) as e:
                print(f"Warning: Could not read proxy list file: {e}")

        # Priority 2: Single proxy from env
        single_proxy = os.environ.get('YTDLP_PROXY')
        if single_proxy and self._validate_proxy_format(single_proxy):
            self.proxies.append(single_proxy)
            print(f"Using single proxy from YTDLP_PROXY")
            return

        # Priority 3: Standard HTTP proxy env vars
        for env_var in ['HTTPS_PROXY', 'HTTP_PROXY', 'https_proxy', 'http_proxy']:
            proxy = os.environ.get(env_var)
            if proxy and self._validate_proxy_format(proxy):
                self.proxies.append(proxy)
                print(f"Using proxy from {env_var}")
                return

    def _validate_proxy_format(self, proxy: str) -> bool:
        """Validate proxy URL format for security."""
        valid_schemes = ('http://', 'https://', 'socks4://', 'socks5://', 'socks5h://')
        if not any(proxy.startswith(scheme) for scheme in valid_schemes):
            print(f"Warning: Invalid proxy format (must start with {valid_schemes}): {proxy[:20]}...")
            return False
        # Basic validation - must have host:port pattern
        try:
            # Remove scheme and auth for validation
            rest = proxy.split('://', 1)[1]
            if '@' in rest:
                rest = rest.split('@', 1)[1]
            # Should have host:port
            if ':' not in rest:
                print(f"Warning: Proxy missing port: {proxy[:30]}...")
                return False
            return True
        except (IndexError, ValueError):
            return False

    def get_next_proxy(self) -> Optional[str]:
        """Get next proxy in rotation (thread-safe)."""
        if not self.proxies:
            return None

        with self._lock:
            # Filter out failed proxies
            available = [p for p in self.proxies if p not in self._failed_proxies]
            if not available:
                # Reset failed proxies if all failed
                if self._failed_proxies:
                    print("All proxies failed, resetting failure counts...")
                    self._failed_proxies.clear()
                    self._proxy_failures.clear()
                    available = self.proxies
                else:
                    return None

            # Round-robin selection
            proxy = available[self._index % len(available)]
            self._index += 1
            return proxy

    def report_failure(self, proxy: str):
        """Report a proxy failure (thread-safe)."""
        if not proxy:
            return
        with self._lock:
            self._proxy_failures[proxy] = self._proxy_failures.get(proxy, 0) + 1
            if self._proxy_failures[proxy] >= self.max_failures:
                self._failed_proxies.add(proxy)
                print(f"Proxy temporarily disabled after {self.max_failures} failures: {proxy[:30]}...")

    def report_success(self, proxy: str):
        """Report a proxy success - reduce failure count (thread-safe)."""
        if not proxy:
            return
        with self._lock:
            if proxy in self._proxy_failures:
                self._proxy_failures[proxy] = max(0, self._proxy_failures[proxy] - 1)

    @property
    def has_proxies(self) -> bool:
        """Check if any proxies are configured."""
        return len(self.proxies) > 0

    def get_stats(self) -> dict:
        """Get proxy pool statistics."""
        with self._lock:
            return {
                'total': len(self.proxies),
                'active': len(self.proxies) - len(self._failed_proxies),
                'failed': len(self._failed_proxies),
            }


@dataclass
class ClusterFileScanner:
    """
    Efficiently scan cluster directories for existing files to skip re-downloads.

    Supports:
    - Multiple cluster paths (local, NFS, etc.)
    - Pre-scanning for O(1) lookup during downloads
    - HDF5 embedding files (checks for processed IDs)
    - Various audio formats (.wav, .mp3, .flac, etc.)
    """
    paths: List[Path] = field(default_factory=list)
    existing_ids: Set[str] = field(default_factory=set, repr=False)
    _scanned: bool = field(default=False, repr=False)

    def add_path(self, path: str | Path):
        """Add a cluster path to scan."""
        p = Path(path).expanduser().resolve()
        if p.exists():
            self.paths.append(p)
        else:
            print(f"Warning: Cluster path does not exist: {p}")

    def add_paths_from_env(self):
        """Load additional cluster paths from AUDIOCAPS_CLUSTER_PATHS env var."""
        paths_env = os.environ.get('AUDIOCAPS_CLUSTER_PATHS', '')
        if paths_env:
            for path in paths_env.split(':'):
                if path.strip():
                    self.add_path(path.strip())

    def scan(self, extensions: Tuple[str, ...] = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')) -> int:
        """
        Pre-scan all cluster paths for existing audio files.

        Returns: Number of existing files found
        """
        if self._scanned:
            return len(self.existing_ids)

        print(f"Scanning {len(self.paths)} cluster path(s) for existing files...")

        for cluster_path in self.paths:
            try:
                # Scan for audio files
                for ext in extensions:
                    for audio_file in cluster_path.rglob(f"*{ext}"):
                        # Extract YouTube ID from filename (handle various naming patterns)
                        youtube_id = self._extract_youtube_id(audio_file.stem)
                        if youtube_id:
                            self.existing_ids.add(youtube_id)

                # Also check for HDF5 embedding files
                for h5_file in cluster_path.rglob("*.h5"):
                    self._scan_h5_for_ids(h5_file)
                for h5_file in cluster_path.rglob("*.hdf5"):
                    self._scan_h5_for_ids(h5_file)

            except PermissionError as e:
                print(f"Warning: Permission denied scanning {cluster_path}: {e}")
            except Exception as e:
                print(f"Warning: Error scanning {cluster_path}: {e}")

        self._scanned = True
        print(f"Found {len(self.existing_ids)} existing files/embeddings across cluster paths")
        return len(self.existing_ids)

    def _extract_youtube_id(self, filename: str) -> Optional[str]:
        """Extract YouTube ID from various filename formats."""
        # YouTube IDs are 11 characters, alphanumeric with - and _
        import re

        # Common patterns:
        # - Direct ID: "dQw4w9WgXcQ.wav"
        # - With prefix: "audiocaps_dQw4w9WgXcQ.wav"
        # - With timestamp: "dQw4w9WgXcQ_30.wav"

        # Try exact 11-char match first
        if re.match(r'^[a-zA-Z0-9_-]{11}$', filename):
            return filename

        # Try to find 11-char YouTube ID pattern
        match = re.search(r'([a-zA-Z0-9_-]{11})', filename)
        if match:
            return match.group(1)

        return None

    def _scan_h5_for_ids(self, h5_path: Path):
        """Scan HDF5 file for already-processed YouTube IDs."""
        try:
            import h5py
            with h5py.File(h5_path, 'r') as f:
                # Check common key patterns for YouTube IDs
                for key in ['youtube_ids', 'ids', 'video_ids', 'sources']:
                    if key in f:
                        dataset = f[key]
                        if hasattr(dataset, '__iter__'):
                            for item in dataset:
                                if isinstance(item, bytes):
                                    item = item.decode('utf-8')
                                if isinstance(item, str):
                                    yt_id = self._extract_youtube_id(item)
                                    if yt_id:
                                        self.existing_ids.add(yt_id)
        except ImportError:
            pass  # h5py not installed, skip HDF5 scanning
        except Exception:
            pass  # Silently skip problematic H5 files

    def exists(self, youtube_id: str) -> bool:
        """Check if a YouTube ID already exists (O(1) after scan)."""
        if not self._scanned:
            self.scan()
        return youtube_id in self.existing_ids

    def filter_needed(self, youtube_ids: List[str]) -> List[str]:
        """Filter list to only IDs that need downloading."""
        if not self._scanned:
            self.scan()
        return [yt_id for yt_id in youtube_ids if yt_id not in self.existing_ids]

    def get_stats(self) -> dict:
        """Get scanner statistics."""
        return {
            'paths_scanned': len(self.paths),
            'existing_files': len(self.existing_ids),
        }


# Global instances (initialized in main)
_proxy_rotator: Optional[ProxyRotator] = None
_cluster_scanner: Optional[ClusterFileScanner] = None


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

def download_audio(youtube_id, output_path, start_time=0, max_retries=3, cookies_file=None,
                   adaptive_rate_limit=True, proxy_rotator: Optional[ProxyRotator] = None):
    """
    Download audio from YouTube using yt-dlp with advanced anti-bot detection and IP rotation.

    Args:
        youtube_id: YouTube video ID
        output_path: Directory to save audio file
        start_time: Start time in seconds for 10-second clip (default: 0)
        max_retries: Number of download attempts
        cookies_file: Path to cookies.txt file
        adaptive_rate_limit: Enable human-like timing and rate limiting
        proxy_rotator: ProxyRotator instance for IP rotation (optional)
    """
    global _proxy_rotator
    if proxy_rotator is None:
        proxy_rotator = _proxy_rotator

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
    current_proxy = None

    for attempt in range(max_retries):
        # Use different strategy for each attempt
        strategy = retry_strategies[min(attempt, len(retry_strategies) - 1)]
        cmd = base_cmd + strategy

        # Add proxy if available (rotate on each attempt)
        if proxy_rotator and proxy_rotator.has_proxies:
            current_proxy = proxy_rotator.get_next_proxy()
            if current_proxy:
                cmd.extend(['--proxy', current_proxy])

        cmd.append(url)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                # Report success to proxy rotator
                if proxy_rotator and current_proxy:
                    proxy_rotator.report_success(current_proxy)
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
                    # Report failure to proxy rotator (likely IP blocked)
                    if proxy_rotator and current_proxy:
                        proxy_rotator.report_failure(current_proxy)

                if any(indicator in stderr_lower for indicator in ['rate', 'quota', 'limit']):
                    if not rate_limited:
                        rate_limited = True
                        # Report failure to proxy rotator
                        if proxy_rotator and current_proxy:
                            proxy_rotator.report_failure(current_proxy)
                        # Adaptive backoff for rate limiting
                        backoff = random.uniform(10.0, 20.0) * (attempt + 1)
                        time.sleep(backoff)

                # Don't retry unavailable/private videos
                if any(indicator in stderr_lower for indicator in [
                    'private', 'unavailable', 'not available', 'removed'
                ]):
                    return False

                # Check for proxy-specific errors
                if any(indicator in stderr_lower for indicator in [
                    'proxy', 'connection refused', 'connection reset', 'timed out'
                ]):
                    if proxy_rotator and current_proxy:
                        proxy_rotator.report_failure(current_proxy)

        except subprocess.TimeoutExpired:
            # Timeout could indicate proxy issue
            if proxy_rotator and current_proxy:
                proxy_rotator.report_failure(current_proxy)
        except Exception as e:
            pass  # Silent - will be counted in stats

        # Exponential backoff with jitter between retries
        if attempt < max_retries - 1:
            backoff = (2 ** attempt) + random.uniform(0, 2)
            time.sleep(backoff)

    return False

def download_single_clip(youtube_id, output_dir, start_time=0, cookies_file=None,
                         cluster_scanner: Optional[ClusterFileScanner] = None):
    """
    Download a single 10-second audio clip with cluster-aware file skipping.

    Args:
        youtube_id: YouTube video ID
        output_dir: Directory to save audio
        start_time: Start time in seconds for the 10-second clip
        cookies_file: Path to cookies.txt file
        cluster_scanner: ClusterFileScanner for checking existing files (optional)

    Returns:
        Tuple of (youtube_id, success, status_message)
    """
    global _cluster_scanner
    if cluster_scanner is None:
        cluster_scanner = _cluster_scanner

    # Check cluster scanner first (O(1) lookup after pre-scan)
    if cluster_scanner and cluster_scanner.exists(youtube_id):
        return (youtube_id, True, "exists_on_cluster")

    # Skip if already downloaded in local output dir
    potential_files = list(output_dir.glob(f"{youtube_id}.*"))
    if potential_files:
        return (youtube_id, True, "already_exists")

    # Download 10-second clip starting at start_time
    success = download_audio(youtube_id, output_dir, start_time=start_time, cookies_file=cookies_file)
    return (youtube_id, success, "success" if success else "failed")


def process_audiocaps_csv(csv_path, output_dir, max_downloads=None, num_workers=4, cookies_file=None,
                          cluster_scanner: Optional[ClusterFileScanner] = None):
    """
    Process AudioCaps CSV and download 10-second audio clips in parallel.

    Supports cluster-aware file skipping to avoid re-downloading files that
    already exist on the cluster.
    """
    global _cluster_scanner
    if cluster_scanner is None:
        cluster_scanner = _cluster_scanner

    print(f"Processing {csv_path}...")

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return

    # Read CSV
    df = pd.read_csv(csv_path)

    # Check if start_time column exists
    has_start_time = 'start_time' in df.columns
    if not has_start_time:
        print("Warning: CSV does not have 'start_time' column - downloading from beginning of videos")
        print("   For accurate 10-second clips, ensure CSV has 'start_time' column")
        df['start_time'] = 0  # Default to start

    # Limit downloads for testing
    if max_downloads:
        df = df.head(max_downloads)

    total_in_csv = len(df)
    print(f"Found {total_in_csv} audio clips in CSV")

    # Pre-filter using cluster scanner for efficiency
    if cluster_scanner:
        youtube_ids = df['youtube_id'].tolist()
        needed_ids = set(cluster_scanner.filter_needed(youtube_ids))
        pre_filter_count = len(df)
        df = df[df['youtube_id'].isin(needed_ids)]
        skipped_cluster = pre_filter_count - len(df)
        print(f"Skipping {skipped_cluster} files already on cluster, {len(df)} remaining to check/download")

    print(f"Downloading 10-second clips" + (f" starting at specified times" if has_start_time else " from video start"))
    print(f"Using {num_workers} parallel workers")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track statistics
    successful = 0
    failed = 0
    already_exists = 0
    exists_on_cluster = 0
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
                cookies_file=cookies_file,
                cluster_scanner=cluster_scanner
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
            elif status == "exists_on_cluster":
                exists_on_cluster += 1
                successful += 1
            elif success:
                successful += 1
            else:
                failed += 1

            processed += 1

            # Print progress every 50 files
            if processed % 50 == 0 or processed == len(df):
                downloaded = successful - already_exists - exists_on_cluster
                skipped_total = already_exists + exists_on_cluster
                print(f"Progress: {processed}/{len(df)} | Downloaded: {downloaded} | Skipped: {skipped_total} (local:{already_exists}, cluster:{exists_on_cluster}) | Failed: {failed}")

    # Print proxy stats if available
    global _proxy_rotator
    if _proxy_rotator and _proxy_rotator.has_proxies:
        proxy_stats = _proxy_rotator.get_stats()
        print(f"Proxy stats: {proxy_stats['active']}/{proxy_stats['total']} active, {proxy_stats['failed']} temporarily failed")

    skipped_total = already_exists + exists_on_cluster
    downloaded_new = successful - skipped_total
    print(f"\nDownload complete: {successful} successful ({downloaded_new} new, {skipped_total} skipped), {failed} failed")

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
    """Main download function with IP rotation and cluster file skipping."""
    global _proxy_rotator, _cluster_scanner
    import argparse

    parser = argparse.ArgumentParser(
        description="Download AudioCaps audio from YouTube with IP rotation and cluster file skipping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  YTDLP_PROXY_LIST          Path to file with proxy URLs (one per line)
  YTDLP_PROXY               Single proxy URL (fallback)
  HTTP_PROXY/HTTPS_PROXY    Standard proxy env vars (fallback)
  AUDIOCAPS_CLUSTER_PATHS   Colon-separated paths to scan for existing files

Proxy file format (one per line):
  socks5://user:pass@host:port
  http://host:port
  socks5h://host:port

Examples:
  # Basic usage
  python download_audiocaps_audio.py --split train

  # With proxy rotation
  YTDLP_PROXY_LIST=proxies.txt python download_audiocaps_audio.py --split train

  # With cluster file skipping
  python download_audiocaps_audio.py --split train --cluster-paths /shared/audio:/nfs/audiocaps

  # Full example with all options
  YTDLP_PROXY_LIST=proxies.txt python download_audiocaps_audio.py \\
      --split train --workers 8 --cookies cookies.txt \\
      --cluster-paths /shared/audio:/nfs/audiocaps
        """
    )
    parser.add_argument("--cookies", type=str, help="Path to cookies.txt file for YouTube authentication")
    parser.add_argument("--max-downloads", type=int, default=None, help="Maximum downloads per split (default: all)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (reduce if rate limited)")
    parser.add_argument("--data-root", type=str, default="experiments/full_training/data/audiocaps",
                        help="Root directory for AudioCaps data")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"],
                        help="Specific split to download (default: all splits)")
    parser.add_argument("--cluster-paths", type=str, default="",
                        help="Colon-separated paths to scan for existing files (e.g., /shared/audio:/nfs/data)")
    parser.add_argument("--proxy-list", type=str, default="",
                        help="Path to file containing proxy URLs (one per line)")
    parser.add_argument("--no-proxy", action="store_true",
                        help="Disable proxy rotation even if proxies are configured")
    parser.add_argument("--no-cluster-scan", action="store_true",
                        help="Disable cluster file scanning (only check local output dir)")
    args = parser.parse_args()

    print("AudioCaps Audio Downloader (Enhanced)")
    print("="*60)
    print("Features: IP rotation, cluster file skipping, anti-bot detection")
    print("="*60)

    # Initialize proxy rotator
    if not args.no_proxy:
        # Allow CLI to override env var
        if args.proxy_list:
            os.environ['YTDLP_PROXY_LIST'] = args.proxy_list

        _proxy_rotator = ProxyRotator()
        if _proxy_rotator.has_proxies:
            print(f"Proxy rotation: ENABLED ({len(_proxy_rotator.proxies)} proxies loaded)")
        else:
            print("Proxy rotation: DISABLED (no proxies configured)")
            print("  Tip: Set YTDLP_PROXY_LIST env var or use --proxy-list")
    else:
        print("Proxy rotation: DISABLED (--no-proxy flag)")

    # Initialize cluster scanner
    if not args.no_cluster_scan:
        _cluster_scanner = ClusterFileScanner()

        # Add paths from CLI
        if args.cluster_paths:
            for path in args.cluster_paths.split(':'):
                if path.strip():
                    _cluster_scanner.add_path(path.strip())

        # Add paths from environment
        _cluster_scanner.add_paths_from_env()

        # Add the output directory itself
        data_dir = Path(args.data_root).expanduser().resolve()
        _cluster_scanner.add_path(data_dir)

        if _cluster_scanner.paths:
            print(f"Cluster scanning: ENABLED ({len(_cluster_scanner.paths)} paths)")
            for p in _cluster_scanner.paths:
                print(f"  - {p}")
            # Pre-scan all paths
            _cluster_scanner.scan()
        else:
            print("Cluster scanning: DISABLED (no paths configured)")
    else:
        print("Cluster scanning: DISABLED (--no-cluster-scan flag)")

    print()

    if args.cookies:
        if not os.path.exists(args.cookies):
            print(f"Warning: Cookies file not found: {args.cookies}")
            print("You can export cookies using a browser extension like 'Get cookies.txt'")
        else:
            print(f"Using cookies from: {args.cookies}")
    else:
        print("No cookies specified - using Android client to bypass bot detection")

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
            process_audiocaps_csv(csv_path, split_audio_dir, args.max_downloads, args.workers, args.cookies,
                                  cluster_scanner=_cluster_scanner)
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