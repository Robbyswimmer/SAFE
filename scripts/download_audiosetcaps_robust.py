#!/usr/bin/env python3
"""
Robust AudioSetCaps downloader for 6M+ audio files with anti-bot detection.

Features:
- Resume capability (SQLite-based state tracking)
- Automatic retries with exponential backoff
- Advanced anti-bot detection measures:
  * Android client API usage (harder to detect than web scraping)
  * User agent rotation (Android/iOS/Desktop)
  * Human-like timing patterns (variable delays, random pauses)
  * Adaptive rate limiting (auto-adjusts on bot detection)
  * Bot error detection and cooldown periods
  * Periodic "break" behavior (simulates human fatigue)
- Rate limiting with jitter to avoid patterns
- Progress tracking and detailed logging
- Error recovery and skip corrupted files
- Parallel downloads with worker pool
- Audio verification to ensure quality
- Cluster file scanning to skip already-downloaded files (AudioCaps, etc.)
- Proxy rotation support for avoiding IP bans

Environment Variables:
- YTDLP_PROXY_LIST: Path to file with proxy URLs (one per line)
- YTDLP_PROXY: Single proxy URL
- AUDIOSETCAPS_SKIP_PATHS: Colon-separated paths to scan for existing files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import re
import threading

import numpy as np

try:
    import yt_dlp
    import pandas as pd
    from huggingface_hub import hf_hub_download
except ImportError as exc:
    raise SystemExit(
        "Required packages missing. Install with:\n"
        "pip install yt-dlp pandas huggingface-hub soundfile"
    ) from exc


# ============================================================================
# Proxy Rotation Support
# ============================================================================

@dataclass
class ProxyRotator:
    """
    Thread-safe proxy rotation for distributed downloading.

    Loads proxies from:
    1. YTDLP_PROXY_LIST env var (path to file with proxy list)
    2. YTDLP_PROXY env var (single proxy)
    3. Standard HTTP_PROXY/HTTPS_PROXY env vars
    """
    proxies: List[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _index: int = field(default=0, repr=False)
    _failed_proxies: Set[str] = field(default_factory=set, repr=False)
    _proxy_failures: dict = field(default_factory=dict, repr=False)
    max_failures: int = 5

    def __post_init__(self):
        if not self.proxies:
            self._load_from_environment()

    def _load_from_environment(self):
        """Load proxy configuration from environment variables."""
        # Priority 1: Proxy list file
        proxy_list_path = os.environ.get('YTDLP_PROXY_LIST')
        if proxy_list_path and os.path.exists(proxy_list_path):
            try:
                with open(proxy_list_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if self._validate_proxy_format(line):
                                self.proxies.append(line)
                if self.proxies:
                    return
            except (IOError, OSError):
                pass

        # Priority 2: Single proxy from env
        single_proxy = os.environ.get('YTDLP_PROXY')
        if single_proxy and self._validate_proxy_format(single_proxy):
            self.proxies.append(single_proxy)
            return

        # Priority 3: Standard HTTP proxy env vars
        for env_var in ['HTTPS_PROXY', 'HTTP_PROXY', 'https_proxy', 'http_proxy']:
            proxy = os.environ.get(env_var)
            if proxy and self._validate_proxy_format(proxy):
                self.proxies.append(proxy)
                return

    def _validate_proxy_format(self, proxy: str) -> bool:
        """Validate proxy URL format."""
        valid_schemes = ('http://', 'https://', 'socks4://', 'socks5://', 'socks5h://')
        if not any(proxy.startswith(scheme) for scheme in valid_schemes):
            return False
        try:
            rest = proxy.split('://', 1)[1]
            if '@' in rest:
                rest = rest.split('@', 1)[1]
            return ':' in rest
        except (IndexError, ValueError):
            return False

    def get_next_proxy(self) -> Optional[str]:
        """Get next proxy in rotation (thread-safe)."""
        if not self.proxies:
            return None
        with self._lock:
            available = [p for p in self.proxies if p not in self._failed_proxies]
            if not available:
                if self._failed_proxies:
                    self._failed_proxies.clear()
                    self._proxy_failures.clear()
                    available = self.proxies
                else:
                    return None
            proxy = available[self._index % len(available)]
            self._index += 1
            return proxy

    def report_failure(self, proxy: str):
        """Report a proxy failure."""
        if not proxy:
            return
        with self._lock:
            self._proxy_failures[proxy] = self._proxy_failures.get(proxy, 0) + 1
            if self._proxy_failures[proxy] >= self.max_failures:
                self._failed_proxies.add(proxy)

    def report_success(self, proxy: str):
        """Report a proxy success."""
        if not proxy:
            return
        with self._lock:
            if proxy in self._proxy_failures:
                self._proxy_failures[proxy] = max(0, self._proxy_failures[proxy] - 1)

    @property
    def has_proxies(self) -> bool:
        return len(self.proxies) > 0


# ============================================================================
# Cluster File Scanner (Skip Already Downloaded)
# ============================================================================

@dataclass
class ClusterFileScanner:
    """
    Scan directories to find already-downloaded files and skip them.

    This prevents re-downloading files that exist in:
    - AudioCaps directories
    - Other AudioSetCaps download locations
    - Shared cluster storage
    """
    paths: List[Path] = field(default_factory=list)
    existing_ids: Set[str] = field(default_factory=set, repr=False)
    _scanned: bool = field(default=False, repr=False)

    def add_path(self, path: str | Path):
        """Add a path to scan."""
        p = Path(path).expanduser().resolve()
        if p.exists():
            self.paths.append(p)

    def add_paths_from_env(self):
        """Load paths from AUDIOSETCAPS_SKIP_PATHS env var."""
        paths_env = os.environ.get('AUDIOSETCAPS_SKIP_PATHS', '')
        if paths_env:
            for path in paths_env.split(':'):
                if path.strip():
                    self.add_path(path.strip())

    def scan(self, extensions: Tuple[str, ...] = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')) -> int:
        """Pre-scan all paths for existing audio files."""
        if self._scanned:
            return len(self.existing_ids)

        for cluster_path in self.paths:
            try:
                for ext in extensions:
                    for audio_file in cluster_path.rglob(f"*{ext}"):
                        youtube_id = self._extract_youtube_id(audio_file.stem)
                        if youtube_id:
                            self.existing_ids.add(youtube_id)
            except (PermissionError, OSError):
                pass

        self._scanned = True
        return len(self.existing_ids)

    def _extract_youtube_id(self, filename: str) -> Optional[str]:
        """Extract YouTube ID from various filename formats."""
        # YouTube IDs are 11 characters
        if re.match(r'^[a-zA-Z0-9_-]{11}$', filename):
            return filename

        # Try to find 11-char pattern
        match = re.search(r'([a-zA-Z0-9_-]{11})', filename)
        if match:
            return match.group(1)
        return None

    def exists(self, youtube_id: str) -> bool:
        """Check if a YouTube ID already exists."""
        if not self._scanned:
            self.scan()
        return youtube_id in self.existing_ids

    def filter_needed(self, youtube_ids: List[str]) -> List[str]:
        """Filter to only IDs that need downloading."""
        if not self._scanned:
            self.scan()
        return [yt_id for yt_id in youtube_ids if yt_id not in self.existing_ids]


# Global instances
_proxy_rotator: Optional[ProxyRotator] = None
_cluster_scanner: Optional[ClusterFileScanner] = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DownloadConfig:
    """Configuration for robust downloading."""

    # Dataset settings
    dataset_name: str = "baijs/AudioSetCaps"
    metadata_file: Optional[str] = None  # Path to local CSV if already downloaded
    output_dir: Path = Path("data/audiosetcaps")

    # Download settings
    max_workers: int = 4  # Parallel download workers
    max_retries: int = 5  # Maximum retry attempts per file
    initial_backoff: float = 1.0  # Initial retry delay (seconds)
    max_backoff: float = 60.0  # Maximum retry delay (seconds)
    rate_limit_delay: float = 0.5  # Delay between downloads (seconds)

    # Anti-bot detection
    randomize_timing: bool = True  # Add random jitter to delays
    use_android_client: bool = True  # Prefer Android client (harder to detect)
    rotate_user_agents: bool = True  # Rotate user agents
    human_like_delays: bool = True  # Use human-like timing patterns
    adaptive_rate_limit: bool = True  # Increase delay if errors detected

    # YouTube download settings
    audio_format: str = "wav"  # Output audio format
    audio_quality: str = "bestaudio"
    sample_rate: int = 48000

    # Resume and tracking
    state_db: Path = field(default_factory=lambda: Path("data/audiosetcaps/.download_state.db"))
    log_file: Path = field(default_factory=lambda: Path("data/audiosetcaps/.download.log"))

    # Limits (for testing)
    max_samples: Optional[int] = None  # None = download all

    # Error handling
    skip_on_error: bool = True  # Skip files that fail after max retries
    verify_audio: bool = True  # Verify downloaded audio is valid


# ============================================================================
# Download State Tracking (SQLite)
# ============================================================================

class DownloadStateTracker:
    """Track download progress using SQLite for resumability."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS downloads (
                audio_id TEXT PRIMARY KEY,
                youtube_id TEXT,
                start_time INTEGER,
                status TEXT,  -- 'pending', 'completed', 'failed', 'skipped'
                file_path TEXT,
                attempts INTEGER DEFAULT 0,
                last_error TEXT,
                last_attempt_time REAL,
                completed_time REAL
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON downloads(status)
        """)
        self.conn.commit()

    def add_samples(self, samples: List[Dict]):
        """Add samples to tracking database."""
        data = [
            (
                self._make_audio_id(s),
                s.get("youtube_id", ""),
                s.get("start_time", 0),
                "pending",
                None,
                0,
                None,
                None,
                None
            )
            for s in samples
        ]
        self.conn.executemany("""
            INSERT OR IGNORE INTO downloads
            (audio_id, youtube_id, start_time, status, file_path, attempts, last_error, last_attempt_time, completed_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        self.conn.commit()

    def _make_audio_id(self, sample: Dict) -> str:
        """Create unique audio ID from sample."""
        yt_id = sample.get("youtube_id", "")
        start = sample.get("start_time", 0)
        return f"{yt_id}_{start:06d}"

    def get_pending_samples(self, limit: Optional[int] = None) -> List[Dict]:
        """Get samples that need to be downloaded."""
        query = "SELECT audio_id, youtube_id, start_time FROM downloads WHERE status = 'pending'"
        if limit:
            query += f" LIMIT {limit}"

        cursor = self.conn.execute(query)
        return [
            {"audio_id": row[0], "youtube_id": row[1], "start_time": row[2]}
            for row in cursor.fetchall()
        ]

    def mark_completed(self, audio_id: str, file_path: str):
        """Mark sample as successfully downloaded."""
        self.conn.execute("""
            UPDATE downloads
            SET status = 'completed', file_path = ?, completed_time = ?
            WHERE audio_id = ?
        """, (file_path, time.time(), audio_id))
        self.conn.commit()

    def mark_failed(self, audio_id: str, error: str, attempts: int):
        """Mark sample as failed."""
        self.conn.execute("""
            UPDATE downloads
            SET status = 'failed', last_error = ?, attempts = ?, last_attempt_time = ?
            WHERE audio_id = ?
        """, (error, attempts, time.time(), audio_id))
        self.conn.commit()

    def mark_skipped(self, audio_id: str, reason: str):
        """Mark sample as skipped."""
        self.conn.execute("""
            UPDATE downloads
            SET status = 'skipped', last_error = ?
            WHERE audio_id = ?
        """, (reason, audio_id))
        self.conn.commit()

    def increment_attempts(self, audio_id: str):
        """Increment attempt counter."""
        self.conn.execute("""
            UPDATE downloads
            SET attempts = attempts + 1, last_attempt_time = ?
            WHERE audio_id = ?
        """, (time.time(), audio_id))
        self.conn.commit()

    def get_stats(self) -> Dict[str, int]:
        """Get download statistics."""
        cursor = self.conn.execute("""
            SELECT status, COUNT(*) FROM downloads GROUP BY status
        """)
        stats = dict(cursor.fetchall())
        stats.setdefault("pending", 0)
        stats.setdefault("completed", 0)
        stats.setdefault("failed", 0)
        stats.setdefault("skipped", 0)
        return stats

    def close(self):
        """Close database connection."""
        self.conn.close()


# ============================================================================
# YouTube Audio Downloader
# ============================================================================

class RobustYouTubeDownloader:
    """Robust YouTube audio downloader with retry logic and anti-bot measures."""

    def __init__(self, config: DownloadConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.last_download_time = 0.0
        self.consecutive_errors = 0
        self.bot_detection_count = 0
        self.current_rate_limit = config.rate_limit_delay
        self.download_count = 0

    def _rate_limit(self):
        """Apply adaptive rate limiting with human-like patterns."""
        elapsed = time.time() - self.last_download_time

        # Use adaptive rate limit if enabled
        base_delay = self.current_rate_limit if self.config.adaptive_rate_limit else self.config.rate_limit_delay

        # Add human-like variability
        if self.config.human_like_delays:
            # Humans don't download at exact intervals
            jitter = random.uniform(-0.2 * base_delay, 0.3 * base_delay)
            delay = max(0.1, base_delay + jitter)

            # Occasionally add longer "thinking" pauses (simulating human behavior)
            if random.random() < 0.05:  # 5% chance
                delay += random.uniform(2, 5)
                self.logger.debug("Adding human-like pause")
        else:
            delay = base_delay

            # Add small jitter even if human_like_delays is off
            if self.config.randomize_timing:
                delay += random.uniform(0, 0.1 * base_delay)

        # Wait if needed
        if elapsed < delay:
            sleep_time = delay - elapsed
            time.sleep(sleep_time)

        self.last_download_time = time.time()

    def _detect_bot_block(self, error_msg: str) -> bool:
        """Detect if error indicates bot detection."""
        bot_indicators = [
            'sign in to confirm',
            'bot',
            'captcha',
            'too many requests',
            'rate limit',
            'forbidden',
            '403',
            '429',
            'unusual traffic',
            'verify you are human',
        ]
        error_lower = error_msg.lower()
        return any(indicator in error_lower for indicator in bot_indicators)

    def _handle_bot_detection(self):
        """Adjust settings when bot detection is suspected."""
        self.bot_detection_count += 1
        self.consecutive_errors += 1

        if self.config.adaptive_rate_limit:
            # Exponentially increase rate limit
            old_limit = self.current_rate_limit
            self.current_rate_limit = min(
                self.current_rate_limit * 2,
                self.config.max_backoff
            )
            self.logger.warning(
                f"Bot detection suspected (count: {self.bot_detection_count}). "
                f"Increasing rate limit: {old_limit:.2f}s → {self.current_rate_limit:.2f}s"
            )

            # Add immediate cooldown
            cooldown = random.uniform(5, 15)
            self.logger.info(f"Cooling down for {cooldown:.1f}s")
            time.sleep(cooldown)

    def _on_successful_download(self):
        """Reset error counters on successful download."""
        if self.consecutive_errors > 0:
            self.consecutive_errors = 0

            # Gradually decrease rate limit if it was increased
            if self.config.adaptive_rate_limit and self.current_rate_limit > self.config.rate_limit_delay:
                self.current_rate_limit = max(
                    self.current_rate_limit * 0.9,
                    self.config.rate_limit_delay
                )
                self.logger.debug(f"Decreasing rate limit to {self.current_rate_limit:.2f}s")

        self.download_count += 1

        # Every 100 downloads, add a longer break (simulate human fatigue)
        if self.config.human_like_delays and self.download_count % 100 == 0:
            break_time = random.uniform(10, 30)
            self.logger.info(f"Taking a break after 100 downloads ({break_time:.1f}s)")
            time.sleep(break_time)

    def _get_random_user_agent(self) -> str:
        """Get random user agent to avoid bot detection."""
        user_agents = [
            # Android clients (helps avoid bot detection)
            'com.google.android.youtube/17.36.4 (Linux; U; Android 11) gzip',
            'com.google.android.youtube/17.31.35 (Linux; U; Android 12) gzip',
            'com.google.android.youtube/18.11.34 (Linux; U; Android 13) gzip',
            # iOS clients
            'com.google.ios.youtube/17.33.2 (iPhone14,3; U; CPU iOS 15_6 like Mac OS X)',
            'com.google.ios.youtube/18.12.2 (iPhone15,2; U; CPU iOS 16_4 like Mac OS X)',
            # Desktop browsers
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
        return random.choice(user_agents)

    def _get_yt_dlp_opts(self, output_path: Path) -> Dict:
        """Get yt-dlp options for downloading with anti-bot detection."""
        # Rotate user agent for each download (if enabled)
        if self.config.rotate_user_agents:
            user_agent = self._get_random_user_agent()
        else:
            user_agent = 'com.google.android.youtube/18.11.34 (Linux; U; Android 13) gzip'

        # Use Android client for better bot evasion (if enabled)
        use_android_client = self.config.use_android_client and 'android' in user_agent.lower()

        opts = {
            'format': self.config.audio_quality,
            'outtmpl': str(output_path.with_suffix('.%(ext)s')),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': self.config.audio_format,
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'no_color': True,
            'extract_flat': False,
            'logger': logging.getLogger('yt-dlp'),  # Use logging instead of stderr
            'postprocessor_args': [
                '-ar', str(self.config.sample_rate),
            ],

            # Anti-bot detection measures
            'user_agent': user_agent,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web'] if use_android_client else ['web'],
                    'player_skip': ['webpage', 'configs'],
                }
            },

            # Network resilience
            'http_chunk_size': 10485760,  # 10MB chunks
            'retries': 3,
            'fragment_retries': 3,
            'skip_unavailable_fragments': True,

            # Additional anti-detection
            'sleep_interval': random.uniform(1, 3),  # Random delay between fragments
            'max_sleep_interval': 5,
            'sleep_interval_subtitles': 0,

            # Disable features that might trigger detection
            'nocheckcertificate': False,  # Keep certificate checking
            'prefer_insecure': False,
            'no_check_certificate': False,

            # Cookies and session management
            'cookiesfrombrowser': None,  # Don't use browser cookies (can be detected)

            # Age gate handling
            'age_limit': None,
        }

        # Add random headers to appear more human-like
        if not use_android_client:
            opts['http_headers'] = {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            }

        # Add proxy if available
        global _proxy_rotator
        if _proxy_rotator and _proxy_rotator.has_proxies:
            proxy = _proxy_rotator.get_next_proxy()
            if proxy:
                opts['proxy'] = proxy
                self._current_proxy = proxy
        else:
            self._current_proxy = None

        return opts

    def download_audio_segment(
        self,
        youtube_id: str,
        start_time: int,
        output_path: Path,
        duration: int = 10,
    ) -> bool:
        """
        Download 10-second audio segment from YouTube video.

        Returns:
            True if successful, False otherwise
        """
        url = f"https://www.youtube.com/watch?v={youtube_id}"

        # Apply rate limiting
        self._rate_limit()

        # Create temporary download path
        temp_path = output_path.with_suffix('.temp.wav')

        try:
            # Download full audio
            ydl_opts = self._get_yt_dlp_opts(temp_path)

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Find the downloaded file
            downloaded_file = None
            for ext in ['wav', 'webm', 'mp4', 'm4a', 'opus']:
                candidate = temp_path.with_suffix(f'.{ext}')
                if candidate.exists():
                    downloaded_file = candidate
                    break

            if not downloaded_file or not downloaded_file.exists():
                raise FileNotFoundError("Downloaded file not found")

            # Extract 10-second segment using soundfile
            audio_data, sr = self._load_and_clip_audio(
                downloaded_file, start_time, duration
            )

            # Save clipped audio
            self._save_audio(output_path, audio_data, sr)

            # Clean up temp file
            if downloaded_file.exists():
                downloaded_file.unlink()

            # Report proxy success
            if _proxy_rotator and getattr(self, '_current_proxy', None):
                _proxy_rotator.report_success(self._current_proxy)

            return True

        except Exception as e:
            # Only log if it's not a common "video unavailable" error
            error_str = str(e).lower()
            if "unavailable" not in error_str and "not available" not in error_str:
                self.logger.debug(f"Failed to download {youtube_id}: {str(e)}")

            # Report proxy failure if it looks like a network/proxy issue
            if _proxy_rotator and getattr(self, '_current_proxy', None):
                if any(ind in error_str for ind in ['proxy', 'connection', 'timeout', '403', '429', 'rate']):
                    _proxy_rotator.report_failure(self._current_proxy)

            # Clean up any temp files
            for ext in ['wav', 'webm', 'mp4', 'm4a', 'opus', 'temp.wav']:
                temp_file = temp_path.with_suffix(f'.{ext}')
                if temp_file.exists():
                    temp_file.unlink()
            return False

    def _load_and_clip_audio(
        self, audio_file: Path, start_time: int, duration: int
    ) -> tuple[np.ndarray, int]:
        """Load audio and extract segment."""
        import soundfile as sf

        data, sr = sf.read(str(audio_file))

        # Convert to mono if stereo
        if data.ndim == 2:
            data = data.mean(axis=1)

        # Extract segment
        start_sample = int(start_time * sr)
        end_sample = start_sample + int(duration * sr)

        if start_sample >= len(data):
            # Start time is beyond audio length, return silence
            segment = np.zeros(int(duration * sr), dtype=np.float32)
        else:
            segment = data[start_sample:end_sample]
            # Pad if necessary
            if len(segment) < int(duration * sr):
                segment = np.pad(
                    segment, (0, int(duration * sr) - len(segment)),
                    mode='constant'
                )

        return segment.astype(np.float32), sr

    def _save_audio(self, output_path: Path, audio_data: np.ndarray, sample_rate: int):
        """Save audio to file."""
        import soundfile as sf

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_data, sample_rate)

    def download_with_retry(
        self,
        sample: Dict,
        output_path: Path,
    ) -> tuple[bool, Optional[str]]:
        """
        Download with automatic retry and exponential backoff.

        Returns:
            (success, error_message)
        """
        youtube_id = sample["youtube_id"]
        start_time = sample["start_time"]

        for attempt in range(self.config.max_retries):
            try:
                success = self.download_audio_segment(
                    youtube_id, start_time, output_path
                )

                if success:
                    # Verify audio if enabled
                    if self.config.verify_audio:
                        if not self._verify_audio(output_path):
                            raise ValueError("Audio verification failed")
                    return True, None

            except Exception as e:
                error = str(e)

                # Check if error indicates bot detection
                is_bot_detected = self._detect_bot_block(error)
                if is_bot_detected:
                    self._handle_bot_detection()

                if attempt < self.config.max_retries - 1:
                    # Exponential backoff with jitter
                    backoff = min(
                        self.config.initial_backoff * (2 ** attempt),
                        self.config.max_backoff
                    )
                    jitter = random.uniform(0, backoff * 0.1)
                    sleep_time = backoff + jitter

                    # Extra delay if bot detection suspected
                    if is_bot_detected:
                        sleep_time += random.uniform(10, 20)

                    self.logger.debug(
                        f"Attempt {attempt + 1} failed for {youtube_id}, "
                        f"retrying in {sleep_time:.1f}s: {error}"
                    )
                    time.sleep(sleep_time)
                else:
                    self.logger.warning(
                        f"All {self.config.max_retries} attempts failed for {youtube_id}: {error}"
                    )
                    return False, error

        return False, "Max retries exceeded"

    def _verify_audio(self, audio_path: Path) -> bool:
        """Verify that downloaded audio is valid."""
        try:
            import soundfile as sf
            data, sr = sf.read(str(audio_path))
            # Check minimum duration (at least 1 second)
            min_samples = sr * 1
            return len(data) >= min_samples and sr > 0
        except Exception:
            return False


# ============================================================================
# Main Download Orchestrator
# ============================================================================

class AudioSetCapsDownloader:
    """Main orchestrator for downloading AudioSetCaps dataset."""

    def __init__(self, config: DownloadConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.state_tracker = DownloadStateTracker(config.state_db)
        self.downloader = RobustYouTubeDownloader(config, self.logger)

        self.start_time = time.time()
        self.completed_count = 0
        self.failed_count = 0
        self.skipped_count = 0

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        self.config.log_file.parent.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("AudioSetCapsDownloader")
        logger.setLevel(logging.INFO)

        # Suppress yt-dlp logging
        yt_dlp_logger = logging.getLogger('yt-dlp')
        yt_dlp_logger.setLevel(logging.CRITICAL)  # Only show critical errors

        # File handler
        fh = logging.FileHandler(self.config.log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def load_metadata(self):
        """Load AudioSetCaps metadata from CSV or HuggingFace."""
        if self.config.metadata_file:
            self.logger.info(f"Loading metadata from CSV: {self.config.metadata_file}")
            samples = self._load_csv_metadata(self.config.metadata_file)
        else:
            self.logger.info(f"Downloading metadata from {self.config.dataset_name}...")
            samples = self._download_metadata_from_hf()

        if self.config.max_samples:
            samples = samples[:self.config.max_samples]

        self.logger.info(f"Loaded {len(samples)} samples")

        # Add to state tracker
        self.state_tracker.add_samples(samples)

        stats = self.state_tracker.get_stats()
        self.logger.info(f"State: {stats}")

    def _load_csv_metadata(self, csv_path: str) -> List[Dict]:
        """Load metadata from CSV file."""
        import pandas as pd

        self.logger.info(f"  Reading CSV file...")
        df = pd.read_csv(csv_path)
        self.logger.info(f"  CSV loaded, parsing {len(df)} rows...")

        # AudioSetCaps CSV format: id (e.g., "Y---1_cCGK4M" or "Y1ZooVL-H9A_30_40"), caption
        samples = []

        for i, row in enumerate(df.itertuples(index=False)):
            if i % 100000 == 0 and i > 0:
                self.logger.info(f"    Parsed {i}/{len(df)} rows...")

            # Parse ID field - three different formats:
            # 1. AudioSetCaps: Y{youtube_id} (e.g., Y---1_cCGK4M) - 11 char YT ID
            # 2. VGGSound: {youtube_id}_{timestamp} (e.g., OxPnZzn1_L8_000883)
            # 3. YouTube-8M: {youtube_id}_{start}_{duration} (e.g., 9eUrEyAR3xk_84_10)
            audio_id = str(row.id)

            # Handle AudioSetCaps format with 'Y' prefix
            if audio_id.startswith('Y') and len(audio_id) == 12:
                youtube_id = audio_id[1:]  # Remove 'Y' prefix
                start_time = 0
            else:
                # Handle VGGSound and YouTube-8M formats with timestamps
                parts = audio_id.split('_')

                # Last part is always numeric (timestamp or duration)
                if len(parts) >= 2 and parts[-1].isdigit():
                    # YouTube ID is everything except the last 1 or 2 numeric parts
                    # VGGSound: ytid_timestamp (2 parts after ytid)
                    # YouTube-8M: ytid_start_duration (2 parts after ytid)

                    # YouTube IDs are 11 characters, so find where it ends
                    # Try to reconstruct by taking parts until we have 11 chars
                    for split_idx in range(1, len(parts)):
                        potential_ytid = '_'.join(parts[:split_idx])
                        if len(potential_ytid) == 11:
                            youtube_id = potential_ytid
                            # Remaining parts are timestamps
                            remaining = parts[split_idx:]
                            if len(remaining) >= 1 and remaining[0].isdigit():
                                start_time = int(remaining[0])
                            else:
                                start_time = 0
                            break
                    else:
                        # Couldn't find valid split, skip
                        continue
                else:
                    # No timestamp info, try as-is
                    youtube_id = audio_id
                    start_time = 0

            # Validate YouTube ID (11 chars, alphanumeric + - and _)
            if len(youtube_id) != 11:
                continue
            if not all(c.isalnum() or c in '-_' for c in youtube_id):
                continue

            sample = {
                "youtube_id": youtube_id,
                "start_time": start_time,
                "caption": str(row.caption if hasattr(row, 'caption') else ""),
            }
            samples.append(sample)

        self.logger.info(f"  ✓ Parsed {len(samples)} samples")
        return samples

    def _download_metadata_from_hf(self) -> List[Dict]:
        """Download metadata CSV files from HuggingFace."""
        # Download the CSV files from the dataset repo
        from huggingface_hub import hf_hub_download

        self.logger.info("Downloading metadata files from HuggingFace...")

        # AudioSetCaps CSV files on HuggingFace
        csv_files = [
            "AudioSetCaps_caption.csv",  # Main AudioSet captions (~1.9M)
            "VGGSound_AudioSetCaps_caption.csv",  # VGGSound captions (~182K)
            "YouTube-8M_AudioSetCaps_caption.csv",  # YouTube-8M captions (~4M)
        ]

        all_samples = []
        for csv_file in csv_files:
            # Stop loading if we already have enough samples
            if self.config.max_samples and len(all_samples) >= self.config.max_samples:
                self.logger.info(f"Reached max_samples limit ({self.config.max_samples}), stopping CSV loading")
                break

            try:
                local_path = hf_hub_download(
                    repo_id=self.config.dataset_name,
                    filename=f"Dataset/{csv_file}",
                    repo_type="dataset",
                )
                self.logger.info(f"Downloaded {csv_file}")
                samples = self._load_csv_metadata(local_path)
                all_samples.extend(samples)
                self.logger.info(f"  Loaded {len(samples)} samples from {csv_file}")
            except Exception as e:
                self.logger.warning(f"Failed to download {csv_file}: {e}")

        self.logger.info(f"Total samples from all files: {len(all_samples)}")
        return all_samples

    def download_single(self, sample: Dict) -> tuple[str, bool, Optional[str], str]:
        """Download a single audio sample.

        Returns:
            (audio_id, success, error, status) where status is one of:
            'downloaded', 'exists_local', 'exists_cluster', 'skipped', 'failed'
        """
        global _cluster_scanner
        audio_id = sample["audio_id"]
        youtube_id = sample["youtube_id"]
        start_time = sample["start_time"]

        # Build output path
        output_filename = f"{audio_id}.{self.config.audio_format}"
        output_path = self.config.output_dir / "audio" / output_filename

        # Skip if already exists locally
        if output_path.exists():
            self.state_tracker.mark_completed(audio_id, str(output_path))
            return audio_id, True, None, "exists_local"

        # Skip if exists in cluster scanner (AudioCaps, etc.)
        if _cluster_scanner and _cluster_scanner.exists(youtube_id):
            self.state_tracker.mark_skipped(audio_id, "exists_on_cluster")
            return audio_id, True, None, "exists_cluster"

        # Increment attempts
        self.state_tracker.increment_attempts(audio_id)

        # Download with retry
        success, error = self.downloader.download_with_retry(sample, output_path)

        if success:
            self.downloader._on_successful_download()
            self.state_tracker.mark_completed(audio_id, str(output_path))
            return audio_id, True, None, "downloaded"
        else:
            # Check attempts
            attempts = self.state_tracker.conn.execute(
                "SELECT attempts FROM downloads WHERE audio_id = ?", (audio_id,)
            ).fetchone()[0]

            if attempts >= self.config.max_retries:
                if self.config.skip_on_error:
                    self.state_tracker.mark_skipped(audio_id, error or "Max retries exceeded")
                    return audio_id, False, error, "skipped"
                else:
                    self.state_tracker.mark_failed(audio_id, error or "Max retries exceeded", attempts)
                    return audio_id, False, error, "failed"

            return audio_id, False, error, "failed"

    def download_all(self):
        """Download all pending samples with parallel workers."""
        pending = self.state_tracker.get_pending_samples()

        if not pending:
            self.logger.info("No pending downloads")
            return

        self.logger.info(f"Starting download of {len(pending)} samples with {self.config.max_workers} workers")

        # Track detailed stats
        downloaded_count = 0
        exists_local_count = 0
        exists_cluster_count = 0

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self.download_single, sample): sample
                for sample in pending
            }

            for i, future in enumerate(as_completed(futures)):
                audio_id, success, error, status = future.result()

                if status == "downloaded":
                    downloaded_count += 1
                    self.completed_count += 1
                    # Show each new download
                    self.logger.info(f"✓ [{downloaded_count}] Downloaded: {audio_id}")
                elif status == "exists_local":
                    exists_local_count += 1
                    self.completed_count += 1
                elif status == "exists_cluster":
                    exists_cluster_count += 1
                    self.skipped_count += 1
                elif status == "skipped":
                    self.skipped_count += 1
                elif status == "failed":
                    self.failed_count += 1

                # Show progress every 100 processed
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - self.start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    self.logger.info(
                        f"Progress: {i + 1}/{len(pending)} | "
                        f"New: {downloaded_count} | Local: {exists_local_count} | "
                        f"Cluster: {exists_cluster_count} | Failed: {self.failed_count} | "
                        f"Rate: {rate:.1f}/s"
                    )

                # Full progress update every 500 samples
                if (i + 1) % 500 == 0:
                    self._print_progress(i + 1, len(pending))

        self._print_final_stats()

    def _print_progress(self, current: int, total: int):
        """Print progress update."""
        elapsed = time.time() - self.start_time
        rate = current / elapsed if elapsed > 0 else 0
        eta_seconds = (total - current) / rate if rate > 0 else 0
        eta = str(timedelta(seconds=int(eta_seconds)))

        self.logger.info(
            f"Progress: {current}/{total} ({100*current/total:.1f}%) | "
            f"Rate: {rate:.1f}/s | ETA: {eta} | "
            f"✓ {self.completed_count} ✗ {self.failed_count} ⊘ {self.skipped_count}"
        )

    def _print_final_stats(self):
        """Print final statistics."""
        stats = self.state_tracker.get_stats()
        elapsed = time.time() - self.start_time

        self.logger.info("\n" + "="*70)
        self.logger.info("Download Complete!")
        self.logger.info("="*70)
        self.logger.info(f"Total time: {timedelta(seconds=int(elapsed))}")
        self.logger.info(f"Completed: {stats['completed']}")
        self.logger.info(f"Failed: {stats['failed']}")
        self.logger.info(f"Skipped: {stats['skipped']}")
        self.logger.info(f"Pending: {stats['pending']}")
        self.logger.info("="*70)

    def close(self):
        """Cleanup resources."""
        self.state_tracker.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    global _proxy_rotator, _cluster_scanner

    parser = argparse.ArgumentParser(
        description="Robust AudioSetCaps downloader for 6M+ audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  YTDLP_PROXY_LIST           Path to file with proxy URLs (one per line)
  YTDLP_PROXY                Single proxy URL
  AUDIOSETCAPS_SKIP_PATHS    Colon-separated paths to scan for existing files

Examples:
  # Basic usage (slow, no proxies)
  python download_audiosetcaps_robust.py --max-workers 2

  # Skip files already in AudioCaps
  python download_audiosetcaps_robust.py --skip-paths /path/to/audiocaps/audio

  # Resume interrupted download
  python download_audiosetcaps_robust.py --skip-metadata

  # Test with small sample
  python download_audiosetcaps_robust.py --max-samples 100
        """
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/audiosetcaps"),
        help="Output directory for downloaded audio"
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default=None,
        help="Path to local CSV metadata file (optional, will download from HF if not provided)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Number of parallel download workers (default: 2, keep low to avoid rate limits)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts per file"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Delay between downloads in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to download (for testing)"
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip metadata loading (resume from existing state)"
    )
    parser.add_argument(
        "--skip-paths",
        type=str,
        default="",
        help="Colon-separated paths to scan for existing files to skip (e.g., AudioCaps dir)"
    )
    parser.add_argument(
        "--proxy-list",
        type=str,
        default="",
        help="Path to file containing proxy URLs (one per line)"
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Disable proxy rotation"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("AudioSetCaps Robust Downloader")
    print("=" * 70)

    # Initialize proxy rotator
    if not args.no_proxy:
        if args.proxy_list:
            os.environ['YTDLP_PROXY_LIST'] = args.proxy_list
        _proxy_rotator = ProxyRotator()
        if _proxy_rotator.has_proxies:
            print(f"✓ Proxy rotation: ENABLED ({len(_proxy_rotator.proxies)} proxies)")
        else:
            print("ℹ Proxy rotation: DISABLED (no proxies configured)")
    else:
        print("ℹ Proxy rotation: DISABLED (--no-proxy)")

    # Initialize cluster scanner for skipping existing files
    _cluster_scanner = ClusterFileScanner()

    # Add skip paths from CLI
    if args.skip_paths:
        for path in args.skip_paths.split(':'):
            if path.strip():
                _cluster_scanner.add_path(path.strip())
                print(f"  Adding skip path: {path.strip()}")

    # Add skip paths from environment
    _cluster_scanner.add_paths_from_env()

    # Add default AudioCaps path if it exists
    default_audiocaps = Path("experiments/full_training/data/audiocaps/audio")
    if default_audiocaps.exists():
        _cluster_scanner.add_path(default_audiocaps)
        print(f"  Adding default AudioCaps path: {default_audiocaps}")

    # Pre-scan for existing files
    if _cluster_scanner.paths:
        print(f"\nScanning {len(_cluster_scanner.paths)} path(s) for existing files...")
        existing_count = _cluster_scanner.scan()
        print(f"✓ Found {existing_count} existing files to skip")
    else:
        print("ℹ No skip paths configured (will download all files)")

    print()

    config = DownloadConfig(
        output_dir=args.output_dir,
        metadata_file=args.metadata_file,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        rate_limit_delay=args.rate_limit,
        max_samples=args.max_samples,
        state_db=args.output_dir / ".download_state.db",
        log_file=args.output_dir / ".download.log",
    )

    downloader = AudioSetCapsDownloader(config)

    try:
        if not args.skip_metadata:
            downloader.load_metadata()

        downloader.download_all()
    finally:
        downloader.close()


if __name__ == "__main__":
    main()
