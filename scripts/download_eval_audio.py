#!/usr/bin/env python3
"""
Download audio files for W&B export samples to enable local evaluation.

Matches predictions from W&B CSV export to AudioCaps validation set,
downloads the corresponding YouTube audio clips, and creates a simple
HTML viewer for side-by-side comparison of predictions vs ground truth.

Usage:
    python scripts/download_eval_audio.py \
        --csv wandb_export_2026-01-05T10_34_54.909-08_00.csv \
        --output eval_audio_review

Requirements:
    pip install yt-dlp
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_audiocaps_val(data_path: Path) -> List[Dict[str, Any]]:
    """Load AudioCaps validation set JSON."""
    val_json = data_path / "audiocaps" / "AudioCaps_val.json"
    if not val_json.exists():
        # Try alternate locations
        alt_paths = [
            data_path / "audiocaps" / "audiocaps_val.json",
            data_path / "AudioCaps_val.json",
        ]
        for alt in alt_paths:
            if alt.exists():
                val_json = alt
                break

    if not val_json.exists():
        raise FileNotFoundError(f"Could not find AudioCaps validation JSON at {val_json}")

    with open(val_json, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_csv_references(csv_path: Path) -> List[Dict[str, Any]]:
    """Parse the W&B export CSV and extract samples."""
    samples = []

    with open(csv_path, "r", encoding="utf-8") as f:
        # The CSV has multi-line reference cells (newline-separated captions)
        content = f.read()

    # Parse using csv module with proper handling
    lines = content.strip().split("\n")

    # Simple parser for this format - references span multiple lines
    samples = []
    current_sample = None

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is a new sample (starts with "What is happening)
        if line.startswith('"What is happening in the audio?"'):
            # Save previous sample
            if current_sample is not None:
                samples.append(current_sample)

            # Parse: "question","prediction","references_first_line
            # Format: "What is happening in the audio?","prediction text","First reference caption
            parts = line.split('","')
            if len(parts) >= 3:
                question = parts[0].strip('"')
                prediction = parts[1]
                # References start here, may continue on next lines
                ref_start = parts[2].strip('"').rstrip('"')
                current_sample = {
                    "question": question,
                    "prediction": prediction,
                    "references": [ref_start] if ref_start else [],
                }
            i += 1
        elif current_sample is not None:
            # This is a continuation of references
            ref_line = line.strip().rstrip('"')
            if ref_line:
                current_sample["references"].append(ref_line)
            i += 1
        else:
            # Skip header or empty lines
            i += 1

    # Don't forget last sample
    if current_sample is not None:
        samples.append(current_sample)

    return samples


def match_sample_to_audiocaps(
    sample: Dict[str, Any],
    audiocaps_data: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Match a sample's references to an AudioCaps entry."""
    sample_refs = set(r.strip().lower() for r in sample["references"])

    for entry in audiocaps_data:
        entry_captions = set(c.strip().lower() for c in entry.get("captions", []))
        # Check if at least 3 captions match (allowing for minor differences)
        matches = len(sample_refs & entry_captions)
        if matches >= 3:
            return entry

        # Also try substring matching for slight variations
        if matches >= 2:
            # Check if remaining ones are close
            for sr in sample_refs:
                for ec in entry_captions:
                    if sr in ec or ec in sr:
                        matches += 0.5
            if matches >= 3:
                return entry

    # Fallback: try matching first caption exactly
    if sample["references"]:
        first_ref = sample["references"][0].strip().lower()
        for entry in audiocaps_data:
            if entry.get("captions") and entry["captions"][0].strip().lower() == first_ref:
                return entry

    return None


def parse_sound_name(sound_name: str) -> Tuple[str, int]:
    """
    Parse sound_name to extract YouTube ID and start time.

    Format: {youtube_id}_{start_ms}.wav
    Example: rqfQRErjfk8_170000.wav -> ("rqfQRErjfk8", 170000)
    """
    name = sound_name.replace(".wav", "").replace(".flac", "").replace(".mp3", "")
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        yt_id = parts[0]
        try:
            start_ms = int(parts[1])
        except ValueError:
            start_ms = 0
        return yt_id, start_ms
    return name, 0


def find_local_audio(
    youtube_id: str,
    start_ms: int,
    data_path: Path,
) -> Optional[Path]:
    """Check if audio file exists locally in various locations."""
    # Try different naming conventions
    start_sec = start_ms // 1000
    name_variants = [
        f"{youtube_id}_{start_ms}.wav",           # e.g., rqfQRErjfk8_170000.wav
        f"{youtube_id}_{start_sec:06d}.wav",      # e.g., rqfQRErjfk8_000170.wav (cluster format)
        f"{youtube_id}_{start_sec}.wav",          # e.g., rqfQRErjfk8_170.wav
        f"{youtube_id}.wav",                       # e.g., rqfQRErjfk8.wav (no timestamp)
    ]

    search_dirs = [
        data_path / "audiocaps" / "audio" / "val",
        data_path / "audiocaps" / "audio" / "val_10s",
        data_path / "audiocaps" / "audio" / "validation",
        data_path / "audiocaps" / "audio",
        data_path / "audiocaps" / "audio" / "train",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for name in name_variants:
            candidate = search_dir / name
            if candidate.exists():
                return candidate

    return None


def download_youtube_clip(
    youtube_id: str,
    start_ms: int,
    output_path: Path,
    duration_sec: int = 10,
    use_cookies: bool = True,
    data_path: Optional[Path] = None,
) -> bool:
    """Download a clip from YouTube using yt-dlp."""
    if output_path.exists():
        print(f"  Already exists: {output_path.name}")
        return True

    # First check if we have the audio locally
    if data_path:
        local_file = find_local_audio(youtube_id, start_ms, data_path)
        if local_file:
            print(f"  Found local file: {local_file.name}")
            # Extract the 10-second segment using ffmpeg
            start_sec = start_ms / 1000.0
            try:
                # Check if file is already ~10 seconds (pre-clipped)
                probe_cmd = ["ffprobe", "-i", str(local_file), "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                file_duration = float(result.stdout.strip()) if result.stdout.strip() else 0

                if file_duration <= 12:  # Already clipped (10s + small tolerance)
                    import shutil
                    shutil.copy(local_file, output_path)
                    return True

                # Need to extract 10-second segment
                print(f"  Extracting 10s segment at {start_sec}s from {file_duration:.1f}s file...")
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(local_file),
                    "-ss", str(start_sec),
                    "-t", "10",
                    "-acodec", "pcm_s16le",
                    "-ar", "48000",
                    "-ac", "1",
                    str(output_path),
                ]
                ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)
                if ffmpeg_result.returncode == 0 and output_path.exists():
                    return True
                else:
                    print(f"  ffmpeg failed, copying full file instead")
                    import shutil
                    shutil.copy(local_file, output_path)
                    return True
            except Exception as e:
                print(f"  Error extracting segment: {e}, copying full file")
                import shutil
                shutil.copy(local_file, output_path)
                return True

    start_sec = start_ms / 1000.0
    url = f"https://www.youtube.com/watch?v={youtube_id}"

    # First, try downloading full audio and use ffmpeg to extract the segment
    temp_path = output_path.with_suffix(".temp.wav")

    # yt-dlp command - download full audio first
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",  # Best quality
        "-o", str(temp_path),
        "--no-playlist",
    ]

    cmd.append(url)

    try:
        print(f"  Downloading full audio...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else "Unknown error"
            # Check for common issues
            if "Video unavailable" in stderr or "Private video" in stderr:
                print(f"  Video unavailable or private")
            elif "Sign in" in stderr or "age" in stderr.lower():
                print(f"  Age-restricted or requires sign-in")
            else:
                print(f"  yt-dlp error: {stderr[:200]}")
            return False

        # yt-dlp may add extra suffixes, find the actual file
        actual_temp = None
        for ext in [".temp.wav", ".temp.wav.wav", ".temp.webm", ".temp.m4a"]:
            candidate = output_path.with_suffix(ext)
            if candidate.exists():
                actual_temp = candidate
                break

        # Also check without .temp
        if actual_temp is None:
            parent = output_path.parent
            stem = output_path.stem
            for f in parent.glob(f"{stem}.temp*"):
                actual_temp = f
                break

        if actual_temp is None or not actual_temp.exists():
            print(f"  Downloaded file not found")
            return False

        # Use ffmpeg to extract the 10-second segment
        print(f"  Extracting {duration_sec}s segment at {start_sec}s...")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite
            "-i", str(actual_temp),
            "-ss", str(start_sec),
            "-t", str(duration_sec),
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "1",
            str(output_path),
        ]

        ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)

        # Clean up temp file
        try:
            actual_temp.unlink()
        except Exception:
            pass

        if ffmpeg_result.returncode == 0 and output_path.exists():
            print(f"  Success!")
            return True
        else:
            print(f"  ffmpeg extraction failed: {ffmpeg_result.stderr[:100] if ffmpeg_result.stderr else 'unknown'}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  Timeout downloading {youtube_id}")
        return False
    except FileNotFoundError as e:
        print(f"  Command not found: {e}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def compute_similarity_score(prediction: str, references: List[str]) -> float:
    """
    Compute similarity score between prediction and references.
    Uses word overlap (Jaccard similarity) with best-matching reference.
    Returns score from 0.0 to 1.0.
    """
    import re

    def tokenize(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    pred_words = set(tokenize(prediction))

    best_score = 0.0
    for ref in references:
        ref_words = set(tokenize(ref))

        # Jaccard similarity: intersection / union
        if pred_words or ref_words:
            intersection = len(pred_words & ref_words)
            union = len(pred_words | ref_words)
            score = intersection / union if union > 0 else 0.0
        else:
            score = 0.0

        best_score = max(best_score, score)

    return best_score


def generate_html_viewer(
    samples: List[Dict[str, Any]],
    output_dir: Path,
    sort_by_similarity: bool = True,
) -> Path:
    """Generate an HTML file for reviewing predictions with audio."""
    html_path = output_dir / "review.html"

    # Compute similarity scores for all samples
    for sample in samples:
        sample["similarity_score"] = compute_similarity_score(
            sample["prediction"],
            sample["references"]
        )

    # Sort by similarity (best matches first)
    if sort_by_similarity:
        samples_sorted = sorted(samples, key=lambda x: x["similarity_score"], reverse=True)
    else:
        samples_sorted = samples

    with_audio = sum(1 for s in samples if s.get("audio_file") and Path(output_dir / s["audio_file"]).exists())
    missing = len(samples) - with_audio

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>SAFE Audio Caption Review</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .sample {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .sample-header {{ display: flex; justify-content: space-between; align-items: center;
                         border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 15px; }}
        .sample-num {{ font-weight: bold; color: #666; cursor: help; }}
        .audio-player {{ margin: 15px 0; }}
        audio {{ width: 100%; }}
        .prediction {{ background: #e3f2fd; padding: 12px; border-radius: 6px; margin: 10px 0; }}
        .prediction-label {{ font-weight: bold; color: #1976d2; margin-bottom: 5px; }}
        .references {{ background: #f1f8e9; padding: 12px; border-radius: 6px; }}
        .references-label {{ font-weight: bold; color: #388e3c; margin-bottom: 5px; }}
        .ref-list {{ margin: 0; padding-left: 20px; }}
        .ref-list li {{ margin: 5px 0; }}
        .status {{ font-size: 12px; color: #999; }}
        .no-audio {{ color: #999; font-style: italic; }}
        .stats {{ background: #fff; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>SAFE Audio Caption Review</h1>
    <div class="stats">
        <strong>Total samples:</strong> {len(samples)} |
        <strong>With audio:</strong> {with_audio}
    </div>
"""

    for i, sample in enumerate(samples_sorted):
        audio_file = sample.get("audio_file")
        has_audio = audio_file and (output_dir / audio_file).exists()
        score = sample["similarity_score"]

        html_content += f"""
    <div class="sample">
        <div class="sample-header">
            <span class="status">{sample.get('youtube_id', 'N/A')} @ {sample.get('start_sec', 0):.1f}s</span>
        </div>
        <div class="audio-player">
"""

        if has_audio:
            html_content += f'            <audio controls><source src="{audio_file}" type="audio/wav"></audio>\n'
        else:
            html_content += f'            <p class="no-audio">Audio not available</p>\n'

        html_content += f"""        </div>
        <div class="prediction">
            <div class="prediction-label">Model Prediction:</div>
            {sample['prediction']}
        </div>
        <div class="references">
            <div class="references-label">Ground Truth References:</div>
            <ul class="ref-list">
"""
        for ref in sample["references"]:
            html_content += f"                <li>{ref}</li>\n"

        html_content += """            </ul>
        </div>
    </div>
"""

    html_content += """
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_path


def main():
    parser = argparse.ArgumentParser(description="Download audio for W&B export evaluation")
    parser.add_argument("--csv", type=str, required=True, help="Path to W&B CSV export")
    parser.add_argument("--output", type=str, default="eval_audio_review", help="Output directory")
    parser.add_argument("--data-path", type=str, default="data", help="Path to data directory with AudioCaps")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples to download")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading, just generate HTML")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    data_path = Path(args.data_path)

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    print(f"Loading AudioCaps validation data...")
    try:
        audiocaps_data = load_audiocaps_val(data_path)
        print(f"  Loaded {len(audiocaps_data)} entries")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Parsing CSV export: {csv_path}")
    samples = parse_csv_references(csv_path)
    print(f"  Found {len(samples)} samples")

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"  Limited to {len(samples)} samples")

    print(f"\nMatching samples to AudioCaps entries...")
    matched = 0
    for sample in samples:
        entry = match_sample_to_audiocaps(sample, audiocaps_data)
        if entry:
            matched += 1
            yt_id, start_ms = parse_sound_name(entry["sound_name"])
            sample["youtube_id"] = yt_id
            sample["start_ms"] = start_ms
            sample["start_sec"] = start_ms / 1000.0
            sample["sound_name"] = entry["sound_name"]
            sample["audio_file"] = f"audio/{yt_id}_{start_ms}.wav"

    print(f"  Matched {matched}/{len(samples)} samples")

    if not args.skip_download:
        print(f"\nDownloading audio clips to {audio_dir}/...")
        downloaded = 0
        failed = 0

        for i, sample in enumerate(samples):
            if "youtube_id" not in sample:
                continue

            yt_id = sample["youtube_id"]
            start_ms = sample["start_ms"]
            output_path = audio_dir / f"{yt_id}_{start_ms}.wav"

            print(f"[{i+1}/{len(samples)}] {yt_id} @ {start_ms/1000:.1f}s")

            if download_youtube_clip(yt_id, start_ms, output_path, data_path=data_path):
                downloaded += 1
            else:
                failed += 1

        print(f"\nDownload complete: {downloaded} succeeded, {failed} failed")

        if failed > 0 and downloaded == 0:
            print(f"\n{'='*60}")
            print("All downloads failed. This is common because many AudioCaps")
            print("YouTube videos are no longer available.")
            print("")
            print("SOLUTIONS:")
            print("1. Export audio during training on cluster:")
            print("   python train_safe.py ... --export-eval-samples --export-eval-samples-audio")
            print("")
            print("2. Sync validation audio from cluster:")
            print("   ./scripts/sync_val_audio.sh user@cluster:/path/to/audio/val/")
            print("")
            print("3. Download AudioCaps dataset officially:")
            print("   python scripts/download_audiocaps.py --split val")
            print(f"{'='*60}")

    print(f"\nGenerating HTML viewer...")
    html_path = generate_html_viewer(samples, output_dir)
    print(f"  Created: {html_path}")

    # Also save a JSON for programmatic access
    json_path = output_dir / "samples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"  Created: {json_path}")

    print(f"\n{'='*60}")
    print(f"Done! Open {html_path} in your browser to review.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
