#!/usr/bin/env python3
"""Download AudioCaps from Hugging Face parquet shards and emit SAFE-ready files."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import io
import numpy as np
import itertools

try:
    from datasets import load_dataset, Audio  # type: ignore
except Exception as exc:  # pragma: no cover - surface helpful hint
    raise SystemExit(
        "datasets package is required. Install with `pip install datasets soundfile`."
    ) from exc

AUDIO_COLUMNS = ("audio_10s", "audio_segment", "clip", "audio")  # Prioritize 10-second clips


def _load_audio_from_path(path: str) -> Tuple[np.ndarray, int]:
    import soundfile as sf  # Local import to avoid mandatory dependency when not needed

    data, sr = sf.read(path)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim == 2:
        data = data.transpose(1, 0)
    return data.astype(np.float32), int(sr)


def _load_audio_from_bytes(blob: bytes) -> Tuple[np.ndarray, int]:
    import soundfile as sf  # Local import to avoid mandatory dependency when not needed

    with io.BytesIO(blob) as buffer:
        data, sr = sf.read(buffer)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim == 2:
        data = data.transpose(1, 0)
    return data.astype(np.float32), int(sr)


def _clip_audio_10s(audio_array: np.ndarray, sample_rate: int, start_time: Optional[int] = None) -> np.ndarray:
    """Extract 10-second clip from audio, optionally starting at start_time."""
    clip_duration = 10.0  # seconds
    clip_samples = int(clip_duration * sample_rate)

    if start_time is not None and start_time > 0:
        start_sample = int(start_time * sample_rate)
    else:
        start_sample = 0

    # Handle mono vs stereo
    if audio_array.ndim == 1:
        end_sample = min(start_sample + clip_samples, len(audio_array))
        clipped = audio_array[start_sample:end_sample]
    else:
        # Assume shape (channels, samples)
        end_sample = min(start_sample + clip_samples, audio_array.shape[1])
        clipped = audio_array[:, start_sample:end_sample]

    # Pad if shorter than 10 seconds
    if audio_array.ndim == 1:
        if len(clipped) < clip_samples:
            clipped = np.pad(clipped, (0, clip_samples - len(clipped)), mode='constant')
    else:
        if clipped.shape[1] < clip_samples:
            pad_amount = clip_samples - clipped.shape[1]
            clipped = np.pad(clipped, ((0, 0), (0, pad_amount)), mode='constant')

    return clipped


def _resolve_audio_field(sample: Dict) -> tuple[Dict, str]:
    """Resolve audio field from sample, returning (audio_payload, field_name)."""
    start_time = sample.get("start_time")

    for key in AUDIO_COLUMNS:
        audio_value = sample.get(key)
        if audio_value is None:
            continue

        audio_array = None
        sampling_rate = None

        if isinstance(audio_value, dict):
            if audio_value.get("array") is not None:
                audio_array = np.asarray(audio_value["array"])
                sampling_rate = audio_value.get("sampling_rate")
            else:
                audio_path = audio_value.get("path")
                if audio_path:
                    audio_array, sampling_rate = _load_audio_from_path(audio_path)
                else:
                    audio_bytes = audio_value.get("bytes")
                    if audio_bytes:
                        audio_array, sampling_rate = _load_audio_from_bytes(audio_bytes)
        elif isinstance(audio_value, str):
            audio_array, sampling_rate = _load_audio_from_path(audio_value)

        if audio_array is not None and sampling_rate is not None:
            # Clip to 10 seconds using start_time if available
            clipped_array = _clip_audio_10s(audio_array, sampling_rate, start_time)
            return {"array": clipped_array, "sampling_rate": sampling_rate}, key

    raise KeyError("Sample does not contain a usable audio field.")


def _collect_captions(sample: Dict) -> List[str]:
    captions: List[str] = []
    if isinstance(sample.get("captions"), (list, tuple)):
        captions.extend(str(cap).strip() for cap in sample["captions"] if str(cap).strip())
    caption = sample.get("caption") or sample.get("text")
    if caption:
        captions.append(str(caption).strip())
    # Ensure unique order-preserving list
    deduped: List[str] = []
    seen = set()
    for cap in captions:
        if cap and cap not in seen:
            deduped.append(cap)
            seen.add(cap)
    if not deduped:
        deduped.append("")
    return deduped


def _build_filename(sample: Dict, split: str, index: int) -> str:
    youtube_id = sample.get("ytid") or sample.get("youtube_id") or sample.get("video_id")
    start_time = sample.get("start_time") or sample.get("start") or sample.get("offset")
    if youtube_id:
        if isinstance(start_time, (int, float)):
            return f"{youtube_id}_{int(start_time):06d}.wav"
        return f"{youtube_id}.wav"
    return f"{split}_{index:06d}.wav"


def _save_audio(target: Path, audio_payload: Dict) -> None:
    array = audio_payload.get("array")
    sr = audio_payload.get("sampling_rate")
    if array is None or sr is None:
        raise ValueError("Audio payload missing 'array' or 'sampling_rate'.")

    np_array = np.asarray(array, dtype=np.float32)
    if np_array.ndim == 1:
        np_array = np_array[np.newaxis, :]

    def _pcm16_write_fallback() -> None:
        import wave

        clipped = np.clip(np_array, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype("<i2")
        channels, length = pcm16.shape
        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(int(channels))
            handle.setsampwidth(2)
            handle.setframerate(int(sr))
            # wave expects frames x channels in row-major order
            handle.writeframes(np.ascontiguousarray(pcm16.T).tobytes())

    try:
        import soundfile as sf  # type: ignore

        sf.write(target, np_array.T, int(sr))
        return
    except Exception:
        pass

    try:
        import torch
        import torchaudio  # type: ignore

        tensor = torch.from_numpy(np_array)
        torchaudio.save(target, tensor, int(sr))
        return
    except Exception:
        try:
            _pcm16_write_fallback()
            return
        except Exception as exc:  # pragma: no cover - fallback seldom used
            raise SystemExit(
                "Failed to save audio. Install either soundfile or torchaudio"
            ) from exc


@dataclass
class SplitRequest:
    hf_name: str
    local_name: str


def parse_split_requests(raw: str) -> List[SplitRequest]:
    requests: List[SplitRequest] = []
    for entry in (part.strip() for part in raw.split(",")):
        if not entry:
            continue
        if ":" in entry:
            hf_name, local_name = (value.strip() for value in entry.split(":", 1))
        else:
            hf_name = local_name = entry
        if not hf_name:
            raise ValueError(f"Invalid split specification: '{entry}'")
        requests.append(SplitRequest(hf_name=hf_name, local_name=local_name or hf_name))
    if not requests:
        raise ValueError("At least one split must be specified")
    return requests


def process_split(
    dataset_name: str,
    request: SplitRequest,
    output_dir: Path,
    overwrite_audio: bool,
    sample_limit: Optional[int],
    cache_dir: Optional[Path],
) -> Tuple[int, int]:
    ds = load_dataset(
        dataset_name,
        split=request.hf_name,
        cache_dir=str(cache_dir) if cache_dir else None,
        streaming=False,
        keep_in_memory=False,
    )

    audio_columns = [col for col in AUDIO_COLUMNS if col in ds.column_names]
    for column in audio_columns:
        try:
            ds = ds.cast_column(column, Audio(decode=False))
        except Exception:
            continue

    audio_root = output_dir / "audio" / request.local_name
    audio_root.mkdir(parents=True, exist_ok=True)

    metadata_entries: List[Dict] = []
    processed = 0
    skipped = 0
    audio_field_counts = {}

    iterator: Iterable
    if sample_limit is None:
        iterator = ds
    else:
        limit = min(len(ds), sample_limit)
        iterator = ds.select(range(limit))

    for idx, sample in enumerate(iterator):
        if idx % 100 == 0:
            print(f"   Processing sample {idx}...", flush=True)

        try:
            audio_payload, audio_field = _resolve_audio_field(sample)
            audio_field_counts[audio_field] = audio_field_counts.get(audio_field, 0) + 1
        except KeyError:
            skipped += 1
            continue

        filename = _build_filename(sample, request.local_name, idx)
        audio_path = audio_root / filename
        rel_path = Path("audio") / request.local_name / filename

        if overwrite_audio or not audio_path.exists():
            _save_audio(audio_path, audio_payload)

        metadata_entries.append(
            {
                "split": request.local_name,
                "sound_name": filename,
                "file_path": str(rel_path.as_posix()),
                "captions": _collect_captions(sample),
            }
        )
        processed += 1

    # Report which audio fields were used
    if audio_field_counts:
        field_summary = ", ".join(f"{field}: {count}" for field, count in sorted(audio_field_counts.items()))
        print(f"   Audio fields used → {field_summary}", flush=True)

    meta_path = output_dir / f"AudioCaps_{request.local_name}.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata_entries, fh, ensure_ascii=False, indent=2)

    return processed, skipped


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Download AudioCaps parquet data from Hugging Face")
    parser.add_argument(
        "--dataset",
        default="jp1924/AudioCaps",
        help="Hugging Face dataset identifier",
    )
    parser.add_argument(
        "--splits",
        default="train:train,validation:val,test:test",
        help="Comma list of splits (format hf_split[:local_name])",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/full_training/data/audiocaps"),
        help="Destination directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="datasets cache directory (defaults to <output-dir>/.hf_cache)",
    )
    parser.add_argument(
        "--overwrite-audio",
        action="store_true",
        help="Re-write WAV files even if they already exist",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit per split (useful for smoke tests)",
    )

    args = parser.parse_args(argv)

    split_requests = parse_split_requests(args.splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output_dir / ".hf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Survive interrupted runs without corrupting cache directories
    safe_cache_dir = cache_dir.resolve()

    for request in split_requests:
        print(f"\n⬇️  Downloading split '{request.hf_name}' as '{request.local_name}'", flush=True)
        processed, skipped = process_split(
            dataset_name=args.dataset,
            request=request,
            output_dir=args.output_dir,
            overwrite_audio=args.overwrite_audio,
            sample_limit=args.max_samples,
            cache_dir=safe_cache_dir,
        )
        print(
            f"   ✓ Saved {processed} samples to {args.output_dir}/audio/{request.local_name}/"
            f" (metadata: AudioCaps_{request.local_name}.json)",
            flush=True,
        )
        if skipped:
            print(f"   ⚠️  Skipped {skipped} samples without audio payload", flush=True)


if __name__ == "__main__":  # pragma: no cover - script entry point guard
    main()
