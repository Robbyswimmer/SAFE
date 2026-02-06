#!/usr/bin/env python3
"""
Prepare standardized JSONL manifests for AVQA and MUSIC-AVQA.

Output schema per line:
  {
    "sample_id": "...",
    "question": "...",
    "answer": "...",
    "question_type": "...",
    "audio_path": "relative/or/absolute/path.wav",
    "image_path": "relative/or/absolute/path.jpg"
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        if "questions" in data and isinstance(data["questions"], list):
            return data["questions"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON format in {path}")


def _first_nonempty(row: Dict[str, Any], keys: Iterable[str]) -> str:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_answer(row: Dict[str, Any]) -> str:
    value = row.get("answer")
    if value is None:
        value = row.get("answers")
    if value is None:
        value = row.get("label")

    if isinstance(value, list):
        if not value:
            return ""
        first = value[0]
        if isinstance(first, dict):
            return str(first.get("answer", "")).strip()
        return str(first).strip()
    if isinstance(value, dict):
        return str(value.get("answer", "")).strip()
    return str(value or "").strip()


def _resolve_path(
    rel_or_abs: str,
    root: Optional[Path],
    fallback_dirs: List[Path],
    exts: Tuple[str, ...],
) -> str:
    if not rel_or_abs and root is None and not fallback_dirs:
        return ""

    candidates: List[Path] = []
    if rel_or_abs:
        p = Path(rel_or_abs).expanduser()
        if p.is_absolute():
            candidates.append(p)
        else:
            if root is not None:
                candidates.append(root / p)
            for d in fallback_dirs:
                candidates.append(d / p)

    for c in candidates:
        if c.exists():
            return str(c)

    if rel_or_abs:
        stem = Path(rel_or_abs).stem
        for d in fallback_dirs:
            for ext in exts:
                c = d / f"{stem}{ext}"
                if c.exists():
                    return str(c)
    return ""


def _video_id(row: Dict[str, Any]) -> str:
    for key in ("video_id", "video", "youtube_id", "vid", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_rows(
    rows: List[Dict[str, Any]],
    *,
    media_root: Optional[Path],
    audio_root: Optional[Path],
    image_root: Optional[Path],
    require_both: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    audio_dirs = [audio_root] if audio_root is not None else []
    image_dirs = [image_root] if image_root is not None else []
    if media_root is not None:
        audio_dirs.extend([media_root / "audio", media_root / "audios"])
        image_dirs.extend([media_root / "images", media_root / "frames"])
    audio_dirs = [p for p in audio_dirs if p is not None]
    image_dirs = [p for p in image_dirs if p is not None]

    for i, row in enumerate(rows):
        q = _first_nonempty(row, ("question", "question_content", "question_text"))
        a = _extract_answer(row)
        if not q or not a:
            continue

        sid = _first_nonempty(row, ("sample_id", "id"))
        vid = _video_id(row)
        if not sid:
            sid = vid or f"sample_{i}"

        raw_audio = _first_nonempty(row, ("audio_path", "audio", "audio_file"))
        raw_image = _first_nonempty(row, ("image_path", "image", "frame_path"))

        if not raw_audio and vid:
            raw_audio = vid
        if not raw_image and vid:
            raw_image = vid

        audio_path = _resolve_path(raw_audio, media_root, audio_dirs, (".wav", ".mp3", ".flac", ".m4a"))
        image_path = _resolve_path(raw_image, media_root, image_dirs, (".jpg", ".jpeg", ".png"))
        qtype = _first_nonempty(row, ("question_type", "type", "task", "category")) or "unknown"

        if require_both and (not audio_path or not image_path):
            continue

        out.append(
            {
                "sample_id": sid,
                "question": q,
                "answer": a,
                "question_type": qtype,
                "audio_path": audio_path,
                "image_path": image_path,
            }
        )
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare AVQA/MUSIC-AVQA manifests")
    p.add_argument("--dataset", type=str, required=True, choices=["avqa", "music_avqa"])
    p.add_argument("--train-json", type=Path, required=True)
    p.add_argument("--val-json", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--media-root", type=Path, default=None)
    p.add_argument("--audio-root", type=Path, default=None)
    p.add_argument("--image-root", type=Path, default=None)
    p.add_argument("--require-both", action="store_true", help="Drop rows missing audio or image.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = _load_rows(args.train_json)
    val_rows = _load_rows(args.val_json)

    train = normalize_rows(
        train_rows,
        media_root=args.media_root,
        audio_root=args.audio_root,
        image_root=args.image_root,
        require_both=args.require_both,
    )
    val = normalize_rows(
        val_rows,
        media_root=args.media_root,
        audio_root=args.audio_root,
        image_root=args.image_root,
        require_both=args.require_both,
    )

    out_dir = args.output_root / "manifests"
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    summary = {
        "dataset": args.dataset,
        "train_in": len(train_rows),
        "train_out": len(train),
        "val_in": len(val_rows),
        "val_out": len(val),
        "train_manifest": str(train_path),
        "val_manifest": str(val_path),
    }
    print("[summary] " + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

