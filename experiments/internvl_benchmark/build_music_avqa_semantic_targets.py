#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


SOURCE_WORDS = [
    "piano", "guitar", "drum", "drums", "violin", "trumpet", "flute", "cello",
    "engine", "car", "vehicle", "motorcycle", "speech", "man", "woman", "crowd",
    "audience", "applause", "clapping", "cheering", "music", "singing", "voice",
]
EVENT_WORDS = [
    "playing", "played", "strumming", "drumming", "clapping", "cheering", "speaking",
    "talking", "idling", "revving", "sputtering", "laughing", "singing", "applauding",
]
ATTR_WORDS = [
    "fast", "slow", "loud", "quiet", "aggressive", "soft", "rhythmic", "rapid",
    "reverberant", "echoing", "background", "indoor", "outdoor",
]
COUNT_WORDS = ["one", "two", "three", "four", "five", "many"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build simple and structured semantic targets from teacher captions.")
    parser.add_argument("--input-manifest", type=str, required=True)
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--raw-caption-field", type=str, default="rich_audio_caption")
    return parser.parse_args()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _first_match(tokens: List[str], lexicon: List[str]) -> str:
    token_set = set(tokens)
    for word in lexicon:
        if word in token_set:
            return word
    return ""


def build_simple_caption(text: str) -> str:
    tokens = _tokenize(text)
    source = _first_match(tokens, SOURCE_WORDS)
    event = _first_match(tokens, EVENT_WORDS)
    attrs = [w for w in ATTR_WORDS if w in set(tokens)][:2]
    parts = [x for x in [source, event] if x]
    parts.extend(attrs)
    if not parts:
        parts = tokens[:6]
    return " ".join(parts).strip()


def build_structured_caption(text: str) -> str:
    tokens = _tokenize(text)
    source = _first_match(tokens, SOURCE_WORDS) or "unknown"
    event = _first_match(tokens, EVENT_WORDS) or "unknown"
    attrs = [w for w in ATTR_WORDS if w in set(tokens)][:2]
    count = _first_match(tokens, COUNT_WORDS)
    segments = [f"source {source}", f"event {event}"]
    if count:
        segments.append(f"count {count}")
    for attr in attrs:
        segments.append(f"attr {attr}")
    return " ; ".join(segments)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_manifest)
    out_path = Path(args.output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            raw = str(row.get(args.raw_caption_field, "") or "").strip()
            row["teacher_caption_simple"] = build_simple_caption(raw)
            row["teacher_caption_structured"] = build_structured_caption(raw)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input_manifest": str(in_path),
                "output_manifest": str(out_path),
                "raw_caption_field": args.raw_caption_field,
                "rows": len(rows),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
