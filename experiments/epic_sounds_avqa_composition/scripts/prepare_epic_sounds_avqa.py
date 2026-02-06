#!/usr/bin/env python3
"""
Prepare EPIC-SOUNDS + EPIC-KITCHENS AV-QA manifests.

Pipeline:
1. Load EPIC-SOUNDS train/validation annotations.
2. Load EPIC-KITCHENS-100 action annotations (+ noun/verb class maps).
3. Match each sound event to an overlapping action in the same video.
4. Build QA samples for audio-only, vision-only, and AV-composition questions.
5. Optionally extract audio clips + RGB keyframes with ffmpeg.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_hhmmss(ts: str) -> float:
    """Parse EPIC timestamp (HH:MM:SS.sss) -> seconds."""
    parts = ts.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp: {ts}")
    h = int(parts[0])
    m = int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def slug(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


@dataclass
class SoundEvent:
    split: str
    video_id: str
    start_sec: float
    end_sec: float
    sound: str


@dataclass
class ActionEvent:
    video_id: str
    start_sec: float
    end_sec: float
    narration: str
    verb: str
    noun: str


@dataclass
class QASample:
    sample_id: str
    split: str
    question_type: str
    video_id: str
    question: str
    answer: str
    sound: str
    noun: str
    verb: str
    narration: str
    clip_start_sec: float
    clip_end_sec: float
    frame_time_sec: float
    audio_path: str
    image_path: str


QUESTION_TEMPLATES = {
    "audio_event": "What sound do you hear?",
    "vision_object": "Which object is being handled?",
    "av_composition": "What is happening to the {noun}?",
}


def load_class_map(csv_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not csv_path.exists():
        return out
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = str(row.get("id", "")).strip()
            key = (row.get("key") or "").strip().lower()
            if idx and key:
                out[idx] = key
    return out


def load_actions(actions_csv: Path, noun_map: Dict[str, str], verb_map: Dict[str, str]) -> Dict[str, List[ActionEvent]]:
    actions_by_video: Dict[str, List[ActionEvent]] = defaultdict(list)

    with actions_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = (row.get("video_id") or "").strip()
            if not video_id:
                continue

            start_ts = row.get("start_timestamp") or row.get("start_time")
            end_ts = row.get("stop_timestamp") or row.get("end_timestamp") or row.get("end_time")
            if not start_ts or not end_ts:
                continue

            narration = normalize_text(row.get("narration") or row.get("narration_text") or "")

            verb = normalize_text(row.get("verb") or "")
            noun = normalize_text(row.get("noun") or "")

            if not verb:
                verb_cls = str(row.get("verb_class", "")).strip()
                if verb_cls in verb_map:
                    verb = verb_map[verb_cls]
            if not noun:
                noun_cls = str(row.get("noun_class", "")).strip()
                if noun_cls in noun_map:
                    noun = noun_map[noun_cls]

            if not noun and row.get("all_nouns"):
                noun = normalize_text(str(row["all_nouns"]).split(",")[0])
            if not verb and row.get("all_verbs"):
                verb = normalize_text(str(row["all_verbs"]).split(",")[0])

            if not narration and noun and verb:
                narration = f"{verb} {noun}"

            try:
                start_sec = parse_hhmmss(str(start_ts))
                end_sec = parse_hhmmss(str(end_ts))
            except ValueError:
                continue

            actions_by_video[video_id].append(
                ActionEvent(
                    video_id=video_id,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    narration=narration,
                    verb=verb,
                    noun=noun,
                )
            )

    for video_id in actions_by_video:
        actions_by_video[video_id].sort(key=lambda x: x.start_sec)

    return actions_by_video


def load_sounds(csv_path: Path, split: str) -> List[SoundEvent]:
    events: List[SoundEvent] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = (row.get("video_id") or "").strip()
            start_ts = row.get("start_timestamp")
            end_ts = row.get("stop_timestamp")
            sound = normalize_text(row.get("class") or row.get("sound") or "")
            if not (video_id and start_ts and end_ts and sound):
                continue
            try:
                start_sec = parse_hhmmss(str(start_ts))
                end_sec = parse_hhmmss(str(end_ts))
            except ValueError:
                continue
            events.append(
                SoundEvent(
                    split=split,
                    video_id=video_id,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    sound=sound,
                )
            )
    return events


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def find_best_action(sound: SoundEvent, actions: Sequence[ActionEvent]) -> Optional[ActionEvent]:
    if not actions:
        return None

    best: Optional[ActionEvent] = None
    best_score = -1.0
    midpoint = 0.5 * (sound.start_sec + sound.end_sec)

    for action in actions:
        ov = overlap(sound.start_sec, sound.end_sec, action.start_sec, action.end_sec)
        dist = abs(midpoint - 0.5 * (action.start_sec + action.end_sec))
        score = ov * 1000.0 - dist
        if score > best_score:
            best_score = score
            best = action

    return best


def find_video_file(video_root: Path, video_id: str) -> Optional[Path]:
    direct = video_root / f"{video_id}.MP4"
    if direct.exists():
        return direct
    direct = video_root / f"{video_id}.mp4"
    if direct.exists():
        return direct

    matches = list(video_root.rglob(f"{video_id}.MP4")) + list(video_root.rglob(f"{video_id}.mp4"))
    if not matches:
        return None
    return matches[0]


def ffmpeg_extract_audio(video_path: Path, start: float, end: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "48000",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ffmpeg_extract_frame(video_path: Path, time_sec: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{time_sec:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_samples(
    sounds: Sequence[SoundEvent],
    actions_by_video: Dict[str, List[ActionEvent]],
    clip_duration: float,
) -> List[QASample]:
    out: List[QASample] = []

    for idx, sound in enumerate(sounds):
        action = find_best_action(sound, actions_by_video.get(sound.video_id, []))

        midpoint = 0.5 * (sound.start_sec + sound.end_sec)
        half = clip_duration / 2.0
        clip_start = max(0.0, midpoint - half)
        clip_end = max(clip_start + 0.2, midpoint + half)

        noun = action.noun if action else ""
        verb = action.verb if action else ""
        narration = action.narration if action else ""

        sid_prefix = f"{sound.split}_{sound.video_id}_{int(sound.start_sec*1000)}_{idx}"

        # Audio question always available
        out.append(
            QASample(
                sample_id=f"{sid_prefix}_qa_audio",
                split=sound.split,
                question_type="audio_event",
                video_id=sound.video_id,
                question=QUESTION_TEMPLATES["audio_event"],
                answer=sound.sound,
                sound=sound.sound,
                noun=noun,
                verb=verb,
                narration=narration,
                clip_start_sec=clip_start,
                clip_end_sec=clip_end,
                frame_time_sec=midpoint,
                audio_path="",
                image_path="",
            )
        )

        # Vision question requires noun
        if noun:
            out.append(
                QASample(
                    sample_id=f"{sid_prefix}_qa_vision",
                    split=sound.split,
                    question_type="vision_object",
                    video_id=sound.video_id,
                    question=QUESTION_TEMPLATES["vision_object"],
                    answer=noun,
                    sound=sound.sound,
                    noun=noun,
                    verb=verb,
                    narration=narration,
                    clip_start_sec=clip_start,
                    clip_end_sec=clip_end,
                    frame_time_sec=midpoint,
                    audio_path="",
                    image_path="",
                )
            )

        # Composition question requires noun+verb
        if noun and verb:
            av_answer = f"{noun} is being {verb}"
            out.append(
                QASample(
                    sample_id=f"{sid_prefix}_qa_av",
                    split=sound.split,
                    question_type="av_composition",
                    video_id=sound.video_id,
                    question=QUESTION_TEMPLATES["av_composition"].format(noun=noun),
                    answer=av_answer,
                    sound=sound.sound,
                    noun=noun,
                    verb=verb,
                    narration=narration,
                    clip_start_sec=clip_start,
                    clip_end_sec=clip_end,
                    frame_time_sec=midpoint,
                    audio_path="",
                    image_path="",
                )
            )

    return out


def maybe_extract_media(
    samples: List[QASample],
    video_root: Path,
    media_root: Path,
    skip_existing: bool,
    dry_run: bool,
) -> List[QASample]:
    updated: List[QASample] = []

    video_cache: Dict[str, Optional[Path]] = {}

    for sample in samples:
        if sample.video_id not in video_cache:
            video_cache[sample.video_id] = find_video_file(video_root, sample.video_id)

        video_file = video_cache[sample.video_id]
        if video_file is None:
            continue

        clip_tag = f"{sample.video_id}_{int(sample.clip_start_sec*1000)}_{int(sample.clip_end_sec*1000)}"
        audio_rel = Path("audio") / sample.split / f"{clip_tag}.wav"
        image_rel = Path("frames") / sample.split / f"{clip_tag}.jpg"

        audio_out = media_root / audio_rel
        image_out = media_root / image_rel

        if not dry_run:
            if not (skip_existing and audio_out.exists()):
                ffmpeg_extract_audio(video_file, sample.clip_start_sec, sample.clip_end_sec, audio_out)
            if not (skip_existing and image_out.exists()):
                ffmpeg_extract_frame(video_file, sample.frame_time_sec, image_out)

        sample.audio_path = str(audio_rel)
        sample.image_path = str(image_rel)
        updated.append(sample)

    return updated


def write_manifest(samples: Sequence[QASample], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.__dict__) + "\n")


def write_summary(samples: Sequence[QASample], out_path: Path) -> None:
    counts: Dict[str, int] = defaultdict(int)
    split_counts: Dict[str, int] = defaultdict(int)

    for s in samples:
        counts[s.question_type] += 1
        split_counts[s.split] += 1

    summary = {
        "num_samples": len(samples),
        "question_type_counts": dict(counts),
        "split_counts": dict(split_counts),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare EPIC-SOUNDS AV-QA manifests")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("experiments/epic_sounds_avqa_composition/data"),
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("experiments/epic_sounds_avqa_composition/data/raw_videos"),
        help="Root containing downloaded EPIC videos",
    )
    parser.add_argument("--clip-duration", type=float, default=4.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--extract-media", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ann_root = args.data_root / "annotations"
    manifest_root = args.data_root / "manifests"
    media_root = args.data_root / "processed"

    sounds_train = load_sounds(ann_root / "EPIC_Sounds_train.csv", split="train")
    sounds_val = load_sounds(ann_root / "EPIC_Sounds_validation.csv", split="validation")

    noun_map = load_class_map(ann_root / "EPIC_100_noun_classes.csv")
    verb_map = load_class_map(ann_root / "EPIC_100_verb_classes.csv")

    actions_train = load_actions(ann_root / "EPIC_100_train.csv", noun_map=noun_map, verb_map=verb_map)
    actions_val = load_actions(ann_root / "EPIC_100_validation.csv", noun_map=noun_map, verb_map=verb_map)

    train_samples = build_samples(sounds_train, actions_by_video=actions_train, clip_duration=args.clip_duration)
    val_samples = build_samples(sounds_val, actions_by_video=actions_val, clip_duration=args.clip_duration)

    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_samples = val_samples[: args.max_val_samples]

    all_samples = train_samples + val_samples

    if args.extract_media:
        all_samples = maybe_extract_media(
            all_samples,
            video_root=args.video_root,
            media_root=media_root,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )

    train_out = [s for s in all_samples if s.split == "train"]
    val_out = [s for s in all_samples if s.split == "validation"]

    write_manifest(train_out, manifest_root / "train.jsonl")
    write_manifest(val_out, manifest_root / "validation.jsonl")
    write_summary(all_samples, manifest_root / "summary.json")

    print(f"[done] train samples: {len(train_out)}")
    print(f"[done] validation samples: {len(val_out)}")
    print(f"[done] manifests: {manifest_root}")


if __name__ == "__main__":
    main()
