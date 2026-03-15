#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


SOURCE_ALIASES: Dict[str, Sequence[str]] = {
    "accordion": ("accordion",),
    "acoustic_guitar": ("acoustic guitar", "guitar", "acoustic_guitar"),
    "bagpipe": ("bagpipe", "bagpipes"),
    "banjo": ("banjo",),
    "bassoon": ("bassoon",),
    "cello": ("cello",),
    "clarinet": ("clarinet",),
    "congas": ("conga", "congas"),
    "drum": ("drum", "drums", "drumming"),
    "electric_bass": ("electric bass", "bass guitar", "electric_bass"),
    "erhu": ("erhu",),
    "flute": ("flute",),
    "guzheng": ("guzheng",),
    "piano": ("piano",),
    "pipa": ("pipa",),
    "saxophone": ("saxophone", "sax"),
    "trumpet": ("trumpet",),
    "tuba": ("tuba",),
    "ukulele": ("ukulele", "ukelele"),
    "violin": ("violin",),
    "xylophone": ("xylophone",),
    "voice": ("voice", "vocal", "vocals", "vocalist", "singer", "singing"),
    "male_voice": ("male vocalist", "male vocal", "male singer", "male voice"),
    "female_voice": ("female vocalist", "female vocal", "female singer", "female voice"),
    "orchestra": ("orchestra", "orchestral", "ensemble"),
    "strings": ("string orchestra", "strings", "string section"),
    "percussion": ("percussion", "percussive"),
}

COUNT_ALIASES: Dict[str, Sequence[str]] = {
    "zero": ("zero", "none", "no", "absent"),
    "one": ("one", "single", "solo", "only"),
    "two": ("two", "duo", "pair"),
    "three": ("three", "trio"),
    "four": ("four", "quartet"),
    "five": ("five",),
    "many": ("many", "multiple", "several", "ensemble"),
}

TEMPO_ALIASES: Dict[str, Sequence[str]] = {
    "fast": ("fast", "rapid", "quick", "lively", "brisk", "up-tempo", "paced"),
    "slow": ("slow", "slower", "gentle", "lyrical", "adagio"),
    "moderate": ("moderate", "moderately", "medium"),
}

LOUDNESS_ALIASES: Dict[str, Sequence[str]] = {
    "loud": ("loud", "strong", "energetic", "dynamic"),
    "soft": ("soft", "quiet", "gentle", "faint"),
    "moderate": ("moderate", "mid-level", "balanced"),
}

TEXTURE_ALIASES: Dict[str, Sequence[str]] = {
    "solo": ("solo", "alone", "only source", "single source"),
    "lead": ("lead", "foreground", "prominent", "dominant", "featured", "most prominent"),
    "accompaniment": ("accompaniment", "accompanied", "support", "backing"),
    "background": ("background", "behind", "faint background"),
    "duet": ("duet",),
    "ensemble": ("ensemble", "band", "orchestra", "full band"),
}

STATE_ALIASES: Dict[str, Sequence[str]] = {
    "continuous": ("continuous", "throughout", "consistently", "always"),
    "intermittent": ("intermittent", "occasional", "occasionally"),
    "only_source": ("only active source", "no other instruments", "no other sources", "sole active source"),
    "no_vocals": ("no vocals", "no voice", "instrumental"),
}

CUE_ALIASES: Dict[str, Sequence[str]] = {
    "plucked": ("plucked", "fingerpicked", "picked", "strummed"),
    "bowed": ("bowed", "bowing"),
    "struck": ("struck", "hit", "hammered"),
    "sustained": ("sustained", "legato", "held"),
    "percussive": ("percussive", "rhythmic", "syncopated"),
    "vibrato": ("vibrato",),
    "reverberant": ("reverberant", "reverb", "echoing", "echo", "concert hall"),
}

ROLE_SPLIT_PATTERNS: Sequence[Tuple[str, str]] = (
    ("foreground", "background"),
    ("lead", "accompaniment"),
    ("prominent", "support"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build richer simple and structured semantic targets from teacher captions.")
    parser.add_argument("--input-manifest", type=str, required=True)
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--raw-caption-field", type=str, default="rich_audio_caption")
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", _normalize_text(text))


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase in text


def _ordered_unique(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _extract_aliases(text: str, alias_map: Dict[str, Sequence[str]]) -> List[str]:
    found: List[Tuple[int, str]] = []
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            idx = text.find(alias)
            if idx >= 0:
                found.append((idx, canonical))
                break
    found.sort(key=lambda x: x[0])
    return _ordered_unique([canonical for _, canonical in found])


def _pick_primary(items: Sequence[str], fallback: str = "unknown") -> str:
    return items[0] if items else fallback


def _pick_secondary(items: Sequence[str]) -> str:
    return items[1] if len(items) > 1 else "none"


def _extract_count(text: str, sources: Sequence[str], textures: Sequence[str]) -> str:
    counts = _extract_aliases(text, COUNT_ALIASES)
    if counts:
        if counts[0] == "one" and len(sources) > 1:
            if len(sources) == 2:
                return "two"
            if len(sources) == 3:
                return "three"
            return "many"
        return counts[0]
    if len(sources) >= 3:
        return "many"
    if len(sources) == 2:
        return "two"
    if len(sources) == 1:
        return "one"
    if "ensemble" in textures:
        return "many"
    return "unknown"


def _extract_event(text: str, sources: Sequence[str]) -> str:
    if any(x in text for x in ("sings", "singing", "singer", "vocal", "voice")):
        return "singing"
    if any(x in text for x in ("plays", "playing", "performance", "performed")):
        return "playing"
    if "clapping" in text:
        return "clapping"
    if sources:
        return "playing"
    return "unknown"


def _extract_roles(text: str, sources: Sequence[str]) -> Tuple[str, str]:
    prominent = "unknown"
    background = "unknown"
    for left_word, right_word in ROLE_SPLIT_PATTERNS:
        if left_word in text and right_word in text:
            left_idx = text.find(left_word)
            right_idx = text.find(right_word)
            ordered = list(sources)
            if ordered:
                prominent = ordered[0]
                if len(ordered) > 1:
                    background = ordered[1]
            if left_idx > right_idx:
                prominent, background = background, prominent
            return prominent, background
    if sources:
        if any(x in text for x in ("prominent", "foreground", "lead", "featured", "most prominent")):
            prominent = sources[0]
        if len(sources) > 1 and any(x in text for x in ("background", "accompaniment", "support", "backing")):
            background = sources[1]
    return prominent, background


def _extract_binary_voice(text: str, sources: Sequence[str], states: Sequence[str]) -> str:
    if any(src in {"voice", "male_voice", "female_voice"} for src in sources):
        return "yes"
    if "no_vocals" in states:
        return "no"
    return "unknown"


def _compress_source_list(sources: Sequence[str]) -> str:
    if not sources:
        return "unknown"
    return ",".join(sources[:4])


def analyze_caption(text: str) -> Dict[str, Any]:
    norm = _normalize_text(text)
    sources = _extract_aliases(norm, SOURCE_ALIASES)
    tempos = _extract_aliases(norm, TEMPO_ALIASES)
    loudness = _extract_aliases(norm, LOUDNESS_ALIASES)
    textures = _extract_aliases(norm, TEXTURE_ALIASES)
    states = _extract_aliases(norm, STATE_ALIASES)
    cues = _extract_aliases(norm, CUE_ALIASES)
    count = _extract_count(norm, sources, textures)
    event = _extract_event(norm, sources)
    prominent, background = _extract_roles(norm, sources)
    voice = _extract_binary_voice(norm, sources, states)

    if "male_voice" in sources and "voice" not in sources:
        sources = ["male_voice"] + [x for x in sources if x != "male_voice"]
    if "female_voice" in sources and "voice" not in sources:
        sources = ["female_voice"] + [x for x in sources if x != "female_voice"]

    if "only_source" in states and count == "unknown":
        count = "one"

    return {
        "sources": sources,
        "primary_source": _pick_primary(sources),
        "secondary_source": _pick_secondary(sources),
        "event": event,
        "count": count,
        "tempo": tempos[0] if tempos else "unknown",
        "loudness": loudness[0] if loudness else "unknown",
        "texture": textures[0] if textures else "unknown",
        "prominent_source": prominent,
        "background_source": background,
        "voice_present": voice,
        "states": states[:3],
        "cues": cues[:3],
    }


def build_simple_caption(text: str) -> str:
    info = analyze_caption(text)
    parts = [
        info["primary_source"],
        info["secondary_source"] if info["secondary_source"] != "none" else "",
        info["event"],
        info["count"],
        info["tempo"],
        info["loudness"],
        info["texture"],
    ]
    parts.extend(info["cues"][:2])
    compact = [x for x in parts if x and x != "unknown"]
    if not compact:
        compact = _tokenize(text)[:10]
    return " ".join(compact).strip()


def build_structured_caption(text: str) -> str:
    info = analyze_caption(text)
    segments = [
        f"sources {_compress_source_list(info['sources'])}",
        f"primary {info['primary_source']}",
        f"secondary {info['secondary_source']}",
        f"event {info['event']}",
        f"count {info['count']}",
        f"tempo {info['tempo']}",
        f"loudness {info['loudness']}",
        f"texture {info['texture']}",
        f"prominent {info['prominent_source']}",
        f"background {info['background_source']}",
        f"voice {info['voice_present']}",
    ]
    if info["states"]:
        segments.append(f"state {','.join(info['states'])}")
    if info["cues"]:
        segments.append(f"cues {','.join(info['cues'])}")
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
            info = analyze_caption(raw)
            row["teacher_caption_simple"] = build_simple_caption(raw)
            row["teacher_caption_structured"] = build_structured_caption(raw)
            row["teacher_caption_semantics"] = info
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
