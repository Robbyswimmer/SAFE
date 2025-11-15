"""Dataset helpers and dataloaders for SAFE training."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "create_safe_dataloader",
    "_collate_multimodal_batch",
    "AudioCapsDataset",
    "VQADataset",
    "AVQADataset",
    "WavCapsDataset",
]


# ---------------------------------------------------------------------------
# Collation utilities
# ---------------------------------------------------------------------------

def _to_tensor(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        try:
            return torch.as_tensor(value)
        except Exception:
            return None
    return None


def _collate_multimodal_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {
            "questions": [],
            "answers": [],
            "images": None,
            "audio": None,
            "has_audio": torch.zeros(0, dtype=torch.bool),
        }

    questions: List[str] = []
    answers: List[Any] = []
    images: List[Any] = []
    audios: List[Any] = []
    has_audio: List[bool] = []

    for sample in batch:
        questions.append(sample.get("question") or sample.get("questions") or "")
        answers.append(sample.get("answers") or sample.get("answer"))

        image = sample.get("images")
        if image is None:
            image = sample.get("image")
        images.append(image)

        audio = sample.get("audio")
        audios.append(audio)
        has_audio.append(audio is not None)

    collated: Dict[str, Any] = {
        "questions": questions,
        "answers": answers,
        "images": images,
        "audio": audios,
        "has_audio": torch.tensor(has_audio, dtype=torch.bool),
    }

    optional_keys = ["sample_id", "difficulty", "question_type"]
    for key in optional_keys:
        values = [sample.get(key) for sample in batch]
        if any(v is not None for v in values):
            collated[key + "s"] = values

    return collated


# ---------------------------------------------------------------------------
# Real dataset placeholders
# ---------------------------------------------------------------------------

class _BaseQADataset(Dataset):
    """Lightweight dataset wrapper around JSON/JSONL files."""

    dataset_name: str = "generic"
    file_stem: str = "data"

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        preferred_file: Optional[Path] = None,
    ):
        self.data_path = Path(data_path).expanduser().resolve()
        self.split = split

        dataset_dir = (self.data_path / self.dataset_name).expanduser().resolve()
        self.dataset_dir = dataset_dir
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Expected directory {dataset_dir} for {self.dataset_name} dataset"
            )

        candidate_paths: List[Path] = []
        if preferred_file is not None:
            candidate_paths.append(Path(preferred_file))

        default_candidates = [
            dataset_dir / f"{self.file_stem}_{split}.jsonl",
            dataset_dir / f"{self.file_stem}_{split}.json",
            dataset_dir / f"{split}.jsonl",
            dataset_dir / f"{split}.json",
        ]
        candidate_paths.extend(default_candidates)

        # Remove duplicates while preserving order
        seen: set[Path] = set()
        candidates: List[Path] = []
        for path in candidate_paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            candidates.append(path)
            seen.add(resolved)

        data_file = next((p for p in candidates if p.exists()), None)
        if data_file is None:
            raise FileNotFoundError(
                f"Could not find data file for {self.dataset_name} split '{split}'. "
                f"Looked for: {', '.join(str(p) for p in candidates)}"
            )

        self.examples: List[Dict[str, Any]] = []
        if data_file.suffix == ".jsonl":
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.examples.append(json.loads(line))
        else:
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
                if not isinstance(data, list):
                    raise ValueError(f"Unexpected format for {data_file}")
                self.examples = data

        if not self.examples:
            raise ValueError(f"No examples found in {data_file}")

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.examples)

    # ------------------------------------------------------------------
    def _extract_answer(self, raw_answer: Any) -> Any:
        if raw_answer is None:
            return ""
        if isinstance(raw_answer, dict) and "answer" in raw_answer:
            return raw_answer["answer"]
        if isinstance(raw_answer, list):
            if not raw_answer:
                return ""
            if isinstance(raw_answer[0], dict):
                counts = Counter(
                    item.get("answer", "") for item in raw_answer if item.get("answer")
                )
                if counts:
                    return counts.most_common(1)[0][0]
                return ""
            counts = Counter(str(a) for a in raw_answer)
            return counts.most_common(1)[0][0]
        return raw_answer

    # ------------------------------------------------------------------
    def _load_audio(self, entry: Dict[str, Any]) -> Any:
        """
        Load audio from file (supports WAV, FLAC, MP3, OGG, etc.).

        Args:
            entry: Metadata entry containing 'audio_path' or 'audio' field

        Returns:
            Tuple of (waveform, sample_rate) or None if loading fails
        """
        raw_audio_path = entry.get("audio") or entry.get("audio_path") or entry.get("file_path")
        split_name = entry.get("split") or self.split
        sound_name = entry.get("sound_name") or entry.get("ytid") or entry.get("id")

        candidate_paths: List[Path] = []

        def _add_candidate(path_value: Any) -> None:
            if not path_value:
                return
            candidate = Path(path_value).expanduser()
            if not candidate.is_absolute():
                candidate = self.dataset_dir / candidate
            candidate = candidate.resolve(strict=False)
            if candidate not in candidate_paths:
                candidate_paths.append(candidate)

        _add_candidate(raw_audio_path)

        if sound_name and split_name:
            base_candidate = Path("audio") / str(split_name) / sound_name
            _add_candidate(base_candidate)

            base = Path(sound_name)
            suffix = base.suffix or ".wav"
            stem = base.stem

            import re

            trimmed_stem = re.sub(r"_(\d+)$", "", stem)
            if trimmed_stem != stem:
                _add_candidate(Path("audio") / str(split_name) / f"{trimmed_stem}{suffix}")
                if suffix.lower() != ".wav":
                    _add_candidate(Path("audio") / str(split_name) / f"{trimmed_stem}.wav")
            elif suffix.lower() != ".wav":
                _add_candidate(Path("audio") / str(split_name) / f"{stem}.wav")

        audio_file = next((candidate for candidate in candidate_paths if candidate and candidate.exists()), None)

        if audio_file is None:
            # Only log first few missing files to avoid spam
            if not hasattr(self, '_missing_audio_count'):
                self._missing_audio_count = 0

            if self._missing_audio_count < 5:
                missing_label = raw_audio_path or sound_name or "<unknown>"
                print(f"[Dataset] Warning: Audio file not found: {missing_label}", flush=True)
                self._missing_audio_count += 1
            elif self._missing_audio_count == 5:
                print(f"[Dataset] Warning: Additional missing audio files will not be logged", flush=True)
                self._missing_audio_count += 1

            return None

        try:
            import torchaudio

            # torchaudio.load() supports WAV, FLAC, MP3, OGG, etc.
            waveform, sample_rate = torchaudio.load(str(audio_file))

            # Convert stereo to mono
            if waveform.dim() == 2 and waveform.size(0) > 1:
                waveform = waveform.mean(dim=0)
            elif waveform.dim() == 2:
                waveform = waveform.squeeze(0)

            # Resample to target sample rate
            target_sample_rate = 48_000
            if sample_rate != target_sample_rate:
                try:
                    waveform = torchaudio.functional.resample(
                        waveform.unsqueeze(0), sample_rate, target_sample_rate
                    ).squeeze(0)
                except Exception:
                    # Fallback to torch interpolation
                    import torch
                    ratio = target_sample_rate / float(sample_rate)
                    num_samples = int(waveform.size(-1) * ratio)
                    waveform = torch.nn.functional.interpolate(
                        waveform.unsqueeze(0).unsqueeze(0),
                        size=num_samples,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)

            return (waveform, target_sample_rate)

        except Exception as e:
            if isinstance(e, TypeError) and "get_src_stream_info" in str(e):
                # torchaudio <-> torio backend sometimes throws TypeError when probing malformed files
                # Fall back to librosa (pure Python) so that a few bad headers don't drop the sample entirely
                try:
                    import librosa
                    import torch

                    target_sample_rate = 48_000
                    audio_np, sample_rate = librosa.load(
                        str(audio_file), sr=target_sample_rate, mono=True
                    )
                    waveform = torch.from_numpy(audio_np).float()
                    return (waveform, target_sample_rate)
                except Exception as inner_exc:
                    e = inner_exc

            # Log first few load failures
            if not hasattr(self, '_load_error_count'):
                self._load_error_count = 0

            if self._load_error_count < 3:
                print(f"[Dataset] Error loading audio {audio_path}: {e}", flush=True)
                self._load_error_count += 1
            elif self._load_error_count == 3:
                print(f"[Dataset] Additional audio load errors will not be logged", flush=True)
                self._load_error_count += 1

            return None

    # ------------------------------------------------------------------
    def _load_image(self, entry: Dict[str, Any]) -> Any:
        image_path = entry.get("image") or entry.get("image_path")
        if not image_path:
            return None
        file_path = self.data_path / image_path
        if not file_path.exists():
            return None
        try:
            from PIL import Image

            image = Image.open(file_path).convert("RGB")
            return image
        except Exception:
            return None

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        entry = self.examples[idx]
        answer_value = entry.get("answers") or entry.get("answer")

        # Debug: Log first sample to verify data loading
        if idx == 0:
            print(f"[DatasetDebug] First sample loaded:", flush=True)
            print(f"  Entry keys: {list(entry.keys())}", flush=True)
            print(f"  Question: '{entry.get('question', '')[:50]}'", flush=True)
            print(f"  Answer: '{answer_value}'", flush=True)
            print(f"  Audio path: '{entry.get('audio_path', entry.get('audio', 'N/A'))}'", flush=True)

        sample = {
            "sample_id": entry.get("id") or entry.get("sample_id"),
            "question": entry.get("question") or entry.get("prompt") or "",
            "answers": answer_value,
            "audio": self._load_audio(entry),
            "images": self._load_image(entry),
            "difficulty": entry.get("difficulty"),
        }
        return sample


class AudioCapsDataset(_BaseQADataset):
    dataset_name = "audiocaps"
    file_stem = "audiocaps"

    def __init__(self, data_path: str | Path, split: str = "train"):
        data_root = Path(data_path)
        dataset_dir = data_root / self.dataset_name
        preferred_file: Optional[Path] = None

        # For evaluation (val split), always prefer the curated multi-caption JSON
        if split.lower().startswith("val"):
            hardcoded = dataset_dir / "audiocaps_val.json"
            if hardcoded.exists():
                preferred_file = hardcoded

        super().__init__(data_path=data_path, split=split, preferred_file=preferred_file)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        entry = self.examples[idx]
        question = entry.get("question") or "What is happening in the audio?"

        # Try multiple field names for answers (datasets use different conventions)
        captions = entry.get("captions")
        if isinstance(captions, (list, tuple)):
            answers = [str(cap).strip() for cap in captions if str(cap).strip()]
        else:
            answers = (
                entry.get("answers") or      # Plural form
                entry.get("answer") or        # Singular form (what full_training data uses)
                entry.get("caption")          # AudioCaps single caption field
            )

        # Debug: Log first few samples to verify answer loading
        if idx < 3:
            print(f"[AudioCapsDebug] Sample {idx}:", flush=True)
            print(f"  Entry keys: {list(entry.keys())}", flush=True)
            print(f"  'answer' field: {entry.get('answer')}", flush=True)
            print(f"  Final answers value: {answers}", flush=True)

        sample = {
            "sample_id": entry.get("id") or entry.get("ytid") or entry.get("sound_name"),
            "question": question,
            "answers": answers,
            "audio": self._load_audio(entry),
            "images": self._load_image(entry),
            "difficulty": entry.get("difficulty"),
        }
        return sample


class VQADataset(_BaseQADataset):
    dataset_name = "vqa"
    file_stem = "vqa"

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        entry = self.examples[idx]
        sample = {
            "sample_id": entry.get("question_id") or entry.get("id"),
            "question": entry.get("question") or entry.get("question_text") or "",
            "answers": entry.get("answers") or entry.get("answer"),
            "images": self._load_image(entry),
            "audio": None,  # VQA does not contain audio
            "difficulty": entry.get("difficulty"),
        }
        return sample


class AVQADataset(_BaseQADataset):
    dataset_name = "avqa"
    file_stem = "avqa"

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        entry = self.examples[idx]
        sample = {
            "sample_id": entry.get("id") or entry.get("sample_id"),
            "question": entry.get("question") or "",
            "answers": entry.get("answers") or entry.get("answer"),
            "audio": self._load_audio(entry),
            "images": self._load_image(entry),
            "difficulty": entry.get("difficulty"),
        }
        return sample


class WavCapsDataset(_BaseQADataset):
    dataset_name = "wavcaps"
    file_stem = "wavcaps"

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        entry = self.examples[idx]

        # WavCaps uses standardized format from download script
        question = entry.get("question") or "What is happening in the audio?"
        answers = entry.get("answer") or entry.get("answers") or entry.get("caption")

        sample = {
            "sample_id": entry.get("id"),
            "question": question,
            "answers": answers,
            "audio": self._load_audio(entry),
            "images": None,  # WavCaps is audio-only
            "subset": entry.get("subset"),  # Track which subset (FreeSound, BBC, etc.)
        }
        return sample


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def create_safe_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader with SAFE's multimodal collate function."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_multimodal_batch,
    )
