"""Dataset helpers and dataloaders for SAFE training."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "create_safe_dataloader",
    "_collate_multimodal_batch",
    "AudioCapsDataset",
    "VQADataset",
    "AVQADataset",
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

    def __init__(self, data_path: str | Path, split: str = "train"):
        self.data_path = Path(data_path)
        self.split = split

        dataset_dir = self.data_path / self.dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Expected directory {dataset_dir} for {self.dataset_name} dataset"
            )

        candidates = [
            dataset_dir / f"{self.file_stem}_{split}.jsonl",
            dataset_dir / f"{self.file_stem}_{split}.json",
            dataset_dir / f"{split}.jsonl",
            dataset_dir / f"{split}.json",
        ]

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
        audio_path = entry.get("audio") or entry.get("audio_path")
        if not audio_path:
            return None
        audio_file = self.data_path / audio_path
        if audio_file.exists():
            try:
                import torchaudio

                waveform, _ = torchaudio.load(audio_file)
                return waveform.mean(dim=0)
            except Exception:
                return None
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
            import sys
            print(f"[DatasetDebug] First sample loaded:", file=sys.stderr, flush=True)
            print(f"  Entry keys: {list(entry.keys())}", file=sys.stderr, flush=True)
            print(f"  Question: '{entry.get('question', '')[:50]}'", file=sys.stderr, flush=True)
            print(f"  Answer: '{answer_value}'", file=sys.stderr, flush=True)
            print(f"  Audio path: '{entry.get('audio_path', entry.get('audio', 'N/A'))}'", file=sys.stderr, flush=True)

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

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        entry = self.examples[idx]
        question = entry.get("question") or "What is happening in the audio?"
        answers = entry.get("answers") or entry.get("caption") or entry.get("captions")
        sample = {
            "sample_id": entry.get("id") or entry.get("ytid"),
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
