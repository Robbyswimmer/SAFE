"""Dataset helpers and dataloaders for SAFE training."""

from __future__ import annotations

import json
import mmap
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

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
    "AudioSetCapsDataset",
]


# ---------------------------------------------------------------------------
# Memory-efficient JSONL reader using line offsets
# ---------------------------------------------------------------------------

class LazyJSONLReader:
    """
    Memory-efficient JSONL reader that stores only line offsets, not full data.

    Instead of loading all JSON objects into memory (~500 bytes each × 400K = 200MB),
    we store only byte offsets (~8 bytes each × 400K = 3.2MB) and read on demand.

    This reduces memory by ~60x for large datasets like WavCaps.
    """

    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)
        self.offsets: List[int] = []
        self._build_index()
        self._file = None  # Opened lazily

    def _build_index(self):
        """Build index of byte offsets for each line."""
        self.offsets = []
        with open(self.filepath, 'rb') as f:
            offset = 0
            for line in f:
                if line.strip():  # Skip empty lines
                    self.offsets.append(offset)
                offset += len(line)

    def _ensure_file_open(self):
        """Ensure file handle is open (lazy initialization for multiprocessing)."""
        if self._file is None or self._file.closed:
            self._file = open(self.filepath, 'r', encoding='utf-8')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Read and parse a single line on demand."""
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError(f"Index {idx} out of range [0, {len(self.offsets)})")

        self._ensure_file_open()
        self._file.seek(self.offsets[idx])
        line = self._file.readline()

        if not line.strip():
            raise ValueError(f"Empty line at index {idx}, offset {self.offsets[idx]}")

        return json.loads(line)

    def __del__(self):
        if hasattr(self, '_file') and self._file and not self._file.closed:
            self._file.close()

    def __getstate__(self):
        """Support pickling for multiprocessing (close file handle)."""
        state = self.__dict__.copy()
        state['_file'] = None  # Don't pickle file handle
        return state

    def __setstate__(self, state):
        """Restore state - file will be reopened lazily on first access."""
        self.__dict__.update(state)
        self._file = None  # Will be reopened by _ensure_file_open()


class CombinedLazyJSONLReader:
    """
    Memory-efficient reader for multiple JSONL files combined.

    Maps global indices to (file_index, local_index) for random access.
    """

    def __init__(self, filepaths: List[Path]):
        # Store filepaths and offsets, but create readers lazily
        self.filepaths: List[Path] = [Path(p) for p in filepaths]
        self.cumulative_lengths: List[int] = [0]
        self._readers: Optional[List[LazyJSONLReader]] = None

        # Build index without keeping readers in memory yet
        for filepath in self.filepaths:
            reader = LazyJSONLReader(filepath)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(reader))
            # Don't store reader yet - will be created lazily

    def _ensure_readers(self):
        """Lazily create readers (needed after unpickling in worker processes)."""
        if self._readers is None:
            self._readers = [LazyJSONLReader(fp) for fp in self.filepaths]

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Map global index to appropriate file and local index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        self._ensure_readers()

        # Find the right file
        for i, cum_len in enumerate(self.cumulative_lengths[1:], 1):
            if idx < cum_len:
                local_idx = idx - self.cumulative_lengths[i - 1]
                return self._readers[i - 1][local_idx]

        raise IndexError(f"Index {idx} out of range")

    def __getstate__(self):
        """Support pickling - don't pickle readers, just paths and lengths."""
        state = self.__dict__.copy()
        state['_readers'] = None  # Will be recreated lazily
        return state

    def __setstate__(self, state):
        """Restore state - readers will be created lazily on first access."""
        self.__dict__.update(state)
        self._readers = None


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

    optional_keys = ["sample_id", "audio_path", "subset", "difficulty", "question_type"]
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

        self._data_file = data_file
        self._use_lazy_loading = False
        self.examples: Any = []  # Can be List[Dict] or LazyJSONLReader

        if data_file.suffix == ".jsonl":
            # Use lazy loading for JSONL files to save memory
            # This reduces memory from ~200MB to ~3MB for large datasets
            self._use_lazy_loading = True
            self.examples = LazyJSONLReader(data_file)
            print(f"[Dataset] Loaded {len(self.examples)} samples from {data_file.name} (lazy mode)", flush=True)
        else:
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
                if not isinstance(data, list):
                    raise ValueError(f"Unexpected format for {data_file}")
                self.examples = data
            print(f"[Dataset] Loaded {len(self.examples)} samples from {data_file.name} (eager mode)", flush=True)

        if len(self.examples) == 0:
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
    def _resolve_audio_file(self, entry: Dict[str, Any]) -> Optional[Path]:
        """
        Resolve an entry's audio to an existing local file path without decoding.
        Mirrors the search logic used in _load_audio.
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
                if candidate.parts and candidate.parts[0] == self.dataset_name:
                    candidate = (self.data_path / candidate).resolve(strict=False)
                else:
                    candidate = (self.dataset_dir / candidate).resolve(strict=False)
            else:
                candidate = candidate.resolve(strict=False)
            if candidate not in candidate_paths:
                candidate_paths.append(candidate)

        _add_candidate(raw_audio_path)

        if sound_name and split_name:
            base_candidate_10s = Path("audio") / f"{split_name}_10s" / sound_name
            _add_candidate(base_candidate_10s)

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

        return next((candidate for candidate in candidate_paths if candidate and candidate.exists()), None)

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
        sound_name = entry.get("sound_name") or entry.get("ytid") or entry.get("id")
        audio_file = self._resolve_audio_file(entry)

        if audio_file is None:
            # Single, concise warning per dataset instance to avoid log spam
            if not hasattr(self, "_missing_audio_warned"):
                print(
                    "[AudioLoad] ⚠️ Some audio files are missing; "
                    "those samples will be skipped during training/evaluation.",
                    flush=True,
                )
                self._missing_audio_warned = True
            return None

        try:
            import torchaudio

            # torchaudio.load() supports WAV, FLAC, MP3, OGG, etc.
            waveform, sample_rate = torchaudio.load(str(audio_file))

            # Optional debug logging of successful loads – disabled by default
            if getattr(self, "_debug_audio_loading", False):
                if not hasattr(self, "_load_success_count"):
                    self._load_success_count = 0
                if self._load_success_count < 3:
                    print(
                        f"[AudioLoad] ✓ Loaded: {audio_file.name} "
                        f"(sr={sample_rate}, shape={waveform.shape})",
                        flush=True,
                    )
                    self._load_success_count += 1

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
                error_label = raw_audio_path or sound_name or str(audio_file)
                print(f"[Dataset] Error loading audio {error_label}: {e}", flush=True)
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
        resolved_audio = self._resolve_audio_file(entry)

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
            "audio_path": str(resolved_audio) if resolved_audio is not None else (entry.get("audio_path") or entry.get("audio")),
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
        resolved_audio = self._resolve_audio_file(entry)

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

        # Debug: log first sample to diagnose missing captions issue
        if idx < 3 and not hasattr(self, '_debug_logged'):
            print(f"[AudioCapsDataset Debug] Sample {idx}:", flush=True)
            print(f"  Entry keys: {list(entry.keys())}", flush=True)
            print(f"  'captions' field: {entry.get('captions')}", flush=True)
            print(f"  'answers' field: {entry.get('answers')}", flush=True)
            print(f"  'answer' field: {entry.get('answer')}", flush=True)
            print(f"  'caption' field: {entry.get('caption')}", flush=True)
            print(f"  Final answers: {answers}", flush=True)
            if idx == 2:
                self._debug_logged = True

        # Avoid expensive audio decoding for samples that have no usable captions.
        has_captions = True
        if answers is None:
            has_captions = False
        elif isinstance(answers, str):
            has_captions = bool(answers.strip())
        elif isinstance(answers, (list, tuple)):
            has_captions = any(str(a).strip() for a in answers)
        else:
            has_captions = bool(str(answers).strip())

        sample = {
            "sample_id": entry.get("id") or entry.get("ytid") or entry.get("sound_name"),
            "question": question,
            "answers": answers,
            "audio_path": str(resolved_audio) if resolved_audio is not None else (entry.get("audio_path") or entry.get("audio")),
            "audio": self._load_audio(entry) if has_captions else None,
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
        resolved_audio = self._resolve_audio_file(entry)

        sample = {
            "sample_id": entry.get("id"),
            "question": question,
            "answers": answers,
            "audio_path": str(resolved_audio) if resolved_audio is not None else (entry.get("audio_path") or entry.get("audio")),
            "audio": self._load_audio(entry),
            "images": None,  # WavCaps is audio-only
            "subset": entry.get("subset"),  # Track which subset (FreeSound, BBC, etc.)
        }
        return sample


class AudioSetCapsDataset(_BaseQADataset):
    """
    AudioSetCaps dataset from Google Drive tar archives.

    Expects JSONL files in format:
    {"id": "youtube_id", "question": "What is happening in the audio?", "answer": "caption"}

    Audio files should be in: data/audiosetcaps/audio/train/{youtube_id}.wav
    """
    dataset_name = "audiosetcaps"
    file_stem = "audiosetcaps"

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        sources: Optional[List[str]] = None,
    ):
        """
        Initialize AudioSetCaps dataset.

        Args:
            data_path: Root data directory
            split: Dataset split (default: "train")
            sources: List of sources to include. Options: ["audiosetcaps", "vggsound", "youtube8m"]
                    If None, includes all available sources.
        """
        self.sources = sources or ["audiosetcaps", "vggsound", "youtube8m"]

        # Load each source separately and combine
        self.data_path = Path(data_path).expanduser().resolve()
        dataset_dir = (self.data_path / self.dataset_name).expanduser().resolve()
        self.dataset_dir = dataset_dir

        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Expected directory {dataset_dir} for {self.dataset_name} dataset"
            )

        # Map source names to JSONL file names
        source_files = {
            "audiosetcaps": f"audiosetcaps_{split}.jsonl",
            "vggsound": f"vggsound_audiosetcaps_{split}.jsonl",
            "youtube8m": f"youtube8m_audiosetcaps_{split}.jsonl",
        }

        # Collect valid source files for lazy loading
        valid_files: List[Path] = []
        loaded_sources = []

        for source in self.sources:
            if source not in source_files:
                print(f"[AudioSetCaps] Warning: Unknown source '{source}', skipping", flush=True)
                continue

            jsonl_file = dataset_dir / source_files[source]
            if not jsonl_file.exists():
                print(f"[AudioSetCaps] Warning: {jsonl_file} not found, skipping {source}", flush=True)
                continue

            valid_files.append(jsonl_file)
            loaded_sources.append(source)

        if not valid_files:
            raise ValueError(
                f"No examples found for AudioSetCaps. Looked for sources: {self.sources}\n"
                f"In directory: {dataset_dir}"
            )

        # Use combined lazy reader for memory efficiency
        self.examples: Any = CombinedLazyJSONLReader(valid_files)

        for source, filepath in zip(loaded_sources, valid_files):
            # Count lines without loading (for logging)
            with open(filepath, 'rb') as f:
                line_count = sum(1 for line in f if line.strip())
            print(f"[AudioSetCaps] Indexed {line_count:,} samples from {source} (lazy mode)", flush=True)

        print(f"[AudioSetCaps] Total samples: {len(self.examples):,} from {len(loaded_sources)} source(s)", flush=True)
        self.split = split

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        entry = self.examples[idx]

        # AudioSetCaps uses standardized format from converter script
        question = entry.get("question") or "What is happening in the audio?"
        answers = entry.get("answer") or entry.get("answers")

        sample = {
            "sample_id": entry.get("id"),
            "question": question,
            "answers": answers,
            "audio": self._load_audio(entry),
            "images": None,  # AudioSetCaps is audio-only
        }
        return sample


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def create_safe_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2,
    *,
    sampler=None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Create a DataLoader with SAFE's multimodal collate function.

    Defaults are chosen to be conservative on memory:
    - num_workers=2 by default; set to 0 on very memory-constrained systems.
    - persistent_workers=False unless explicitly requested.
    - prefetch_factor=1 to avoid buffering many batches in RAM.
    - pin_memory=False unless explicitly requested.

    For distributed training, pass a DistributedSampler via the `sampler` argument.
    When a sampler is provided, `shuffle` is ignored (controlled by sampler).

    Callers (e.g., training scripts) can still override these via the keyword
    arguments when needed.
    """

    # Conservative memory defaults unless explicitly overridden.
    resolved_persistent_workers = False if persistent_workers is None else persistent_workers
    resolved_prefetch_factor = (
        1 if prefetch_factor is None and num_workers > 0 else prefetch_factor
    )
    resolved_pin_memory = False if pin_memory is None else pin_memory

    # When using a sampler, shuffle must be False (sampler controls ordering)
    effective_shuffle = shuffle if sampler is None else False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate_multimodal_batch,
        persistent_workers=resolved_persistent_workers if num_workers > 0 else False,
        pin_memory=resolved_pin_memory,
        prefetch_factor=resolved_prefetch_factor if num_workers > 0 else None,
    )
