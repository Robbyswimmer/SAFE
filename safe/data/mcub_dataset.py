"""
MCUB (Multimodal Commonality Understanding Benchmark) dataset loader.

From: "Model Composition for Multimodal Large Language Models" (ACL 2024)
GitHub: https://github.com/THUNLP-MT/ModelCompose

MCUB tests cross-modal alignment by asking models to identify shared semantic
attributes across different modality inputs (audio, point cloud, image, video).

This implementation supports the audio + point cloud subset for the composition
ablation study comparing pre-FFN vs KV augmentation architectures.
"""

from __future__ import annotations

import json
import torch
import numpy as np
import torchaudio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader


class MCUBDataset(Dataset):
    """
    MCUB evaluation dataset for multimodal commonality understanding.

    Supports three evaluation conditions:
    1. Audio only: provide audio, mask point cloud
    2. Point cloud only: provide point cloud, mask audio
    3. Both modalities: provide both audio and point cloud

    Each sample contains a question about what the modalities have in common,
    with multiple-choice answers.
    """

    dataset_name = "mcub"

    def __init__(
        self,
        data_path: str | Path,
        modalities: List[str] = ["audio", "pointcloud"],
        split: str = "test",
        audio_sample_rate: int = 48000,
        audio_max_length: float = 10.0,
        num_points: int = 1024,
        annotation_file: Optional[str] = None,
        answer_file: Optional[str] = None,
    ):
        """
        Initialize MCUB dataset.

        Args:
            data_path: Root directory containing MCUB data
            modalities: Which modalities to load ("audio", "pointcloud", or both)
            split: Dataset split (typically "test" for evaluation)
            audio_sample_rate: Target sample rate for audio
            audio_max_length: Maximum audio length in seconds
            num_points: Number of points for point cloud sampling
            annotation_file: Path to annotation JSON (auto-detected if not specified)
            answer_file: Path to answer JSON (auto-detected if not specified)
        """
        self.data_path = Path(data_path)
        self.modalities = modalities
        self.split = split
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_length = audio_max_length
        self.num_points = num_points

        # Load answers first (to merge with annotations)
        self.answers_by_id = self._load_answers(answer_file)

        # Load annotations
        self.samples = self._load_annotations(annotation_file)

        print(f"[MCUB] Loaded {len(self.samples)} samples", flush=True)
        print(f"[MCUB] Modalities: {modalities}", flush=True)
        print(f"[MCUB] Answers loaded: {len(self.answers_by_id)}", flush=True)

    def _load_answers(self, answer_file: Optional[str]) -> Dict[str, str]:
        """Load ground truth answers from MCUB-answer.json."""
        answers_by_id = {}

        # Try to find answer file
        if answer_file:
            ans_path = Path(answer_file)
        else:
            possible_paths = [
                self.data_path / "test" / "MCUB-answer.json",
                self.data_path / "MCUB-answer.json",
                self.data_path / "mcub_answers.json",
            ]
            ans_path = None
            for p in possible_paths:
                if p.exists():
                    ans_path = p
                    break

        if ans_path is None or not ans_path.exists():
            print(f"[MCUB] Warning: Answer file not found, answers will be empty")
            return answers_by_id

        print(f"[MCUB] Loading answers from: {ans_path}")
        with open(ans_path, "r") as f:
            answer_data = json.load(f)

        # Parse answers - extract from gpt response in conversations
        for sample in answer_data:
            sample_id = sample.get("id", "")
            conversations = sample.get("conversations", [])
            for conv in conversations:
                if conv.get("from") == "gpt" and conv.get("value"):
                    answers_by_id[sample_id] = conv["value"]
                    break

        return answers_by_id

    def _load_annotations(self, annotation_file: Optional[str]) -> List[Dict]:
        """Load MCUB annotations from JSON file."""
        # Try to find annotation file
        if annotation_file:
            ann_path = Path(annotation_file)
        else:
            # Common paths in ModelCompose structure
            possible_paths = [
                # MCUB-3 with audio + pointcloud
                self.data_path / "test" / "MCUB-3-audio-video-pointcloud.json",
                self.data_path / "test" / "MCUB-3-image-audio-pointcloud.json",
                self.data_path / "MCUB-3-audio-video-pointcloud.json",
                self.data_path / "MCUB-3-image-audio-pointcloud.json",
                # MCUB-4 (all modalities)
                self.data_path / "test" / "MCUB-4.json",
                self.data_path / "MCUB-4.json",
                # Generic paths
                self.data_path / "mcub_audio_pointcloud.json",
                self.data_path / "mcub" / "audio_pointcloud.json",
                self.data_path / "annotations" / f"mcub_{self.split}.json",
                self.data_path / f"mcub_{self.split}.json",
                self.data_path / "mcub.json",
            ]

            ann_path = None
            for p in possible_paths:
                if p.exists():
                    ann_path = p
                    break

            if ann_path is None:
                # List what we do have for debugging
                available = list(self.data_path.glob("*.json"))
                raise FileNotFoundError(
                    f"Could not find MCUB annotations in {self.data_path}. "
                    f"Available JSON files: {available}"
                )

        with open(ann_path, "r") as f:
            annotations = json.load(f)

        # Handle different annotation formats
        if isinstance(annotations, dict):
            # Format: {"data": [...]} or {"samples": [...]}
            if "data" in annotations:
                samples = annotations["data"]
            elif "samples" in annotations:
                samples = annotations["samples"]
            else:
                # Assume the dict values are samples
                samples = list(annotations.values())
        else:
            samples = annotations

        # Filter to samples that have our required modalities
        filtered = []
        for sample in samples:
            # Check modal_inputs structure (ModelCompose format)
            modal_inputs = sample.get("modal_inputs", {})

            has_audio = (
                "audio" in modal_inputs or
                "audio" in sample or
                "audio_path" in sample
            )
            has_pc = (
                "point" in modal_inputs or  # ModelCompose uses "point"
                "pointcloud" in modal_inputs or
                "pointcloud" in sample or
                "pointcloud_path" in sample or
                "point_cloud" in sample
            )

            # Check if sample has the modalities we need
            if "audio" in self.modalities and not has_audio:
                continue
            if "pointcloud" in self.modalities and not has_pc:
                continue

            filtered.append(sample)

        return filtered

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, audio_info: Union[str, Dict]) -> Optional[torch.Tensor]:
        """Load and preprocess audio file."""
        if isinstance(audio_info, str):
            audio_path = self.data_path / audio_info
        elif isinstance(audio_info, dict):
            audio_path = self.data_path / audio_info.get("path", audio_info.get("file", ""))
        else:
            return None

        if not audio_path.exists():
            # Try without data_path prefix
            audio_path = Path(audio_info) if isinstance(audio_info, str) else Path(audio_info.get("path", ""))

        if not audio_path.exists():
            print(f"[MCUB] Warning: Audio file not found: {audio_path}")
            return None

        try:
            waveform, sr = torchaudio.load(audio_path)

            # Resample if needed
            if sr != self.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.audio_sample_rate)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Truncate or pad to max length
            max_samples = int(self.audio_max_length * self.audio_sample_rate)
            if waveform.shape[-1] > max_samples:
                waveform = waveform[..., :max_samples]
            elif waveform.shape[-1] < max_samples:
                padding = max_samples - waveform.shape[-1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            return waveform.squeeze(0)  # (samples,)

        except Exception as e:
            print(f"[MCUB] Error loading audio {audio_path}: {e}")
            return None

    def _load_pointcloud(self, pc_info: Union[str, Dict]) -> Optional[torch.Tensor]:
        """Load and preprocess point cloud file."""
        if isinstance(pc_info, str):
            pc_path = self.data_path / pc_info
        elif isinstance(pc_info, dict):
            pc_path = self.data_path / pc_info.get("path", pc_info.get("file", ""))
        else:
            return None

        if not pc_path.exists():
            # Try without data_path prefix
            pc_path = Path(pc_info) if isinstance(pc_info, str) else Path(pc_info.get("path", ""))

        if not pc_path.exists():
            print(f"[MCUB] Warning: Point cloud file not found: {pc_path}")
            return None

        try:
            # Support various formats
            suffix = pc_path.suffix.lower()

            if suffix == ".npy":
                pc = np.load(pc_path)
            elif suffix == ".npz":
                data = np.load(pc_path)
                pc = data["points"] if "points" in data else data[list(data.keys())[0]]
            elif suffix == ".ply":
                pc = self._load_ply(pc_path)
            elif suffix == ".pts":
                pc = np.loadtxt(pc_path)
            else:
                print(f"[MCUB] Unsupported point cloud format: {suffix}")
                return None

            # Ensure float32
            pc = pc.astype(np.float32)

            # Sample or pad to num_points
            if len(pc) > self.num_points:
                indices = np.random.choice(len(pc), self.num_points, replace=False)
                pc = pc[indices]
            elif len(pc) < self.num_points:
                pad_size = self.num_points - len(pc)
                pad_indices = np.random.choice(len(pc), pad_size, replace=True)
                pc = np.concatenate([pc, pc[pad_indices]], axis=0)

            # Take only xyz (first 3 columns)
            if pc.shape[-1] > 3:
                pc = pc[:, :3]

            return torch.from_numpy(pc).float()

        except Exception as e:
            print(f"[MCUB] Error loading point cloud {pc_path}: {e}")
            return None

    def _load_ply(self, path: Path) -> np.ndarray:
        """Load PLY file (basic ASCII parser)."""
        points = []
        header_ended = False
        vertex_count = 0

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not header_ended:
                    if line.startswith("element vertex"):
                        vertex_count = int(line.split()[-1])
                    elif line == "end_header":
                        header_ended = True
                else:
                    if len(points) < vertex_count:
                        coords = [float(x) for x in line.split()[:3]]
                        points.append(coords)

        return np.array(points, dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample in SAFE-compatible format.

        Returns dict with:
            - sample_id: Unique identifier
            - question: The commonality question
            - answers: Correct answer(s)
            - choices: Multiple choice options (if available)
            - audio: Audio waveform tensor (if loaded)
            - pointcloud: Point cloud tensor (if loaded)
            - valid: Whether sample loaded successfully
        """
        sample = self.samples[idx]

        # Get sample ID
        sample_id = sample.get("id", sample.get("sample_id", f"mcub_{idx}"))

        # Parse ModelCompose conversation format
        conversations = sample.get("conversations", [])
        modal_inputs = sample.get("modal_inputs", {})

        # Extract question from conversation
        question = ""
        answers = ""
        choices = None

        for conv in conversations:
            if conv.get("from") == "human":
                question = conv.get("value", "")
                # Extract choices from question text (A. xxx \nB. yyy format)
                if "\nA." in question or "\nA. " in question:
                    choices = self._parse_choices(question)
            elif conv.get("from") == "gpt":
                answers = conv.get("value", "")

        # Fallback for non-conversation format
        if not question:
            question = sample.get("question", sample.get("text", "What do these have in common?"))

        if not answers:
            # Try to get answer from answer file (by ID)
            if sample_id in self.answers_by_id:
                answers = self.answers_by_id[sample_id]
            elif "answer" in sample:
                answers = sample["answer"]
            elif "answers" in sample:
                answers = sample["answers"]
            elif "label" in sample:
                answers = sample["label"]

        if choices is None:
            choices = sample.get("choices", sample.get("options", None))

        # Load modalities
        audio = None
        pointcloud = None
        valid = True

        if "audio" in self.modalities:
            # Try modal_inputs first (ModelCompose format)
            audio_info = modal_inputs.get("audio")
            if audio_info and isinstance(audio_info, list):
                audio_info = audio_info[0]  # Take first audio file
            if not audio_info:
                audio_info = sample.get("audio", sample.get("audio_path"))
            if audio_info:
                audio = self._load_audio(audio_info)
                if audio is None:
                    valid = False

        if "pointcloud" in self.modalities:
            # Try modal_inputs first - ModelCompose uses "point"
            pc_info = modal_inputs.get("point") or modal_inputs.get("pointcloud")
            if pc_info and isinstance(pc_info, list):
                pc_info = pc_info[0]  # Take first point cloud file
            if not pc_info:
                pc_info = sample.get("pointcloud", sample.get("pointcloud_path", sample.get("point_cloud")))
            if pc_info:
                pointcloud = self._load_pointcloud(pc_info)
                if pointcloud is None:
                    valid = False

        return {
            "sample_id": sample_id,
            "question": question,
            "answers": answers,
            "choices": choices,
            "audio": audio,
            "pointcloud": pointcloud,
            "valid": valid,
            "images": None,  # For compatibility with collate functions
            "raw_sample": sample,  # Keep original for debugging
        }

    def _parse_choices(self, question: str) -> List[str]:
        """Parse multiple choice options from question text."""
        choices = []
        import re
        # Match patterns like "A. Water transportation" or "A) Water transportation"
        pattern = r'([A-D])[.\)]\s*([^\n]+)'
        matches = re.findall(pattern, question)
        for letter, text in matches:
            choices.append(f"{letter}. {text.strip()}")
        return choices if choices else None


def mcub_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for MCUB batches.

    Handles variable-length audio and ensures consistent tensor shapes.
    """
    # Filter out invalid samples
    valid_batch = [s for s in batch if s["valid"]]
    if not valid_batch:
        raise ValueError("No valid samples in batch")

    result = {
        "sample_ids": [s["sample_id"] for s in valid_batch],
        "questions": [s["question"] for s in valid_batch],
        "answers": [s["answers"] for s in valid_batch],
        "choices": [s["choices"] for s in valid_batch],
    }

    # Stack audio if present
    audio_tensors = [s["audio"] for s in valid_batch if s["audio"] is not None]
    if audio_tensors:
        # Pad to same length
        max_len = max(t.shape[-1] for t in audio_tensors)
        padded = []
        for t in audio_tensors:
            if t.shape[-1] < max_len:
                padding = max_len - t.shape[-1]
                t = torch.nn.functional.pad(t, (0, padding))
            padded.append(t)
        result["audio"] = torch.stack(padded)
    else:
        result["audio"] = None

    # Stack point clouds if present
    pc_tensors = [s["pointcloud"] for s in valid_batch if s["pointcloud"] is not None]
    if pc_tensors:
        result["pointcloud"] = torch.stack(pc_tensors)
    else:
        result["pointcloud"] = None

    return result


def create_mcub_dataloader(
    data_path: str | Path,
    modalities: List[str] = ["audio", "pointcloud"],
    batch_size: int = 1,
    num_workers: int = 4,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create MCUB evaluation dataloader.

    Args:
        data_path: Root directory containing MCUB data
        modalities: Which modalities to load
        batch_size: Batch size (typically 1 for evaluation)
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for MCUBDataset

    Returns:
        DataLoader for MCUB evaluation
    """
    dataset = MCUBDataset(
        data_path=data_path,
        modalities=modalities,
        **dataset_kwargs,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for evaluation
        num_workers=num_workers,
        collate_fn=mcub_collate_fn,
        pin_memory=True,
    )


# Also support a synthetic MCUB dataset for testing without real data
class SyntheticMCUBDataset(Dataset):
    """
    Synthetic MCUB-like dataset for testing composition without real MCUB data.

    Generates random audio and point cloud pairs with synthetic questions.
    Useful for debugging the composition evaluation pipeline.
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_classes: int = 10,
        audio_sample_rate: int = 48000,
        audio_length: float = 5.0,
        num_points: int = 1024,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.audio_sample_rate = audio_sample_rate
        self.audio_length = audio_length
        self.num_points = num_points

        # Generate class labels
        self.labels = np.random.randint(0, num_classes, num_samples)
        self.class_names = [f"category_{i}" for i in range(num_classes)]

        print(f"[SyntheticMCUB] Created {num_samples} synthetic samples", flush=True)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        label = self.labels[idx]
        class_name = self.class_names[label]

        # Generate synthetic audio (random noise with class-specific frequency)
        num_audio_samples = int(self.audio_length * self.audio_sample_rate)
        freq = 200 + label * 100  # Different frequency per class
        t = torch.linspace(0, self.audio_length, num_audio_samples)
        audio = torch.sin(2 * np.pi * freq * t) * 0.5 + torch.randn(num_audio_samples) * 0.1

        # Generate synthetic point cloud (random points in class-specific region)
        center = np.array([label % 3, label // 3, 0]) * 2
        pointcloud = np.random.randn(self.num_points, 3).astype(np.float32) * 0.5 + center

        return {
            "sample_id": f"synthetic_{idx}",
            "question": "What category does this belong to?",
            "answers": class_name,
            "choices": self.class_names[:4],  # First 4 classes as choices
            "audio": audio.float(),
            "pointcloud": torch.from_numpy(pointcloud).float(),
            "valid": True,
            "images": None,
            "label": label,
        }
