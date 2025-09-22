"""Run overfitting ablation experiments for SAFE.

This script trains Stage A on a tiny audio-centric subset to measure
vision-language regression under different retention mechanisms.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from safe.models.safe_model import SAFEModel
from safe.training.stage_a import StageATrainer
from safe.data.datasets import (
    AudioCapsDataset,
    AVQADataset,
    VQADataset,
    create_safe_dataloader,
)

try:
    import librosa
except ImportError as exc:  # pragma: no cover - optional dependency
    librosa = None


@dataclass
class ExperimentConfig:
    variant: str
    subset_size: int
    seed: int
    train_source: str
    val_source: str
    train_batch_size: int
    val_batch_size: int
    num_epochs: int
    learning_rate_projector: float
    learning_rate_adapter: float
    fisher_weight: float
    retention_loss_weight: float
    enable_null_space: bool
    null_space_rank: int
    null_space_min_samples: int
    null_space_refresh_interval: int
    output_dir: str


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ManifestAudioCapsDataset(Dataset):
    """Dataset backed by a packaged AudioCaps subset."""

    def __init__(self, pack_root: Path, resample_rate: int = 48_000):
        self.pack_root = pack_root
        self.resample_rate = resample_rate
        self.logger = logging.getLogger(__name__)

        manifest_path = pack_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Expected manifest at {manifest_path}. Run package_subset.py first."
            )

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = self.manifest.get("entries", [])
        if not entries:
            raise ValueError("Manifest contains no entries")

        if librosa is None:
            raise ImportError(
                "librosa is required to load audio from the packaged subset."
            )

        valid_entries: List[Dict[str, object]] = []
        missing_paths: List[str] = []
        for entry in entries:
            rel = entry.get("audio_relpath")
            if not rel:
                continue
            audio_path = self.pack_root / rel
            if audio_path.exists():
                entry["_abs_audio_path"] = audio_path
                valid_entries.append(entry)
            else:
                missing_paths.append(str(audio_path))

        if missing_paths:
            preview = ", ".join(missing_paths[:3])
            self.logger.warning(
                "Skipping %d missing audio files (e.g., %s)",
                len(missing_paths),
                preview,
            )

        if not valid_entries:
            raise FileNotFoundError(
                "No valid audio files found in data pack. Did you copy the clips?"
            )

        self.entries = valid_entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        entry = self.entries[idx]
        audio_path = entry.get("_abs_audio_path")
        if audio_path is None:
            audio_relpath = entry["audio_relpath"]
            audio_path = self.pack_root / audio_relpath
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio clip: {audio_path}")

        waveform, _ = librosa.load(audio_path, sr=self.resample_rate)
        audio_tensor = torch.tensor(waveform, dtype=torch.float32)

        question = "Describe the audio."  # simple prompt for overfitting QA
        caption = entry.get("caption", "")

        return {
            "sample_id": entry.get("youtube_id"),
            "question": question,
            "answers": caption,
            "audio": audio_tensor,
            "images": None,
            "question_type": "audio_dependent",
        }


class ManifestVQADataset(Dataset):
    """Dataset backed by a VQA validation manifest."""

    def __init__(self, manifest_path: Path, pack_root: Optional[Path] = None):
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Validation manifest not found: {manifest_path}. Create it with create_val_manifest.py"
            )
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.entries = data.get("entries", [])
        if not self.entries:
            raise ValueError("Validation manifest is empty")
        self.pack_root = pack_root

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        entry = self.entries[idx]
        images = None

        image_filename = entry.get("image_filename")
        if self.pack_root is not None and image_filename:
            image_path = self.pack_root / "images" / image_filename
            if image_path.exists():
                from PIL import Image
                import torchvision.transforms as T

                image = Image.open(image_path).convert("RGB")
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                ])
                images = transform(image)

        return {
            "sample_id": entry.get("question_id"),
            "question": entry.get("question", ""),
            "answers": entry.get("majority_answer", ""),
            "audio": None,
            "images": images,
            "question_type": "visual_only",
        }


def _load_mock_dataset(size: int, seed: int, modality_distribution: Optional[Dict[str, float]] = None):
    from tests.fixtures.mock_datasets import MockSAFEDataset

    return MockSAFEDataset(
        size=size,
        modality_distribution=modality_distribution,
        seed=seed,
    )


def _build_audio_subset(dataset: Dataset, subset_size: int, seed: int) -> Subset:
    """Select a reproducible audio-containing subset."""
    rng = random.Random(seed)

    audio_indices: List[int] = []

    # Optimised path for mock dataset (uses precomputed specs)
    sample_specs = getattr(dataset, "sample_specs", None)
    if sample_specs is not None:
        for idx, spec in enumerate(sample_specs):
            if getattr(spec, "has_audio", False):
                audio_indices.append(idx)
    else:
        # Generic path – iterate until enough samples gathered
        for idx in range(len(dataset)):
            sample = dataset[idx]
            audio = sample.get("audio") if isinstance(sample, dict) else None
            has_audio = False
            if audio is None:
                has_audio = False
            elif isinstance(audio, torch.Tensor):
                has_audio = audio.numel() > 0
            else:
                has_audio = True
            if has_audio:
                audio_indices.append(idx)
            if len(audio_indices) >= subset_size * 2:
                # Early stop once we have a healthy margin
                break

    if len(audio_indices) < subset_size:
        raise ValueError(
            f"Not enough audio examples to draw {subset_size} samples (found {len(audio_indices)})"
        )

    selected = rng.sample(audio_indices, subset_size)
    selected.sort()
    return Subset(dataset, selected)


def build_train_dataset(
    source: str,
    subset_size: int,
    seed: int,
    data_path: Path,
    pack_root: Optional[Path] = None,
) -> Dataset:
    source = source.lower()
    if source == "mock":
        raw_dataset = _load_mock_dataset(size=max(subset_size * 4, 1000), seed=seed)
        subset = _build_audio_subset(raw_dataset, subset_size=subset_size, seed=seed)
        return subset
    elif source == "audiocaps":
        raw_dataset = AudioCapsDataset(data_path=data_path, split="train")
        subset = _build_audio_subset(raw_dataset, subset_size=subset_size, seed=seed)
        return subset
    elif source == "avqa":
        raw_dataset = AVQADataset(data_path=data_path, split="train")
        subset = _build_audio_subset(raw_dataset, subset_size=subset_size, seed=seed)
        return subset
    elif source == "pack":
        if pack_root is None:
            raise ValueError("pack_root must be provided when train-source='pack'")
        dataset = ManifestAudioCapsDataset(pack_root=pack_root)
        if subset_size and subset_size < len(dataset):
            indices = list(range(subset_size))
            return Subset(dataset, indices)
        return dataset
    else:
        raise ValueError(f"Unsupported train-source '{source}'")


def build_val_dataset(
    source: str,
    size: int,
    seed: int,
    data_path: Path,
    pack_root: Optional[Path] = None,
) -> Dataset:
    source = source.lower()
    if source == "mock_vl":
        # Bias towards VL without audio to measure regression cleanly
        modality_dist = {"visual_only": 0.7, "text_only": 0.2, "audio_visual": 0.1}
        dataset = _load_mock_dataset(size=size, seed=seed + 1, modality_distribution=modality_dist)
        return dataset
    elif source == "vqa":
        dataset = VQADataset(data_path=data_path, split="val")
        if size and size < len(dataset):
            indices = list(range(size))
            return Subset(dataset, indices)
        return dataset
    elif source == "pack_vl":
        if pack_root is None:
            raise ValueError("pack_root must be provided when val-source='pack_vl'")
        manifest_path = pack_root / "val_manifest.json"
        dataset = ManifestVQADataset(manifest_path, pack_root=pack_root)
        if size and size < len(dataset):
            indices = list(range(size))
            return Subset(dataset, indices)
        return dataset
    elif source == "pack_audio":
        if pack_root is None:
            raise ValueError("pack_root must be provided when val-source='pack_audio'")
        dataset = ManifestAudioCapsDataset(pack_root=pack_root)
        if size and size < len(dataset):
            indices = list(range(size))
            return Subset(dataset, indices)
        return dataset
    else:
        raise ValueError(f"Unsupported val-source '{source}'")


def build_stage_a_config(config: ExperimentConfig) -> Dict[str, float | int | bool | str]:
    """Translate ExperimentConfig into StageATrainer keyword configuration."""
    return {
        "learning_rate_projector": config.learning_rate_projector,
        "learning_rate_adapter": config.learning_rate_adapter,
        "num_epochs": config.num_epochs,
        "eval_steps": 1_000,  # effectively disables mid-epoch evals
        "save_steps": 1_000,
        "logging_steps": 50,
        "max_eval_batches": None,
        "retention_loss_weight": config.retention_loss_weight,
        "fisher_weight": config.fisher_weight,
        "enable_null_space": config.enable_null_space,
        "null_space_rank": config.null_space_rank,
        "null_space_min_samples": config.null_space_min_samples,
        "null_space_refresh_interval": config.null_space_refresh_interval,
        "output_dir": config.output_dir,
    }


def configure_variant(variant: str, base_config: ExperimentConfig) -> ExperimentConfig:
    variant = variant.lower()
    cfg = base_config

    if variant == "no_retention":
        cfg.retention_loss_weight = 0.0
        cfg.fisher_weight = 0.0
        cfg.enable_null_space = False
    elif variant == "soft_retention":
        cfg.retention_loss_weight = 1.0
        cfg.fisher_weight = 0.1
        cfg.enable_null_space = False
    elif variant == "full_safe":
        cfg.retention_loss_weight = 1.0
        cfg.fisher_weight = 0.1
        cfg.enable_null_space = True
    else:
        raise ValueError(f"Unknown variant '{variant}'")

    cfg.variant = variant
    return cfg


def run_experiment(args: argparse.Namespace) -> None:
    print(f"[DEBUG] Starting run_experiment with variant: {args.variant}", flush=True)
    output_root = Path(args.output_root).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{timestamp}_{args.variant}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Created run directory: {run_dir}", flush=True)

    set_random_seeds(args.seed)
    print(f"[DEBUG] Set random seeds: {args.seed}", flush=True)

    data_path = Path(args.data_path).expanduser().resolve()

    pack_root = Path(args.pack_root).expanduser().resolve() if args.pack_root else None

    train_dataset = build_train_dataset(
        source=args.train_source,
        subset_size=args.subset_size,
        seed=args.seed,
        data_path=data_path,
        pack_root=pack_root,
    )
    val_dataset = build_val_dataset(
        source=args.val_source,
        size=args.val_size,
        seed=args.seed,
        data_path=data_path,
        pack_root=pack_root,
    )

    train_loader = create_safe_dataloader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = create_safe_dataloader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Base model setup (reuse demo config for now)
    print(f"[DEBUG] Creating SAFEModel...", flush=True)
    model = SAFEModel()
    print(f"[DEBUG] SAFEModel created successfully. Model type: {model.base_vl.model_type}", flush=True)

    print(f"[DEBUG] Creating experiment config...", flush=True)
    base_config = ExperimentConfig(
        variant=args.variant,
        subset_size=args.subset_size,
        seed=args.seed,
        train_source=args.train_source,
        val_source=args.val_source,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        num_epochs=args.num_epochs,
        learning_rate_projector=args.lr_projector,
        learning_rate_adapter=args.lr_adapter,
        fisher_weight=0.0,
        retention_loss_weight=0.0,
        enable_null_space=False,
        null_space_rank=args.null_space_rank,
        null_space_min_samples=args.null_space_min_samples,
        null_space_refresh_interval=args.null_space_refresh,
        output_dir=str(run_dir / "checkpoints"),
    )

    print(f"[DEBUG] Configuring variant: {args.variant}", flush=True)
    variant_config = configure_variant(args.variant, base_config)
    stage_a_config = build_stage_a_config(variant_config)
    print(f"[DEBUG] Stage A config created", flush=True)

    print(f"[DEBUG] Creating StageATrainer...", flush=True)
    trainer = StageATrainer(
        safe_model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=stage_a_config,
        curriculum_config=None,
    )
    print(f"[DEBUG] StageATrainer created successfully", flush=True)

    print(f"[DEBUG] Starting trainer.train()...", flush=True)
    final_metrics = trainer.train()
    print(f"[DEBUG] trainer.train() completed", flush=True)

    # Persist artefacts
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    subset_indices_path = run_dir / "subset_indices.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(variant_config), f, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    # Store the absolute indices from the parent dataset for reproducibility
    subset: Subset = train_dataset  # type: ignore[assignment]
    if isinstance(subset, Subset) and hasattr(subset, "indices"):
        indices = list(subset.indices)  # type: ignore[attr-defined]
        with open(subset_indices_path, "w", encoding="utf-8") as f:
            json.dump(indices, f, indent=2)

    print("\nRun complete.")
    print(f"  Config saved to:   {config_path}")
    print(f"  Metrics saved to:  {metrics_path}")
    print(f"  Checkpoints under: {stage_a_config['output_dir']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAFE overfitting ablation runner")
    parser.add_argument("--variant", type=str, choices=["no_retention", "soft_retention", "full_safe"],
                        help="Retention configuration to run")
    parser.add_argument("--subset-size", type=int, default=400,
                        help="Number of audio samples to overfit")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subset selection and training")
    parser.add_argument("--train-source", type=str, default="mock",
                        help="Training dataset source (mock, audiocaps, avqa)")
    parser.add_argument("--val-source", type=str, default="pack_vl",
                        help="Validation dataset source (mock_vl, vqa, pack_vl, pack_audio)")
    parser.add_argument("--val-size", type=int, default=500,
                        help="Number of validation samples (if applicable)")
    parser.add_argument("--train-batch-size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=16,
                        help="Validation batch size")
    parser.add_argument("--num-epochs", type=int, default=30,
                        help="Number of Stage A epochs")
    parser.add_argument("--lr-projector", type=float, default=1e-4,
                        help="Learning rate for audio projector")
    parser.add_argument("--lr-adapter", type=float, default=5e-5,
                        help="Learning rate for fusion adapter")
    parser.add_argument("--null-space-rank", type=int, default=8,
                        help="Rank of protected subspace when null-space editing is active")
    parser.add_argument("--null-space-min-samples", type=int, default=64,
                        help="Minimum gradient samples before constructing null-space basis")
    parser.add_argument("--null-space-refresh", type=int, default=2000,
                        help="Refresh interval (steps) for recomputing null-space basis")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of dataloader worker processes")
    parser.add_argument("--data-path", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--pack-root", type=str, default="data_pack",
                        help="Root of packaged subset (used when train/val-source='pack*')")
    parser.add_argument("--output-root", type=str, default="experiments/overfitting/runs",
                        help="Directory to store experiment outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
