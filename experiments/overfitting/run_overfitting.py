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
from configs.model_configs import DEMO_CONFIG, FULL_CONFIG, MULTIMODAL_CONFIG
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
    distillation_weight: float
    enable_null_space: bool
    null_space_rank: int
    null_space_min_samples: int
    null_space_refresh_interval: int
    output_dir: str
    max_eval_batches: Optional[int]
    max_audio_eval_batches: int
    max_vl_eval_batches: int
    eval_with_audio_gate: bool
    eval_audio_gate_comparison: bool
    eval_logging_steps: int
    debug_logging: bool


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


class MixedValidationDataset(Dataset):
    """Mixed validation dataset combining audio and VL samples."""

    def __init__(self, pack_root: Path, total_size: int = 100, audio_ratio: float = 0.5, seed: int = 42):
        self.pack_root = pack_root
        self.total_size = total_size
        self.audio_ratio = audio_ratio
        self.seed = seed
        
        # Calculate split sizes
        self.audio_size = int(total_size * audio_ratio)
        self.vl_size = total_size - self.audio_size
        
        # Load audio samples from training manifest
        self.audio_dataset = ManifestAudioCapsDataset(pack_root=pack_root)
        audio_indices = list(range(min(self.audio_size, len(self.audio_dataset))))
        self.audio_subset = Subset(self.audio_dataset, audio_indices)
        
        # Load VL samples from VL manifest
        manifest_path = pack_root / "val_manifest.json"
        self.vl_dataset = ManifestVQADataset(manifest_path, pack_root=pack_root)
        vl_indices = list(range(min(self.vl_size, len(self.vl_dataset))))
        self.vl_subset = Subset(self.vl_dataset, vl_indices)
        
        # Create combined index mapping
        self.indices = []
        # Add audio indices
        for i in range(len(self.audio_subset)):
            self.indices.append(('audio', i))
        # Add VL indices  
        for i in range(len(self.vl_subset)):
            self.indices.append(('vl', i))
            
        # Shuffle with fixed seed for reproducibility
        import random
        rng = random.Random(seed)
        rng.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        dataset_type, dataset_idx = self.indices[idx]
        if dataset_type == 'audio':
            return self.audio_subset[dataset_idx]
        else:
            return self.vl_subset[dataset_idx]


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
    elif source == "pack_train_audio":
        if pack_root is None:
            raise ValueError("pack_root must be provided when val-source='pack_train_audio'")
        # Use training audio manifest for validation subset
        dataset = ManifestAudioCapsDataset(pack_root=pack_root)
        # Take first 50-100 samples for consistent validation subset
        val_size = min(size or 100, len(dataset), 100)  # Cap at 100 samples
        indices = list(range(val_size))
        return Subset(dataset, indices)
    elif source == "pack_mixed":
        if pack_root is None:
            raise ValueError("pack_root must be provided when val-source='pack_mixed'")
        # Mixed validation with both audio and VL samples
        total_size = size or 100
        return MixedValidationDataset(pack_root=pack_root, total_size=total_size, audio_ratio=0.5, seed=seed)
    else:
        raise ValueError(f"Unsupported val-source '{source}'")


def build_stage_a_config(config: ExperimentConfig) -> Dict[str, float | int | bool | str]:
    """Translate ExperimentConfig into StageATrainer keyword configuration."""
    max_eval_batches = config.max_eval_batches
    if max_eval_batches is not None and max_eval_batches <= 0:
        max_eval_batches = None
    return {
        "learning_rate_projector": config.learning_rate_projector,
        "learning_rate_adapter": config.learning_rate_adapter,
        "num_epochs": config.num_epochs,
        "eval_steps": 1_000,  # effectively disables mid-epoch evals
        "save_steps": 1_000,
        "logging_steps": 50,
        "eval_logging_steps": max(1, config.eval_logging_steps),
        "max_eval_batches": max_eval_batches,
        "debug_logging": config.debug_logging,
        "retention_loss_weight": config.retention_loss_weight,
        "distillation_weight": config.distillation_weight,
        "fisher_weight": config.fisher_weight,
        "enable_null_space": config.enable_null_space,
        "null_space_rank": config.null_space_rank,
        "null_space_min_samples": config.null_space_min_samples,
        "null_space_refresh_interval": config.null_space_refresh_interval,
        "output_dir": config.output_dir,
    }


def _save_variant_checkpoint(safe_model: SAFEModel, variant: str, run_dir: Path) -> None:
    """Save only the parameters that were actually trained for each variant."""
    # Create checkpoints directory  
    checkpoint_dir = Path("experiments/overfitting/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create variant-specific checkpoint path
    checkpoint_path = checkpoint_dir / f"{variant}_model.pt"
    
    if variant == "no_retention":
        # No additional parameters trained - just save a marker file
        marker_path = checkpoint_dir / f"{variant}_marker.txt"
        with open(marker_path, "w") as f:
            f.write("no_retention variant - uses base model only\n")
        print(f"No checkpoint saved for {variant} (uses base model only)")
        print(f"Marker saved to: {marker_path}")
        
    elif variant == "soft_retention":
        # Save only retention/distillation components (projectors, adapters)
        state_dict = {}
        for name, param in safe_model.named_parameters():
            # Save only trainable parameters that aren't part of the base LLM
            if param.requires_grad and not name.startswith('base_vl.llm.'):
                state_dict[name] = param.cpu()
        
        if state_dict:
            torch.save({
                'variant': variant,
                'model_state_dict': state_dict,
                'trained_params': list(state_dict.keys())
            }, checkpoint_path)
            print(f"Soft retention checkpoint saved to: {checkpoint_path}")
            print(f"Saved {len(state_dict)} trained parameter tensors")
        else:
            print(f"Warning: No trainable parameters found for {variant}")
            
    elif variant == "full_safe":
        # Save all SAFE model parameters
        state_dict = safe_model.state_dict()
        torch.save({
            'variant': variant,
            'model_state_dict': state_dict,
            'full_model': True
        }, checkpoint_path)
        print(f"Full SAFE checkpoint saved to: {checkpoint_path}")
        print(f"Saved {len(state_dict)} total parameter tensors")
    
    else:
        print(f"Warning: Unknown variant '{variant}' - no checkpoint saved")


def configure_variant(variant: str, base_config: ExperimentConfig) -> ExperimentConfig:
    variant = variant.lower()
    cfg = base_config

    if variant == "no_retention":
        cfg.retention_loss_weight = 0.0
        cfg.distillation_weight = 0.0
        cfg.fisher_weight = 0.0
        cfg.enable_null_space = False
    elif variant == "soft_retention":
        cfg.retention_loss_weight = 0.2
        cfg.distillation_weight = 0.2
        cfg.fisher_weight = 0.0
        cfg.enable_null_space = False
    elif variant == "full_safe":
        cfg.retention_loss_weight = 1.0
        cfg.distillation_weight = 1.0
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
    model_configs = {
        "demo": DEMO_CONFIG,
        "full": FULL_CONFIG,
        "multimodal": MULTIMODAL_CONFIG,
    }
    if args.model_config not in model_configs:
        raise ValueError(f"Unknown model config '{args.model_config}'. Available: {list(model_configs)}")

    selected_model_config = model_configs[args.model_config]
    print(f"[DEBUG] Using model config '{args.model_config}'", flush=True)

    safe_model_keys = {
        "llm_model_name",
        "vision_model_name",
        "audio_encoder_type",
        "audio_encoder_config",
        "projector_type",
        "num_audio_tokens",
        "projector_config",
        "fusion_type",
        "fusion_layer_indices",
        "lora_rank",
        "fusion_config",
        "freeze_base_vl",
        "freeze_audio_encoder",
        "llm_hidden_size",
        "audio_embed_dim",
    }

    model_kwargs = {
        key: value
        for key, value in selected_model_config.items()
        if key in safe_model_keys
    }

    print(f"[DEBUG] Creating SAFEModel...", flush=True)
    model = SAFEModel(**model_kwargs)
    print(
        f"[DEBUG] SAFEModel created successfully. Model type: {model.base_vl.model_type}",
        flush=True,
    )

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
        distillation_weight=1.0,
        enable_null_space=False,
        null_space_rank=args.null_space_rank,
        null_space_min_samples=args.null_space_min_samples,
        null_space_refresh_interval=args.null_space_refresh,
        output_dir=str(run_dir / "checkpoints"),
        max_eval_batches=None if args.max_eval_batches <= 0 else args.max_eval_batches,
        max_audio_eval_batches=args.max_audio_eval_batches,
        max_vl_eval_batches=args.max_vl_eval_batches,
        eval_with_audio_gate=args.eval_with_audio_gate,
        eval_audio_gate_comparison=args.eval_audio_gate_comparison,
        eval_logging_steps=max(1, args.eval_logging_steps),
        debug_logging=args.debug_logging,
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

    # Save variant-specific model parameters (only what was actually trained)
    _save_variant_checkpoint(model, args.variant, run_dir)

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
                        help="Validation dataset source (mock_vl, vqa, pack_vl, pack_audio, pack_train_audio, pack_mixed)")
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
    parser.add_argument(
        "--model-config",
        type=str,
        choices=["demo", "full", "multimodal"],
        default="full",
        help="Model backbone configuration to use",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=4,
        help="Limit number of validation batches during evaluation (<=0 uses full set)",
    )
    parser.add_argument(
        "--max-audio-eval-batches",
        type=int,
        default=4,
        help="Maximum number of audio evaluation batches (samples with has_audio=True)",
    )
    parser.add_argument(
        "--max-vl-eval-batches",
        type=int,
        default=4,
        help="Maximum number of VL evaluation batches (samples with has_audio=False)",
    )
    parser.add_argument(
        "--eval-with-audio-gate",
        action="store_true",
        default=True,
        help="Enable audio during evaluation",
    )
    parser.add_argument(
        "--eval-audio-gate-comparison",
        action="store_true",
        help="Run evaluation with both audio gate on/off to measure VL drift",
    )
    parser.add_argument(
        "--eval-logging-steps",
        type=int,
        default=1,
        help="Number of eval batches between progress messages",
    )
    parser.add_argument(
        "--debug-logging",
        action="store_true",
        help="Enable verbose debug logging inside trainer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
