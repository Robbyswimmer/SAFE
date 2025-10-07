"""Stage A training driver for the full-scale SAFE experiment."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from configs.model_configs import DEMO_CONFIG, FULL_CONFIG, MULTIMODAL_CONFIG
from configs.retention_variants import get_variant_config, RETENTION_VARIANTS
from safe.data.datasets import AudioCapsDataset, VQADataset, WavCapsDataset, create_safe_dataloader
from safe.models.safe_model import SAFEModel
from safe.training.stage_a import StageATrainer


@dataclass
class TrainingConfig:
    variant: str
    seed: int
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
    train_accuracy_interval: int
    train_accuracy_warmup: int
    train_eval_batches: int
    generation_max_new_tokens: int


class CombinedAudioDataset(Dataset):
    """Combine AudioCaps and WavCaps with configurable ratio."""

    def __init__(
        self,
        audiocaps_dataset: Optional[Dataset] = None,
        wavcaps_dataset: Optional[Dataset] = None,
        *,
        wavcaps_ratio: float = 0.5,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.datasets = []

        if audiocaps_dataset is not None:
            self.datasets.append(("audiocaps", audiocaps_dataset))

        if wavcaps_dataset is not None:
            self.datasets.append(("wavcaps", wavcaps_dataset))

        if not self.datasets:
            raise ValueError("At least one dataset (AudioCaps or WavCaps) must be provided")

        # Build index map based on ratio
        self.index_map: List[Tuple[str, int]] = []

        if len(self.datasets) == 1:
            # Only one dataset, use all samples
            name, dataset = self.datasets[0]
            self.index_map = [(name, idx) for idx in range(len(dataset))]
        else:
            # Both datasets: mix based on ratio
            audiocaps_count = len(audiocaps_dataset) if audiocaps_dataset else 0
            wavcaps_count = len(wavcaps_dataset) if wavcaps_dataset else 0

            # Calculate samples to use from each dataset
            total_samples = audiocaps_count + int(wavcaps_count * wavcaps_ratio)

            self.index_map = [
                ("audiocaps", idx) for idx in range(audiocaps_count)
            ] + [
                ("wavcaps", idx) for idx in range(int(wavcaps_count * wavcaps_ratio))
            ]

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.index_map)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        dataset_name, local_idx = self.index_map[idx]
        for name, dataset in self.datasets:
            if name == dataset_name:
                return dataset[local_idx]
        raise IndexError(f"Invalid index: {idx}")


class CombinedValidationDataset(Dataset):
    """Mix AudioCaps and VQA examples for joint evaluation."""

    def __init__(
        self,
        audio_dataset: Dataset,
        vl_dataset: Dataset,
        *,
        max_audio_samples: Optional[int] = None,
        max_vl_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.audio_dataset = audio_dataset
        self.vl_dataset = vl_dataset

        audio_count = len(audio_dataset) if max_audio_samples is None else min(len(audio_dataset), max_audio_samples)
        vl_count = len(vl_dataset) if max_vl_samples is None else min(len(vl_dataset), max_vl_samples)

        self.index_map: List[Tuple[str, int]] = [
            ("audio", idx) for idx in range(audio_count)
        ] + [
            ("vl", idx) for idx in range(vl_count)
        ]

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.index_map)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        domain, local_idx = self.index_map[idx]
        if domain == "audio":
            return self.audio_dataset[local_idx]
        return self.vl_dataset[local_idx]


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_variant(variant: str, base_config: TrainingConfig) -> TrainingConfig:
    """
    Configure training based on retention variant.

    Uses centralized retention variant configs from configs/retention_variants.py
    """
    variant = variant.lower()
    cfg = base_config

    # Get retention variant configuration
    try:
        retention_config = get_variant_config(variant)
    except ValueError as e:
        # List available variants
        available = ", ".join(RETENTION_VARIANTS.keys())
        raise ValueError(
            f"Unknown variant '{variant}'. Available variants: {available}"
        ) from e

    # Apply retention configuration
    cfg.retention_loss_weight = retention_config.retention_loss_weight
    cfg.distillation_weight = retention_config.distillation_weight
    cfg.fisher_weight = retention_config.fisher_weight
    cfg.enable_null_space = retention_config.enable_null_space
    cfg.null_space_rank = retention_config.null_space_rank
    cfg.null_space_min_samples = retention_config.null_space_min_samples
    cfg.null_space_refresh_interval = retention_config.null_space_refresh_interval

    cfg.variant = variant
    return cfg


def build_stage_a_config(cfg: TrainingConfig) -> Dict[str, object]:
    max_eval_batches = cfg.max_eval_batches
    if max_eval_batches is not None and max_eval_batches <= 0:
        max_eval_batches = None

    return {
        "learning_rate_projector": cfg.learning_rate_projector,
        "learning_rate_adapter": cfg.learning_rate_adapter,
        "num_epochs": cfg.num_epochs,
        "eval_steps": 5_000,
        "save_steps": 5_000,
        "logging_steps": 100,
        "eval_logging_steps": max(1, cfg.eval_logging_steps),
        "max_eval_batches": max_eval_batches,
        "max_audio_eval_batches": cfg.max_audio_eval_batches,
        "max_vl_eval_batches": cfg.max_vl_eval_batches,
        "eval_with_audio_gate": cfg.eval_with_audio_gate,
        "eval_audio_gate_comparison": cfg.eval_audio_gate_comparison,
        "debug_logging": cfg.debug_logging,
        "retention_loss_weight": cfg.retention_loss_weight,
        "distillation_weight": cfg.distillation_weight,
        "fisher_weight": cfg.fisher_weight,
        "enable_null_space": cfg.enable_null_space,
        "null_space_rank": cfg.null_space_rank,
        "null_space_min_samples": cfg.null_space_min_samples,
        "null_space_refresh_interval": cfg.null_space_refresh_interval,
        "train_accuracy_interval": cfg.train_accuracy_interval,
        "train_accuracy_warmup": cfg.train_accuracy_warmup,
        "generation_max_new_tokens": cfg.generation_max_new_tokens,
        "output_dir": cfg.output_dir,
    }


def run_experiment(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{timestamp}_{args.variant}"
    run_dir.mkdir(parents=True, exist_ok=False)

    set_random_seeds(args.seed)

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root '{data_root}' does not exist")

    # Load AudioCaps dataset
    audiocaps_train = AudioCapsDataset(data_path=data_root, split=args.train_split)
    print(f"Loaded AudioCaps train: {len(audiocaps_train)} samples")

    # Optionally load WavCaps
    wavcaps_train = None
    if args.use_wavcaps:
        wavcaps_path = data_root / "wavcaps"
        if wavcaps_path.exists():
            try:
                wavcaps_train = WavCapsDataset(data_path=data_root, split="train")
                print(f"Loaded WavCaps train: {len(wavcaps_train)} samples")
                print(f"Using WavCaps ratio: {args.wavcaps_ratio}")
            except Exception as e:
                print(f"Warning: Failed to load WavCaps: {e}")
                print("Continuing with AudioCaps only")
        else:
            print(f"Warning: WavCaps directory not found at {wavcaps_path}")
            print("Continuing with AudioCaps only")

    # Combine datasets if WavCaps is available
    if wavcaps_train is not None:
        train_dataset = CombinedAudioDataset(
            audiocaps_dataset=audiocaps_train,
            wavcaps_dataset=wavcaps_train,
            wavcaps_ratio=args.wavcaps_ratio,
            shuffle=not args.disable_train_shuffle,
            seed=args.seed,
        )
        print(f"Combined training dataset: {len(train_dataset)} samples")
    else:
        train_dataset = audiocaps_train

    # Validation datasets (only AudioCaps for audio validation)
    val_audio_dataset = AudioCapsDataset(data_path=data_root, split=args.val_audio_split)
    val_vl_dataset = VQADataset(data_path=data_root, split=args.val_vqa_split)

    val_dataset = CombinedValidationDataset(
        val_audio_dataset,
        val_vl_dataset,
        max_audio_samples=args.max_audio_val_samples if args.max_audio_val_samples > 0 else None,
        max_vl_samples=args.max_vqa_val_samples if args.max_vqa_val_samples > 0 else None,
        shuffle=not args.disable_val_shuffle,
        seed=args.seed,
    )

    train_loader = create_safe_dataloader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=not args.disable_train_shuffle,
        num_workers=args.num_workers,
    )
    val_loader = create_safe_dataloader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_configs = {
        "demo": DEMO_CONFIG,
        "full": FULL_CONFIG,
        "multimodal": MULTIMODAL_CONFIG,
    }
    if args.model_config not in model_configs:
        raise ValueError(f"Unknown model config '{args.model_config}'. Options: {list(model_configs)}")

    selected_model_config = model_configs[args.model_config]
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

    model_kwargs = {key: value for key, value in selected_model_config.items() if key in safe_model_keys}
    model = SAFEModel(**model_kwargs)

    base_config = TrainingConfig(
        variant=args.variant,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        num_epochs=args.num_epochs,
        learning_rate_projector=args.lr_projector,
        learning_rate_adapter=args.lr_adapter,
        fisher_weight=0.0,
        retention_loss_weight=0.0,
        distillation_weight=0.0,
        enable_null_space=False,
        null_space_rank=args.null_space_rank,
        null_space_min_samples=args.null_space_min_samples,
        null_space_refresh_interval=args.null_space_refresh,
        output_dir=str(run_dir / "checkpoints"),
        max_eval_batches=args.max_eval_batches if args.max_eval_batches > 0 else None,
        max_audio_eval_batches=args.max_audio_eval_batches,
        max_vl_eval_batches=args.max_vl_eval_batches,
        eval_with_audio_gate=not args.disable_eval_audio_gate,
        eval_audio_gate_comparison=args.eval_audio_gate_comparison,
        eval_logging_steps=args.eval_logging_steps,
        debug_logging=args.debug_logging,
        train_accuracy_interval=args.train_accuracy_interval,
        train_accuracy_warmup=args.train_accuracy_warmup,
        train_eval_batches=args.train_eval_batches,
        generation_max_new_tokens=args.generation_max_new_tokens,
    )

    variant_config = configure_variant(args.variant, base_config)
    stage_a_config = build_stage_a_config(variant_config)

    trainer = StageATrainer(
        safe_model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=stage_a_config,
        curriculum_config=None,
    )

    final_metrics = trainer.train()

    train_metrics = None
    if args.train_eval_batches > 0:
        train_metrics = trainer.evaluate(
            max_batches=args.train_eval_batches,
            dataloader=train_loader,
            description="Training",
            split_batches=False,
        )

    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"

    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(variant_config), fh, indent=2)

    metrics_payload: Dict[str, Dict[str, float]] = {"validation": final_metrics}
    if train_metrics is not None:
        metrics_payload["training"] = train_metrics

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)

    print(f"Run complete. Artefacts saved to {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-scale SAFE Stage A training")
    parser.add_argument("--variant", choices=["no_retention", "soft_retention", "fisher_retention", "nullspace_retention", "full_retention"], help="Retention configuration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-root", type=str, default="experiments/full_training/data", help="Dataset root")
    parser.add_argument("--train-split", type=str, default="train", help="AudioCaps split for training")
    parser.add_argument("--val-audio-split", type=str, default="val", help="AudioCaps split for validation")
    parser.add_argument("--use-wavcaps", action="store_true", help="Include WavCaps dataset for training")
    parser.add_argument("--wavcaps-ratio", type=float, default=0.5, help="Ratio of WavCaps samples to use (0.0-1.0)")
    parser.add_argument("--val-vqa-split", type=str, default="val", help="VQA split for validation")
    parser.add_argument("--train-batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=64, help="Validation batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker count")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr-projector", type=float, default=2e-4, help="Learning rate for projector")
    parser.add_argument("--lr-adapter", type=float, default=1e-4, help="Learning rate for adapter")
    parser.add_argument("--null-space-rank", type=int, default=8, help="Null-space rank when enabled")
    parser.add_argument("--null-space-min-samples", type=int, default=128, help="Minimum samples for null-space")
    parser.add_argument("--null-space-refresh", type=int, default=4000, help="Null-space refresh interval")
    parser.add_argument("--max-eval-batches", type=int, default=-1, help="Limit validation batches (<=0 = full)")
    parser.add_argument("--max-audio-eval-batches", type=int, default=32, help="Audio eval batch cap")
    parser.add_argument("--max-vl-eval-batches", type=int, default=64, help="VL eval batch cap")
    parser.add_argument("--max-audio-val-samples", type=int, default=4096, help="Audio validation sample cap (<=0 disables)")
    parser.add_argument("--max-vqa-val-samples", type=int, default=4096, help="VQA validation sample cap (<=0 disables)")
    parser.add_argument("--eval-logging-steps", type=int, default=10, help="Eval logging frequency")
    parser.add_argument("--train-accuracy-interval", type=int, default=0, help="Interval for train accuracy logging")
    parser.add_argument("--train-accuracy-warmup", type=int, default=5, help="Warmup epochs before accuracy logging")
    parser.add_argument("--train-eval-batches", type=int, default=0, help="Evaluate training split after fit")
    parser.add_argument("--generation-max-new-tokens", type=int, default=32, help="Generation token budget")
    parser.add_argument("--output-root", type=str, default="experiments/full_training/runs", help="Run output directory")
    parser.add_argument("--model-config", choices=["demo", "full", "multimodal"], default="full", help="Model config preset")
    parser.add_argument("--disable-train-shuffle", action="store_true", help="Disable shuffling for training dataloader")
    parser.add_argument("--disable-val-shuffle", action="store_true", help="Disable shuffling when mixing validation datasets")
    parser.add_argument("--disable-eval-audio-gate", action="store_true", help="Evaluate without audio gating")
    parser.add_argument("--eval-audio-gate-comparison", action="store_true", help="Run with and without audio gate during eval")
    parser.add_argument("--debug-logging", action="store_true", help="Enable verbose trainer logging")

    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
