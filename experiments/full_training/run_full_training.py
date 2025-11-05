"""Stage A training driver for the full-scale SAFE experiment."""

from __future__ import annotations

print("[run_full_training.py] Script starting...", flush=True)
import sys
sys.stdout.flush()
sys.stderr.flush()

print("[run_full_training.py] Importing standard library modules...", flush=True)
sys.stdout.flush()
import argparse
import json
import os
import random
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

print("[run_full_training.py] Importing numpy...", flush=True)
sys.stdout.flush()
import numpy as np

print("[run_full_training.py] Importing torch...", flush=True)
sys.stdout.flush()
import torch
from torch.utils.data import Dataset

print("[run_full_training.py] Importing configs...", flush=True)
sys.stdout.flush()
from configs.model_configs import DEMO_CONFIG, FULL_CONFIG, MULTIMODAL_CONFIG
from configs.retention_variants import get_variant_config, RETENTION_VARIANTS

print("[run_full_training.py] Importing SAFE datasets...", flush=True)
sys.stdout.flush()
from safe.data.datasets import AudioCapsDataset, VQADataset, WavCapsDataset, create_safe_dataloader

print("[run_full_training.py] Importing SAFE model...", flush=True)
sys.stdout.flush()
from safe.models.safe_model import SAFEModel

print("[run_full_training.py] Importing StageATrainer...", flush=True)
sys.stdout.flush()
from safe.training.stage_a import StageATrainer

print("[run_full_training.py] ✓ All imports complete", flush=True)
sys.stdout.flush()


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
    max_audio_eval_samples: int
    max_vl_eval_samples: int
    eval_with_audio_gate: bool
    eval_audio_gate_comparison: bool
    eval_logging_steps: int
    debug_logging: bool
    train_accuracy_interval: int
    train_accuracy_warmup: int
    train_eval_batches: int
    generation_max_new_tokens: int
    audio_generation_max_new_tokens: int
    gradient_accumulation_steps: int
    disable_bertscore: bool
    progress_log_timeout: int
    save_audio_csv: bool
    csv_min_accuracy: float
    csv_max_samples: int
    enable_scst_finetune: bool = False
    scst_epochs: int = 1
    scst_learning_rate: float = 5e-6
    scst_num_samples: int = 1
    scst_sample_top_p: float = 0.9
    scst_sample_temperature: float = 0.9
    scst_reward_metric: str = "cider"
    scst_patience_epochs: int = 2
    scst_improvement_threshold: float = 1e-4


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


class DatasetSubset(Dataset):
    """Lightweight subset wrapper with deterministic ordering."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


class AudioValidationMixDataset(Dataset):
    """Blend multiple audio validation datasets with controllable ratios."""

    def __init__(
        self,
        sources: Sequence[Tuple[str, Dataset]],
        *,
        weights: Optional[Sequence[float]] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if not sources:
            raise ValueError("At least one audio dataset must be provided")

        self.sources: List[Tuple[str, Dataset]] = list(sources)

        rng = random.Random(seed)
        self._index_map: List[Tuple[int, int]] = []
        self._source_counts: Dict[str, int] = {}

        # Prepare per-source indices (optionally shuffled for diversity)
        per_source_indices: List[List[int]] = []
        for name, dataset in self.sources:
            indices = list(range(len(dataset)))
            if shuffle:
                rng.shuffle(indices)
            per_source_indices.append(indices)

        total_available = sum(len(indices) for indices in per_source_indices)
        max_samples = int(max_samples) if max_samples is not None and max_samples > 0 else None
        total_samples = total_available if max_samples is None else min(max_samples, total_available)

        raw_weights: List[float]
        if weights is None:
            raw_weights = [1.0] * len(self.sources)
        else:
            if len(weights) != len(self.sources):
                raise ValueError("weights length must match number of sources")
            raw_weights = [max(0.0, float(w)) for w in weights]

        weight_sum = sum(raw_weights)
        if weight_sum <= 0:
            raw_weights = [1.0] * len(self.sources)
            weight_sum = float(len(self.sources))

        # Initial allocation using fractional targets
        target_floats = [total_samples * (w / weight_sum) for w in raw_weights]
        base_counts: List[int] = []
        fractional_parts: List[Tuple[float, int]] = []
        for idx, (target, indices) in enumerate(zip(target_floats, per_source_indices)):
            capacity = len(indices)
            base = min(int(math.floor(target)), capacity)
            base_counts.append(base)
            fractional_parts.append((target - base, idx))

        assigned = sum(base_counts)
        leftover = max(0, total_samples - assigned)

        if leftover > 0:
            # Prioritize sources with largest fractional remainder and available capacity
            fractional_parts.sort(reverse=True)
            for _, idx in fractional_parts:
                if leftover <= 0:
                    break
                capacity = len(per_source_indices[idx]) - base_counts[idx]
                if capacity <= 0:
                    continue
                base_counts[idx] += 1
                leftover -= 1

        if leftover > 0:
            # Distribute any remaining samples to sources with spare capacity
            for idx in range(len(per_source_indices)):
                if leftover <= 0:
                    break
                capacity = len(per_source_indices[idx]) - base_counts[idx]
                if capacity <= 0:
                    continue
                take = min(capacity, leftover)
                base_counts[idx] += take
                leftover -= take

        # Build final index map and keep per-source counts for reporting
        for source_idx, (name, _) in enumerate(self.sources):
            count = base_counts[source_idx]
            if count <= 0:
                continue
            selected = per_source_indices[source_idx][:count]
            self._source_counts[name] = count
            for item_idx in selected:
                self._index_map.append((source_idx, item_idx))

        if shuffle:
            rng.shuffle(self._index_map)

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int):
        source_idx, local_idx = self._index_map[idx]
        name, dataset = self.sources[source_idx]
        sample = dataset[local_idx]
        if isinstance(sample, dict):
            sample = dict(sample)
            sample.setdefault("dataset_name", name)
        return sample

    def get_source_counts(self) -> Dict[str, int]:
        """Return number of samples drawn from each source."""

        return dict(self._source_counts)


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
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


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
        "variant": cfg.variant,
        "eval_steps": 5_000,
        "save_steps": 5_000,
        "logging_steps": 100,
        "eval_logging_steps": max(1, cfg.eval_logging_steps),
        "progress_log_timeout": max(0, cfg.progress_log_timeout),
        "max_eval_batches": max_eval_batches,
        "max_audio_eval_batches": cfg.max_audio_eval_batches,
        "max_vl_eval_batches": cfg.max_vl_eval_batches,
        "max_audio_eval_samples": cfg.max_audio_eval_samples,
        "max_vl_eval_samples": cfg.max_vl_eval_samples,
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
        "audio_generation_max_new_tokens": cfg.audio_generation_max_new_tokens,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "disable_bertscore": cfg.disable_bertscore,
        "output_dir": cfg.output_dir,
        "save_audio_csv": cfg.save_audio_csv,
        "csv_min_accuracy": cfg.csv_min_accuracy,
        "csv_max_samples": cfg.csv_max_samples,
        "enable_scst_finetune": cfg.enable_scst_finetune,
        "scst_epochs": cfg.scst_epochs,
        "scst_learning_rate": cfg.scst_learning_rate,
        "scst_num_samples": cfg.scst_num_samples,
        "scst_sample_top_p": cfg.scst_sample_top_p,
        "scst_sample_temperature": cfg.scst_sample_temperature,
        "scst_reward_metric": cfg.scst_reward_metric,
        "scst_patience_epochs": cfg.scst_patience_epochs,
        "scst_improvement_threshold": cfg.scst_improvement_threshold,
    }


def run_experiment(args: argparse.Namespace) -> None:
    print(f"\n[run_experiment] Function called with variant={args.variant}", flush=True)
    import sys
    sys.stdout.flush()

    print(f"[run_experiment] Creating output directories...", flush=True)
    sys.stdout.flush()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{timestamp}_{args.variant}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"[run_experiment] ✓ Output directory created: {run_dir}", flush=True)
    sys.stdout.flush()

    print(f"[run_experiment] Setting random seeds (seed={args.seed})...", flush=True)
    sys.stdout.flush()
    set_random_seeds(args.seed)
    print(f"[run_experiment] ✓ Random seeds set", flush=True)
    sys.stdout.flush()

    print(f"[run_experiment] Checking data root: {args.data_root}...", flush=True)
    sys.stdout.flush()
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root '{data_root}' does not exist")
    print(f"[run_experiment] ✓ Data root exists: {data_root}", flush=True)
    sys.stdout.flush()

    # Load AudioCaps dataset
    print(f"[run_experiment] Loading AudioCaps dataset (split={args.train_split})...", flush=True)
    sys.stdout.flush()
    audiocaps_train = AudioCapsDataset(data_path=data_root, split=args.train_split)
    print(f"Loaded AudioCaps train: {len(audiocaps_train)} samples", flush=True)
    sys.stdout.flush()

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

    # Validation datasets (blend AudioCaps with WavCaps when available)
    val_vl_dataset = VQADataset(data_path=data_root, split=args.val_vqa_split)

    val_audio_sources: List[Tuple[str, Dataset]] = []
    val_audio_weights: List[float] = []

    audiocaps_val = AudioCapsDataset(data_path=data_root, split=args.val_audio_split)
    print(f"Loaded AudioCaps val: {len(audiocaps_val)} samples", flush=True)

    wavcaps_val = None
    wavcaps_share = 0.0
    if args.use_wavcaps:
        wavcaps_share = max(0.0, min(1.0, float(args.val_wavcaps_share)))
        if wavcaps_share > 0:
            try:
                wavcaps_val = WavCapsDataset(data_path=data_root, split=args.val_wavcaps_split)
                print(f"Loaded WavCaps val: {len(wavcaps_val)} samples", flush=True)
            except Exception as exc:
                print(f"Warning: Failed to load WavCaps val split ({exc}).", flush=True)
                fallback_samples = int(args.val_wavcaps_sample_size)
                if fallback_samples > 0 and wavcaps_train is not None:
                    sample_rng = random.Random(args.seed + 97)
                    total_wavcaps = len(wavcaps_train)
                    sample_size = min(total_wavcaps, fallback_samples)
                    sampled_indices = sample_rng.sample(range(total_wavcaps), sample_size)
                    sampled_indices.sort()
                    wavcaps_val = DatasetSubset(wavcaps_train, sampled_indices)
                    print(
                        f"Using {sample_size} WavCaps training samples for validation (seed={args.seed + 97}).",
                        flush=True,
                    )
                else:
                    print("Falling back to AudioCaps-only validation.", flush=True)
                    wavcaps_val = None
                    wavcaps_share = 0.0

    audio_val_limit = args.max_audio_val_samples if args.max_audio_val_samples > 0 else None

    audio_share = 1.0
    if wavcaps_val is not None:
        audio_share = 1.0 - wavcaps_share

    if audio_share > 0:
        val_audio_sources.append(("audiocaps", audiocaps_val))
        val_audio_weights.append(audio_share)

    if wavcaps_val is not None and wavcaps_share > 0:
        val_audio_sources.append(("wavcaps", wavcaps_val))
        val_audio_weights.append(wavcaps_share)

    if not val_audio_sources:
        # Fallback: ensure at least AudioCaps is present
        val_audio_sources.append(("audiocaps", audiocaps_val))
        val_audio_weights.append(1.0)
        wavcaps_share = 0.0

    audio_mix_dataset = AudioValidationMixDataset(
        val_audio_sources,
        weights=val_audio_weights,
        max_samples=audio_val_limit,
        shuffle=not args.disable_val_shuffle,
        seed=args.seed,
    )

    mix_counts = audio_mix_dataset.get_source_counts()
    mix_summary = ", ".join(f"{name}: {count}" for name, count in mix_counts.items())
    print(f"Validation audio mix → {mix_summary}", flush=True)

    val_dataset = CombinedValidationDataset(
        audio_mix_dataset,
        val_vl_dataset,
        max_audio_samples=None,
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

    # Log dataset sizes
    train_size = len(train_dataset) if hasattr(train_dataset, '__len__') else 'unknown'
    val_size = len(val_dataset) if hasattr(val_dataset, '__len__') else 'unknown'
    print(f"📊 Dataset sizes - Train: {train_size} samples, Val: {val_size} samples", flush=True)

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

    print(f"\n{'='*70}", flush=True)
    print(f"Initializing SAFEModel with config: {args.model_config}", flush=True)
    print(f"  LLM: {model_kwargs.get('llm_model_name', 'N/A')}", flush=True)
    print(f"  Vision: {model_kwargs.get('vision_model_name', 'N/A')}", flush=True)
    print(f"  Audio: {model_kwargs.get('audio_encoder_type', 'N/A')}", flush=True)
    print(f"{'='*70}", flush=True)
    import sys
    sys.stdout.flush()
    sys.stderr.flush()

    model = SAFEModel(**model_kwargs)

    print(f"✅ SAFEModel initialized successfully", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

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
        max_audio_eval_samples=args.max_audio_eval_samples,
        max_vl_eval_samples=args.max_vl_eval_samples,
        eval_with_audio_gate=not args.disable_eval_audio_gate,
        eval_audio_gate_comparison=args.eval_audio_gate_comparison,
        eval_logging_steps=args.eval_logging_steps,
        debug_logging=args.debug_logging,
        train_accuracy_interval=args.train_accuracy_interval,
        train_accuracy_warmup=args.train_accuracy_warmup,
        train_eval_batches=args.train_eval_batches,
        generation_max_new_tokens=args.generation_max_new_tokens,
        audio_generation_max_new_tokens=args.audio_generation_max_new_tokens,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        disable_bertscore=args.disable_bertscore,
        progress_log_timeout=args.progress_log_timeout,
        save_audio_csv=args.save_audio_csv,
        csv_min_accuracy=args.csv_min_accuracy,
        csv_max_samples=args.csv_max_samples,
        enable_scst_finetune=args.enable_scst_finetune,
        scst_epochs=args.scst_epochs,
        scst_learning_rate=args.scst_learning_rate,
        scst_num_samples=args.scst_num_samples,
        scst_sample_top_p=args.scst_sample_top_p,
        scst_sample_temperature=args.scst_sample_temperature,
        scst_reward_metric=args.scst_reward_metric,
        scst_patience_epochs=args.scst_patience_epochs,
        scst_improvement_threshold=args.scst_improvement_threshold,
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
    parser.add_argument("--val-wavcaps-share", type=float, default=0.5, help="Fraction of audio validation samples to draw from WavCaps (0.0-1.0)")
    parser.add_argument("--val-wavcaps-split", type=str, default="val", help="WavCaps split for validation when available")
    parser.add_argument("--val-wavcaps-sample-size", type=int, default=2048, help="If validation split is missing, sample this many examples from WavCaps train (<=0 disables)")
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
    parser.add_argument("--max-audio-eval-batches", type=int, default=0, help="Audio eval batch cap (<=0 disables)")
    parser.add_argument("--max-vl-eval-batches", type=int, default=0, help="VL eval batch cap (<=0 disables)")
    parser.add_argument("--max-audio-eval-samples", type=int, default=0, help="Audio evaluation sample cap (<=0 disables = full eval)")
    parser.add_argument("--max-vl-eval-samples", type=int, default=600, help="VL evaluation sample cap (<=0 disables)")
    parser.add_argument("--max-audio-val-samples", type=int, default=4096, help="Audio validation sample cap (<=0 disables)")
    parser.add_argument("--max-vqa-val-samples", type=int, default=4096, help="VQA validation sample cap (<=0 disables)")
    parser.add_argument("--eval-logging-steps", type=int, default=10, help="Eval logging frequency")
    parser.add_argument("--progress-log-timeout", type=int, default=600, help="Seconds between forced progress logs (0 disables time-based fallback)")
    parser.add_argument("--train-accuracy-interval", type=int, default=0, help="Interval for train accuracy logging")
    parser.add_argument("--train-accuracy-warmup", type=int, default=5, help="Warmup epochs before accuracy logging")
    parser.add_argument("--train-eval-batches", type=int, default=0, help="Evaluate training split after fit")
    parser.add_argument("--generation-max-new-tokens", type=int, default=32, help="Generation token budget")
    parser.add_argument("--audio-generation-max-new-tokens", type=int, default=20, help="Generation token budget for audio-only tasks")
    parser.add_argument("--output-root", type=str, default="experiments/full_training/runs", help="Run output directory")
    parser.add_argument("--model-config", choices=["demo", "full", "multimodal"], default="full", help="Model config preset")
    parser.add_argument("--disable-train-shuffle", action="store_true", help="Disable shuffling for training dataloader")
    parser.add_argument("--disable-val-shuffle", action="store_true", help="Disable shuffling when mixing validation datasets")
    parser.add_argument("--disable-eval-audio-gate", action="store_true", help="Evaluate without audio gating")
    parser.add_argument("--eval-audio-gate-comparison", action="store_true", help="Run with and without audio gate during eval")
    parser.add_argument("--debug-logging", action="store_true", help="Enable verbose trainer logging")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps for effective larger batch size")
    parser.add_argument("--disable-bertscore", action="store_true", help="Disable BERTScore evaluation (use token F1 only)")
    parser.add_argument("--save-audio-csv", action="store_true", help="Save audio evaluation samples to CSV")
    parser.add_argument("--csv-min-accuracy", type=float, default=0.45, help="Minimum audio accuracy to trigger CSV export")
    parser.add_argument("--csv-max-samples", type=int, default=500, help="Maximum audio samples to save to CSV")
    parser.add_argument("--enable-scst-finetune", action="store_true", help="Enable SCST fine-tuning after plateau")
    parser.add_argument("--scst-epochs", type=int, default=1, help="Number of SCST fine-tune epochs")
    parser.add_argument("--scst-learning-rate", type=float, default=5e-6, help="Learning rate during SCST")
    parser.add_argument("--scst-num-samples", type=int, default=1, help="Samples per clip for SCST updates")
    parser.add_argument("--scst-sample-top-p", type=float, default=0.9, help="Top-p for SCST sampling")
    parser.add_argument("--scst-sample-temperature", type=float, default=0.9, help="Temperature for SCST sampling")
    parser.add_argument(
        "--scst-reward-metric",
        choices=["cider", "spice", "spider"],
        default="cider",
        help="Reward metric for SCST",
    )
    parser.add_argument("--scst-patience-epochs", type=int, default=2, help="Epochs without improvement before SCST")
    parser.add_argument("--scst-improvement-threshold", type=float, default=1e-4, help="Minimum improvement to reset SCST patience")

    return parser.parse_args()


if __name__ == "__main__":
    print("[run_full_training.py] Entering main block...", flush=True)
    import sys
    sys.stdout.flush()
    print("[run_full_training.py] Parsing arguments...", flush=True)
    sys.stdout.flush()
    args = parse_args()
    print("[run_full_training.py] ✓ Arguments parsed", flush=True)
    sys.stdout.flush()
    print("[run_full_training.py] Starting experiment...", flush=True)
    sys.stdout.flush()
    run_experiment(args)
