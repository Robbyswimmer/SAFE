#!/usr/bin/env python3
"""
train_audio_classification.py - Audio classification training script

A clean, focused training script for audio classification tasks.
Uses CLAP audio encoder with a classification head.

Supports:
- AVE (Audio-Visual Event) dataset
- Standard audio classification with cross-entropy loss
- Accuracy, F1, and confusion matrix metrics
- Mixed precision training
- Distributed training support
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Optional: Weights & Biases
try:
    import wandb
except ImportError:
    wandb = None

# SAFE imports
from safe.models.audio_encoders import CLAPAudioEncoder


# ============================================================================
# SECTION 1: UTILITIES
# ============================================================================

def setup_distributed() -> Dict[str, Any]:
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

        return {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "is_main": rank == 0,
            "distributed": True,
        }
    else:
        return {
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "is_main": True,
            "distributed": False,
        }


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0):
    """Set random seeds for reproducibility."""
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# SECTION 2: AVE DATASET
# ============================================================================

# AVE dataset has 28 event categories
AVE_CATEGORIES = [
    "Church bell", "Male speech, man speaking", "Bark", "Fixed-wing aircraft, airplane",
    "Race car, auto racing", "Female speech, woman speaking", "Helicopter", "Violin, fiddle",
    "Flute", "Ukulele", "Frying (food)", "Truck", "Shofar", "Motorcycle", "Acoustic guitar",
    "Train horn", "Clock", "Banjo", "Goat", "Baby cry, infant cry", "Bus", "Chainsaw",
    "Cat", "Horse", "Toilet flush", "Rodents, rats, mice", "Accordion", "Mandolin"
]

AVE_LABEL_TO_IDX = {label: idx for idx, label in enumerate(AVE_CATEGORIES)}
AVE_IDX_TO_LABEL = {idx: label for label, idx in AVE_LABEL_TO_IDX.items()}


class AVEDataset(Dataset):
    """
    Audio-Visual Event (AVE) dataset for audio classification.

    Expected directory structure:
        data_path/
            ave/
                train.json (or train.jsonl)
                val.json (or val.jsonl)
                test.json (or test.jsonl)
                audio/
                    train/
                    val/
                    test/

    JSON format:
        [
            {
                "id": "video_id",
                "audio": "audio/train/video_id.wav",
                "label": "Church bell",
                "category_id": 0
            },
            ...
        ]

    Or JSONL format (one JSON object per line).
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        sample_rate: int = 48000,
        max_length: float = 10.0,
        label_map: Optional[Dict[str, int]] = None,
    ):
        self.data_path = Path(data_path).expanduser().resolve()
        self.split = split
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_samples = int(sample_rate * max_length)

        # Use provided label map or default AVE categories
        self.label_map = label_map or AVE_LABEL_TO_IDX
        self.num_classes = len(self.label_map)

        # Find dataset directory
        dataset_dir = self.data_path / "ave"
        if not dataset_dir.exists():
            # Try without subdirectory (data_path is directly the ave folder)
            dataset_dir = self.data_path
        self.dataset_dir = dataset_dir

        # Find data file
        data_file = self._find_data_file(split)
        self.examples = self._load_data(data_file)

        print(f"[AVEDataset] Loaded {len(self.examples)} samples from {data_file.name} ({split})", flush=True)

        # Build label statistics
        label_counts = defaultdict(int)
        for ex in self.examples:
            label_counts[ex.get("label", "unknown")] += 1
        self._label_counts = dict(label_counts)

    def _find_data_file(self, split: str) -> Path:
        """Find the data file for the given split."""
        candidates = [
            self.dataset_dir / f"{split}.json",
            self.dataset_dir / f"{split}.jsonl",
            self.dataset_dir / f"ave_{split}.json",
            self.dataset_dir / f"ave_{split}.jsonl",
            self.dataset_dir / f"{split}_data.json",
            self.dataset_dir / f"{split}_data.jsonl",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not find data file for AVE split '{split}'. "
            f"Looked for: {', '.join(str(p) for p in candidates)}"
        )

    def _load_data(self, data_file: Path) -> List[Dict[str, Any]]:
        """Load data from JSON or JSONL file."""
        examples = []

        if data_file.suffix == ".jsonl":
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        examples.append(json.loads(line))
        else:
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    examples = data["data"]
                elif isinstance(data, list):
                    examples = data
                else:
                    raise ValueError(f"Unexpected JSON format in {data_file}")

        return examples

    def _resolve_audio_path(self, entry: Dict[str, Any]) -> Optional[Path]:
        """Resolve audio path from entry."""
        audio_path = entry.get("audio") or entry.get("audio_path") or entry.get("file_path")

        if not audio_path:
            return None

        audio_path = Path(audio_path)

        # Try different path resolutions
        candidates = [
            audio_path if audio_path.is_absolute() else None,
            self.dataset_dir / audio_path,
            self.data_path / audio_path,
            self.dataset_dir / "audio" / self.split / audio_path.name,
        ]

        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate

        return None

    def _load_audio(self, entry: Dict[str, Any]) -> Optional[Tuple[torch.Tensor, int]]:
        """Load and preprocess audio file."""
        audio_file = self._resolve_audio_path(entry)

        if audio_file is None:
            return None

        try:
            import torchaudio

            waveform, sr = torchaudio.load(str(audio_file))

            # Convert to mono
            if waveform.dim() == 2 and waveform.size(0) > 1:
                waveform = waveform.mean(dim=0)
            elif waveform.dim() == 2:
                waveform = waveform.squeeze(0)

            # Resample if needed
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform.unsqueeze(0), sr, self.sample_rate
                ).squeeze(0)

            # Truncate or pad
            if waveform.size(-1) > self.max_samples:
                waveform = waveform[:self.max_samples]
            elif waveform.size(-1) < self.max_samples:
                padding = self.max_samples - waveform.size(-1)
                waveform = F.pad(waveform, (0, padding))

            return (waveform, self.sample_rate)

        except Exception as e:
            if not hasattr(self, "_load_error_count"):
                self._load_error_count = 0
            if self._load_error_count < 3:
                print(f"[AVEDataset] Error loading audio {audio_file}: {e}", flush=True)
                self._load_error_count += 1
            return None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.examples[idx]

        # Get label
        label_str = entry.get("label") or entry.get("category")
        if label_str is None:
            # Try to use category_id directly
            label_idx = entry.get("category_id", 0)
        else:
            label_idx = self.label_map.get(label_str, 0)

        # Load audio
        audio = self._load_audio(entry)

        return {
            "sample_id": entry.get("id") or entry.get("video_id") or idx,
            "audio": audio,
            "label": label_idx,
            "label_str": label_str or AVE_IDX_TO_LABEL.get(label_idx, "unknown"),
        }

    def get_label_weights(self) -> torch.Tensor:
        """Compute inverse frequency weights for class balancing."""
        counts = torch.zeros(self.num_classes)
        for ex in self.examples:
            label_str = ex.get("label") or ex.get("category")
            if label_str:
                idx = self.label_map.get(label_str, 0)
            else:
                idx = ex.get("category_id", 0)
            counts[idx] += 1

        # Inverse frequency with smoothing
        weights = 1.0 / (counts + 1.0)
        weights = weights / weights.sum() * self.num_classes
        return weights


def collate_classification_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for classification batches."""
    # Filter out samples with missing audio
    valid_batch = [s for s in batch if s.get("audio") is not None]

    if not valid_batch:
        return {
            "audio": None,
            "labels": torch.tensor([], dtype=torch.long),
            "sample_ids": [],
            "label_strs": [],
        }

    audios = []
    labels = []
    sample_ids = []
    label_strs = []

    for sample in valid_batch:
        audios.append(sample["audio"])
        labels.append(sample["label"])
        sample_ids.append(sample["sample_id"])
        label_strs.append(sample["label_str"])

    return {
        "audio": audios,  # List of (waveform, sr) tuples
        "labels": torch.tensor(labels, dtype=torch.long),
        "sample_ids": sample_ids,
        "label_strs": label_strs,
    }


# ============================================================================
# SECTION 3: AUDIO CLASSIFICATION MODEL
# ============================================================================

class AudioClassifier(nn.Module):
    """
    Audio classification model using CLAP encoder.

    Architecture:
        Audio -> CLAP Encoder (frozen) -> Projection -> Classification Head -> Logits
    """

    def __init__(
        self,
        num_classes: int = 28,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Initialize CLAP encoder
        print("[AudioClassifier] Initializing CLAP audio encoder...", flush=True)
        self.audio_encoder = CLAPAudioEncoder(freeze=freeze_encoder)
        self.audio_embed_dim = self.audio_encoder.audio_embed_dim  # 512

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.audio_embed_dim),
            nn.Linear(self.audio_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Initialize classifier weights
        self._init_weights()

        print(f"[AudioClassifier] Initialized with {num_classes} classes", flush=True)

    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        audio: List[Tuple[torch.Tensor, int]],
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            audio: List of (waveform, sample_rate) tuples
            return_embeddings: If True, also return audio embeddings

        Returns:
            logits: Classification logits (batch_size, num_classes)
            embeddings: Audio embeddings (batch_size, embed_dim) if return_embeddings=True
        """
        # Encode audio
        audio_embeddings = self.audio_encoder(audio)  # (batch_size, audio_embed_dim)

        # Classify
        logits = self.classifier(audio_embeddings)  # (batch_size, num_classes)

        if return_embeddings:
            return logits, audio_embeddings
        return logits

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get trainable parameters (classifier only if encoder is frozen)."""
        return [p for p in self.parameters() if p.requires_grad]


# ============================================================================
# SECTION 4: METRICS
# ============================================================================

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).float().sum()
    return (correct / len(labels)).item()


def compute_metrics(
    all_preds: List[int],
    all_labels: List[int],
    num_classes: int,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Returns:
        Dictionary with accuracy, per-class metrics, and confusion matrix
    """
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()

    # Per-class metrics
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    per_class_pred_total = defaultdict(int)

    for pred, label in zip(all_preds, all_labels):
        per_class_total[label] += 1
        per_class_pred_total[pred] += 1
        if pred == label:
            per_class_correct[label] += 1

    # Precision, Recall, F1 per class
    per_class_metrics = {}
    precisions = []
    recalls = []
    f1s = []

    for cls in range(num_classes):
        tp = per_class_correct[cls]
        total_true = per_class_total[cls]
        total_pred = per_class_pred_total[cls]

        precision = tp / total_pred if total_pred > 0 else 0
        recall = tp / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        cls_name = label_names[cls] if label_names else str(cls)
        per_class_metrics[cls_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": total_true,
        }

        if total_true > 0:  # Only include classes with samples
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    # Macro averages
    macro_precision = np.mean(precisions) if precisions else 0
    macro_recall = np.mean(recalls) if recalls else 0
    macro_f1 = np.mean(f1s) if f1s else 0

    # Confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(all_preds, all_labels):
        confusion_matrix[label, pred] += 1

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "confusion_matrix": confusion_matrix.tolist(),
    }


# ============================================================================
# SECTION 5: TRAINING LOOP
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    dist_info: Dict[str, Any],
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        if batch["audio"] is None or len(batch["labels"]) == 0:
            continue

        audio = batch["audio"]
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast(enabled=args.fp16):
            logits = model(audio)
            loss = criterion(logits, labels)

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accumulate metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)
        num_batches += 1

        # Log progress
        if dist_info["is_main"] and (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / num_batches
            accuracy = total_correct / total_samples if total_samples > 0 else 0
            elapsed = time.time() - start_time
            samples_per_sec = total_samples / elapsed if elapsed > 0 else 0

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | "
                f"LR: {lr:.2e} | {samples_per_sec:.1f} samples/s",
                flush=True,
            )

    # Compute epoch metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "samples": total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    label_names: Optional[List[str]] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate model on validation/test set."""
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        if batch["audio"] is None or len(batch["labels"]) == 0:
            continue

        audio = batch["audio"]
        labels = batch["labels"].to(device)

        logits = model(audio)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

        total_loss += loss.item()
        num_batches += 1

    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels, num_classes, label_names)
    metrics["loss"] = total_loss / num_batches if num_batches > 0 else 0
    metrics["num_samples"] = len(all_labels)

    return metrics


# ============================================================================
# SECTION 6: MAIN TRAINING FUNCTION
# ============================================================================

def main(args: argparse.Namespace):
    """Main training function."""
    # Setup distributed training
    dist_info = setup_distributed()
    device = torch.device(f"cuda:{dist_info['local_rank']}" if torch.cuda.is_available() else "cpu")

    if dist_info["is_main"]:
        print("=" * 60)
        print("Audio Classification Training")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Distributed: {dist_info['distributed']} (world_size={dist_info['world_size']})")
        print(f"Data path: {args.data_path}")
        print(f"Output dir: {args.output_dir}")
        print()

    # Set seed
    set_seed(args.seed, dist_info["rank"])

    # Create output directory
    output_dir = Path(args.output_dir)
    if dist_info["is_main"]:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    if dist_info["is_main"]:
        print("[Data] Loading datasets...")

    train_dataset = AVEDataset(
        data_path=args.data_path,
        split="train",
        sample_rate=48000,
        max_length=10.0,
    )

    val_dataset = AVEDataset(
        data_path=args.data_path,
        split="val",
        sample_rate=48000,
        max_length=10.0,
    )

    num_classes = train_dataset.num_classes
    label_names = AVE_CATEGORIES if num_classes == 28 else None

    if dist_info["is_main"]:
        print(f"[Data] Train samples: {len(train_dataset)}")
        print(f"[Data] Val samples: {len(val_dataset)}")
        print(f"[Data] Num classes: {num_classes}")
        print()

    # Create data loaders
    train_sampler = DistributedSampler(train_dataset) if dist_info["distributed"] else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist_info["distributed"] else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_classification_batch,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_classification_batch,
        pin_memory=True,
    )

    # Create model
    if dist_info["is_main"]:
        print("[Model] Creating audio classifier...")

    model = AudioClassifier(
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_encoder=not args.train_encoder,
    )
    model = model.to(device)

    if dist_info["distributed"]:
        model = DDP(model, device_ids=[dist_info["local_rank"]])

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    if dist_info["is_main"]:
        print(f"[Model] Total parameters: {total_params / 1e6:.2f}M")
        print(f"[Model] Trainable parameters: {trainable_params / 1e6:.2f}M")
        print()

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Create scheduler
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=args.learning_rate * 0.01,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # Create gradient scaler for mixed precision
    scaler = GradScaler() if args.fp16 else None

    # Initialize wandb
    if wandb is not None and args.wandb and dist_info["is_main"]:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"audio-clf-{time.strftime('%Y%m%d-%H%M%S')}",
            config=vars(args),
        )

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0

    if dist_info["is_main"]:
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)

    for epoch in range(1, args.num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if dist_info["is_main"]:
            print(f"\nEpoch {epoch}/{args.num_epochs}")
            print("-" * 40)

        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
            dist_info=dist_info,
        )

        if dist_info["is_main"]:
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")

        # Evaluate
        if epoch % args.eval_frequency == 0 or epoch == args.num_epochs:
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                num_classes=num_classes,
                label_names=label_names,
                max_batches=args.max_eval_batches,
            )

            if dist_info["is_main"]:
                print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
                print(f"  Val F1 (macro): {val_metrics['macro_f1']:.4f}")

                # Log to wandb
                if wandb is not None and args.wandb:
                    wandb.log({
                        "epoch": epoch,
                        "train/loss": train_metrics["loss"],
                        "train/accuracy": train_metrics["accuracy"],
                        "val/loss": val_metrics["loss"],
                        "val/accuracy": val_metrics["accuracy"],
                        "val/macro_f1": val_metrics["macro_f1"],
                        "val/macro_precision": val_metrics["macro_precision"],
                        "val/macro_recall": val_metrics["macro_recall"],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    })

                # Save best model
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                    best_epoch = epoch

                    checkpoint_path = output_dir / "best_model.pt"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict() if dist_info["distributed"] else model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": val_metrics["accuracy"],
                        "val_metrics": val_metrics,
                        "args": vars(args),
                    }, checkpoint_path)
                    print(f"  Saved best model (acc={best_val_acc:.4f})")

        # Save periodic checkpoint
        if dist_info["is_main"] and epoch % args.save_frequency == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.module.state_dict() if dist_info["distributed"] else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }, checkpoint_path)

    # Final summary
    if dist_info["is_main"]:
        print("\n" + "=" * 60)
        print("Training Complete")
        print("=" * 60)
        print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
        print(f"Model saved to: {output_dir}")

        if wandb is not None and args.wandb:
            wandb.finish()

    cleanup_distributed()


# ============================================================================
# SECTION 7: ARGUMENT PARSER
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train audio classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to data directory containing AVE dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/audio_classification",
        help="Output directory for checkpoints and logs"
    )

    # Model arguments
    parser.add_argument(
        "--hidden-dim", type=int, default=512,
        help="Hidden dimension of classification head"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate"
    )
    parser.add_argument(
        "--train-encoder", action="store_true",
        help="Fine-tune the audio encoder (default: frozen)"
    )

    # Training arguments
    parser.add_argument(
        "--num-epochs", type=int, default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.1,
        help="Warmup ratio of total steps"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="Max gradient norm for clipping (0 to disable)"
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.1,
        help="Label smoothing factor"
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Use mixed precision training"
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval-frequency", type=int, default=1,
        help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--max-eval-batches", type=int, default=None,
        help="Max batches for evaluation (for debugging)"
    )

    # Logging arguments
    parser.add_argument(
        "--log-interval", type=int, default=10,
        help="Log every N batches"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=10,
        help="Save checkpoint every N epochs"
    )

    # Wandb arguments
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="audio-classification",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="W&B run name"
    )

    # Other arguments
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of data loading workers"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
