#!/usr/bin/env python3
"""
train_ave_classifier.py - Simple Audio Classification with CLAP + MLP

Clean research setup:
    Audio -> CLAP Encoder (frozen) -> MLP Classifier -> 28 classes

This is a straightforward baseline to verify audio features work for AVE classification.
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torchaudio
except ImportError:
    torchaudio = None


# ============================================================================
# AVE Dataset
# ============================================================================

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
    """Simple AVE dataset loader."""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        sample_rate: int = 48000,
        max_length: float = 10.0,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_length)

        # Find data file
        split_files = {"train": "trainSet.txt", "val": "valSet.txt", "test": "testSet.txt"}
        data_file = self.data_path / split_files.get(split, f"{split}Set.txt")

        if not data_file.exists():
            # Try ave subdirectory
            data_file = self.data_path / "ave" / split_files.get(split, f"{split}Set.txt")

        self.examples = self._load_data(data_file)
        print(f"[AVEDataset] {split}: {len(self.examples)} samples")

    def _load_data(self, data_file: Path) -> List[Dict]:
        """Load AVE data file (format: category&video_id&quality)."""
        examples = []
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("&")
                if len(parts) >= 2:
                    category, video_id = parts[0], parts[1]
                    if category in AVE_LABEL_TO_IDX:
                        examples.append({
                            "category": category,
                            "video_id": video_id,
                            "label_idx": AVE_LABEL_TO_IDX[category],
                        })
        return examples

    def _find_audio(self, video_id: str) -> Optional[Path]:
        """Find audio file for video ID."""
        # Try different locations
        candidates = [
            self.data_path / f"{self.split}/audio/{video_id}.wav",
            self.data_path / f"audio/{video_id}.wav",
            self.data_path / f"{video_id}.wav",
            self.data_path / "ave" / f"{self.split}/audio/{video_id}.wav",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load and preprocess audio."""
        waveform, sr = torchaudio.load(str(path))

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)

        # Pad or truncate
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]
        elif waveform.shape[0] < self.max_samples:
            padding = self.max_samples - waveform.shape[0]
            waveform = F.pad(waveform, (0, padding))

        return waveform

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        audio_path = self._find_audio(ex["video_id"])

        if audio_path is None:
            # Return zeros if audio not found (will be filtered in training)
            waveform = torch.zeros(self.max_samples)
            valid = False
        else:
            try:
                waveform = self._load_audio(audio_path)
                valid = True
            except Exception as e:
                waveform = torch.zeros(self.max_samples)
                valid = False

        return {
            "waveform": waveform,
            "sample_rate": self.sample_rate,
            "label": ex["label_idx"],
            "category": ex["category"],
            "valid": valid,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate batch, filtering invalid samples."""
    valid_batch = [b for b in batch if b["valid"]]
    if not valid_batch:
        valid_batch = batch[:1]  # Keep at least one

    return {
        "waveform": torch.stack([b["waveform"] for b in valid_batch]),
        "sample_rate": valid_batch[0]["sample_rate"],
        "label": torch.tensor([b["label"] for b in valid_batch], dtype=torch.long),
        "category": [b["category"] for b in valid_batch],
    }


# ============================================================================
# Model
# ============================================================================

class AudioClassifier(nn.Module):
    """
    Simple audio classifier: CLAP encoder + MLP head.

    Architecture:
        Audio -> CLAP (frozen, 512-dim) -> MLP -> 28 classes
    """

    def __init__(
        self,
        num_classes: int = 28,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load CLAP encoder
        print("[Model] Loading CLAP encoder...")
        from safe.models.audio_encoders import CLAPAudioEncoder
        self.encoder = CLAPAudioEncoder(freeze=freeze_encoder)
        self.encoder_dim = 512  # CLAP output dimension
        print(f"[Model] CLAP encoder frozen: {freeze_encoder}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Initialize
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Model] Parameters: {trainable:,} trainable / {total:,} total")

    def forward(self, waveform: torch.Tensor, sample_rate: int = 48000) -> torch.Tensor:
        """
        Forward pass.

        Args:
            waveform: (batch, samples) audio waveform
            sample_rate: audio sample rate

        Returns:
            logits: (batch, num_classes)
        """
        # Encode audio with CLAP
        # CLAPAudioEncoder.forward() accepts (batch, samples) tensor directly
        with torch.no_grad():
            embeddings = self.encoder(waveform)  # (batch, 512)

        # Move embeddings to same device as classifier and ensure float
        embeddings = embeddings.to(waveform.device).float()

        # Classify
        logits = self.classifier(embeddings)
        return logits

    def get_embeddings(self, waveform: torch.Tensor, sample_rate: int = 48000) -> torch.Tensor:
        """Get CLAP embeddings without classification."""
        with torch.no_grad():
            embeddings = self.encoder(waveform)

        return embeddings.to(waveform.device).float()


# ============================================================================
# Training
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(loader):
        waveform = batch["waveform"].to(device)
        labels = batch["label"].to(device)
        sr = batch["sample_rate"]

        optimizer.zero_grad()

        logits = model(waveform, sr)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        # Metrics
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()

        total_loss += loss.item() * labels.size(0)
        total_correct += correct
        total_samples += labels.size(0)

        if batch_idx % 20 == 0:
            acc = correct / labels.size(0) * 100
            print(f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f} | Acc: {acc:.1f}%")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples * 100,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    for batch in loader:
        waveform = batch["waveform"].to(device)
        labels = batch["label"].to(device)
        sr = batch["sample_rate"]

        logits = model(waveform, sr)
        loss = F.cross_entropy(logits, labels)

        preds = logits.argmax(dim=-1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Per-class accuracy
    class_correct = {i: 0 for i in range(28)}
    class_total = {i: 0 for i in range(28)}
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples * 100,
        "class_correct": class_correct,
        "class_total": class_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Train AVE audio classifier")
    parser.add_argument("--data-path", type=str, required=True, help="Path to AVE dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/ave_classifier", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="AVE-Classification", help="Wandb project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    args = parser.parse_args()

    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Wandb
    if args.wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"ave-clf-{time.strftime('%Y%m%d-%H%M%S')}",
            config=vars(args),
        )

    # Data
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)

    train_dataset = AVEDataset(args.data_path, split="train")
    test_dataset = AVEDataset(args.data_path, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)

    model = AudioClassifier(
        num_classes=28,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_encoder=True,
    )
    model = model.to(device)

    # Optimizer
    optimizer = AdamW(
        model.classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    best_acc = 0.0

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Train | Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%")

        # Evaluate
        test_metrics = evaluate(model, test_loader, device)
        print(f"Test  | Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['accuracy']:.2f}%")

        scheduler.step()

        # Log
        if args.wandb and wandb is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "test/loss": test_metrics["loss"],
                "test/accuracy": test_metrics["accuracy"],
                "lr": scheduler.get_last_lr()[0],
            })

        # Save best
        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": best_acc,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> New best accuracy: {best_acc:.2f}%")

        # Save latest
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": test_metrics["accuracy"],
            "args": vars(args),
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "latest_model.pt"))

    # Final results
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print("="*60)

    # Print per-class accuracy for best model
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = evaluate(model, test_loader, device)

    print("\nPer-class accuracy:")
    for idx in range(28):
        cat = AVE_IDX_TO_LABEL[idx]
        correct = final_metrics["class_correct"][idx]
        total = final_metrics["class_total"][idx]
        acc = correct / total * 100 if total > 0 else 0
        print(f"  {cat:35s}: {acc:5.1f}% ({correct}/{total})")

    if args.wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
