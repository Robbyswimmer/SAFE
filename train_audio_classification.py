#!/usr/bin/env python3
"""
train_audio_classification.py - Generative Audio Classification Training

Uses the EXACT same architecture and training objective as train_safe.py (captioning),
but with single-word answers for classification.

Architecture:
    Audio -> CLAP Encoder (frozen)
          -> Audio Projector (trainable)
          -> Multi-layer Fusion via hooks at fusion_layer_indices
          -> LLaVA 1.5 13B LLM (frozen)
          -> Generate text (LM loss on answer tokens)

This proves the SAFE architecture works by showing a frozen LLM can
correctly generate category names (e.g., "Dog", "Church bell") from audio alone.

Supports:
- AVE (Audio-Visual Event) dataset with 28 categories
- Language modeling loss (same as captioning)
- Generation-based evaluation (generate text, match to labels)
- Mixed precision training
- Distributed training support
- WANDB logging
"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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

# SAFE imports - use SAFEModel directly for exact architecture match
from safe.models.safe_model import SAFEModel
from configs.model_configs import get_config


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


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """
    AdamW with standard no-decay rules (matches train_safe.py intent).

    Critical: avoid decaying scalar scale parameters (e.g., output_scale/residual_scale)
    and norms/biases, otherwise the model can suppress audio by driving scales to ~0.
    """

    base_model = model.module if hasattr(model, "module") else model
    safe_model = getattr(base_model, "safe_model", None)
    if safe_model is None:
        raise RuntimeError("Expected model.safe_model to exist for optimizer building.")

    def _use_weight_decay(param_name: str, param: torch.nn.Parameter) -> bool:
        if not getattr(param, "requires_grad", False):
            return False
        if param.ndim <= 1:
            return False
        name = str(param_name).lower()
        if name.endswith(".bias") or name.endswith("bias"):
            return False
        if "layernorm" in name or "layer_norm" in name or ".norm" in name or "norm." in name:
            return False
        if name.endswith(("output_scale", "residual_scale")):
            return False
        return True

    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for name, param in safe_model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        if _use_weight_decay(name, param):
            decay.append(param)
        else:
            no_decay.append(param)

    if not decay and not no_decay:
        raise RuntimeError("No trainable SAFE parameters found for optimizer.")

    param_groups: List[Dict[str, Any]] = []
    if decay:
        param_groups.append({"params": decay, "lr": learning_rate, "weight_decay": float(weight_decay), "name": "decay"})
    if no_decay:
        param_groups.append({"params": no_decay, "lr": learning_rate, "weight_decay": 0.0, "name": "no_decay"})

    return AdamW(param_groups, lr=learning_rate, weight_decay=0.0)


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

# Simplified labels for generation (easier for LLM to produce)
AVE_SIMPLE_LABELS = {
    "Church bell": "Church bell",
    "Male speech, man speaking": "Male speech",
    "Bark": "Dog barking",
    "Fixed-wing aircraft, airplane": "Airplane",
    "Race car, auto racing": "Race car",
    "Female speech, woman speaking": "Female speech",
    "Helicopter": "Helicopter",
    "Violin, fiddle": "Violin",
    "Flute": "Flute",
    "Ukulele": "Ukulele",
    "Frying (food)": "Frying",
    "Truck": "Truck",
    "Shofar": "Shofar",
    "Motorcycle": "Motorcycle",
    "Acoustic guitar": "Guitar",
    "Train horn": "Train horn",
    "Clock": "Clock",
    "Banjo": "Banjo",
    "Goat": "Goat",
    "Baby cry, infant cry": "Baby crying",
    "Bus": "Bus",
    "Chainsaw": "Chainsaw",
    "Cat": "Cat",
    "Horse": "Horse",
    "Toilet flush": "Toilet flush",
    "Rodents, rats, mice": "Rodents",
    "Accordion": "Accordion",
    "Mandolin": "Mandolin"
}

# Reverse mapping for matching generated text to labels
SIMPLE_TO_ORIGINAL = {v.lower(): k for k, v in AVE_SIMPLE_LABELS.items()}


class AVEDataset(Dataset):
    """
    Audio-Visual Event (AVE) dataset for audio classification.

    Returns both numeric label indices and label strings for generative training.
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
            dataset_dir = self.data_path
        self.dataset_dir = dataset_dir

        # Find and load data file
        data_file = self._find_data_file(split)
        self.examples = self._load_data(data_file)

        print(f"[AVEDataset] Loaded {len(self.examples)} samples from {data_file.name} ({split})", flush=True)
        print(f"[AVEDataset] Dataset dir: {self.dataset_dir}", flush=True)

        # Verify audio files exist - sample check
        found_count = 0
        missing_count = 0
        for i, ex in enumerate(self.examples[:10]):
            audio_path = self._resolve_audio_path(ex)
            if audio_path:
                found_count += 1
                if i == 0:
                    print(f"[AVEDataset] Sample audio path: {audio_path}", flush=True)
            else:
                missing_count += 1
                if missing_count <= 3:
                    audio_name = ex.get('audio') or ex.get('audio_path')
                    print(f"[AVEDataset] Missing audio for: {audio_name}", flush=True)

        print(f"[AVEDataset] Audio check (first 10): {found_count} found, {missing_count} missing", flush=True)

    def _find_data_file(self, split: str) -> Path:
        """Find the data file for the given split."""
        split_to_txt = {
            "train": "trainSet.txt",
            "val": "valSet.txt",
            "test": "testSet.txt",
        }

        candidates = [
            self.dataset_dir / split_to_txt.get(split, f"{split}Set.txt"),
            self.dataset_dir / f"{split}.json",
            self.dataset_dir / f"{split}.jsonl",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Could not find data file for split '{split}' in {self.dataset_dir}")

    def _load_data(self, data_file: Path) -> List[Dict[str, Any]]:
        """Load data from file."""
        if data_file.suffix == ".txt":
            return self._load_ave_text_format(data_file)
        elif data_file.suffix == ".jsonl":
            examples = []
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        examples.append(json.loads(line))
            return examples
        else:
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                elif isinstance(data, list):
                    return data
                else:
                    raise ValueError(f"Unexpected JSON format in {data_file}")

    def _load_ave_text_format(self, data_file: Path) -> List[Dict[str, Any]]:
        """
        Load data from original AVE text format.
        Format: Category&VideoID&Quality&StartTime&EndTime
        """
        examples = []

        with open(data_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("&")
                if len(parts) != 5:
                    continue

                category, video_id, quality, start_time, end_time = parts
                category = category.strip()
                video_id = video_id.strip()
                start_time = start_time.strip()
                end_time = end_time.strip()

                audio_filename = f"{video_id}_{start_time}_{end_time}.wav"

                examples.append({
                    "id": f"{video_id}_{start_time}_{end_time}",
                    "video_id": video_id,
                    "label": category,
                    "audio": audio_filename,
                })

        return examples

    def _resolve_audio_path(self, entry: Dict[str, Any]) -> Optional[Path]:
        """Resolve audio path from entry."""
        audio_path = entry.get("audio") or entry.get("audio_path")
        if not audio_path:
            return None

        audio_path = Path(audio_path)

        split_to_dir = {"train": "train", "val": "test", "test": "test"}
        split_dir = split_to_dir.get(self.split, self.split)

        candidates = [
            audio_path if audio_path.is_absolute() else None,
            self.dataset_dir / split_dir / "audio" / audio_path.name,
            self.dataset_dir / "train" / "audio" / audio_path.name,
            self.dataset_dir / "test" / "audio" / audio_path.name,
            self.dataset_dir / "audio" / audio_path.name,
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

        # Get label string
        label_str = entry.get("label") or entry.get("category")
        if isinstance(label_str, str):
            label_str = label_str.strip()

        # Get label index
        label_idx = self.label_map.get(label_str, -1) if label_str else -1

        # Get simplified label for generation target
        simple_label = AVE_SIMPLE_LABELS.get(label_str, label_str) if label_str else "unknown"

        # Load audio
        audio = self._load_audio(entry)

        # Skip invalid samples
        if label_idx < 0:
            audio = None

        return {
            "sample_id": entry.get("id") or idx,
            "audio": audio,
            "label": label_idx,
            "label_str": label_str or "unknown",
            "target_text": simple_label,  # Text target for generation
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for generative classification."""
    # Filter out samples with missing audio or invalid labels
    valid_batch = [s for s in batch if s.get("audio") is not None and int(s.get("label", -1)) >= 0]

    if not valid_batch:
        return {
            "audio": None,
            "labels": torch.tensor([], dtype=torch.long),
            "label_strs": [],
            "target_texts": [],
            "sample_ids": [],
        }

    audios = []
    labels = []
    label_strs = []
    target_texts = []
    sample_ids = []

    for sample in valid_batch:
        audios.append(sample["audio"])
        labels.append(sample["label"])
        label_strs.append(sample["label_str"])
        target_texts.append(sample["target_text"])
        sample_ids.append(sample["sample_id"])

    return {
        "audio": audios,  # List of (waveform, sr) tuples
        "labels": torch.tensor(labels, dtype=torch.long),
        "label_strs": label_strs,
        "target_texts": target_texts,  # For training: these are the answers
        "sample_ids": sample_ids,
    }


# ============================================================================
# SECTION 3: GENERATIVE CLASSIFIER (SAME AS CAPTIONING)
# ============================================================================

class SAFEGenerativeClassifier(nn.Module):
    """
    Audio classification via generation using EXACT SAME architecture as train_safe.py.

    Instead of a classification head, we:
    1. Train with LM loss on answer tokens (same as captioning)
    2. Evaluate by generating text and matching to labels

    Architecture:
        Audio -> CLAP Encoder (frozen)
              -> Audio Projector (trainable)
              -> Multi-layer Fusion via hooks at fusion_layer_indices
              -> LLaVA 1.5 13B LLM (frozen)
              -> Generate category name

    Frozen: CLAP encoder, LLaVA 13B, CLIP vision
    Trainable: Audio projector, Fusion adapters (SimpleFusionAdapter at each layer)
    """

    PROMPT = "What is in this sound? Answer in a few words."

    def __init__(
        self,
        model_config: str = "phase1",
        fusion_layer_indices: Optional[List[int]] = None,
        use_ffn: Optional[bool] = None,
    ):
        super().__init__()

        # 1. Load config (same as train_safe.py)
        config = get_config(model_config)
        print(f"[SAFEGenerativeClassifier] Using config: {model_config}", flush=True)

        # Override fusion layers if specified
        if fusion_layer_indices is not None:
            config["fusion_layer_indices"] = fusion_layer_indices
            if "fusion_config" in config and "modalities" in config["fusion_config"]:
                config["fusion_config"]["modalities"]["audio"]["layer_indices"] = fusion_layer_indices

        # Optional FFN override
        if "fusion_config" in config and isinstance(config["fusion_config"], dict):
            if use_ffn is not None:
                config["fusion_config"]["use_ffn"] = bool(use_ffn)
            print(f"[SAFEGenerativeClassifier] Fusion FFN: {config['fusion_config'].get('use_ffn', True)}", flush=True)

        # 2. Initialize SAFEModel (EXACT same as train_safe.py)
        print("[SAFEGenerativeClassifier] Initializing SAFEModel...", flush=True)
        self.safe_model = SAFEModel(
            llm_model_name=config.get("llm_model_name", "llava-hf/llava-1.5-13b-hf"),
            vision_model_name=config.get("vision_model_name", "openai/clip-vit-large-patch14"),
            audio_encoder_type=config.get("audio_encoder_type", "clap"),
            audio_encoder_config=config.get("audio_encoder_config"),
            projector_type=config.get("projector_type", "standard"),
            num_audio_tokens=config.get("num_audio_tokens", 8),
            projector_config=config.get("projector_config"),
            fusion_type=config.get("fusion_type", "multilayer"),
            fusion_layer_indices=config.get("fusion_layer_indices"),
            lora_rank=config.get("lora_rank", 8),
            fusion_config=config.get("fusion_config"),
            freeze_base_vl=True,
            freeze_audio_encoder=True,
            label_smoothing=float(config.get("label_smoothing", 0.0) or 0.0),
            llm_hidden_size=config.get("llm_hidden_size", 5120),
            audio_embed_dim=config.get("audio_embed_dim", 512),
        )
        print("[SAFEGenerativeClassifier] ✓ SAFEModel initialized", flush=True)

        # Store config for reference
        self.config = config

        # Enable audio training mode
        self.safe_model.enable_audio_training()

        print(f"[SAFEGenerativeClassifier] Fusion layers: {config.get('fusion_layer_indices')}", flush=True)
        print(f"[SAFEGenerativeClassifier] Prompt: '{self.PROMPT}'", flush=True)

        # Debug info
        if hasattr(self.safe_model.fusion_adapter, 'fusion_adapters'):
            adapter_keys = list(self.safe_model.fusion_adapter.fusion_adapters.keys())
            print(f"[SAFEGenerativeClassifier] Fusion adapter keys: {adapter_keys}", flush=True)
            print(f"[SAFEGenerativeClassifier] enable_midlayer_fusion: {self.safe_model.enable_midlayer_fusion}", flush=True)

    def forward(
        self,
        audio: List[Tuple[torch.Tensor, int]],
        target_texts: List[str],
        label_smoothing: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (EXACT same as train_safe.py captioning).

        Args:
            audio: List of (waveform, sample_rate) tuples
            target_texts: List of target strings (e.g., ["Dog barking", "Church bell"])
            label_smoothing: Label smoothing factor for loss

        Returns:
            Dict with 'loss' and optionally 'logits'
        """
        batch_size = len(audio)
        device = next(self.safe_model.audio_projector.parameters()).device

        # Mirror train_safe.py: build tensors via prepare_multimodal_inputs, then call SAFEModel.forward()
        # with (input_ids, attention_mask, labels, audio_tokens, audio_attention_mask).
        inputs = self.safe_model.prepare_multimodal_inputs(
            text=[self.PROMPT] * batch_size,
            images=None,
            audio=audio,
            answers=target_texts,
            device=device,
            training_mode=True,
        )

        # Debug: ensure input_ids exists
        if "input_ids" not in inputs or inputs["input_ids"] is None:
            raise ValueError(
                f"prepare_multimodal_inputs returned invalid result. "
                f"Keys: {list(inputs.keys())}, input_ids: {inputs.get('input_ids')}"
            )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.to(device)
        audio_tokens = inputs.get("audio_tokens")
        if audio_tokens is not None:
            audio_tokens = audio_tokens.to(device)
        audio_attention_mask = inputs.get("audio_attention_mask")
        if audio_attention_mask is not None:
            audio_attention_mask = audio_attention_mask.to(device)

        # Set label smoothing on the SAFEModel (SAFEModel.forward uses self.label_smoothing).
        if label_smoothing is not None:
            try:
                self.safe_model.label_smoothing = float(label_smoothing)
            except Exception:
                pass

        return self.safe_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            audio_tokens=audio_tokens,
            audio_attention_mask=audio_attention_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        audio: List[Tuple[torch.Tensor, int]],
        max_new_tokens: int = 20,
        num_beams: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> List[str]:
        """
        Generate text predictions for classification.

        Args:
            audio: List of (waveform, sample_rate) tuples
            max_new_tokens: Maximum tokens to generate
            num_beams: Beam search width (1 = greedy)
            temperature: Sampling temperature
            do_sample: Whether to sample

        Returns:
            List of generated strings
        """
        batch_size = len(audio)

        # Generate using SAFEModel (same as inference in train_safe.py)
        generated_ids = self.safe_model.generate(
            text=[self.PROMPT] * batch_size,
            images=None,
            audio=audio,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample,
        )

        # Decode generated token IDs to strings
        tokenizer = self.safe_model.base_vl.tokenizer
        if tokenizer is None:
            tokenizer = self.safe_model.base_vl.processor.tokenizer

        # Handle both tensor and list outputs
        if torch.is_tensor(generated_ids):
            generated_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        elif isinstance(generated_ids, list):
            # Could be list of tensors or list of strings
            if len(generated_ids) > 0 and torch.is_tensor(generated_ids[0]):
                generated_texts = [
                    tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in generated_ids
                ]
            else:
                # Already strings
                generated_texts = generated_ids
        else:
            generated_texts = [str(generated_ids)]

        # Extract only the generated response (after ASSISTANT:)
        cleaned_texts = []
        for text in generated_texts:
            # LLaVA format: "USER: ... ASSISTANT: <response>"
            if "ASSISTANT:" in text:
                response = text.split("ASSISTANT:")[-1].strip()
            elif "assistant:" in text.lower():
                response = text.lower().split("assistant:")[-1].strip()
            else:
                # Just take the text as-is
                response = text.strip()
            cleaned_texts.append(response)

        return cleaned_texts

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get trainable parameters (same as train_safe.py)."""
        return list(self.safe_model.get_trainable_parameters())

    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        if mode:
            self.safe_model.enable_audio_training()
        return self

    def eval(self):
        """Set eval mode."""
        super().eval()
        self.safe_model.eval()
        return self


# ============================================================================
# SECTION 4: LABEL MATCHING
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def match_generated_to_label(generated: str, label_names: List[str]) -> Tuple[int, float]:
    """
    Match generated text to one of the label names.

    Returns:
        (predicted_index, confidence_score)
    """
    gen_norm = normalize_text(generated)

    best_idx = -1
    best_score = 0.0

    for idx, label in enumerate(label_names):
        label_norm = normalize_text(label)
        simple_norm = normalize_text(AVE_SIMPLE_LABELS.get(label, label))

        # Exact match
        if gen_norm == label_norm or gen_norm == simple_norm:
            return idx, 1.0

        # Check if label is contained in generated text
        if label_norm in gen_norm or simple_norm in gen_norm:
            score = len(label_norm) / max(len(gen_norm), 1)
            if score > best_score:
                best_score = score
                best_idx = idx

        # Check if generated text is contained in label
        if gen_norm in label_norm or gen_norm in simple_norm:
            score = len(gen_norm) / max(len(label_norm), 1)
            if score > best_score:
                best_score = score
                best_idx = idx

        # Word overlap
        gen_words = set(gen_norm.split())
        label_words = set(label_norm.split()) | set(simple_norm.split())
        overlap = len(gen_words & label_words)
        if overlap > 0:
            score = overlap / max(len(gen_words | label_words), 1)
            if score > best_score:
                best_score = score
                best_idx = idx

    return best_idx, best_score


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
    global_step: int,
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        audio = batch["audio"]
        target_texts = batch["target_texts"]

        if audio is None or len(audio) == 0:
            continue

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast(enabled=args.fp16):
            outputs = model(
                audio=audio,
                target_texts=target_texts,
                label_smoothing=args.label_smoothing,
            )
            loss = outputs.get("loss")

        if loss is None:
            continue

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                base_model = model.module if hasattr(model, "module") else model
                params = base_model.get_trainable_params()
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.max_grad_norm > 0:
                base_model = model.module if hasattr(model, "module") else model
                params = base_model.get_trainable_params()
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        # Log progress
        if dist_info["is_main"] and (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / num_batches
            elapsed = time.time() - start_time
            samples_per_sec = (num_batches * len(audio)) / elapsed if elapsed > 0 else 0

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | {samples_per_sec:.1f} samples/s",
                flush=True,
            )

            # Log gradient norms once per epoch
            if batch_idx + 1 == args.log_interval:
                base_model = model.module if hasattr(model, "module") else model

                proj_grad_norm = 0.0
                proj_count = 0
                for p in base_model.safe_model.audio_projector.parameters():
                    if p.grad is not None:
                        proj_grad_norm += p.grad.norm().item() ** 2
                        proj_count += 1
                proj_grad_norm = proj_grad_norm ** 0.5 if proj_count > 0 else 0.0

                fusion_grad_norm = 0.0
                fusion_count = 0
                for p in base_model.safe_model.fusion_adapter.parameters():
                    if p.grad is not None:
                        fusion_grad_norm += p.grad.norm().item() ** 2
                        fusion_count += 1
                fusion_grad_norm = fusion_grad_norm ** 0.5 if fusion_count > 0 else 0.0

                print(
                    f"  [Gradients] Projector: {proj_grad_norm:.4f} ({proj_count} tensors) | "
                    f"Fusion: {fusion_grad_norm:.4f} ({fusion_count} tensors)",
                    flush=True,
                )

            # WANDB logging
            if wandb is not None and args.wandb:
                wandb.log({
                    "train/loss_step": avg_loss,
                    "train/lr": lr,
                    "train/samples_per_sec": samples_per_sec,
                    "train/global_step": global_step,
                }, step=global_step)

    avg_loss = total_loss / max(num_batches, 1)

    return {"loss": avg_loss}, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    dist_info: Dict[str, Any],
    label_names: List[str],
) -> Dict[str, float]:
    """Evaluate by generating text and matching to labels."""
    model.eval()
    base_model = model.module if hasattr(model, "module") else model

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    num_batches = 0

    # For detailed analysis
    predictions = []
    all_generated = []

    for batch_idx, batch in enumerate(dataloader):
        audio = batch["audio"]
        labels = batch["labels"]
        target_texts = batch["target_texts"]

        if audio is None or len(audio) == 0:
            continue

        # Compute loss (for monitoring)
        with autocast(enabled=args.fp16):
            outputs = base_model(
                audio=audio,
                target_texts=target_texts,
                label_smoothing=0.0,
            )
            loss = outputs.get("loss")
            if loss is not None:
                total_loss += loss.item()
                num_batches += 1

        # Generate predictions
        generated_texts = base_model.generate(
            audio=audio,
            max_new_tokens=20,
            num_beams=1,
            temperature=1.0,
            do_sample=False,
        )

        # Match generated text to labels
        for gen_text, true_label in zip(generated_texts, labels.tolist()):
            pred_idx, score = match_generated_to_label(gen_text, label_names)
            predictions.append((pred_idx, true_label, gen_text))
            all_generated.append(gen_text)

            if pred_idx == true_label:
                total_correct += 1
            total_samples += 1

        # Log some examples
        if dist_info["is_main"] and batch_idx == 0:
            print(f"  [Eval Examples]", flush=True)
            for i in range(min(3, len(generated_texts))):
                true_label_name = label_names[labels[i].item()]
                print(f"    True: '{true_label_name}' | Generated: '{generated_texts[i]}'", flush=True)

    accuracy = total_correct / max(total_samples, 1)
    avg_loss = total_loss / max(num_batches, 1)

    # Compute per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for pred_idx, true_idx, _ in predictions:
        class_total[true_idx] += 1
        if pred_idx == true_idx:
            class_correct[true_idx] += 1

    # Macro F1 approximation
    per_class_acc = []
    for idx in range(len(label_names)):
        if class_total[idx] > 0:
            per_class_acc.append(class_correct[idx] / class_total[idx])
    macro_acc = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_accuracy": macro_acc,
        "total_samples": total_samples,
    }


# ============================================================================
# SECTION 6: MAIN
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generative Audio Classification Training")

    # Data
    parser.add_argument("--data-path", type=str, required=True, help="Path to AVE dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/ave_generative", help="Output directory")

    # Model
    parser.add_argument("--model-config", type=str, default="phase1", help="Model config name")
    parser.add_argument("--fusion-layer-indices", type=str, default=None,
                        help="Comma-separated fusion layer indices (e.g., '12,24,36')")
    parser.add_argument("--use-ffn", action="store_true", help="Enable FFN in fusion adapter")

    # Training
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing")

    # Mixed precision
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable WANDB logging")
    parser.add_argument("--wandb-project", type=str, default="SAFE_2", help="WANDB project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WANDB run name")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N batches")
    parser.add_argument("--save-frequency", type=int, default=10, help="Save every N epochs")

    # Other
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout (unused in generative)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup distributed
    dist_info = setup_distributed()
    device = torch.device(f"cuda:{dist_info['local_rank']}" if torch.cuda.is_available() else "cpu")

    # Set seed
    set_seed(args.seed, dist_info["rank"])

    if dist_info["is_main"]:
        print("=" * 60)
        print("Generative Audio Classification Training")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Distributed: {dist_info['distributed']} (world_size={dist_info['world_size']})")
        print(f"Data path: {args.data_path}")
        print(f"Output dir: {args.output_dir}")
        print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

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
        split="test",  # Use test set for validation
        sample_rate=48000,
        max_length=10.0,
    )

    label_names = AVE_CATEGORIES

    if dist_info["is_main"]:
        print(f"[Data] Train samples: {len(train_dataset)}")
        print(f"[Data] Val samples: {len(val_dataset)}")
        print(f"[Data] Num classes: {len(label_names)}")
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
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    # Create model
    if dist_info["is_main"]:
        print("[Model] Creating SAFE generative classifier...")

    fusion_layers = None
    if args.fusion_layer_indices:
        fusion_layers = [int(x.strip()) for x in args.fusion_layer_indices.split(",")]

    model = SAFEGenerativeClassifier(
        model_config=args.model_config,
        fusion_layer_indices=fusion_layers,
        use_ffn=args.use_ffn if args.use_ffn else None,
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
    base_model = model.module if hasattr(model, "module") else model
    optimizer = build_optimizer(
        model=model,
        learning_rate=args.learning_rate,
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

    # Mixed precision scaler
    scaler = GradScaler() if args.fp16 else None

    # WANDB init
    if dist_info["is_main"] and args.wandb and wandb is not None:
        run_name = args.wandb_run_name or f"gen-clf-{args.model_config}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

    # Training loop
    if dist_info["is_main"]:
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print()

    best_accuracy = 0.0
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if dist_info["is_main"]:
            print(f"Epoch {epoch}/{args.num_epochs}")
            print("-" * 40)

        # Train
        train_metrics, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
            dist_info=dist_info,
            global_step=global_step,
        )

        # Evaluate
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            args=args,
            dist_info=dist_info,
            label_names=label_names,
        )

        if dist_info["is_main"]:
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Val Macro Acc: {val_metrics['macro_accuracy']:.4f}")

            # WANDB epoch logging
            if args.wandb and wandb is not None:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/macro_accuracy": val_metrics["macro_accuracy"],
                }, step=global_step)

            # Save best model
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                save_path = os.path.join(args.output_dir, "best_model.pt")
                base_model = model.module if hasattr(model, "module") else model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": {
                        "audio_projector": base_model.safe_model.audio_projector.state_dict(),
                        "fusion_adapter": base_model.safe_model.fusion_adapter.state_dict(),
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": best_accuracy,
                    "config": args.model_config,
                    "fusion_layers": fusion_layers,
                }, save_path)
                print(f"  Saved best model (acc={best_accuracy:.4f})")

            # Periodic save
            if epoch % args.save_frequency == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
                base_model = model.module if hasattr(model, "module") else model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": {
                        "audio_projector": base_model.safe_model.audio_projector.state_dict(),
                        "fusion_adapter": base_model.safe_model.fusion_adapter.state_dict(),
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": val_metrics["accuracy"],
                }, save_path)

        print()

    # Final summary
    if dist_info["is_main"]:
        print("=" * 60)
        print("Training Complete")
        print("=" * 60)
        print(f"Best Validation Accuracy: {best_accuracy:.4f}")
        print(f"Model saved to: {args.output_dir}")

        if args.wandb and wandb is not None:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
