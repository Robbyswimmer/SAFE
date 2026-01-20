#!/usr/bin/env python3
"""
train_audio_llm_probe.py

Step-1 "LLM readout" for AVE classification:
  audio -> CLAP (frozen) -> projector (trainable) -> SAFE fusion (trainable) -> frozen LLM
  pooled LLM hidden state -> linear head (trainable) -> 28-way classification

This avoids free-form generation and directly measures whether audio fusion changes
LLM representations in a way that supports classification.
"""

import argparse
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except ImportError:
    wandb = None

from configs.model_configs import get_config
from safe.models.safe_model import SAFEModel


AVE_CATEGORIES = [
    "Church bell",
    "Male speech, man speaking",
    "Bark",
    "Fixed-wing aircraft, airplane",
    "Race car, auto racing",
    "Female speech, woman speaking",
    "Helicopter",
    "Violin, fiddle",
    "Flute",
    "Ukulele",
    "Frying (food)",
    "Truck",
    "Shofar",
    "Motorcycle",
    "Acoustic guitar",
    "Train horn",
    "Clock",
    "Banjo",
    "Goat",
    "Baby cry, infant cry",
    "Bus",
    "Chainsaw",
    "Cat",
    "Horse",
    "Toilet flush",
    "Rodents, rats, mice",
    "Accordion",
    "Mandolin",
]

AVE_LABEL_TO_IDX = {label: idx for idx, label in enumerate(AVE_CATEGORIES)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AVEDataset(Dataset):
    """
    AVE loader for audio classification.
    Expects annotations in the standard AVE format:
      category&video_id&quality&start&end
    Audio files are typically named:
      {video_id}_{start}_{end}.wav
    And stored under:
      train/audio/ and test/audio/
    """

    def __init__(
        self,
        data_path: str,
        split: str,
        sample_rate: int = 48000,
        max_length: float = 10.0,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.sample_rate = int(sample_rate)
        self.max_length = float(max_length)

        split_files = {"train": "trainSet.txt", "val": "valSet.txt", "test": "testSet.txt"}
        data_file = self.data_path / split_files.get(split, f"{split}Set.txt")
        if not data_file.exists():
            data_file = self.data_path / "ave" / split_files.get(split, f"{split}Set.txt")
        if not data_file.exists():
            raise FileNotFoundError(f"Could not find split file for '{split}': {data_file}")

        self.examples = self._load_data(data_file)

        # Quick path sanity check
        check_n = min(50, len(self.examples))
        found = 0
        for i in range(check_n):
            if self._resolve_audio_path(self.examples[i]["audio_name"]) is not None:
                found += 1
        print(f"[AVEDataset] {split}: {len(self.examples)} samples", flush=True)
        if check_n > 0:
            print(f"[AVEDataset] {split}: audio found for {found}/{check_n} sample check", flush=True)

    def _load_data(self, data_file: Path) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("&")
                if len(parts) < 2:
                    continue
                category = parts[0].strip()
                video_id = parts[1].strip()
                if category not in AVE_LABEL_TO_IDX:
                    continue
                start = parts[3].strip() if len(parts) > 3 else "0"
                end = parts[4].strip() if len(parts) > 4 else "10"
                audio_name = f"{video_id}_{start}_{end}.wav"
                examples.append(
                    {
                        "category": category,
                        "label": AVE_LABEL_TO_IDX[category],
                        "video_id": video_id,
                        "start": start,
                        "end": end,
                        "audio_name": audio_name,
                    }
                )
        return examples

    def _resolve_audio_path(self, audio_name: str) -> Optional[Path]:
        candidates = [
            self.data_path / "train" / "audio" / audio_name,
            self.data_path / "test" / "audio" / audio_name,
            self.data_path / "val" / "audio" / audio_name,
            self.data_path / "audio" / audio_name,
            self.data_path / audio_name,
            self.data_path / "AVE" / audio_name,
            self.data_path / "ave" / audio_name,
            # Split-specific paths (matches train_ave_classifier.py)
            self.data_path / "AVE" / self.split / "audio" / audio_name,
            self.data_path / "ave" / self.split / "audio" / audio_name,
            self.data_path / self.split / "audio" / audio_name,
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        audio_path = self._resolve_audio_path(ex["audio_name"])
        return {
            "audio": str(audio_path) if audio_path is not None else None,
            "label": int(ex["label"]),
            "category": ex["category"],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    audio: List[str] = []
    labels: List[int] = []
    categories: List[str] = []

    for item in batch:
        if not item.get("audio"):
            continue
        # SAFE/CLAP audio encoder supports raw file paths directly; do not wrap in tuples.
        audio.append(str(item["audio"]))
        labels.append(int(item["label"]))
        categories.append(str(item["category"]))

    if not audio:
        return {"audio": None, "labels": None, "categories": []}

    return {
        "audio": audio,
        "labels": torch.tensor(labels, dtype=torch.long),
        "categories": categories,
    }


class SAFELLMProbe(nn.Module):
    PROMPT = "What is happening in the audio?"

    def __init__(self, config: Dict[str, Any], num_classes: int):
        super().__init__()

        constructor_keys = {
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
            "label_smoothing",
            "llm_hidden_size",
            "audio_embed_dim",
        }
        constructor_config = {k: v for k, v in config.items() if k in constructor_keys}
        self.safe_model = SAFEModel(**constructor_config)

        # Ensure only audio components train
        self.safe_model.enable_audio_training()
        hidden_size = int(config.get("llm_hidden_size", 5120))
        self.head = nn.Linear(hidden_size, num_classes)

        # Ensure the underlying HF model actually returns hidden states when requested.
        # Some configurations ignore call-time flags unless config is set.
        try:
            llm = self.safe_model.base_vl.llm
            for candidate in [llm, getattr(llm, "language_model", None), getattr(llm, "model", None)]:
                if candidate is None or not hasattr(candidate, "config"):
                    continue
                try:
                    candidate.config.output_hidden_states = True
                except Exception:
                    pass
                try:
                    candidate.config.return_dict = True
                except Exception:
                    pass
        except Exception:
            pass

    def freeze_safe(self, freeze: bool = True) -> None:
        """Freeze/unfreeze SAFE trainable components (projector + fusion)."""
        for p in self.safe_model.get_trainable_parameters():
            p.requires_grad = not freeze

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = list(self.safe_model.get_trainable_parameters())
        params.extend(list(self.head.parameters()))
        return params

    def get_safe_params(self) -> List[nn.Parameter]:
        return list(self.safe_model.get_trainable_parameters())

    def get_head_params(self) -> List[nn.Parameter]:
        return list(self.head.parameters())

    def _pool_by_audio_attention(
        self,
        hidden: torch.Tensor,
        top_k: int = 8,
        candidate_span: int = 16,  # Only consider last N positions (answer-decision span)
    ) -> torch.Tensor:
        """
        Pool hidden states at positions with highest audio attention mass.

        IMPROVEMENTS over naive top-k:
        1. Restricts to last `candidate_span` positions (answer-decision tokens)
           - Avoids picking junk like early prompt tokens or punctuation
        2. Aggregates attention across ALL fusion layers (max per position)
           - Ensures we don't miss signal from any injection site
        3. Logs attention entropy to detect uniform (non-discriminative) attention

        Args:
            hidden: (batch, seq_len, hidden_size)
            top_k: number of top positions to pool
            candidate_span: only consider last N positions as candidates

        Returns:
            pooled: (batch, hidden_size)
        """
        batch_size, seq_len, hidden_size = hidden.shape

        # Restrict to answer-decision span (last N positions)
        span = min(candidate_span, seq_len)
        hidden_span = hidden[:, -span:, :]  # (batch, span, hidden_size)

        # Try to get audio attention weights from ALL fusion layers
        audio_attn_per_pos = None
        layer_contributions = {}
        try:
            if hasattr(self.safe_model, 'kv_hook_manager') and self.safe_model.kv_hook_manager is not None:
                attn_weights = self.safe_model.kv_hook_manager.get_attention_weights()
                if attn_weights:
                    # Aggregate across ALL layers using MAX (captures strongest signal)
                    layer_attn_list = []
                    for layer_idx, weights in sorted(attn_weights.items()):
                        # weights: (batch, heads, q_len, n_audio)
                        # Sum attention to audio, average over heads: (batch, q_len)
                        layer_attn = weights.sum(dim=-1).mean(dim=1)
                        # Only take the last `span` positions
                        layer_attn = layer_attn[:, -span:]
                        layer_attn_list.append(layer_attn)
                        layer_contributions[layer_idx] = layer_attn.mean().item()

                    # Stack and take MAX across layers per position
                    stacked = torch.stack(layer_attn_list, dim=0)  # (n_layers, batch, span)
                    audio_attn_per_pos, max_layer_idx = stacked.max(dim=0)  # (batch, span)

                    # Compute entropy of attention distribution (non-uniform = good)
                    # Entropy over audio tokens, averaged over positions
                    with torch.no_grad():
                        # Use last layer's full attention for entropy
                        last_layer_idx = max(attn_weights.keys())
                        last_weights = attn_weights[last_layer_idx][:, :, -span:, :]  # (B, H, span, n_audio)
                        # Normalize to get distribution over audio tokens
                        attn_dist = last_weights.mean(dim=1)  # (B, span, n_audio)
                        attn_dist = attn_dist / (attn_dist.sum(dim=-1, keepdim=True) + 1e-8)
                        # Entropy: -sum(p * log(p))
                        entropy = -(attn_dist * (attn_dist + 1e-8).log()).sum(dim=-1).mean()
                        n_audio = last_weights.shape[-1]
                        max_entropy = float(torch.tensor(n_audio).float().log())
                        self._last_attn_entropy = entropy.item()
                        self._last_max_entropy = max_entropy
        except Exception as e:
            if not hasattr(self, "_audio_attn_error_logged"):
                self._audio_attn_error_logged = True
                print(f"[LLMProbe] Warning: Error getting attention: {e}", flush=True)

        if audio_attn_per_pos is None:
            # HARD FAIL: audio_attn pooling requires attention weights
            # If we get here, something is broken in the KV augmentation pipeline
            if not hasattr(self, "_audio_attn_fallback_count"):
                self._audio_attn_fallback_count = 0
            self._audio_attn_fallback_count += 1

            if self._audio_attn_fallback_count == 1:
                print("\n" + "!" * 60, flush=True)
                print("CRITICAL ERROR: No audio attention weights available!", flush=True)
                print("  This means KV augmentation is NOT working correctly.", flush=True)
                print("  Possible causes:", flush=True)
                print("    1. kv_hook_manager.set_return_attention_weights(True) not called", flush=True)
                print("    2. Attention modules not wrapped", flush=True)
                print("    3. Audio tokens not injected", flush=True)
                print("  Falling back to last-k pooling but results will be INVALID.", flush=True)
                print("!" * 60 + "\n", flush=True)

            # Still fall back but results will be garbage
            pooled = hidden_span[:, -top_k:, :].mean(dim=1)
        else:
            # Pool top-k positions by audio attention (within restricted span)
            k = min(top_k, span)
            _, top_indices = audio_attn_per_pos.topk(k, dim=1)  # (batch, k)

            # Gather hidden states at top positions
            top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
            top_hidden = hidden_span.gather(1, top_indices_expanded)  # (batch, k, hidden_size)

            # Mean pool the top-k
            pooled = top_hidden.mean(dim=1)

            # Log detailed stats once
            if not hasattr(self, "_audio_attn_stats_logged"):
                self._audio_attn_stats_logged = True
                with torch.no_grad():
                    mean_attn = audio_attn_per_pos.mean().item()
                    max_attn = audio_attn_per_pos.max().item()
                    top_k_mean = audio_attn_per_pos.gather(1, top_indices).mean().item()

                    # Find which layer contributes most
                    if layer_contributions:
                        max_layer = max(layer_contributions, key=layer_contributions.get)
                        max_layer_val = layer_contributions[max_layer]
                    else:
                        max_layer, max_layer_val = "?", 0

                    entropy_info = ""
                    if hasattr(self, '_last_attn_entropy'):
                        entropy_ratio = self._last_attn_entropy / self._last_max_entropy
                        entropy_info = f" entropy={self._last_attn_entropy:.3f}/{self._last_max_entropy:.3f}={entropy_ratio:.2f}"
                        if entropy_ratio > 0.95:
                            entropy_info += " ⚠️UNIFORM"

                print(f"[LLMProbe] audio_attn pooling: span={span} top-{k} "
                      f"mean={mean_attn:.4f} max={max_attn:.4f} top_mean={top_k_mean:.4f} "
                      f"max_layer={max_layer}({max_layer_val:.4f}){entropy_info}", flush=True)

        return pooled

    def forward(
        self,
        audio: List[str],
        device: torch.device,
        pooling: str = "last",
    ) -> torch.Tensor:
        batch_size = len(audio)
        inputs = self.safe_model.prepare_multimodal_inputs(
            text=[self.PROMPT] * batch_size,
            images=None,
            audio=audio,
            answers=None,
            device=device,
            training_mode=False,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        audio_tokens = inputs.get("audio_tokens")
        if audio_tokens is not None:
            audio_tokens = audio_tokens.to(device)
        audio_attention_mask = inputs.get("audio_attention_mask")
        if audio_attention_mask is not None:
            audio_attention_mask = audio_attention_mask.to(device)

        # Fallback: capture the final hidden states via a pre-hook on the output embedding
        # module (often `lm_head`). This is robust when HF wrappers ignore hidden-state flags.
        hidden_capture: Dict[str, Optional[torch.Tensor]] = {"last": None}
        hook_handle = None
        try:
            llm = self.safe_model.base_vl.llm
            head_module = None
            for candidate_owner in [
                llm,
                getattr(llm, "language_model", None),
                getattr(llm, "model", None),
            ]:
                if candidate_owner is None:
                    continue
                try:
                    get_out = getattr(candidate_owner, "get_output_embeddings", None)
                    if callable(get_out):
                        head_module = get_out()
                except Exception:
                    head_module = None
                if head_module is not None:
                    break
                head_module = getattr(candidate_owner, "lm_head", None)
                if head_module is not None:
                    break

            if head_module is not None:
                def _capture_head_input(_module, inputs):
                    try:
                        if inputs and torch.is_tensor(inputs[0]):
                            hidden_capture["last"] = inputs[0]
                    except Exception:
                        hidden_capture["last"] = None

                hook_handle = head_module.register_forward_pre_hook(_capture_head_input)
        except Exception:
            hook_handle = None

        # Enable attention weight capture for audio_attn pooling
        # This MUST be set BEFORE the forward pass
        if hasattr(self.safe_model, 'kv_hook_manager') and self.safe_model.kv_hook_manager is not None:
            self.safe_model.kv_hook_manager.set_return_attention_weights(True)

        outputs = self.safe_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_tokens=audio_tokens,
            audio_attention_mask=audio_attention_mask,
            labels=None,
            output_hidden_states=True,
            return_dict=True,
        )
        if hook_handle is not None:
            try:
                hook_handle.remove()
            except Exception:
                pass

        hidden = outputs.get("hidden_states")
        if hidden is None:
            hidden = hidden_capture.get("last")
        if hidden is None:
            if not hasattr(self, "_hidden_debug_logged"):
                self._hidden_debug_logged = True
                try:
                    llm = self.safe_model.base_vl.llm
                    cfg = getattr(llm, "config", None)
                    cfg_hs = getattr(cfg, "output_hidden_states", None) if cfg is not None else None
                    cfg_rd = getattr(cfg, "return_dict", None) if cfg is not None else None
                except Exception:
                    cfg_hs = None
                    cfg_rd = None
                print(
                    "[LLMProbeDebug] hidden_states missing. "
                    f"input_ids={tuple(input_ids.shape)} "
                    f"attn_sum={int(attention_mask.sum().item()) if attention_mask is not None else 'None'} "
                    f"audio_tokens={tuple(audio_tokens.shape) if audio_tokens is not None else None} "
                    f"audio_numel={int(audio_tokens.numel()) if audio_tokens is not None else 0} "
                    f"enable_midlayer_fusion={getattr(self.safe_model, 'enable_midlayer_fusion', None)} "
                    f"llm.config.output_hidden_states={cfg_hs} "
                    f"llm.config.return_dict={cfg_rd}",
                    flush=True,
                )
            raise RuntimeError("SAFEModel did not return hidden_states; expected last hidden state tensor.")

        if pooling == "mean":
            # Mean over all tokens (dilutes signal)
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.to(hidden.dtype).unsqueeze(-1)  # (B, T, 1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        elif pooling == "audio_attn":
            # Pool positions with highest audio attention mass (top-k)
            # This targets tokens that are actually using audio information
            pooled = self._pool_by_audio_attention(hidden, top_k=8)

        elif pooling == "last" or pooling == "prompt_last":
            # Last prompt token - where model "decides" the answer
            # This is the most informative position for classification
            if attention_mask is None:
                pooled = hidden[:, -1, :]
            else:
                lengths = attention_mask.long().sum(dim=1).clamp_min(1) - 1
                pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]

        else:
            raise ValueError(f"Unknown pooling mode: {pooling}")

        # Lightweight sanity checks (printed once)
        if not hasattr(self, "_probe_stats_logged"):
            self._probe_stats_logged = True
            with torch.no_grad():
                pooled_var = float(pooled.float().var(dim=0).mean().item())
                pooled_norm = float(pooled.float().norm(dim=-1).mean().item())
            attn_summary = None
            try:
                attn_summary = self.safe_model.get_last_attention_summary()
            except Exception:
                attn_summary = None
            print(
                f"[LLMProbe] pooled_norm={pooled_norm:.3f} pooled_var={pooled_var:.6e} "
                f"attn_summary={'yes' if attn_summary is not None else 'no'}",
                flush=True,
            )

        logits = self.head(pooled.float())
        return logits


def _split_decay(params: List[nn.Parameter]) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for p in params:
        if not p.requires_grad:
            continue
        if p.ndim == 1:
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def build_optimizer(
    model: SAFELLMProbe,
    safe_lr: float,
    head_lr: float,
    safe_weight_decay: float,
    head_weight_decay: float,
) -> torch.optim.Optimizer:
    safe_decay, safe_no_decay = _split_decay(model.get_safe_params())
    head_decay, head_no_decay = _split_decay(model.get_head_params())

    param_groups = [
        {"name": "safe_decay", "params": safe_decay, "lr": safe_lr, "weight_decay": safe_weight_decay},
        {"name": "safe_no_decay", "params": safe_no_decay, "lr": safe_lr, "weight_decay": 0.0},
        {"name": "head_decay", "params": head_decay, "lr": head_lr, "weight_decay": head_weight_decay},
        {"name": "head_no_decay", "params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
    ]
    param_groups = [g for g in param_groups if g["params"]]

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def load_safe_checkpoint_into(model: SAFELLMProbe, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Try loading directly into SAFEModel inside the probe.
    adapted: Dict[str, torch.Tensor] = {}
    skipped = 0
    safe_sd = model.safe_model.state_dict()
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[len("module.") :]
        if key in safe_sd:
            adapted[key] = v
        else:
            skipped += 1

    missing, unexpected = model.safe_model.load_state_dict(adapted, strict=False)
    print(f"[Checkpoint] Loaded {len(adapted)} tensors into SAFEModel", flush=True)
    if skipped:
        print(f"[Checkpoint] Skipped {skipped} tensors (not in SAFEModel)", flush=True)
    if missing:
        print(f"[Checkpoint] Missing {len(missing)} keys (first 5): {missing[:5]}", flush=True)
    if unexpected:
        print(f"[Checkpoint] Unexpected {len(unexpected)} keys (first 5): {unexpected[:5]}", flush=True)


def train_epoch(
    model: SAFELLMProbe,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> Tuple[Dict[str, float], int]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    num_batches = 0
    start = time.time()

    for batch_idx, batch in enumerate(loader):
        audio = batch["audio"]
        labels = batch["labels"]
        if audio is None or labels is None or labels.numel() == 0:
            continue

        # Warmups
        if args.force_gate is not None:
            try:
                model.safe_model.set_gate(float(args.force_gate))
            except Exception:
                pass
        elif args.gate_warmup_steps > 0:
            warmup_steps = max(1, int(args.gate_warmup_steps))
            progress = min(1.0, float(global_step) / float(warmup_steps))
            gate = float(args.gate_warmup_start) + (1.0 - float(args.gate_warmup_start)) * progress
            try:
                model.safe_model.set_gate(gate)
            except Exception:
                pass

        if args.scale_min_warmup_epochs > 0:
            try:
                model.safe_model.set_scale_minimum_warmup(
                    epoch=epoch - 1,
                    warmup_epochs=int(args.scale_min_warmup_epochs),
                    start_min=float(args.scale_min_start),
                    end_min=float(args.scale_min_end),
                )
            except Exception:
                pass

        if args.residual_warmup_epochs > 0 and (epoch - 1) < args.residual_warmup_epochs:
            try:
                model.safe_model.set_residual_scale_warmup(
                    epoch=epoch - 1,
                    warmup_epochs=int(args.residual_warmup_epochs),
                    start_scale=float(args.residual_warmup_start),
                    end_scale=1.0,
                )
            except Exception:
                pass

        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Head warmup: freeze HEAD LR for early steps so SAFE learns first
        # This is CRITICAL: without this, head dominates and SAFE never learns
        if getattr(args, "head_warmup_steps", 0) > 0:
            warmup_steps = int(args.head_warmup_steps)
            head_lr = float(args.head_learning_rate)
            # Freeze head during warmup, unfreeze after
            effective_head_lr = 0.0 if global_step < warmup_steps else head_lr
            for group in optimizer.param_groups:
                if str(group.get("name", "")).startswith("head_"):
                    group["lr"] = effective_head_lr
            # Log once when HEAD LR turns on (warmup complete)
            if (
                global_step == warmup_steps
                and not hasattr(model, "_head_warmup_complete_logged")
            ):
                model._head_warmup_complete_logged = True
                lr_by_group = {g.get("name", f"g{idx}"): g.get("lr") for idx, g in enumerate(optimizer.param_groups)}
                print(f"\n  [LR] HEAD warmup complete at step={global_step}. Head unfrozen. lrs={lr_by_group}\n", flush=True)

        with autocast(enabled=args.fp16):
            logits = model(audio=audio, device=device, pooling=args.pooling)
            loss = F.cross_entropy(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), args.max_grad_norm)
            optimizer.step()

        preds = logits.argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())
        total_seen += int(labels.numel())
        total_loss += float(loss.item()) * int(labels.numel())
        num_batches += 1
        global_step += 1

        if (batch_idx + 1) % args.log_interval == 0:
            elapsed = max(1e-6, time.time() - start)
            sps = (total_seen / elapsed)
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss: {total_loss / max(total_seen, 1):.4f} | "
                f"Acc: {total_correct / max(total_seen, 1):.4f} | {sps:.1f} samples/s",
                flush=True,
            )

            # Log gradients/LRs on first log interval, and once right after HEAD warmup completes.
            should_log_grads = not hasattr(model, "_grad_logged")
            if getattr(args, "head_warmup_steps", 0) and hasattr(model, "_head_warmup_complete_logged"):
                should_log_grads = should_log_grads or not hasattr(model, "_grad_after_warmup_logged")
            if should_log_grads:
                if hasattr(model, "_grad_logged"):
                    model._grad_after_warmup_logged = True
                else:
                    model._grad_logged = True
                with torch.no_grad():
                    head_g = 0.0
                    head_n = 0
                    for p in model.get_head_params():
                        if p.grad is not None:
                            head_g += float(p.grad.float().norm().item()) ** 2
                            head_n += 1
                    head_g = head_g ** 0.5 if head_n else 0.0

                    safe_g = 0.0
                    safe_n = 0
                    for p in model.get_safe_params():
                        if p.grad is not None:
                            safe_g += float(p.grad.float().norm().item()) ** 2
                            safe_n += 1
                    safe_g = safe_g ** 0.5 if safe_n else 0.0

                lr_by_group = {g.get("name", f"g{idx}"): g.get("lr") for idx, g in enumerate(optimizer.param_groups)}

                # Log ΔQ/Q ratio from KV augmentation (key diagnostic for training progress)
                dq_q_str = ""
                entropy_str = ""
                if hasattr(model.safe_model, 'kv_hook_manager') and model.safe_model.kv_hook_manager is not None:
                    diag = model.safe_model.kv_hook_manager.get_diagnostics()
                    if diag:
                        dq_ratios = []
                        entropies = []
                        for layer_idx, layer_diag in diag.items():
                            if 'delta_q_rms' in layer_diag and 'q_rms' in layer_diag:
                                q_rms = max(layer_diag['q_rms'], 1e-8)
                                ratio = layer_diag['delta_q_rms'] / q_rms
                                dq_ratios.append(ratio)
                            if 'normalized_entropy' in layer_diag:
                                entropies.append(layer_diag['normalized_entropy'])
                        if dq_ratios:
                            avg_dq_q = sum(dq_ratios) / len(dq_ratios)
                            dq_q_str = f" ΔQ/Q={avg_dq_q*100:.4f}%"
                        if entropies:
                            avg_ent = sum(entropies) / len(entropies)
                            entropy_str = f" entropy={avg_ent:.3f}"

                print(f"  [Gradients] head={head_g:.4f} ({head_n}) safe={safe_g:.4f} ({safe_n}){dq_q_str}{entropy_str} lrs={lr_by_group}", flush=True)

            if wandb is not None and args.wandb:
                wandb.log(
                    {
                        "train/loss": total_loss / max(total_seen, 1),
                        "train/acc": total_correct / max(total_seen, 1),
                        "train/gate": getattr(model.safe_model, "_default_gate", 1.0),
                        "epoch": epoch,
                    },
                    step=global_step,
                )

    return {"loss": total_loss / max(total_seen, 1), "acc": total_correct / max(total_seen, 1)}, global_step


@torch.no_grad()
def evaluate(
    model: SAFELLMProbe,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    if args.force_gate is not None:
        model.safe_model.set_gate(float(args.force_gate))
    else:
        model.safe_model.set_gate(1.0)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch in loader:
        audio = batch["audio"]
        labels = batch["labels"]
        if audio is None or labels is None or labels.numel() == 0:
            continue
        labels = labels.to(device)
        with autocast(enabled=args.fp16):
            logits = model(audio=audio, device=device, pooling=args.pooling)
            loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())
        total_seen += int(labels.numel())
        total_loss += float(loss.item()) * int(labels.numel())

    return {"loss": total_loss / max(total_seen, 1), "acc": total_correct / max(total_seen, 1)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAFE LLM-probe audio classification (AVE)")
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs/ave_llm_probe")
    p.add_argument("--model-config", type=str, default="phase1")
    p.add_argument("--fusion-layer-indices", type=str, default="1", help="Comma-separated fusion layers (default: 1)")
    p.add_argument(
        "--fusion-injection-point",
        type=str,
        default=None,
        choices=["pre_ffn", "post_layer"],
        help="Override fusion injection point for hooks (debugging).",
    )
    p.add_argument(
        "--fusion-mode",
        type=str,
        default=None,
        choices=["residual", "film"],
        help="Fusion update type (bottleneck only).",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--safe-learning-rate", type=float, default=6e-5, help="LR for projector+fusion")
    p.add_argument("--head-learning-rate", type=float, default=1e-3, help="LR for linear probe head")
    p.add_argument("--safe-weight-decay", type=float, default=0.01)
    p.add_argument("--head-weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--pooling", type=str, default="last",
                   choices=["last", "prompt_last", "mean", "audio_attn"],
                   help="Pooling strategy: 'last'/'prompt_last' (last token), "
                        "'mean' (all tokens), 'audio_attn' (top-k by audio attention)")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument(
        "--force-gate",
        type=float,
        default=None,
        help="If set, override SAFE fusion gate to this constant (disables gate warmup).",
    )

    # Warmups (enabled by default)
    p.add_argument("--gate-warmup-steps", type=int, default=500)
    p.add_argument("--gate-warmup-start", type=float, default=0.1)
    p.add_argument("--scale-min-warmup-epochs", type=int, default=5)
    p.add_argument("--scale-min-start", type=float, default=0.5)
    p.add_argument("--scale-min-end", type=float, default=1.0)
    p.add_argument("--residual-warmup-epochs", type=int, default=5)
    p.add_argument("--residual-warmup-start", type=float, default=0.1)

    # Head warmup: freeze head for N steps to force SAFE to learn first
    p.add_argument("--head-warmup-steps", type=int, default=0,
                   help="Freeze classifier head for first N steps to force SAFE learning")

    # Logging
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="SAFE")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--load-checkpoint", type=str, default=None, help="Load a SAFE checkpoint (e.g., from train_safe.py)")
    p.add_argument(
        "--head-only",
        action="store_true",
        help="Freeze projector+fusion and train only the linear head.",
    )
    p.add_argument(
        "--head-warmup-steps",
        type=int,
        default=0,
        help="If >0, train head-only for this many optimizer steps (SAFE LR=0), then unfreeze SAFE LR.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_config(args.model_config)
    if args.fusion_layer_indices:
        layers = [int(x.strip()) for x in str(args.fusion_layer_indices).split(",") if x.strip()]
        config["fusion_layer_indices"] = layers
    if args.fusion_injection_point is not None:
        config.setdefault("fusion_config", {})
        config["fusion_config"]["injection_point"] = str(args.fusion_injection_point)
        print(f"[Config] fusion_injection_point={args.fusion_injection_point}", flush=True)
    if args.fusion_mode is not None:
        config.setdefault("fusion_config", {})
        config["fusion_config"]["fusion_mode"] = str(args.fusion_mode)
        print(f"[Config] fusion_mode={args.fusion_mode}", flush=True)

    # === CRITICAL: Print and verify config at startup ===
    print("\n" + "=" * 60, flush=True)
    print("ACTIVE CONFIGURATION (verify these match your intent!):", flush=True)
    print(f"  model_config: {args.model_config}", flush=True)
    print(f"  fusion_layer_indices: {config['fusion_layer_indices']}", flush=True)
    print(f"  pooling: {args.pooling}", flush=True)
    fusion_mode = config.get("fusion_config", {}).get("fusion_mode", "additive")
    print(f"  fusion_mode: {fusion_mode}", flush=True)
    print("=" * 60 + "\n", flush=True)

    # Hard assertion for kv_augment mode
    if args.model_config == "kv_augment":
        expected_layers = [16, 24, 32]
        actual_layers = config["fusion_layer_indices"]
        if actual_layers != expected_layers:
            print(f"⚠️  WARNING: kv_augment expected layers {expected_layers}, got {actual_layers}", flush=True)
            print(f"   This may be intentional if testing different configs.", flush=True)

    train_ds = AVEDataset(args.data_path, split="train")
    test_ds = AVEDataset(args.data_path, split="test")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = SAFELLMProbe(config=config, num_classes=len(AVE_CATEGORIES)).to(device)

    # === Verify KV augmentation is set up if using audio_attn pooling ===
    if args.pooling == "audio_attn":
        has_kv_manager = (
            hasattr(model.safe_model, 'kv_hook_manager')
            and model.safe_model.kv_hook_manager is not None
        )
        if not has_kv_manager:
            raise RuntimeError(
                "FATAL: pooling='audio_attn' requires kv_hook_manager but it's not initialized!\n"
                "Check that fusion_mode='kv_augment' is set in config."
            )
        print(f"✓ KV hook manager verified for audio_attn pooling", flush=True)

    if args.load_checkpoint:
        print(f"[Checkpoint] Loading: {args.load_checkpoint}", flush=True)
        load_safe_checkpoint_into(model, args.load_checkpoint, device=device)

    if args.head_only:
        print("[Mode] head-only: freezing SAFE trainables (projector+fusion)", flush=True)
        model.freeze_safe(True)

    optimizer = build_optimizer(
        model=model,
        safe_lr=float(args.safe_learning_rate),
        head_lr=float(args.head_learning_rate),
        safe_weight_decay=float(args.safe_weight_decay),
        head_weight_decay=float(args.head_weight_decay),
    )
    scaler = GradScaler() if args.fp16 else None

    if args.wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"llm-probe-ave-layer{args.fusion_layer_indices}",
            config=vars(args),
        )

    best = -1.0
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}\n" + "-" * 40, flush=True)
        train_metrics, global_step = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            epoch=epoch,
            global_step=global_step,
        )
        val_metrics = evaluate(model=model, loader=test_loader, device=device, args=args)
        print(f"Train | loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f}", flush=True)
        print(f"Test  | loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f}", flush=True)

        if args.wandb and wandb is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "val/loss": val_metrics["loss"],
                    "val/acc": val_metrics["acc"],
                },
                step=global_step,
            )

        if val_metrics["acc"] > best:
            best = val_metrics["acc"]
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best,
                "config": config,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> New best acc: {best:.4f}", flush=True)

    if args.wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
