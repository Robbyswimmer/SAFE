import math
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import wandb
import os
from pathlib import Path
import logging
import textwrap
import traceback
import time
from collections import defaultdict
from dataclasses import dataclass
import re
import csv

# Set CPU threads for predictable performance
torch.set_num_threads(min(8, os.cpu_count() or 2))

from ..models.safe_model import SAFEModel
from ..models.base_vl import BaseVLModel
from ..data.datasets import create_safe_dataloader
from ..data.curriculum import CurriculumManager, CurriculumConfig, ProgressionStatus
from .losses import AudioTaskLoss


@dataclass
class AccuracyResult:
    """Container for per-sample accuracy with auxiliary metrics."""

    score: float
    metrics: Dict[str, Any]


class StageATrainer:
    """
    Stage A trainer for supervised warm-start of SAFE model with curriculum learning.
    
    Trains projector + LoRA adapter while keeping base VL & audio encoder frozen.
    Uses progressive curriculum with adaptive audio ratios and difficulty progression.
    """
    
    def __init__(
        self,
        safe_model: SAFEModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict = None,
        curriculum_config: Union[str, Path, Dict, CurriculumConfig] = None
    ):
        self.safe_model = safe_model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Log dataset sizes
        train_size = len(train_dataloader.dataset) if hasattr(train_dataloader.dataset, '__len__') else 'unknown'
        val_size = len(val_dataloader.dataset) if hasattr(val_dataloader.dataset, '__len__') else 'unknown'
        print(f"📊 Dataset sizes - Train: {train_size} samples, Val: {val_size} samples", flush=True)

        # Initialize curriculum learning
        if curriculum_config is not None:
            if isinstance(curriculum_config, CurriculumConfig):
                self.curriculum_manager = CurriculumManager(curriculum_config)
            else:
                self.curriculum_manager = CurriculumManager(curriculum_config)
            self.use_curriculum = True
            print(f"Curriculum learning enabled with {self.curriculum_manager.config.get_num_stages()} stages", flush=True)
        else:
            self.curriculum_manager = None
            self.use_curriculum = False
            print("Using traditional fixed-epoch training", flush=True)
        
        # Default configuration
        self.config = {
            "learning_rate_projector": 2e-4,
            "learning_rate_adapter": 1e-4,
            "learning_rate_audio_embeddings": 1e-4,
            "weight_decay": 0.01,
            "num_epochs": 10 if not self.use_curriculum else None,  # Curriculum controls epochs
            "warmup_steps": 1000,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
            "audio_loss_weight": 1.0,
            "save_steps": 5000,
            "eval_steps": 1000,
            "logging_steps": 100,
            "progress_log_timeout": 600,
            "output_dir": "./checkpoints/stage_a",
            "validation_frequency": 5 if self.use_curriculum else 1000,  # More frequent validation for curriculum
            "sample_log_interval": 500,
            "sample_log_limit": 5,
            "sample_log_examples": 3,
            "audio_label_smoothing": 0.1,
            "enable_audio_sanity_checks": True,
            "log_waveform_stats": True,
            "waveform_log_limit": 8,
            "log_attention_probe": True,
            "attention_probe_log_limit": 8,
            "log_grad_norms": True,
            "grad_log_interval": 1,
            "grad_log_limit": 20,
            "sanitize_nan_gradients": True,
            "grad_sanitize_clip": 1e3,
            "param_sanitize_clip": 1e3,
            "train_accuracy_interval": 0,
            "train_accuracy_warmup": 5,
            "generation_max_new_tokens": 12,
            "audio_generation_max_new_tokens": 20,
            "vl_max_new_tokens": 4,  # Encourage one-word/short answers for VQA
            "gradient_accumulation_steps": 1,
            "audio_bertscore_threshold": 0.7,
            "gate_warmup_steps": 2000,
            "disable_bertscore": False,  # Set to True to skip BERTScore (use token F1 only)
            "max_audio_eval_batches": 0,
            "max_vl_eval_batches": 0,
            "max_audio_eval_samples": 0,
            "max_vl_eval_samples": 0,
            # Modality-specific generation tuning
            "vl_repetition_penalty": 1.0,
            "audio_repetition_penalty": 1.1,
            # Only suppress EOS early in training for audio batches (not VL)
            "suppress_eos_for_audio_early_steps": True,
            "audio_contrastive_weight": 0.0,
            "audio_contrastive_temperature": 0.07,
            "audio_contrastive_answer_max_length": 48,
            "audio_num_beams": 1,
            "audio_num_return_sequences": None,
            "audio_length_penalty": 1.0,
            "audio_rerank_with_clap": False,
            "audio_rerank_logprob_weight": 1.0,
            "audio_rerank_clap_weight": 0.0,
            "audio_rerank_ngram_weight": 0.0,
            "audio_rerank_coverage_weight": 0.0,
            "audio_rerank_cider_weight": 0.0,
            "audio_rerank_spider_weight": 0.0,
            "audio_rerank_tag_vocab": None,
            "audio_rerank_num_tags": 3,
            "audio_rerank_tag_weight": 0.0,
            "audio_num_samples": 0,
            "audio_sample_top_p": 0.9,
            "audio_sample_temperature": 1.0,
            "enable_scst_finetune": False,
            "scst_epochs": 1,
            "scst_learning_rate": 5e-6,
            "scst_num_samples": 1,
            "scst_sample_top_p": 0.9,
            "scst_sample_temperature": 0.9,
            "scst_reward_metric": "cider",
            "scst_patience_epochs": 2,
            "scst_improvement_threshold": 1e-4,
        }

        if config:
            self.config.update(config)

        if isinstance(config, dict) and config.get("variant") is not None:
            self.config["variant"] = config.get("variant")
        elif "variant" not in self.config:
            self.config["variant"] = "unknown"

        self.grad_accum_steps = max(1, int(self.config.get("gradient_accumulation_steps", 1)))
        self._micro_step = 0

        self.audio_contrastive_weight = float(self.config.get("audio_contrastive_weight", 0.0))
        self.audio_contrastive_temperature = float(self.config.get("audio_contrastive_temperature", 0.07))
        self.audio_contrastive_answer_max_length = int(
            self.config.get("audio_contrastive_answer_max_length", 48)
        )
        self.audio_contrastive_metric_weight = float(
            self.config.get("audio_contrastive_metric_weight", 0.0)
        )
        self.audio_contrastive_negative_threshold = float(
            self.config.get("audio_contrastive_negative_threshold", 0.0)
        )
        self.audio_contrastive_max_negatives = int(
            self.config.get("audio_contrastive_max_negatives", 0)
        )

        self.scst_enabled = bool(self.config.get("enable_scst_finetune", False))
        self.scst_patience_epochs = max(1, int(self.config.get("scst_patience_epochs", 2)))
        self.scst_min_delta = float(self.config.get("scst_improvement_threshold", 1e-4))
        self._best_audio_accuracy = float("-inf")
        self._epochs_since_improvement = 0
        self._scst_triggered = False

        self.log_interval = max(1, int(self.config.get("logging_steps", 50)))
        self.config["logging_steps"] = self.log_interval

        timeout_value = float(self.config.get("progress_log_timeout", 0) or 0)
        self.progress_log_timeout = timeout_value if timeout_value > 0 else float("inf")
        self._last_progress_log_time = time.time()

        self.eval_interval = max(1, int(self.config.get("eval_steps", 1000)))
        self.config["eval_steps"] = self.eval_interval

        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        # Create output directory
        Path(self.config["output_dir"]).mkdir(parents=True, exist_ok=True)

        # Initialize loss functions (will be updated by curriculum)
        self._setup_loss_functions()
        
        # Setup optimizer
        self._setup_optimizer()

        self.audio_rerank_num_tags = int(self.config.get("audio_rerank_num_tags", 3) or 0)
        self.audio_rerank_tag_weight = float(self.config.get("audio_rerank_tag_weight", 0.0))
        self.audio_rerank_cider_weight = float(self.config.get("audio_rerank_cider_weight", 0.0))
        self.audio_rerank_spider_weight = float(self.config.get("audio_rerank_spider_weight", 0.0))
        self._tag_vocab_terms: Optional[List[str]] = None
        self._tag_vocab_embeddings: Optional[torch.Tensor] = None
        self._cider_scorer = None
        self._spice_scorer = None

        # Derive a dataset-aware warmup schedule so tiny overfit packs are not stuck in warmup
        nominal_epochs = self.config.get("num_epochs") or 1
        self.total_train_steps = max(1, len(train_dataloader) * max(1, int(nominal_epochs)))
        warmup_ratio = float(self.config.get("warmup_ratio", 0.0))
        ratio_warmup = int(self.total_train_steps * warmup_ratio) if warmup_ratio > 0 else self.config["warmup_steps"]
        effective_warmup = max(1, min(self.config["warmup_steps"], ratio_warmup))
        effective_warmup = min(effective_warmup, self.total_train_steps)
        if effective_warmup != self.config["warmup_steps"]:
            print(
                f"[StageATrainer] Adjusted warmup steps from {self.config['warmup_steps']} to {effective_warmup} based on dataset size.",
                flush=True,
            )
        self.warmup_steps = effective_warmup
        self.config["warmup_steps"] = effective_warmup

        # Setup scheduler (will be updated by curriculum if needed)
        if self.use_curriculum:
            # Curriculum will manage learning rate
            self.scheduler = None
        else:
            total_steps = self.total_train_steps
            cold_steps = max(1, total_steps - self.warmup_steps)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cold_steps)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.sample_log_interval = max(1, int(self.config.get("sample_log_interval", 500)))
        self.sample_log_limit = int(self.config.get("sample_log_limit", 5))
        self.sample_log_examples = max(1, int(self.config.get("sample_log_examples", 3)))
        self.sample_logs_emitted = 0
        self._gate_warm_counter = 0
        self._robust_error_logged = False
        self._teacher_shape_warned = False
        self._teacher_broadcast_warned = False
        self._shape_debug_once = False
        self._eval_shape_debug_once = False
        self._sanity_checks_completed = False
        self._attention_logs_emitted = 0
        self._grad_logs_emitted = 0
        
        # EMA tracking for learning curves
        self.audio_loss_ema = None
        self.ema_decay = 0.98  # 30-50 step EMA (alpha = 1 - decay = 0.02)
        self.ablation_check_interval = 100
        self.last_ablation_step = -1

        waveform_logs = int(self.config.get("waveform_log_limit", 8))
        waveform_debug = bool(self.config.get("log_waveform_stats", False))
        attention_probe = bool(self.config.get("log_attention_probe", False))
        attention_log_limit = int(self.config.get("attention_probe_log_limit", 5))

        if hasattr(self.safe_model, "configure_audio_debug"):
            self.safe_model.configure_audio_debug(waveform_debug, waveform_logs)

        if hasattr(self.safe_model, "configure_attention_probe"):
            self.safe_model.configure_attention_probe(attention_probe, attention_log_limit)

        # Move model to GPU if available
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            print(f"Moving SAFE model to GPU device: {device}", flush=True)
            self.safe_model.to_device(f"cuda:{device}")
            
            # Verify device placement worked
            if self.safe_model.verify_device_placement(f"cuda:{device}"):
                print(f"✅ SAFE model device placement verified successfully", flush=True)
            else:
                print(f"❌ SAFE model device placement FAILED - aborting", flush=True)
                raise RuntimeError("Device placement verification failed")
        else:
            print(f"CUDA not available, keeping model on CPU", flush=True)

        # Ensure base VL model is in eval mode for retention comparison
        self.safe_model.base_vl.eval()
        
        # Fix tokenizer truncation warnings (QoL improvement)
        tok = self.safe_model.base_vl.tokenizer
        if getattr(tok, "model_max_length", None) in (None, 1e30):
            tok.model_max_length = 2048  # pick a sane context limit
        
        # Cache trainable parameter references (needed for gradient ops)
        self.trainable_params = {
            name: param
            for name, param in self.safe_model.named_parameters()
            if param.requires_grad
        }
        self.trainable_param_list = list(self.trainable_params.values())
        
        # DEBUG: Print trainable parameter analysis
        total_params = sum(p.numel() for p in self.safe_model.parameters())
        trainable_count = sum(p.numel() for p in self.trainable_param_list)
        
        print(f"\n🔍 TRAINABLE PARAMETER ANALYSIS:", flush=True)
        print(f"Total model parameters: {total_params:,}", flush=True)
        print(f"Trainable parameters: {trainable_count:,} ({100*trainable_count/total_params:.2f}%)", flush=True)
        
        # Show audio-related trainable components
        audio_trainable = 0
        base_trainable = 0
        for name, param in self.trainable_params.items():
            if any(keyword in name.lower() for keyword in ['audio', 'projector', 'fusion', 'lora']):
                audio_trainable += param.numel()
                print(f"✅ Audio: {name} - {param.numel():,} params", flush=True)
            elif name.startswith('base_vl.'):
                base_trainable += param.numel()
                print(f"⚠️  Base: {name} - {param.numel():,} params", flush=True)
        
        print(f"Audio components: {audio_trainable:,} trainable", flush=True)
        print(f"Base model: {base_trainable:,} trainable", flush=True)
        
        if trainable_count == 0:
            print("❌ CRITICAL: NO TRAINABLE PARAMETERS! Training will not work.", flush=True)
        elif audio_trainable == 0:
            print("❌ CRITICAL: NO AUDIO COMPONENTS TRAINABLE! Audio learning impossible.", flush=True)
        elif base_trainable > 0:
            print("⚠️  WARNING: Base model not frozen - may cause catastrophic forgetting!", flush=True)
        else:
            print("✅ Parameter setup looks correct for modality addition.", flush=True)
        print("=" * 60, flush=True)

        # Metrics tracking
        self.training_stats = {
            "total_loss": [],
            "audio_task_loss": [],
            "audio_gain_score": []
        }

        # Initialize BERTScore for audio caption evaluation (lazy load to avoid startup delays)
        self._bertscore_metric = None
        self._bertscore_failed = False

        # Curriculum state
        self.current_stage_config = None
        self.baseline_metrics = None

        # Initialize CSV export tracking
        self._last_audio_accuracy = 0.0

    def _get_image_token_id(self):
        """Get the image token ID for LLaVA models."""
        # Be defensive across processors/tokenizers
        bv = self.safe_model.base_vl
        tok = getattr(bv, "tokenizer", None)
        proc = getattr(bv, "processor", None)

        image_token = None
        if proc is not None and hasattr(proc, "image_token"):
            image_token = proc.image_token  # e.g., "<image>"
        if image_token is None:
            # Common default in Llava; still try tokenizer first
            image_token = "<image>"

        if tok is not None and hasattr(tok, "convert_tokens_to_ids"):
            # Handle both AddedToken objects and string tokens
            if hasattr(image_token, 'content'):
                # It's an AddedToken, get the string content
                token_str = image_token.content
            else:
                # It's already a string
                token_str = str(image_token)
            
            # convert_tokens_to_ids expects a list of tokens
            token_ids = tok.convert_tokens_to_ids([token_str])
            return token_ids[0] if token_ids else None

        # Last-resort fallback (LLaVA often uses 32000)
        return getattr(bv, "image_token_id", 32000)

    def _forward_llava_teacher(self, base_inputs: dict):
        """Run the frozen base LLaVA model on each sample to keep logits aligned with the SAFE batch."""
        input_ids = base_inputs.get("input_ids")
        attention_mask = base_inputs.get("attention_mask")
        labels = base_inputs.get("labels")
        pixel_values = base_inputs.get("pixel_values")

        # LLaVA checkpoints often run in fp16. Ensure any floating point inputs
        # that we pass to the frozen teacher (e.g. pixel values) match the
        # underlying model dtype to avoid matmul dtype mismatches inside the
        # HuggingFace layers.
        model_dtype = getattr(self.safe_model.base_vl.llm, "dtype", None)
        if model_dtype is None:
            try:
                model_dtype = next(self.safe_model.base_vl.llm.parameters()).dtype
            except StopIteration:
                model_dtype = None

        if model_dtype is not None:
            for key in ("pixel_values", "inputs_embeds"):
                value = base_inputs.get(key)
                if torch.is_tensor(value) and value.is_floating_point() and value.dtype != model_dtype:
                    base_inputs[key] = value.to(dtype=model_dtype)
                    if key == "pixel_values":
                        pixel_values = base_inputs[key]

        if input_ids is None:
            return self.safe_model.base_vl(**base_inputs)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            if isinstance(labels, torch.Tensor) and labels.dim() == 1:
                labels = labels.unsqueeze(0)
            if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)

        batch_size = input_ids.size(0)
        image_token_id = self.safe_model._get_image_token_id()
        has_img_tokens = (input_ids == image_token_id).any(dim=1)

        if pixel_values is None:
            pixel_values = torch.empty(0)

        def _slice_inputs(indices: torch.Tensor, keep_pixels: bool) -> dict:
            if indices is None or indices.numel() == 0:
                return None
            subset = {}
            for key, value in base_inputs.items():
                if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.size(0) == batch_size:
                    subset[key] = value.index_select(0, indices.to(value.device))
                else:
                    subset[key] = value
            if not keep_pixels:
                subset.pop("pixel_values", None)
            return subset

        idx_mm = torch.nonzero(has_img_tokens, as_tuple=False).flatten()
        idx_text = torch.nonzero(~has_img_tokens, as_tuple=False).flatten()

        outputs_ordered: List[Optional[torch.Tensor]] = [None] * batch_size
        loss_values: List[torch.Tensor] = []

        def _run_subset(sub_inputs, indices):
            if sub_inputs is None:
                return
            result = self.safe_model.base_vl(**sub_inputs)
            logits_subset = result.get("logits") if isinstance(result, dict) else getattr(result, "logits", None)
            if logits_subset is None:
                raise RuntimeError("Base LLaVA teacher returned no logits for a subset.")
            for slot, tensor in zip(indices.tolist(), logits_subset):
                outputs_ordered[slot] = tensor.unsqueeze(0)
            loss_tensor = result.get("loss") if isinstance(result, dict) else getattr(result, "loss", None)
            if loss_tensor is not None:
                loss_values.append(loss_tensor)

        _run_subset(_slice_inputs(idx_mm, keep_pixels=True), idx_mm)
        _run_subset(_slice_inputs(idx_text, keep_pixels=False), idx_text)

        if any(o is None for o in outputs_ordered):
            missing = [i for i, o in enumerate(outputs_ordered) if o is None]
            raise RuntimeError(f"Base LLaVA teacher failed to return logits for samples: {missing}")

        logits = torch.cat(outputs_ordered, dim=0)

        class OutputsWrapper(dict):
            pass

        wrapped = OutputsWrapper()
        wrapped["logits"] = logits
        if loss_values:
            wrapped["loss"] = torch.stack(loss_values).mean()
        return wrapped

    def _sanitize_input_ids_batch(self, input_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if input_ids is None:
            return None
        sanitized = self.safe_model.sanitize_input_ids_for_base(input_ids)
        if sanitized is None:
            return None
        if not torch.is_tensor(sanitized):
            return sanitized
        if input_ids.dim() == 2 and sanitized.dim() == 1:
            per_sample = []
            for row in input_ids:
                row_clean = self.safe_model.sanitize_input_ids_for_base(row)
                if torch.is_tensor(row_clean) and row_clean.dim() == 1:
                    row_clean = row_clean.unsqueeze(0)
                per_sample.append(row_clean)
            sanitized = torch.cat(per_sample, dim=0)
        return sanitized
    
    def _setup_loss_functions(self):
        """Setup loss functions with curriculum-aware weights."""
        if self.use_curriculum and self.curriculum_manager.current_stage:
            stage_config = self.curriculum_manager.get_current_config()
            loss_weights = stage_config.get("loss_weights", {})
        else:
            loss_weights = {}

        self.audio_task_loss = AudioTaskLoss(
            task_type="qa",
            label_smoothing=float(self.config.get("audio_label_smoothing", 0.1)),
            debug=False,
        )

        self.audio_loss_weight = float(loss_weights.get("audio_task_loss", self.config.get("audio_loss_weight", 1.0)))

        print(f"[LossSetup] ✓ audio_weight={self.audio_loss_weight}", flush=True)

    def _shorten_text_for_log(self, text: Any, limit: int = 80) -> str:
        if text is None:
            return ""

        display = str(text).strip()
        if not display:
            return ""

        display = display.replace("\n", " ")
        display = re.sub(r"\s+", " ", display)

        # Repair truncated assistant prefixes (common when prompts are trimmed)
        display = display.replace("SSISTANT:", "ASSISTANT:")
        display = display.replace("ISTANT:", "ASSISTANT:")
        display = display.replace("ANT:", "ASSISTANT:")

        # Collapse to the segment after the final ASSISTANT marker if present
        lower_display = display.lower()
        if "assistant:" in lower_display:
            parts = re.split(r"assistant:\s*", display, flags=re.IGNORECASE)
            # Keep a short context before the last ASSISTANT for reference when available
            if len(parts) >= 2:
                prefix = parts[-2].strip()
                answer = parts[-1].strip()
                if prefix:
                    display = f"{prefix} → {answer}"
                else:
                    display = answer

        if len(display) > limit:
            display = display[: limit - 1].rstrip() + "…"

        return display

    def update_curriculum_config(self):
        """Update training configuration based on current curriculum stage."""
        if not self.use_curriculum:
            return
            
        stage_config = self.curriculum_manager.get_current_config()
        if stage_config is None:
            return
            
        self.current_stage_config = stage_config
        
        # Update loss function weights
        self._setup_loss_functions()
        
        # Update learning rates if specified
        lr_multiplier = stage_config.get("learning_rate_multiplier", 1.0)
        if lr_multiplier != 1.0:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "projector":
                    param_group["lr"] = self.config["learning_rate_projector"] * lr_multiplier
                elif param_group["name"] == "adapter":
                    param_group["lr"] = self.config["learning_rate_adapter"] * lr_multiplier
                elif param_group["name"] == "audio_tokens":
                    base_lr = self.config.get(
                        "learning_rate_audio_embeddings",
                        self.config["learning_rate_projector"],
                    )
                    param_group["lr"] = base_lr * lr_multiplier
        
        print(f"Updated to curriculum stage: {stage_config['stage_name']}", flush=True)
        print(f"  Audio ratio: {stage_config['audio_ratio']:.2f}", flush=True)
        print(f"  Difficulty filter: {stage_config['difficulty_filter']}", flush=True)
        print(f"  LR multiplier: {lr_multiplier:.2f}", flush=True)
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for different components."""
        param_groups = []
        
        # Projector parameters
        projector_params = list(self.safe_model.audio_projector.parameters())
        if projector_params:
            projector_lr = self.config["learning_rate_projector"]
            param_groups.append({
                "params": projector_params,
                "lr": projector_lr,
                "base_lr": projector_lr,  # Store base learning rate for warmup
                "name": "projector"
            })
        
        # Fusion adapter parameters (LoRA)
        adapter_params = list(self.safe_model.fusion_adapter.parameters())
        if adapter_params:
            adapter_lr = self.config["learning_rate_adapter"]
            param_groups.append({
                "params": adapter_params,
                "lr": adapter_lr,
                "base_lr": adapter_lr,  # Store base learning rate for warmup
                "name": "adapter"
            })

        # Audio token embedding parameters (if present)
        audio_token_module = getattr(self.safe_model, "audio_token_embeddings", None)
        if audio_token_module is not None:
            audio_token_params = [
                p for p in audio_token_module.parameters() if p.requires_grad
            ]
            if audio_token_params:
                token_lr = self.config.get(
                    "learning_rate_audio_embeddings",
                    self.config["learning_rate_projector"],
                )
                param_groups.append({
                    "params": audio_token_params,
                    "lr": token_lr,
                    "base_lr": token_lr,
                    "name": "audio_tokens"
                })

        if not param_groups:
            raise ValueError("No trainable parameters found!")
        
        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.config["weight_decay"],
            betas=(0.9, 0.999)
        )
        
        # Parameter validation
        trainable_count = sum(p.numel() for group in param_groups for p in group["params"] if p.requires_grad)
        print(f"Trainable parameters: {trainable_count:,}", flush=True)
        
        # base_lr is already set in parameter groups above
    
    def _apply_warmup(self, step: int):
        """Apply learning rate warmup (assign, don't multiply)."""
        factor = min(1.0, step / max(1, getattr(self, "warmup_steps", self.config["warmup_steps"])))
        for g in self.optimizer.param_groups:
            base_lr = g.get("base_lr", g["lr"])
            # Ensure base_lr is a number, not a sequence
            if isinstance(base_lr, (list, tuple)):
                print(f"Warning: base_lr is a sequence: {base_lr}, using first element", flush=True)
                base_lr = base_lr[0]
            g["lr"] = float(base_lr) * factor

    def _clone_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        cloned: Dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.detach().clone()
            elif isinstance(value, list):
                cloned[key] = list(value)
            else:
                cloned[key] = value
        return cloned

    def _select_batch_indices(
        self,
        inputs: Dict[str, Any],
        indices: torch.Tensor,
        clone: bool = True,
    ) -> Dict[str, Any]:
        if indices.numel() == 0:
            return {}

        if indices.dtype != torch.long:
            indices = indices.to(torch.long)

        base_tensor = inputs.get("input_ids")
        if not isinstance(base_tensor, torch.Tensor):
            base_tensor = next(
                (v for v in inputs.values() if isinstance(v, torch.Tensor) and v.dim() > 0),
                None,
            )
        batch_size = base_tensor.size(0) if isinstance(base_tensor, torch.Tensor) else None

        subset: Dict[str, Any] = {}
        index_list = indices.tolist()

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and batch_size is not None and value.size(0) == batch_size:
                device_indices = indices.to(value.device)
                sliced = value.index_select(0, device_indices)
                subset[key] = sliced.detach().clone() if clone else sliced
            elif isinstance(value, list) and batch_size is not None and len(value) == batch_size:
                subset[key] = [value[int(i)] for i in index_list]
            else:
                if isinstance(value, torch.Tensor) and clone:
                    subset[key] = value.detach().clone()
                else:
                    subset[key] = value

        return subset

    def _compute_audio_loss(self, inputs: Dict[str, Any], gate: float) -> Optional[float]:
        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")
        if labels is None:
            return None

        was_training = self.safe_model.training
        self.safe_model.eval()
        try:
            with torch.no_grad():
                outputs = self.safe_model(**inputs, gate=gate)
                logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
                if logits is None:
                    return None
                loss = self.audio_task_loss(logits, labels, attention_mask)
                return float(loss.detach().item())
        finally:
            if was_training:
                self.safe_model.train()

    def _sanitize_audio_gradients(self) -> None:
        """Replace NaN/Inf grads on audio modules to keep optimization stable."""
        if not self.config.get("sanitize_nan_gradients", True):
            return

        clip_value = float(self.config.get("grad_sanitize_clip", 1e3))

        for name, param in self.safe_model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue

            lower = name.lower()
            if (
                "audio_projector" not in lower
                and "fusion_adapter" not in lower
                and "lora" not in lower
                and "audio_token_embeddings" not in lower
            ):
                continue

            grad = param.grad
            if not grad.dtype.is_floating_point:
                continue

            if torch.isnan(grad).any() or torch.isinf(grad).any():
                torch.nan_to_num_(grad, nan=0.0, posinf=clip_value, neginf=-clip_value)
                grad.clamp_(-clip_value, clip_value)


    def _sanitize_audio_parameters(self) -> None:
        """Clamp/sanitize audio pathway parameters if they ever contain NaNs/Infs."""
        clip_value = float(self.config.get("param_sanitize_clip", 1e3))

        for name, param in self.safe_model.named_parameters():
            if not param.requires_grad:
                continue

            lower = name.lower()
            if (
                "audio_projector" not in lower
                and "fusion_adapter" not in lower
                and "lora" not in lower
                and "audio_token_embeddings" not in lower
            ):
                continue

            data = param.data
            if torch.isnan(data).any() or torch.isinf(data).any():
                torch.nan_to_num_(data, nan=0.0, posinf=clip_value, neginf=-clip_value)
                data.clamp_(-clip_value, clip_value)


    def _check_silent_audio(self, batch: Dict) -> torch.Tensor:
        """Check for silent/zero audio clips and return has_audio mask."""
        has_audio = batch.get("has_audio", None)
        audio_data = batch.get("audio", None)

        if has_audio is None or audio_data is None:
            return has_audio if has_audio is not None else torch.tensor([])

        # DISABLED: Let the audio masking in safe_model.py handle this
        # The pre-filtering here was too aggressive and causing audio loss to be 0
        # Just log statistics without filtering
        silent_threshold = 1e-8
        silent_count = 0
        for audio_tensor in audio_data:
            waveform = None
            if isinstance(audio_tensor, torch.Tensor):
                waveform = audio_tensor
            elif isinstance(audio_tensor, tuple) and audio_tensor:
                candidate = audio_tensor[0]
                if isinstance(candidate, torch.Tensor):
                    waveform = candidate
            elif isinstance(audio_tensor, dict):
                candidate = audio_tensor.get("waveform")
                if isinstance(candidate, torch.Tensor):
                    waveform = candidate

            if waveform is None:
                continue

            max_amplitude = torch.abs(waveform).max().item() if waveform.numel() > 0 else 0.0
            if max_amplitude < silent_threshold:
                silent_count += 1

        if silent_count > 0:
            print(
                f"[SilentAudioCheck] Found {silent_count}/{len(audio_data)} silent samples (not filtered)",
                flush=True,
            )

        return has_audio  # Return unchanged

    def train_step(self, batch: Dict) -> Tuple[Dict[str, float], bool]:
        """
        Single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Tuple of (loss metrics, optimizer_step_performed)
        """
        self.safe_model.train()
        self.safe_model.enable_audio_training()  # Only train audio components

        if self._micro_step == 0:
            self.optimizer.zero_grad(set_to_none=True)

        # Move batch to device
        device = next(self.safe_model.parameters()).device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Prepare inputs for SAFE model (device-consistent masks)
        has_audio = batch.get(
            "has_audio",
            torch.zeros(len(batch["questions"]), dtype=torch.bool, device=device),
        )
        if isinstance(has_audio, torch.Tensor):
            has_audio = has_audio.to(device=device, dtype=torch.bool)
        else:
            has_audio = torch.tensor(has_audio, dtype=torch.bool, device=device)

        # Filter silent audio clips
        if self.config.get("filter_silent_audio", True):
            has_audio = self._check_silent_audio(batch)
            if isinstance(has_audio, torch.Tensor):
                has_audio = has_audio.to(device=device, dtype=torch.bool)

        if has_audio is None or has_audio.numel() == 0 or not bool(has_audio.any()):
            return {"audio_task_loss": 0.0, "total_loss": 0.0}, False

        batch_questions = list(batch.get("questions", []))
        batch_images = batch.get("images", None)
        batch_audio = batch.get("audio", None)
        batch_size = len(batch_questions)
        reference_answer_list = self._build_training_answer_list(batch.get("answers"), batch_size)

        (
            batch_questions,
            batch_images,
            batch_audio,
            sampled_answers,
            has_audio,
            reference_answer_list,
        ) = self._expand_audio_caption_training_batch(
            batch_questions,
            batch_images,
            batch_audio,
            reference_answer_list,
            has_audio,
        )

        has_audio = has_audio.to(device=device, dtype=torch.bool)
        audio_indices = torch.where(has_audio)[0]
        if audio_indices.numel() == 0:
            return {"audio_task_loss": 0.0, "total_loss": 0.0}, False

        # Create input tensors for training - apply answers for supervised learning
        inputs = self.safe_model.prepare_multimodal_inputs(
            text=batch_questions,
            images=batch_images if batch_images is not None else None,
            audio=batch_audio if batch_audio is not None else None,
            answers=sampled_answers,
            device=device,
            training_mode=True  # Apply answers during training
        )

        # Drive fusion gate warmup using micro-step aware schedule
        if hasattr(self.safe_model, "set_gate_warmup"):
            warmup_steps = int(self.config.get("gate_warmup_steps", self.warmup_steps))
            warmup_steps = max(1, warmup_steps)
            self._gate_warm_counter += 1
            self.safe_model.set_gate_warmup(self._gate_warm_counter, warmup_steps)

        audio_token_tensor = inputs.get("audio_tokens")

        # Forward pass through SAFE model
        safe_outputs = self.safe_model(**inputs)

        safe_logits = safe_outputs.get("logits") if isinstance(safe_outputs, dict) else getattr(safe_outputs, "logits", None)

        # Audio loss only (no retention)
        if safe_logits is not None and audio_indices.numel() > 0:
            labels = inputs.get("labels")
            if labels is not None:
                max_safe = safe_logits.size(0)
                max_labels = labels.size(0)
                max_bound = min(max_safe, max_labels)
                audio_indices = audio_indices[audio_indices < max_bound]
            else:
                audio_indices = audio_indices[:0]

            if audio_indices.numel() > 0 and labels is not None:
                audio_logits = safe_logits[audio_indices]
                audio_labels = labels[audio_indices]
                audio_mask = inputs.get("attention_mask", None)
                if audio_mask is not None:
                    audio_mask = audio_mask[audio_indices]
                audio_loss = self.audio_task_loss(audio_logits, audio_labels, audio_mask)
            else:
                # Create zero loss while maintaining gradient connection via safe_logits
                audio_loss = safe_logits.sum() * 0.0
        else:
            if safe_logits is not None:
                # Create zero loss while maintaining gradient connection via safe_logits
                audio_loss = safe_logits.sum() * 0.0
            else:
                # No logits available - create zero tensor with gradients
                audio_loss = torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = self.audio_loss_weight * audio_loss

        contrastive_raw = None
        if (
            self.audio_contrastive_weight > 0.0
            and audio_token_tensor is not None
            and audio_indices.numel() > 0
        ):
            contrastive_raw = self._compute_audio_contrastive_loss(
                audio_token_tensor,
                sampled_answers,
                audio_indices,
                reference_answers=reference_answer_list,
            )
            if contrastive_raw is not None:
                total_loss = total_loss + (self.audio_contrastive_weight * contrastive_raw)

        # Ensure total_loss has gradients
        if not total_loss.requires_grad:
            if safe_logits is None or not safe_logits.requires_grad:
                # No gradient path (e.g., gate=0) -> skip this micro step quietly
                return {"audio_task_loss": 0.0, "total_loss": 0.0}, False
            else:
                raise RuntimeError("Cannot create loss with gradients - model outputs don't require grad")

        loss_dict = {
            "audio_task_loss": audio_loss,
            "total_loss": total_loss,
        }
        if contrastive_raw is not None:
            loss_dict["audio_contrastive_loss"] = contrastive_raw * self.audio_contrastive_weight

        log_metrics: Dict[str, float] = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                log_metrics[key] = float(value.detach().cpu().item())
            else:
                log_metrics[key] = float(value)

        # Backward pass with gradient accumulation support
        total_loss = loss_dict["total_loss"]
        loss_scale = 1.0 / float(self.grad_accum_steps)
        (total_loss * loss_scale).backward()

        # Clean up NaN/Inf gradients from audio pathway before diagnostics
        self._sanitize_audio_gradients()

        self._micro_step += 1
        ready_to_step = self._micro_step >= self.grad_accum_steps

        if ready_to_step:
            # Optional additional parameter sanitization before inspection
            self._sanitize_audio_parameters()

            # Gradient clipping with NaN detection and prevention
            if self.trainable_param_list:
                grad_clip = float(self.config.get("grad_sanitize_clip", 1e3))
                for param in self.trainable_param_list:
                    if param.grad is None:
                        continue
                    grad = param.grad
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        torch.nan_to_num_(grad, nan=0.0, posinf=grad_clip, neginf=-grad_clip)
                        grad.clamp_(-grad_clip, grad_clip)
                        if torch.isnan(grad).any() or torch.isinf(grad).any():
                            grad.zero_()

                torch.nn.utils.clip_grad_norm_(
                    self.trainable_param_list,
                    self.config["max_grad_norm"],
                )

            # Store pre-update parameters for update norm calculation
            pre_update_params = {}
            if self.config.get("log_update_norms", True):
                for name, param in self.safe_model.named_parameters():
                    if param.requires_grad:
                        pre_update_params[name] = param.data.clone()

            # Optimizer step
            self.optimizer.step()
            self._sanitize_audio_parameters()

            self._micro_step = 0

            # Learning rate scheduling (fixed: use local_step, assign don't multiply)
            local_step = self.global_step + 1  # caller increments after returning

            if local_step <= self.warmup_steps:
                self._apply_warmup(local_step)
            elif self.scheduler is not None:
                self.scheduler.step()

        return log_metrics, ready_to_step
    
    def evaluate(
        self,
        max_batches: Optional[int] = None,
        dataloader: Optional[DataLoader] = None,
        description: str = "Validation",
        split_batches: Optional[bool] = None,
        save_audio_samples_csv: Optional[str] = None,
        max_csv_samples: int = 500,
    ) -> Dict[str, float]:
        """
        Evaluate model on a specified dataset.

        Args:
            max_batches: Optional limit on number of batches to evaluate
            dataloader: Optional dataloader to evaluate. Defaults to validation loader
            description: Human readable description used in logs
            split_batches: Override whether to perform audio/VL split evaluation
            save_audio_samples_csv: Optional path to save audio sample predictions to CSV
            max_csv_samples: Maximum number of audio samples to save to CSV (default: 500)

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"[Eval] Starting evaluation: {description}", flush=True)
        previous_mode = self.safe_model.training
        self.safe_model.eval()

        # Clear GPU cache before evaluation to free memory from training
        print(f"[Eval] Clearing GPU cache...", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Log GPU memory status
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[Eval] GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved", flush=True)

        data_source = dataloader if dataloader is not None else self.val_dataloader
        if data_source is None:
            raise ValueError("No dataloader available for evaluation")

        print(f"[Eval] DataLoader ready, beginning batch iteration...", flush=True)

        dataset_len = None
        dataset_obj = getattr(data_source, "dataset", None)
        if dataset_obj is not None:
            try:
                dataset_len = len(dataset_obj)
            except TypeError:
                dataset_len = None

        if dataset_len == 0:
            print(f"Warning: {description} dataset is empty!", flush=True)
            return {
                "loss": 0.0,
                "audio_loss": 0.0,
                "vl_loss": 0.0,
                "retention_loss": 0.0,
                "audio_accuracy": 0.0,
                "vl_safe_accuracy": 0.0,
                "vl_base_accuracy": 0.0,
                "retention_score": 0.0
            }
        
        total_samples = 0
        audio_samples = 0
        vl_samples = 0
        
        # Metrics for audio-dependent samples
        audio_correct = 0
        audio_total = 0
        audio_exact = 0.0
        audio_token_f1 = 0.0
        audio_bertscore = 0.0
        audio_caption_predictions: List[str] = []
        audio_caption_references: List[List[str]] = []

        # CSV export collection for audio samples
        csv_samples = [] if save_audio_samples_csv else None

        # Metrics for VL retention
        vl_safe_correct = 0
        vl_base_correct = 0
        vl_total = 0
        
        eval_losses = {
            "total_loss": 0.0,
            "audio_task_loss": 0.0,
        }
        
        device = next(self.safe_model.parameters()).device
        eval_logging_steps = self.config.get("eval_logging_steps", 20)
        
        # Get split batch evaluation parameters
        max_audio_eval_batches = int(self.config.get("max_audio_eval_batches", 0) or 0)
        max_vl_eval_batches = int(self.config.get("max_vl_eval_batches", 0) or 0)
        max_audio_eval_samples = int(self.config.get("max_audio_eval_samples", 0) or 0)
        max_vl_eval_samples = int(self.config.get("max_vl_eval_samples", 0) or 0)

        if split_batches is None:
            split_batches = any(
                (
                    max_audio_eval_batches > 0,
                    max_vl_eval_batches > 0,
                    max_audio_eval_samples > 0,
                    max_vl_eval_samples > 0,
                )
            )

        if not split_batches:
            max_audio_eval_batches = 0
            max_vl_eval_batches = 0
            max_audio_eval_samples = 0
            max_vl_eval_samples = 0

        if split_batches:
            audio_batch_limit = max_audio_eval_batches if max_audio_eval_batches > 0 else None
            vl_batch_limit = max_vl_eval_batches if max_vl_eval_batches > 0 else None
            audio_sample_limit = max_audio_eval_samples if max_audio_eval_samples > 0 else None
            vl_sample_limit = max_vl_eval_samples if max_vl_eval_samples > 0 else None
        else:
            audio_batch_limit = None
            vl_batch_limit = None
            audio_sample_limit = None
            vl_sample_limit = None

        collected_audio_batches = 0
        collected_vl_batches = 0

        eval_with_audio_gate = self.config.get("eval_with_audio_gate", True)
        eval_audio_gate_comparison = self.config.get("eval_audio_gate_comparison", False)
        
        # Save current gate state to restore after evaluation
        saved_gate = None
        if hasattr(self.safe_model, 'get_gate'):
            try:
                saved_gate = self.safe_model.get_gate()
            except:
                saved_gate = 1.0  # Default fallback
        elif hasattr(self.safe_model, '_default_gate'):
            saved_gate = self.safe_model._default_gate
        else:
            saved_gate = 1.0  # Safe default
        
        try:
            with torch.no_grad():
                # Separate audio and VL batches if split evaluation is configured
                if split_batches:
                    def _format_limit(sample_limit: Optional[int], batch_limit: Optional[int]) -> str:
                        sample_str = str(sample_limit) if sample_limit is not None else "∞"
                        batch_str = str(batch_limit) if batch_limit is not None else "∞"
                        return f"{sample_str} samples / {batch_str} batches"

                    print(
                        "Using split evaluation on "
                        f"{description} set with limits: audio {_format_limit(audio_sample_limit, audio_batch_limit)}, "
                        f"VL {_format_limit(vl_sample_limit, vl_batch_limit)}",
                        flush=True,
                    )

                    audio_batches: List[Dict[str, Any]] = []
                    vl_batches: List[Dict[str, Any]] = []
                    audio_samples_collected = 0
                    vl_samples_collected = 0

                    def _can_collect(sample_limit: Optional[int], collected_samples: int,
                                     batch_limit: Optional[int], collected_batches: int) -> bool:
                        sample_ok = True if sample_limit is None else collected_samples < sample_limit
                        batch_ok = True if batch_limit is None else collected_batches < batch_limit
                        return sample_ok and batch_ok

                    # Collect batches and separate them
                    batch_count = 0
                    print(f"[Eval] Starting batch collection loop...", flush=True)
                    for batch in data_source:
                        batch_count += 1
                        if batch_count % 10 == 1:
                            print(f"[Eval] Processing batch {batch_count}, collected audio={audio_samples_collected}, vl={vl_samples_collected}", flush=True)

                        # Move batch to device
                        for key in batch:
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].to(device)

                        # Determine if batch has audio samples
                        has_audio = batch.get(
                            "has_audio",
                            torch.zeros(len(batch["questions"]), dtype=torch.bool, device=device),
                        )
                        if isinstance(has_audio, torch.Tensor):
                            has_audio = has_audio.to(device=device, dtype=torch.bool)
                        else:
                            has_audio = torch.tensor(has_audio, dtype=torch.bool, device=device)

                        # Debug batch composition
                        audio_indices = torch.nonzero(has_audio, as_tuple=False).flatten()
                        vl_indices = torch.nonzero(~has_audio, as_tuple=False).flatten()

                        if audio_indices.numel() > 0 and _can_collect(audio_sample_limit, audio_samples_collected, audio_batch_limit, len(audio_batches)):
                            if audio_sample_limit is not None:
                                remaining = audio_sample_limit - audio_samples_collected
                                if remaining <= 0:
                                    audio_indices = audio_indices[:0]
                                else:
                                    audio_indices = audio_indices[:remaining]
                            if audio_indices.numel() > 0:
                                audio_subset = self._select_batch_indices(batch, audio_indices, clone=True)
                                if audio_subset:
                                    audio_batches.append(audio_subset)
                                    added = len(audio_subset.get("questions", []))
                                    if added == 0:
                                        tensor_candidate = next(
                                            (
                                                v
                                                for v in audio_subset.values()
                                                if isinstance(v, torch.Tensor) and v.dim() > 0
                                            ),
                                            None,
                                        )
                                        if tensor_candidate is not None:
                                            added = int(tensor_candidate.size(0))
                                    audio_samples_collected += added
                        elif audio_indices.numel() > 0 and audio_batch_limit is not None:
                            pass  # Audio batch limit reached
                        elif audio_indices.numel() > 0 and audio_sample_limit is not None:
                            pass  # Audio sample limit reached

                        if vl_indices.numel() > 0 and _can_collect(vl_sample_limit, vl_samples_collected, vl_batch_limit, len(vl_batches)):
                            if vl_sample_limit is not None:
                                remaining_vl = vl_sample_limit - vl_samples_collected
                                if remaining_vl <= 0:
                                    vl_indices = vl_indices[:0]
                                else:
                                    vl_indices = vl_indices[:remaining_vl]
                            if vl_indices.numel() > 0:
                                vl_subset = self._select_batch_indices(batch, vl_indices, clone=True)
                                if vl_subset:
                                    vl_batches.append(vl_subset)
                                    added_vl = len(vl_subset.get("questions", []))
                                    if added_vl == 0:
                                        tensor_candidate_vl = next(
                                            (
                                                v
                                                for v in vl_subset.values()
                                                if isinstance(v, torch.Tensor) and v.dim() > 0
                                            ),
                                            None,
                                        )
                                        if tensor_candidate_vl is not None:
                                            added_vl = int(tensor_candidate_vl.size(0))
                                    vl_samples_collected += added_vl
                        elif vl_indices.numel() > 0 and vl_batch_limit is not None:
                            pass  # VL batch limit reached
                        elif vl_indices.numel() > 0 and vl_sample_limit is not None:
                            pass  # VL sample limit reached

                        batch_count += 1

                        audio_done = not _can_collect(audio_sample_limit, audio_samples_collected, audio_batch_limit, len(audio_batches))
                        vl_done = not _can_collect(vl_sample_limit, vl_samples_collected, vl_batch_limit, len(vl_batches))
                        if audio_done and vl_done:
                            break

                    # Create combined evaluation list
                    eval_batch_list = []
                    for i, batch in enumerate(audio_batches):
                        eval_batch_list.append((f"AUDIO-{i+1}", batch))
                    for i, batch in enumerate(vl_batches):
                        eval_batch_list.append((f"VL-{i+1}", batch))

                    batch_iterable = eval_batch_list
                    total_eval_batches = len(eval_batch_list)
                    collected_audio_batches = len(audio_batches)
                    collected_vl_batches = len(vl_batches)
                    print(
                        f"Split evaluation prepared {collected_audio_batches} audio + {collected_vl_batches} VL batches (total={total_eval_batches})",
                        flush=True,
                    )
                else:
                    # Original evaluation logic
                    batch_iterable = data_source
                    if max_batches is not None and max_batches > 0:
                        from itertools import islice

                        batch_iterable = islice(data_source, max_batches)
                        total_eval_batches = max_batches
                    else:
                        total_eval_batches = len(data_source) if hasattr(data_source, "__len__") else None
                    if max_batches:
                        print(
                            f"Standard evaluation on first {max_batches} {description.lower()} batches...",
                            flush=True,
                        )
                    else:
                        batch_msg = (
                            f"{total_eval_batches}"
                            if isinstance(total_eval_batches, int)
                            else "an unknown number of"
                        )
                        print(
                            f"Standard evaluation on {batch_msg} {description.lower()} batches...",
                            flush=True,
                        )

                for batch_idx, batch_data in enumerate(batch_iterable, start=1):
                    # Handle both split evaluation format (batch_label, batch) and standard format
                    if isinstance(batch_data, tuple) and len(batch_data) == 2:
                        batch_label, batch = batch_data
                    else:
                        batch = batch_data
                        batch_label = f"batch"
                        # Move batch to device for standard evaluation
                        for key in batch:
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].to(device)
                    
                    has_audio = batch.get(
                        "has_audio",
                        torch.zeros(len(batch["questions"]), dtype=torch.bool, device=device),
                    )
                    if isinstance(has_audio, torch.Tensor):
                        has_audio = has_audio.to(device=device, dtype=torch.bool)
                    else:
                        has_audio = torch.tensor(has_audio, dtype=torch.bool, device=device)
                    
                    # Prepare inputs - always pass images/audio if available, don't gate on flags
                    # Hard assert audio presence for audio batches
                    if batch_label.startswith("AUDIO"):
                        assert batch.get("audio_tokens", None) is not None or (batch.get("has_audio", torch.zeros(1))).any(), \
                            f"Eval set has no usable audio (tokens/mask missing) for batch {batch_label}"
                        # Removed: print(f"[AUDIO_EVAL] Audio batch confirmed: {batch_label}", flush=True)
                    
                    # IMPORTANT: Preserve original answers from batch BEFORE preprocessing
                    # These are needed for accuracy computation after generation
                    original_answers = batch.get("answers", [])

                    # DEBUG: Check what's in the batch
                    # Removed verbose BatchDebug logging
                    pass

                    # Ensure include_audio_tokens=True for audio samples to guarantee audio placeholders in prompts
                    has_audio_data = batch.get("audio", None) is not None
                    inputs = self.safe_model.prepare_multimodal_inputs(
                        text=batch["questions"],
                        images=batch.get("images", None),
                        audio=batch.get("audio", None),
                        answers=None,  # Do NOT pass answers during evaluation to prevent contamination
                        device=device,
                        include_audio_tokens=has_audio_data,  # Critical: ensure audio placeholders for audio samples
                        training_mode=False  # Inference mode - no answer application
                    )

                    # Add preserved answers back to inputs for accuracy computation
                    inputs["original_answers"] = original_answers

                    # Apply audio gate control if configured
                    if eval_audio_gate_comparison and hasattr(self.safe_model, 'set_gate'):
                        # Run evaluation with both audio gate on/off to measure VL drift
                        if batch_label.startswith("AUDIO"):
                            # For audio batches, use gate=1.0 (with audio)
                            self.safe_model.set_gate(1.0)
                        else:
                            # For VL batches, use gate=0.0 (without audio) to measure drift
                            self.safe_model.set_gate(0.0)
                    elif not eval_with_audio_gate and hasattr(self.safe_model, 'set_gate'):
                        # Disable audio entirely if configured
                        self.safe_model.set_gate(0.0)
                    
                    # Handle pixel_values based on model type (AFTER gate settings)
                    if self.safe_model.base_vl.model_type == "llava":
                        # Only LLaVA uses <image> tokens - check for image token presence
                        image_token_id = self._get_image_token_id()
                        input_ids = inputs.get("input_ids")
                        if input_ids is not None:
                            if input_ids.dim() == 1:
                                input_ids = input_ids.unsqueeze(0)
                            
                            has_img_tokens = (input_ids == image_token_id).any(dim=1)
                            
                            # If no samples have image tokens but we have pixel_values, remove them
                            if not torch.any(has_img_tokens) and "pixel_values" in inputs:
                                inputs.pop("pixel_values")
                            # If image tokens are present but pixel_values missing, log warning and skip
                            elif torch.any(has_img_tokens) and "pixel_values" not in inputs:
                                print(
                                    f"Warning: Found <image> tokens but no pixel_values in {batch_label}. Skipping multimodal processing.",
                                    flush=True
                                )
                    elif self.safe_model.base_vl.model_type == "blip2":
                        # BLIP-2 doesn't use <image> tokens, so keep pixel_values as-is if they exist
                        pass
                    
                    # SAFE model forward pass
                    safe_outputs = self.safe_model(**inputs)

                    safe_logits = safe_outputs.get("logits") if isinstance(safe_outputs, dict) else getattr(safe_outputs, "logits", None)

                    if safe_logits is not None and torch.any(has_audio):
                        audio_indices = torch.where(has_audio)[0]
                        labels = inputs.get("labels")
                        if labels is not None:
                            max_safe = safe_logits.size(0)
                            max_labels = labels.size(0)
                            max_bound = min(max_safe, max_labels)
                            audio_indices = audio_indices[audio_indices < max_bound]
                        else:
                            audio_indices = audio_indices[:0]

                        if audio_indices.numel() > 0 and labels is not None:
                            audio_logits = safe_logits[audio_indices]
                            audio_labels = labels[audio_indices]
                            audio_mask = inputs.get("attention_mask", None)
                            if audio_mask is not None:
                                audio_mask = audio_mask[audio_indices]
                            batch_audio_loss = self.audio_task_loss(audio_logits, audio_labels, audio_mask)
                        else:
                            batch_audio_loss = safe_logits.sum() * 0.0
                    else:
                        if safe_logits is not None:
                            batch_audio_loss = safe_logits.sum() * 0.0
                        else:
                            batch_audio_loss = torch.tensor(0.0, device=device, requires_grad=True)

                    batch_size = len(batch["questions"])
                    eval_losses["audio_task_loss"] += batch_audio_loss.detach().item() * batch_size
                    eval_losses["total_loss"] += (batch_audio_loss.detach().item() * batch_size)

                    safe_results_batch, base_results_batch = self._compute_robust_accuracy(
                        safe_outputs, None, inputs, has_audio, batch
                    )
                    # Removed verbose progress logging

                    # Audio-dependent accuracy
                    if torch.any(has_audio):
                        audio_indices = torch.where(has_audio)[0]
                        if len(audio_indices) > 0:
                            audio_correct += sum(safe_results_batch[int(i)].score for i in audio_indices)
                            audio_exact += sum(safe_results_batch[int(i)].metrics.get("exact", 0.0) for i in audio_indices)
                            audio_token_f1 += sum(safe_results_batch[int(i)].metrics.get("token_f1", 0.0) for i in audio_indices)
                            audio_bertscore += sum(safe_results_batch[int(i)].metrics.get("bertscore", 0.0) for i in audio_indices)
                            for idx in audio_indices.tolist():
                                metrics = safe_results_batch[int(idx)].metrics if idx < len(safe_results_batch) else {}
                                pred_text = metrics.get("prediction_raw") or metrics.get("prediction")
                                ref_list = metrics.get("references_raw") or metrics.get("references")
                                if pred_text and ref_list:
                                    if isinstance(ref_list, str):
                                        ref_candidates = [ref_list]
                                    else:
                                        ref_candidates = [str(ref).strip() for ref in ref_list if str(ref).strip()]
                                    clean_pred = str(pred_text).strip()
                                    if clean_pred and ref_candidates:
                                        audio_caption_predictions.append(clean_pred)
                                        audio_caption_references.append(ref_candidates)
                                        # Log first sample to validate reference count
                                        if len(audio_caption_references) == 1:
                                            print(f"[RefValidation] First sample has {len(ref_candidates)} reference(s)", flush=True)

                                # Collect CSV samples if enabled (up to max_csv_samples)
                                if csv_samples is not None and len(csv_samples) < max_csv_samples:
                                    # Get prompt/question for this sample
                                    prompt = batch.get("questions", [""])[int(idx)] if int(idx) < len(batch.get("questions", [])) else ""
                                    prediction = str(pred_text).strip() if pred_text else ""
                                    # Get ground truth as string
                                    if isinstance(ref_list, str):
                                        ground_truth = ref_list
                                    elif isinstance(ref_list, list) and len(ref_list) > 0:
                                        ground_truth = ref_list[0]  # Use first reference
                                    else:
                                        ground_truth = str(ref_list) if ref_list else ""

                                    csv_samples.append({
                                        "prompt": prompt,
                                        "model_output": prediction,
                                        "ground_truth": ground_truth,
                                        "accuracy": safe_results_batch[int(idx)].score
                                    })
                            audio_total += len(audio_indices)
                            audio_samples += len(audio_indices)

                    # VL retention accuracy
                    vl_indices_list: List[int] = []
                    if has_audio is None:
                        vl_indices_list = list(range(batch_size))
                    elif isinstance(has_audio, torch.Tensor):
                        if has_audio.numel() > 0:
                            vl_mask = (~has_audio).detach()
                            if vl_mask.is_cuda:
                                vl_mask = vl_mask.to("cpu")
                            vl_indices_list = vl_mask.nonzero(as_tuple=False).view(-1).tolist()
                    else:
                        try:
                            vl_indices_list = [idx for idx, flag in enumerate(has_audio) if not bool(flag)]
                        except Exception:
                            vl_indices_list = list(range(batch_size))

                    if vl_indices_list:
                        vl_safe_correct += sum(safe_results_batch[idx].score for idx in vl_indices_list)
                        vl_base_correct += sum(base_results_batch[idx].score for idx in vl_indices_list)
                        vl_total += len(vl_indices_list)
                        vl_samples += len(vl_indices_list)
                    
                    total_samples += batch_size
    
                    if eval_logging_steps and (
                        batch_idx % eval_logging_steps == 0
                        or (isinstance(total_eval_batches, int) and batch_idx == total_eval_batches)
                    ):
                        print(
                            f"[Eval] Processed {batch_idx}/{total_eval_batches if isinstance(total_eval_batches, int) else '?'} batches",
                            flush=True
                        )
            
        except Exception as e:
            print(f"ERROR during evaluation: {e}", flush=True)
            raise
        finally:
            # Always restore original gate state to prevent regression in training
            if saved_gate is not None:
                try:
                    if hasattr(self.safe_model, 'set_gate'):
                        self.safe_model.set_gate(saved_gate)
                    elif hasattr(self.safe_model, '_default_gate'):
                        self.safe_model._default_gate = saved_gate
                    print(f"🔧 Restored audio gate to: {saved_gate}", flush=True)
                except Exception as exc:
                    print(f"[EvalGate] WARNING: Failed to restore gate state ({exc})", flush=True)
            if previous_mode:
                try:
                    self.safe_model.train()
                    self.safe_model.enable_audio_training()
                except Exception as exc:
                    print(f"[Eval] WARNING: Failed to restore training mode ({exc})", flush=True)
        
        # Normalize losses (avoid division by zero)
        if total_samples > 0:
            for key in eval_losses:
                eval_losses[key] /= total_samples
        else:
            print("Warning: No validation samples processed!", flush=True)
            for key in eval_losses:
                eval_losses[key] = 0.0
        
        # Compute accuracy metrics
        audio_accuracy = audio_correct / max(audio_total, 1)
        audio_exact_avg = audio_exact / max(audio_total, 1)
        audio_token_f1_avg = audio_token_f1 / max(audio_total, 1)
        audio_bertscore_avg = audio_bertscore / max(audio_total, 1)
        vl_safe_accuracy = vl_safe_correct / max(vl_total, 1)
        vl_base_accuracy = vl_base_correct / max(vl_total, 1)
        
        eval_metrics = {
            **eval_losses,
            "audio_accuracy": audio_accuracy,
            "audio_exact_match": audio_exact_avg,
            "audio_token_f1": audio_token_f1_avg,
            "audio_bertscore": audio_bertscore_avg,
            "vl_safe_accuracy": vl_safe_accuracy,
            "vl_base_accuracy": vl_base_accuracy,
            "audio_gain": audio_accuracy,  # Simplified audio gain metric
            "audio_samples": audio_samples,
            "vl_samples": vl_samples,
            "total_samples": total_samples
        }

        caption_metrics = self._compute_standard_caption_metrics(
            audio_caption_predictions,
            audio_caption_references,
        )
        if caption_metrics:
            eval_metrics.update(caption_metrics)

        residual_value = None
        fusion_adapter = getattr(self.safe_model, "fusion_adapter", None)
        if fusion_adapter is not None and hasattr(fusion_adapter, "residual_scale"):
            residual_param = fusion_adapter.residual_scale
            residual_cap = getattr(fusion_adapter, "residual_scale_max", None)
            if residual_cap is not None:
                residual_value = float(torch.clamp(residual_param, 0.0, float(residual_cap)).item())
            else:
                residual_value = float(residual_param.item())
            eval_metrics["residual_scale"] = residual_value
            print(f"[Eval] Residual scale: {residual_value:.4f}", flush=True)

        # Report cumulative evaluation results
        print(f"\n=== {description.upper()} EVALUATION COMPLETE ===", flush=True)
        if split_batches:
            def _summary_limit(sample_limit: Optional[int], batch_limit: Optional[int]) -> str:
                sample_str = str(sample_limit) if sample_limit is not None else "∞"
                batch_str = str(batch_limit) if batch_limit is not None else "∞"
                return f"limit {sample_str} samples / {batch_str} batches"

            print(f"Split Evaluation Summary:", flush=True)
            print(
                f"  Audio: {audio_samples} samples across {collected_audio_batches} batches ("
                f"{_summary_limit(audio_sample_limit, audio_batch_limit)})",
                flush=True,
            )
            print(
                f"  VL: {vl_samples} samples across {collected_vl_batches} batches ("
                f"{_summary_limit(vl_sample_limit, vl_batch_limit)})",
                flush=True,
            )
        else:
            print(f"Standard Evaluation Summary:", flush=True)
            print(f"  Total samples: {total_samples} (Audio: {audio_samples}, VL: {vl_samples})", flush=True)
        
        print(f"\nAccuracy Results:", flush=True)
        if audio_total > 0:
            audio_ci_lower, audio_ci_upper = self._compute_confidence_interval(audio_correct, audio_total)
            print(f"  🎵 Audio Task Accuracy: {audio_accuracy:.3f} ({audio_correct}/{audio_total}) "
                  f"[95% CI: {audio_ci_lower:.1f}%-{audio_ci_upper:.1f}%]", flush=True)
        if vl_total > 0:
            vl_safe_ci_lower, vl_safe_ci_upper = self._compute_confidence_interval(vl_safe_correct, vl_total)
            vl_base_ci_lower, vl_base_ci_upper = self._compute_confidence_interval(vl_base_correct, vl_total)
            print(f"  🔥 SAFE VL Accuracy: {vl_safe_accuracy:.3f} ({vl_safe_correct}/{vl_total}) "
                  f"[95% CI: {vl_safe_ci_lower:.1f}%-{vl_safe_ci_upper:.1f}%]", flush=True)
            print(f"  📊 Base VL Accuracy: {vl_base_accuracy:.3f} ({vl_base_correct}/{vl_total}) "
                  f"[95% CI: {vl_base_ci_lower:.1f}%-{vl_base_ci_upper:.1f}%]", flush=True)

        if caption_metrics:
            print(f"\nCaption Quality Metrics:", flush=True)
            caption_order = [
                "audio_bleu_1",
                "audio_bleu_2",
                "audio_bleu_3",
                "audio_bleu_4",
                "audio_meteor",
                "audio_rouge_l",
                "audio_cider",
                "audio_spice",
                "audio_spider",
            ]
            for key in caption_order:
                if key in caption_metrics:
                    # CIDEr/SPICE/SPIDEr are on 0-100 scale, show 1 decimal; others are 0-1 scale, show 3 decimals
                    if key in ["audio_cider", "audio_spice", "audio_spider"]:
                        print(f"  - {key.replace('audio_', '').upper()}: {caption_metrics[key]:.1f}", flush=True)
                    else:
                        print(f"  - {key.replace('audio_', '').upper()}: {caption_metrics[key]:.3f}", flush=True)
            if "audio_caption_samples" in caption_metrics:
                print(f"  - CAPTION_SAMPLES: {int(caption_metrics['audio_caption_samples'])}", flush=True)

        print(f"==============================\n", flush=True)

        # Save audio samples to CSV if requested
        if csv_samples is not None and len(csv_samples) > 0:
            try:
                # Create directory if it doesn't exist
                csv_path = Path(save_audio_samples_csv)
                csv_path.parent.mkdir(parents=True, exist_ok=True)

                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['prompt', 'model_output', 'ground_truth', 'accuracy']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_samples)

                print(f"✓ Saved {len(csv_samples)} audio samples to {save_audio_samples_csv}", flush=True)
            except Exception as e:
                print(f"⚠️  Failed to save CSV: {e}", flush=True)

        return eval_metrics

    def evaluate_gate_zero_retention(
        self,
        max_batches: Optional[int] = None,
        dataloader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, float]:
        """
        Evaluate SAFE model with gate=0 and compare to original VL baseline.
        This validates the core retention property: SAFE(gate=0) ≈ Original_VL

        Args:
            max_batches: Optional limit on number of batches to evaluate
            dataloader: Optional dataloader (defaults to validation VL-only data)

        Returns:
            Dictionary with retention validation metrics
        """
        previous_mode = self.safe_model.training
        self.safe_model.eval()

        data_source = dataloader if dataloader is not None else self.val_dataloader
        if data_source is None:
            print("Warning: No dataloader available for gate=0 retention validation", flush=True)
            return {"gate_zero_accuracy": 0.0, "original_accuracy": 0.0, "retention_degradation": 0.0}

        # Save current gate state
        saved_gate = None
        if hasattr(self.safe_model, 'get_gate'):
            try:
                saved_gate = self.safe_model.get_gate()
            except:
                saved_gate = 1.0
        elif hasattr(self.safe_model, '_default_gate'):
            saved_gate = self.safe_model._default_gate
        else:
            saved_gate = 1.0

        print("\n" + "="*60, flush=True)
        print("GATE=0 RETENTION VALIDATION", flush=True)
        print("="*60, flush=True)

        gate_zero_correct = 0
        gate_zero_total = 0

        try:
            # Set gate to 0 for SAFE model
            if hasattr(self.safe_model, 'set_gate'):
                self.safe_model.set_gate(0.0)
                print("✓ Set SAFE model gate=0", flush=True)

            with torch.no_grad():
                batch_count = 0
                for batch_idx, batch in enumerate(data_source):
                    if max_batches is not None and batch_idx >= max_batches:
                        break

                    batch_count += 1

                    # Move to device
                    device = next(self.safe_model.parameters()).device
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}

                    # Filter out audio data for VL-only evaluation
                    inputs_vl_only = {k: v for k, v in inputs.items()
                                     if k not in ['audio', 'audio_features', 'has_audio']}

                    # Check if this is actually VL-only data
                    has_audio = inputs.get('has_audio')
                    if has_audio is not None and torch.any(has_audio):
                        continue  # Skip batches with audio for pure VL retention test

                    # Generate predictions with SAFE(gate=0)
                    generation_config = {
                        "max_new_tokens": self.config.get("generation_max_new_tokens", 32),
                        "do_sample": False,
                        "temperature": None,
                        "top_p": None,
                    }

                    try:
                        generated_ids = self.safe_model.generate(
                            **inputs_vl_only,
                            **generation_config
                        )

                        # Decode and extract answers
                        for i in range(generated_ids.size(0)):
                            gen_text = self.safe_model.get_tokenizer().decode(
                                generated_ids[i],
                                skip_special_tokens=True
                            )
                            pred_answer = self._extract_answer(gen_text)

                            # Get ground truth
                            gt_answer = batch.get("answer")
                            if gt_answer is not None:
                                if isinstance(gt_answer, torch.Tensor):
                                    gt_answer = gt_answer[i]
                                elif isinstance(gt_answer, list):
                                    gt_answer = gt_answer[i]

                                # Compute accuracy
                                acc = self._compute_answer_accuracy(pred_answer, gt_answer)
                                gate_zero_correct += acc
                                gate_zero_total += 1

                    except Exception as e:
                        if batch_idx == 0:
                            print(f"Warning: Generation failed in gate=0 eval: {e}", flush=True)
                        continue

                print(f"Evaluated {batch_count} batches, {gate_zero_total} samples", flush=True)

        finally:
            # Restore gate
            if hasattr(self.safe_model, 'set_gate') and saved_gate is not None:
                self.safe_model.set_gate(saved_gate)
                print(f"✓ Restored gate to {saved_gate}", flush=True)

            self.safe_model.train(previous_mode)

        # Compute metrics
        gate_zero_accuracy = (gate_zero_correct / gate_zero_total * 100) if gate_zero_total > 0 else 0.0

        # For comparison, we would need to run original VL baseline
        # For now, track gate=0 accuracy over time
        # Degradation = (baseline_acc - gate_zero_acc)
        # We'll track this metric and expect it to stay near 0

        print(f"\n{'='*60}", flush=True)
        print(f"Gate=0 Accuracy: {gate_zero_accuracy:.2f}%", flush=True)
        print(f"Samples Evaluated: {gate_zero_total}", flush=True)
        print(f"{'='*60}\n", flush=True)

        metrics = {
            "gate_zero_accuracy": gate_zero_accuracy,
            "gate_zero_samples": gate_zero_total,
        }

        # TODO: Add comparison to original VL baseline for true retention metric
        # For now, track gate=0 accuracy and manually compare to baseline

        return metrics

    def _compute_confidence_interval(
        self,
        correct: float,
        total: int,
        confidence: float = 0.95,
        num_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for accuracy using bootstrap.

        Args:
            correct: Number of correct predictions
            total: Total number of samples
            confidence: Confidence level (default 0.95)
            num_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (lower_bound, upper_bound) as percentages
        """
        if total == 0:
            return (0.0, 0.0)

        import numpy as np

        accuracy = correct / total

        # Generate bootstrap samples
        bootstrap_accs = []
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility

        for _ in range(num_bootstrap):
            # Sample with replacement
            sample = rng.binomial(total, accuracy)
            bootstrap_acc = sample / total
            bootstrap_accs.append(bootstrap_acc)

        bootstrap_accs = np.array(bootstrap_accs)

        # Compute percentiles
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(bootstrap_accs, lower_percentile) * 100
        upper_bound = np.percentile(bootstrap_accs, upper_percentile) * 100

        return (lower_bound, upper_bound)

    def _compute_robust_accuracy(self, safe_outputs, base_outputs, inputs, has_audio, batch):
        """
        Compute robust accuracy using proper answer generation and matching.
        
        Args:
            safe_outputs: SAFE model outputs
            base_outputs: Base VL model outputs  
            inputs: Model inputs
            has_audio: Audio sample mask
            batch: Original batch data
            
        Returns:
            Tuple of (safe_accuracies, base_accuracies) for each sample in batch
        """
        batch_size = len(batch.get("questions", inputs.get("input_ids", [])))
        safe_results = [AccuracyResult(0.0, {"exact": 0.0}) for _ in range(batch_size)]
        base_results = [AccuracyResult(0.0, {"exact": 0.0}) for _ in range(batch_size)]
        
        try:
            # Get ground truth answers (preserved from original batch before preprocessing)
            if "original_answers" in inputs:
                gt_answers = inputs["original_answers"]
            elif "answers" in batch:
                # Fallback for backwards compatibility
                gt_answers = batch["answers"]
            elif "labels" in inputs:
                # Decode labels to text (skip padding tokens)
                labels = inputs["labels"]
                gt_answers = []
                for i in range(batch_size):
                    # Remove padding tokens (-100) and decode
                    label_tokens = labels[i][labels[i] != -100]
                    if len(label_tokens) > 0:
                        try:
                            answer = self.safe_model.base_vl.tokenizer.decode(
                                label_tokens, skip_special_tokens=True
                            ).strip()
                            gt_answers.append(answer)
                        except:
                            gt_answers.append("")
                    else:
                        gt_answers.append("")
            else:
                # No ground truth available
                return safe_results, base_results
            
            # Generate predictions using proper generation (without labels)
            # Removed verbose progress logging
            # During training we receive inputs where ground-truth answers have been
            # appended directly to the prompt (teacher forcing). Generating from those
            # tensors leaks the correct answer which leads to empty generations and
            # zero accuracy. Build a clean copy of the prompt that strips answer
            # tokens so that generation truly reflects model predictions.
            gen_inputs = {}
            clone_keys = {"input_ids", "attention_mask"}
            for key, value in inputs.items():
                if key == "labels":
                    continue
                if isinstance(value, torch.Tensor) and key in clone_keys:
                    # Clone tensors we intend to mutate so the training graph remains
                    # untouched for backpropagation.
                    gen_inputs[key] = value.detach().clone()
                else:
                    gen_inputs[key] = value

            labels = inputs.get("labels")
            attention_mask = gen_inputs.get("attention_mask")
            input_ids = gen_inputs.get("input_ids")

            if (
                isinstance(labels, torch.Tensor)
                and isinstance(attention_mask, torch.Tensor)
                and isinstance(input_ids, torch.Tensor)
                # Training-mode batches have -100 for prompt tokens; evaluation batches
                # copy the prompt into labels directly. Only adjust when we detect the
                # -100 sentinel within the attended region.
                and torch.any((labels == -100) & (attention_mask > 0))
            ):
                pad_token_id = self.safe_model.base_vl.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = getattr(self.safe_model.base_vl.tokenizer, "eos_token_id", 0)

                # Strip answer tokens by truncating everything from the first
                # supervised position onwards. The labels tensor marks answer tokens
                # with their actual ids while the prompt remains -100.
                answer_positions = (labels != -100) & (attention_mask > 0)
                for row in range(labels.size(0)):
                    row_positions = torch.nonzero(answer_positions[row], as_tuple=False)
                    if row_positions.numel() == 0:
                        continue
                    first_answer_idx = int(row_positions[0].item())
                    # Zero-out future attention so generation treats these slots as
                    # padding and produce new tokens instead of echoing labels.
                    attention_mask[row, first_answer_idx:] = 0
                    input_ids[row, first_answer_idx:] = pad_token_id
            # Removed verbose progress logging
            batch_audio = batch.get("audio")
            batch_questions = batch.get("questions")
            references = gt_answers if gt_answers else batch.get("answers")

            safe_pred_tokens = self._generate_predictions(
                gen_inputs,
                "safe",
                batch_audio=batch_audio,
                questions=batch_questions,
                references=references,
            )

            # ALWAYS generate BASE predictions independently
            # BASE is the frozen baseline - must never copy SAFE, regardless of retention variant
            # This ensures valid comparison even in no_retention experiments
            base_pred_tokens = self._generate_predictions(
                gen_inputs,
                "base",
                batch_audio=batch_audio,
                questions=batch_questions,
                references=references,
            )

            # Removed verbose progress logging
            
            for i in range(batch_size):
                try:
                    # Check bounds before accessing prediction tokens
                    if i >= len(safe_pred_tokens) or i >= len(base_pred_tokens):
                        safe_results[i] = AccuracyResult(0.0, {"exact": 0.0})
                        base_results[i] = AccuracyResult(0.0, {"exact": 0.0})
                        continue
                    
                    # Decode SAFE model predictions and extract answer
                    safe_pred_full = self.safe_model.base_vl.tokenizer.decode(
                        safe_pred_tokens[i], skip_special_tokens=True
                    ).strip()
                    # Decode base model predictions and extract answer
                    base_pred_full = self.safe_model.base_vl.tokenizer.decode(
                        base_pred_tokens[i], skip_special_tokens=True
                    ).strip()

                    is_audio_sample = False
                    if has_audio is not None:
                        if isinstance(has_audio, torch.Tensor):
                            if i < has_audio.numel():
                                is_audio_sample = bool(has_audio[i].item())
                        else:
                            try:
                                is_audio_sample = bool(has_audio[i])
                            except Exception:
                                is_audio_sample = False

                    audio_only_sample = is_audio_sample and not self._batch_contains_pixels(batch, i)

                    if audio_only_sample:
                        safe_pred_extracted = self._extract_answer(safe_pred_full, mode="audio")
                        base_pred_extracted = self._extract_answer(base_pred_full, mode="audio")

                        gt_raw = gt_answers[i] if i < len(gt_answers) else ""

                        # Preserve display strings prior to normalization for logging
                        gt_display = str(gt_raw).strip() if gt_raw is not None else ""
                        safe_display = safe_pred_extracted
                        base_display = base_pred_extracted

                        # Enable metric debugging for first 2 samples
                        debug_metrics = (i < 2)
                        safe_metrics = self._compute_audio_caption_metrics(
                            safe_pred_extracted, gt_raw, debug=debug_metrics
                        )
                        base_metrics = self._compute_audio_caption_metrics(
                            base_pred_extracted, gt_raw, debug=debug_metrics
                        )

                        safe_results[i] = AccuracyResult(safe_metrics["composite"], safe_metrics)
                        base_results[i] = AccuracyResult(base_metrics["composite"], base_metrics)
                    else:
                        safe_pred = self._clean_answer(self._extract_answer(safe_pred_full))
                        base_pred = self._clean_answer(self._extract_answer(base_pred_full))

                        gt_raw = gt_answers[i] if i < len(gt_answers) else ""
                        gt_answer = self._clean_answer(gt_raw)

                        gt_display = gt_answer
                        safe_display = safe_pred
                        base_display = base_pred

                        # Compute answer-level accuracy (exact match or fuzzy match)
                        safe_score = self._compute_answer_accuracy(safe_pred, gt_answer)
                        base_score = self._compute_answer_accuracy(base_pred, gt_answer)
                        safe_results[i] = AccuracyResult(safe_score, {"exact": safe_score})
                        base_results[i] = AccuracyResult(base_score, {"exact": base_score})

                    # Debug output for first few samples
                    safe_acc_value = safe_results[i].score
                    base_acc_value = base_results[i].score

                    if i < 2:  # Only log first 2 samples to avoid spam
                        # Removed: print(f"[AccuracyDebug] Sample {i}:", flush=True)
                        gt_log = self._shorten_text_for_log(gt_display)
                        safe_raw_log = self._shorten_text_for_log(safe_pred_full)
                        base_raw_log = self._shorten_text_for_log(base_pred_full)

                        print(f"  GT: {gt_log or '∅'}", flush=True)
                        print(
                            f"  SAFE: pred='{safe_display}' | raw='{safe_raw_log}'",
                            flush=True,
                        )
                        print(
                            f"  BASE: pred='{base_display}' | raw='{base_raw_log}'",
                            flush=True,
                        )
                        print(
                            f"  ACC: SAFE={safe_acc_value:.3f}, BASE={base_acc_value:.3f}",
                            flush=True,
                        )

                except Exception as e:
                    # Handle decoding errors gracefully
                    safe_results[i] = AccuracyResult(0.0, {"exact": 0.0})
                    base_results[i] = AccuracyResult(0.0, {"exact": 0.0})
                    
        except Exception as e:
            # Handle any errors gracefully
            if not self._robust_error_logged:
                self.logger.exception("Error in robust accuracy computation")
                self._robust_error_logged = True
            else:
                self.logger.warning("Error in robust accuracy computation: %s", e)
            print(f"Warning: Error in robust accuracy computation: {e}", flush=True)
            pass

        return safe_results, base_results
    
    def _normalize_vqa_answer(self, answer: str) -> str:
        import re

        answer = answer.lower().strip()
        answer = answer.replace("\n", " ")
        answer = answer.replace("-", " ")
        answer = re.sub(r"[^a-z0-9 ]", "", answer)

        contractions = {
            "cant": "cannot",
            "won't": "will not",
            "aint": "is not",
        }
        for key, value in contractions.items():
            answer = answer.replace(key, value)

        # Remove leading stop words/prepositions commonly added by the model
        stop_words = {"a", "an", "the", "on", "in", "at", "of"}
        tokens = [tok for tok in answer.split() if tok]
        while tokens and tokens[0] in stop_words:
            tokens.pop(0)
        answer = " ".join(tokens)

        number_map = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        words = [w for w in answer.split() if w not in {"a", "an", "the"}]
        words = [number_map.get(w, w) for w in words]
        return " ".join(words)

    def _prepare_gt_answers(self, gt_answer: Any) -> List[str]:
        if gt_answer is None:
            return []
        if isinstance(gt_answer, str):
            return [gt_answer]
        if isinstance(gt_answer, dict):
            value = gt_answer.get("answer")
            return [value] if value else []
        if isinstance(gt_answer, (list, tuple)):
            answers = []
            for item in gt_answer:
                if isinstance(item, dict):
                    value = item.get("answer")
                    if value:
                        answers.append(value)
                elif isinstance(item, str):
                    answers.append(item)
            return answers
        return [str(gt_answer)]

    def _build_training_answer_list(self, answers: Any, batch_size: int) -> List[Any]:
        if batch_size <= 0:
            return []
        if answers is None:
            return [""] * batch_size

        if torch.is_tensor(answers):
            if answers.dim() == 0:
                answers_list = [answers.item()]
            else:
                answers_list = answers.detach().cpu().tolist()
        elif isinstance(answers, (list, tuple)):
            answers_list = list(answers)
        else:
            answers_list = [answers]

        if len(answers_list) < batch_size:
            answers_list.extend([""] * (batch_size - len(answers_list)))
        elif len(answers_list) > batch_size:
            answers_list = answers_list[:batch_size]

        return answers_list

    def _expand_audio_caption_training_batch(
        self,
        questions: Sequence[str],
        images: Optional[Sequence[Any]],
        audio: Optional[Sequence[Any]],
        reference_answers: Sequence[Any],
        has_audio_mask: torch.Tensor,
    ) -> Tuple[List[str], Optional[List[Any]], Optional[List[Any]], List[str], torch.Tensor, List[Sequence[str]]]:
        question_list = list(questions)

        if isinstance(images, torch.Tensor):
            if images.dim() == 0:
                image_list: Optional[List[Any]] = [images]
            else:
                image_list = [images[idx] for idx in range(images.size(0))]
        elif isinstance(images, (list, tuple)):
            image_list = list(images)
        else:
            image_list = None

        if isinstance(audio, torch.Tensor):
            if audio.dim() == 0:
                audio_list: Optional[List[Any]] = [audio]
            else:
                audio_list = [audio[idx] for idx in range(audio.size(0))]
        elif isinstance(audio, (list, tuple)):
            audio_list = list(audio)
        else:
            audio_list = None

        if isinstance(has_audio_mask, torch.Tensor):
            has_audio_flags = has_audio_mask.detach().cpu().tolist()
        else:
            has_audio_flags = list(has_audio_mask)

        expanded_questions: List[str] = []
        expanded_answers: List[str] = []
        expanded_references: List[Sequence[str]] = []
        expanded_images: Optional[List[Any]] = [] if image_list is not None else None
        expanded_audio: Optional[List[Any]] = [] if audio_list is not None else None
        expanded_has_audio: List[bool] = []

        total_items = max(len(question_list), len(reference_answers))

        for idx in range(total_items):
            question = question_list[idx] if idx < len(question_list) else ""
            has_audio_flag = bool(has_audio_flags[idx]) if idx < len(has_audio_flags) else False

            refs_raw = self._prepare_gt_answers(reference_answers[idx] if idx < len(reference_answers) else "")
            normalized_refs: List[str] = []
            seen: Set[str] = set()
            for candidate in refs_raw:
                normalized = self._normalize_audio_caption(candidate)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    normalized_refs.append(normalized)

            if not normalized_refs:
                normalized_refs = [""]

            reference_tuple = tuple(normalized_refs)

            def append_sample(answer_text: str, audio_flag: bool) -> None:
                expanded_questions.append(question)
                if expanded_images is not None:
                    image_value = image_list[idx] if image_list is not None and idx < len(image_list) else None
                    expanded_images.append(image_value)
                if expanded_audio is not None:
                    audio_value = audio_list[idx] if audio_list is not None and idx < len(audio_list) else None
                    expanded_audio.append(audio_value)
                expanded_answers.append(answer_text)
                expanded_references.append(reference_tuple)
                expanded_has_audio.append(audio_flag)

            if has_audio_flag:
                for answer_text in normalized_refs:
                    append_sample(answer_text, True)
            else:
                append_sample(normalized_refs[0], False)

        expanded_has_audio_tensor = torch.tensor(expanded_has_audio, dtype=torch.bool)

        return (
            expanded_questions,
            expanded_images,
            expanded_audio,
            expanded_answers,
            expanded_has_audio_tensor,
            expanded_references,
        )

    def _compute_answer_accuracy(self, pred_answer: str, gt_answer: Any) -> float:
        """Compute answer accuracy with proper handling for single vs. multiple ground truths."""

        gt_answers = [a for a in self._prepare_gt_answers(gt_answer) if a]
        if not gt_answers or not pred_answer:
            return 0.0

        pred_norm = self._normalize_vqa_answer(pred_answer)
        if not pred_norm:
            return 0.0

        matches = 0
        for ans in gt_answers:
            if self._normalize_vqa_answer(ans) == pred_norm:
                matches += 1

        # For single ground truth: binary accuracy (0.0 or 1.0)
        # For multiple ground truths: VQA-style consensus (matches/3.0, capped at 1.0)
        if len(gt_answers) == 1:
            return 1.0 if matches > 0 else 0.0
        else:
            return min(1.0, matches / 3.0)

    def _normalize_audio_caption(self, text: Any) -> str:
        if text is None:
            return ""

        if isinstance(text, (list, tuple)):
            text = " ".join(str(t) for t in text if t)
        elif isinstance(text, dict):
            value = text.get("answer") or text.get("text")
            text = value if value is not None else ""

        import re
        import unicodedata

        normalized = unicodedata.normalize("NFKC", str(text))
        normalized = normalized.replace("\u2019", "'")  # Normalise curly apostrophes
        normalized = normalized.lower()

        # Collapse possessives before stripping punctuation so "dog's" -> "dogs"
        normalized = re.sub(r"'s\b", "s", normalized)

        # Remove residual apostrophes and punctuation (keep alphanumerics + whitespace)
        normalized = re.sub(r"'", " ", normalized)
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)

        tokens = [tok for tok in normalized.split() if tok]
        if not tokens:
            return ""

        number_map = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "fifteen": "15",
            "sixteen": "16",
            "seventeen": "17",
            "eighteen": "18",
            "nineteen": "19",
            "twenty": "20",
        }

        # DISABLED: Article removal is not standard for AudioCaps/Clotho benchmarks
        # Keeping articles preserves caption length and improves n-gram overlap
        # articles = {"a", "an", "the"}
        cleaned_tokens: List[str] = []
        for tok in tokens:
            # DISABLED: Don't skip articles - keep them for proper BLEU/CIDEr evaluation
            # if tok in articles:
            #     continue
            cleaned_tokens.append(number_map.get(tok, tok))

        if not cleaned_tokens:
            return ""

        return " ".join(cleaned_tokens)

    def _get_bertscore_metric(self):
        """Lazy load BERTScore metric."""
        # Check if BERTScore is disabled via config
        if self.config.get("disable_bertscore", False):
            if not self._bertscore_failed:
                print("ℹ️  BERTScore disabled via config (using token F1 only)", flush=True)
                self._bertscore_failed = True
            return None

        if self._bertscore_metric is None and not self._bertscore_failed:
            try:
                import logging
                import evaluate

                # Suppress logging errors from evaluate library in SLURM environment
                # The evaluate library tries to log warnings to a closed stderr handle
                old_level = logging.root.level
                logging.root.setLevel(logging.CRITICAL)  # Suppress all logs below CRITICAL

                try:
                    self._bertscore_metric = evaluate.load("bertscore")
                finally:
                    logging.root.setLevel(old_level)  # Restore original level

                print("✅ BERTScore metric loaded successfully", flush=True)
            except Exception as e:
                self._bertscore_failed = True
                print(f"⚠️  BERTScore unavailable (will use token F1 only): {e}", flush=True)
        return self._bertscore_metric

    def _tokenize_caption(self, text: str) -> List[str]:
        """Tokenize caption into words (keeps duplicates for F1)."""
        normalized = self._normalize_audio_caption(text)
        if not normalized:
            return []
        return normalized.split()

    def _compute_token_f1(self, pred_caption: str, gt_caption: str) -> float:
        """Compute token-level F1 score using multisets (preserves duplicates)."""
        from collections import Counter

        pred_tokens = self._tokenize_caption(pred_caption)
        gt_tokens = self._tokenize_caption(gt_caption)

        if not pred_tokens or not gt_tokens:
            return 0.0

        pred_counts = Counter(pred_tokens)
        gt_counts = Counter(gt_tokens)
        overlap = sum(min(pred_counts[token], gt_counts[token]) for token in pred_counts.keys() | gt_counts.keys())

        if overlap == 0:
            return 0.0

        precision = overlap / sum(pred_counts.values())
        recall = overlap / sum(gt_counts.values())
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def _compute_bertscore(self, pred_caption: str, gt_captions: List[str]) -> float:
        """Compute BERTScore between prediction and ground truth captions."""
        bertscore = self._get_bertscore_metric()
        if bertscore is None or not pred_caption or not gt_captions:
            return 0.0

        try:
            valid_refs = [gt for gt in gt_captions if gt]
            if not valid_refs:
                return 0.0

            predictions = [pred_caption] * len(valid_refs)

            # CRITICAL: Flush before the blocking bertscore call
            import sys
            sys.stdout.flush()
            sys.stderr.flush()

            results = bertscore.compute(
                predictions=predictions,
                references=valid_refs,
                lang="en",
                model_type="distilbert-base-uncased",  # Faster than BERT
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=8,  # Add batch size to prevent OOM/hang
                verbose=False,  # Disable internal logging
            )

            # Log completion for first few computations
            import sys
            sys.stdout.flush()

            if not results or "f1" not in results:
                return 0.0
            scores = results["f1"]
            return max(scores) if scores else 0.0
        except Exception as e:
            if not self._bertscore_failed:
                print(f"⚠️  BERTScore computation failed: {e}", flush=True)
                self._bertscore_failed = True
            return 0.0

    def _compute_audio_caption_metrics(self, pred_caption: Any, gt_caption: Any, debug: bool = False) -> Dict[str, Any]:
        """Compute granular audio caption metrics and a composite score."""

        pred_raw = str(pred_caption).strip() if pred_caption is not None else ""
        pred_norm = self._normalize_audio_caption(pred_caption)
        if not pred_norm:
            return {
                "composite": 0.0,
                "exact": 0.0,
                "token_f1": 0.0,
                "bertscore": 0.0,
                "prediction": "",
                "prediction_raw": pred_raw,
                "references": [],
                "references_raw": [],
            }

        gt_candidates = [str(ans).strip() for ans in self._prepare_gt_answers(gt_caption) if str(ans).strip()]
        gt_norms = [self._normalize_audio_caption(ans) for ans in gt_candidates]
        gt_norms = [ans for ans in gt_norms if ans]
        if not gt_norms:
            return {
                "composite": 0.0,
                "exact": 0.0,
                "token_f1": 0.0,
                "bertscore": 0.0,
                "prediction": pred_norm,
                "prediction_raw": pred_raw,
                "references": [],
                "references_raw": gt_candidates,
            }

        exact_match = 1.0 if pred_norm in gt_norms else 0.0
        token_f1 = max((self._compute_token_f1(pred_norm, gt) for gt in gt_norms), default=0.0)
        bertscore_raw = self._compute_bertscore(pred_norm, gt_norms)
        bert_threshold = float(self.config.get("audio_bertscore_threshold", 0.7))
        bertscore = bertscore_raw if bertscore_raw >= bert_threshold else 0.0

        if exact_match == 1.0:
            composite = 1.0
        else:
            composite = 0.0
            composite += 0.6 * token_f1
            composite += 0.4 * bertscore
            composite = min(composite, 1.0)

        if debug:
            print(
                "  [Metrics] Exact={:.3f}, TokenF1={:.3f}, BERT={:.3f} (thr={:.2f}) → Composite={:.3f}".format(
                    exact_match, token_f1, bertscore_raw, bert_threshold, composite
                ),
                flush=True,
            )

        return {
            "composite": composite,
            "exact": exact_match,
            "token_f1": token_f1,
            "bertscore": bertscore_raw,
            "prediction": pred_norm,
            "prediction_raw": pred_raw,
            "references": gt_norms,
            "references_raw": gt_candidates,
        }

    def _log_caption_metric_warning(self, metric_name: str, error: Exception) -> None:
        """Emit a single warning per caption metric failure."""

        if not hasattr(self, "_caption_metric_warnings"):
            self._caption_metric_warnings = set()
        if metric_name not in self._caption_metric_warnings:
            print(f"⚠️  {metric_name} metric unavailable: {error}", flush=True)
            self._caption_metric_warnings.add(metric_name)

    def _compute_standard_caption_metrics(
        self,
        predictions: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """Compute standard captioning metrics for SAFE audio predictions."""

        if not predictions or not references:
            return {}

        paired_data: List[Tuple[str, List[str]]] = []
        for pred, refs in zip(predictions, references):
            pred_clean = self._normalize_audio_caption(str(pred).strip())
            refs_clean = [
                self._normalize_audio_caption(str(ref).strip())
                for ref in refs
                if str(ref).strip()
            ]
            pred_clean = pred_clean.strip()
            refs_clean = [ref for ref in refs_clean if ref.strip()]
            if pred_clean and refs_clean:
                paired_data.append((pred_clean, refs_clean))

        if not paired_data:
            return {}

        preds, refs = zip(*paired_data)
        preds_list = list(preds)
        refs_list = [list(r) for r in refs]

        # Validate reference count (AudioCaps should have ~5 references per sample)
        ref_counts = [len(r) for r in refs_list]
        if ref_counts:
            avg_refs = sum(ref_counts) / len(ref_counts)
            min_refs = min(ref_counts)
            max_refs = max(ref_counts)
            print(f"[RefValidation] References per sample: avg={avg_refs:.1f}, min={min_refs}, max={max_refs}, total_samples={len(refs_list)}", flush=True)
            if avg_refs < 2.0:
                print(f"⚠️  WARNING: Low reference count (avg={avg_refs:.1f}). AudioCaps should have ~5 references per sample for proper CIDEr computation!", flush=True)

        try:
            import evaluate
        except Exception as exc:
            self._log_caption_metric_warning("evaluate", exc)
            return {}

        metrics: Dict[str, float] = {}

        metric_cache = getattr(self, "_caption_metric_cache", None)
        if metric_cache is None:
            metric_cache = {}
            self._caption_metric_cache = metric_cache

        def _metric(name: str, **load_kwargs):
            """Load caption metric once and reuse across evaluations."""

            cache_key = (name, tuple(sorted(load_kwargs.items())))
            if cache_key not in metric_cache:
                metric_cache[cache_key] = evaluate.load(name, **load_kwargs)
            return metric_cache[cache_key]

        try:
            bleu_metric = _metric("bleu")
            bleu_result = bleu_metric.compute(predictions=preds_list, references=refs_list)
            if bleu_result:
                precisions = bleu_result.get("precisions", [])
                for n in range(min(4, len(precisions))):
                    metrics[f"audio_bleu_{n + 1}"] = float(precisions[n])
                if "bleu" in bleu_result:
                    metrics["audio_bleu"] = float(bleu_result["bleu"])
        except Exception as exc:
            self._log_caption_metric_warning("BLEU", exc)

        try:
            meteor_metric = _metric("meteor")
            meteor_result = meteor_metric.compute(predictions=preds_list, references=refs_list)
            if meteor_result and "meteor" in meteor_result:
                metrics["audio_meteor"] = float(meteor_result["meteor"])
        except Exception as exc:
            self._log_caption_metric_warning("METEOR", exc)

        try:
            rouge_metric = _metric("rouge")
            rouge_scores = []
            for pred, refs in zip(preds_list, refs_list):
                best_score = 0.0
                for ref in refs:
                    try:
                        rouge_result = rouge_metric.compute(predictions=[pred], references=[ref])
                        best_score = max(best_score, float(rouge_result.get("rougeL", 0.0)))
                    except Exception as rouge_exc:
                        self._log_caption_metric_warning("ROUGE-L", rouge_exc)
                        best_score = max(best_score, 0.0)
                rouge_scores.append(best_score)
            if rouge_scores:
                metrics["audio_rouge_l"] = float(sum(rouge_scores) / len(rouge_scores))
        except Exception as exc:
            self._log_caption_metric_warning("ROUGE-L", exc)

        # Use pycocoevalcap directly for CIDEr/SPICE (not available in evaluate)
        try:
            from pycocoevalcap.cider.cider import Cider
            from pycocoevalcap.spice.spice import Spice

            # Convert to pycocoevalcap format: {id: [refs]} and {id: [pred]}
            print(f"[MetricCompute] Converting {len(refs_list)} samples to pycocoevalcap format...", flush=True)
            gts = {str(i): refs for i, refs in enumerate(refs_list)}
            res = {str(i): [pred] for i, pred in enumerate(preds_list)}
            print(f"[MetricCompute] Format conversion complete. Starting CIDEr computation...", flush=True)

            # Compute CIDEr
            try:
                cider_scorer = Cider()
                print(f"[MetricCompute] CIDEr scorer initialized, computing score...", flush=True)
                cider_score, _ = cider_scorer.compute_score(gts, res)
                # Scale to 0-100 range (standard reporting format)
                metrics["audio_cider"] = float(cider_score) * 100.0
                print(f"[MetricCompute] CIDEr computed: {metrics['audio_cider']:.2f}", flush=True)
            except Exception as cider_exc:
                print(f"[MetricCompute] CIDEr computation FAILED: {cider_exc}", flush=True)
                self._log_caption_metric_warning("CIDEr", cider_exc)

            # Compute SPICE
            try:
                print(f"[MetricCompute] Starting SPICE computation (this spawns Java process)...", flush=True)
                # Free GPU memory before spawning Java process
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                spice_scorer = Spice()
                print(f"[MetricCompute] SPICE scorer initialized, computing score...", flush=True)
                spice_score, _ = spice_scorer.compute_score(gts, res)
                # Scale to 0-100 range (standard reporting format)
                metrics["audio_spice"] = float(spice_score) * 100.0
                print(f"[MetricCompute] SPICE computed: {metrics['audio_spice']:.2f}", flush=True)
            except Exception as spice_exc:
                print(f"[MetricCompute] SPICE computation FAILED: {spice_exc}", flush=True)
                self._log_caption_metric_warning("SPICE", spice_exc)

            # Compute SPIDEr (average of CIDEr and SPICE)
            if "audio_cider" in metrics and "audio_spice" in metrics:
                metrics["audio_spider"] = (metrics["audio_cider"] + metrics["audio_spice"]) / 2.0

        except ImportError as import_exc:
            self._log_caption_metric_warning("pycocoevalcap (CIDEr/SPICE/SPIDEr)", import_exc)

        metrics["audio_caption_samples"] = float(len(preds_list))

        return metrics

    def _resolve_answer_text(self, answer: Any) -> str:
        if hasattr(self.safe_model, "_select_training_answer"):
            resolved = self.safe_model._select_training_answer(answer)
            if resolved:
                return resolved

        if answer is None:
            return ""
        if isinstance(answer, str):
            return answer
        if isinstance(answer, dict):
            return str(answer.get("answer") or answer.get("text") or "")
        if isinstance(answer, (list, tuple)):
            for item in answer:
                text = self._resolve_answer_text(item)
                if text:
                    return text
            return ""
        if torch.is_tensor(answer):
            if answer.dim() == 0:
                return str(answer.item())
            tokens = answer.detach().cpu().tolist()
            tokenizer = self.safe_model.base_vl.tokenizer
            return tokenizer.decode(tokens, skip_special_tokens=True)
        return str(answer)

    def _gather_answer_texts(self, answers: Any, indices: torch.Tensor) -> List[str]:
        if answers is None or indices.numel() == 0:
            return []

        texts: List[str] = []
        if isinstance(answers, (list, tuple)):
            answers_list = list(answers)
        elif torch.is_tensor(answers):
            answers_list = answers.detach().cpu().tolist()
        else:
            answers_list = [answers]

        for idx in indices.tolist():
            candidate = None
            if isinstance(answers_list, list) and idx < len(answers_list):
                candidate = answers_list[idx]
            elif torch.is_tensor(answers_list):
                if answers_list.dim() == 1:
                    candidate = answers_list[idx].item()
                else:
                    candidate = answers_list[idx]
            elif isinstance(answers_list, tuple) and idx < len(answers_list):
                candidate = answers_list[idx]
            else:
                candidate = answers_list

            text = self._resolve_answer_text(candidate)
            normalized = self._normalize_audio_caption(text)
            if normalized:
                texts.append(normalized)

        return texts

    def _compute_audio_contrastive_loss(
        self,
        audio_tokens: torch.Tensor,
        answers: Any,
        audio_indices: torch.Tensor,
        reference_answers: Optional[Sequence[Any]] = None,
    ) -> Optional[torch.Tensor]:
        if audio_tokens is None or audio_indices.numel() == 0:
            return None

        if audio_tokens.dim() == 2:
            token_matrix = audio_tokens.unsqueeze(1)
        else:
            token_matrix = audio_tokens

        if token_matrix.size(0) <= audio_indices.max().item():
            return None

        selected = token_matrix.index_select(0, audio_indices.to(audio_tokens.device))
        audio_vecs = selected.mean(dim=1) if selected.dim() == 3 else selected

        texts = self._gather_answer_texts(answers, audio_indices)
        if len(texts) != audio_vecs.size(0):
            paired = [
                (vec, text)
                for vec, text in zip(audio_vecs, texts)
                if text.strip()
            ]
            if not paired:
                return None
            audio_vecs = torch.stack([p[0] for p in paired], dim=0)
            texts = [p[1] for p in paired]

        if not texts:
            return None

        device = audio_vecs.device
        audio_vecs = F.normalize(audio_vecs, dim=-1)

        text_vecs = self._embed_texts(texts, device)
        if text_vecs.numel() == 0:
            return None
        text_vecs = F.normalize(text_vecs, dim=-1)

        reference_sets: List[List[str]] = []
        if reference_answers is not None:
            for idx in audio_indices.tolist():
                raw_answer = reference_answers[idx] if idx < len(reference_answers) else None
                ref_texts = [
                    self._normalize_audio_caption(ans)
                    for ans in self._prepare_gt_answers(raw_answer)
                    if ans
                ]
                reference_sets.append([ref for ref in ref_texts if ref])
        else:
            reference_sets = [[] for _ in texts]

        temperature = max(self.audio_contrastive_temperature, 1e-5)

        extra_texts: List[str] = []
        threshold = self.audio_contrastive_negative_threshold
        max_neg = self.audio_contrastive_max_negatives
        if threshold > 0.0 and max_neg > 0 and reference_sets:
            for i, ref_list in enumerate(reference_sets):
                for j, other_refs in enumerate(reference_sets):
                    if i == j:
                        continue
                    for candidate in other_refs:
                        if candidate in texts:
                            continue
                        score = self._compute_ngram_overlap_score(candidate, ref_list)
                        if score >= threshold:
                            extra_texts.append(candidate)
            dedup: List[str] = []
            seen = set()
            for text in extra_texts:
                if text not in seen:
                    dedup.append(text)
                    seen.add(text)
                if len(dedup) >= max_neg:
                    break
            extra_texts = dedup
        else:
            extra_texts = []

        if extra_texts:
            neg_vecs = self._embed_texts(extra_texts, device)
            neg_vecs = F.normalize(neg_vecs, dim=-1) if neg_vecs.numel() > 0 else torch.empty(0, text_vecs.size(1), device=device)
            text_vecs_extended = torch.cat([text_vecs, neg_vecs], dim=0)
        else:
            text_vecs_extended = text_vecs

        logits = torch.matmul(audio_vecs, text_vecs_extended.transpose(0, 1)) / temperature
        targets = torch.arange(audio_vecs.size(0), device=device)

        weights = torch.ones(audio_vecs.size(0), device=device)
        if self.audio_contrastive_metric_weight != 0.0 and reference_sets:
            for idx, refs in enumerate(reference_sets):
                if not refs:
                    continue
                score = self._compute_ngram_overlap_score(texts[idx], refs)
                weights[idx] += self.audio_contrastive_metric_weight * score
        if weights.numel() > 0:
            weights = weights / weights.mean().clamp_min(1e-6)

        loss_a2t = F.cross_entropy(logits, targets, reduction='none')
        loss_a2t = (loss_a2t * weights).mean()

        logits_t2a = torch.matmul(text_vecs, audio_vecs.transpose(0, 1)) / temperature
        loss_t2a = F.cross_entropy(logits_t2a, targets, reduction='none')
        loss_t2a = (loss_t2a * weights).mean()

        return 0.5 * (loss_a2t + loss_t2a)

    # ------------------------------------------------------------------
    # Reranking helpers
    # ------------------------------------------------------------------

    def _rerank_audio_candidates(
        self,
        candidate_seqs: torch.Tensor,
        prompt_lengths: torch.Tensor,
        has_audio_mask: torch.Tensor,
        batch_audio: Optional[Sequence[Any]],
        questions: Sequence[str],
        references: Sequence[Any],
        device: torch.device,
    ) -> torch.Tensor:
        batch_size, num_candidates, seq_len = candidate_seqs.shape

        processed_questions = list(questions)
        if len(processed_questions) < batch_size:
            processed_questions.extend([""] * (batch_size - len(processed_questions)))
        processed_questions = processed_questions[:batch_size]

        processed_audio = list(batch_audio) if batch_audio is not None else [None] * batch_size
        if len(processed_audio) < batch_size:
            processed_audio.extend([None] * (batch_size - len(processed_audio)))
        processed_audio = processed_audio[:batch_size]

        processed_refs: List[List[str]] = []
        for idx in range(batch_size):
            if idx < len(references):
                ref_set = [
                    self._normalize_audio_caption(ans)
                    for ans in self._prepare_gt_answers(references[idx])
                    if ans
                ]
            else:
                ref_set = []
            processed_refs.append([ref for ref in ref_set if ref])

        clap_audio_embeds = None
        clap_weight = float(self.config.get("audio_rerank_clap_weight", 0.0))
        if clap_weight > 0.0:
            clap_audio_embeds = self._compute_clap_audio_embeddings(processed_audio)

        logprob_weight = float(self.config.get("audio_rerank_logprob_weight", 1.0))
        ngram_weight = float(self.config.get("audio_rerank_ngram_weight", 0.0))
        coverage_weight = float(self.config.get("audio_rerank_coverage_weight", 0.0))

        tokenizer = self.safe_model.base_vl.tokenizer
        text_embed_cache: Dict[str, torch.Tensor] = {}

        best_sequences = []
        for sample_idx in range(batch_size):
            candidates = candidate_seqs[sample_idx]
            if not has_audio_mask[sample_idx] or processed_refs[sample_idx] == []:
                best_sequences.append(candidates[0])
                continue

            question = processed_questions[sample_idx] if sample_idx < len(processed_questions) else ""
            audio_payload = processed_audio[sample_idx] if sample_idx < len(processed_audio) else None
            reference_texts = processed_refs[sample_idx]
            prompt_len = int(prompt_lengths[sample_idx].item())
            audio_embed = None
            if clap_audio_embeds is not None and sample_idx < len(clap_audio_embeds):
                audio_embed = clap_audio_embeds[sample_idx]
            predicted_tags: List[str] = []
            if (
                self.audio_rerank_tag_weight > 0.0
                and self.audio_rerank_num_tags > 0
                and audio_embed is not None
            ):
                predicted_tags = self._predict_audio_tags(audio_embed)
            best_score = float("-inf")
            best_idx = 0

            for cand_idx in range(num_candidates):
                candidate_tokens = candidates[cand_idx]
                if prompt_len < candidate_tokens.size(0):
                    caption_tokens = candidate_tokens[prompt_len:]
                else:
                    caption_tokens = candidate_tokens.new_empty(0)
                caption_text = tokenizer.decode(caption_tokens, skip_special_tokens=True).strip()
                caption_text = self._normalize_audio_caption(caption_text)
                if not caption_text:
                    continue

                total_score = 0.0

                if logprob_weight != 0.0:
                    logprob = self._score_candidate_logprob(
                        question,
                        audio_payload,
                        caption_text,
                        device,
                    )
                    total_score += logprob_weight * logprob

                if clap_weight > 0.0 and audio_embed is not None:
                    if caption_text not in text_embed_cache:
                        text_embed_cache[caption_text] = self._compute_clap_text_embedding(caption_text)
                    text_embed = text_embed_cache.get(caption_text)
                    if text_embed is not None and text_embed.numel() > 0:
                        clap_sim = torch.nn.functional.cosine_similarity(
                            audio_embed.to(text_embed.device),
                            text_embed,
                            dim=0,
                        ).item()
                        total_score += clap_weight * clap_sim

                if ngram_weight > 0.0:
                    ngram_score = self._compute_ngram_overlap_score(caption_text, reference_texts)
                    total_score += ngram_weight * ngram_score

                if coverage_weight > 0.0:
                    coverage = self._compute_coverage_score(caption_text, reference_texts)
                    total_score += coverage_weight * coverage

                cider_score = 0.0
                spice_score = 0.0
                if (self.audio_rerank_cider_weight > 0.0 or self.audio_rerank_spider_weight > 0.0) and reference_texts:
                    cider_score = self._compute_cider_score(caption_text, reference_texts)
                    if self.audio_rerank_cider_weight > 0.0:
                        total_score += self.audio_rerank_cider_weight * cider_score
                if self.audio_rerank_spider_weight > 0.0 and reference_texts:
                    if spice_score == 0.0:
                        spice_score = self._compute_spice_score(caption_text, reference_texts)
                    spider_score = 0.5 * (cider_score + spice_score)
                    total_score += self.audio_rerank_spider_weight * spider_score

                if self.audio_rerank_tag_weight > 0.0 and predicted_tags:
                    tag_bonus = self._score_tag_bonus(caption_text, predicted_tags)
                    total_score += self.audio_rerank_tag_weight * tag_bonus

                if total_score > best_score:
                    best_score = total_score
                    best_idx = cand_idx

            best_sequences.append(candidates[best_idx])

        return torch.stack(best_sequences, dim=0)

    def _score_candidate_logprob(
        self,
        question: str,
        audio_data: Any,
        candidate_text: str,
        device: torch.device,
    ) -> float:
        candidate_text = candidate_text.strip()
        if not candidate_text:
            return float("-inf")

        with torch.no_grad():
            inputs = self.safe_model.prepare_multimodal_inputs(
                text=[question or ""],
                images=None,
                audio=[audio_data] if audio_data is not None else None,
                answers=[candidate_text],
                device=device,
                training_mode=True,
            )

            outputs = self.safe_model(**inputs)
            logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
            labels = inputs.get("labels")
            if logits is None or labels is None:
                return float("-inf")

            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            seq_len = min(shift_logits.size(-2), shift_labels.size(-1))
            shift_logits = shift_logits[..., :seq_len, :]
            shift_labels = shift_labels[..., :seq_len]
            mask = shift_labels != -100
            if not mask.any():
                return float("-inf")

            log_probs = F.log_softmax(shift_logits, dim=-1)
            gathered = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            token_log_probs = gathered.masked_select(mask)
            if token_log_probs.numel() == 0:
                return float("-inf")

            avg_log_prob = token_log_probs.mean()
            alpha = float(self.config.get("audio_length_penalty", 1.0))
            length = max(int(mask.sum().item()), 1)
            if alpha != 1.0:
                length_norm = ((5 + length) ** alpha) / ((5 + 1) ** alpha)
                score = avg_log_prob / length_norm
            else:
                score = avg_log_prob

            return float(score.detach().cpu().item())

    def _tokenize_caption(self, text: str) -> List[str]:
        normalized = self._normalize_audio_caption(text)
        if not normalized:
            return []
        return normalized.split()

    def _ngram_f1(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        if len(pred_tokens) < n or len(ref_tokens) < n:
            return 0.0
        from collections import Counter

        pred_ngrams = Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
        ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
        overlap = sum(min(pred_ngrams[gram], ref_ngrams[gram]) for gram in pred_ngrams)
        if overlap == 0:
            return 0.0
        precision = overlap / max(sum(pred_ngrams.values()), 1)
        recall = overlap / max(sum(ref_ngrams.values()), 1)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _compute_ngram_overlap_score(self, prediction: str, references: Sequence[str]) -> float:
        pred_tokens = self._tokenize_caption(prediction)
        if not pred_tokens or not references:
            return 0.0
        best = 0.0
        for ref in references:
            ref_tokens = self._tokenize_caption(ref)
            if not ref_tokens:
                continue
            uni = self._ngram_f1(pred_tokens, ref_tokens, 1)
            bi = self._ngram_f1(pred_tokens, ref_tokens, 2)
            best = max(best, 0.5 * (uni + bi))
        return best

    def _compute_coverage_score(self, prediction: str, references: Sequence[str]) -> float:
        if not references:
            return 0.0
        pred_tokens = set(self._tokenize_caption(prediction))
        ref_tokens = set()
        for ref in references:
            ref_tokens.update(self._tokenize_caption(ref))
        if not ref_tokens:
            return 0.0
        overlap = len(pred_tokens & ref_tokens)
        return overlap / len(ref_tokens)

    def _resolve_clap_encoder(self):
        registry = getattr(self.safe_model, "modality_registry", None)
        if registry is None:
            return None
        try:
            components = registry.get("audio")
        except Exception:
            return None
        encoder = getattr(components, "encoder", None)
        if encoder is None:
            return None
        if hasattr(encoder, "clap_encoder") and encoder.clap_encoder is not None:
            return encoder.clap_encoder
        return encoder if hasattr(encoder, "model") and hasattr(encoder.model, "get_audio_embedding_from_data") else None

    def _compute_clap_audio_embeddings(self, audio_batch: Sequence[Any]) -> Optional[List[Optional[torch.Tensor]]]:
        clap_encoder = self._resolve_clap_encoder()
        if clap_encoder is None or not audio_batch:
            return None
        valid_audio = []
        index_map = []
        for idx, audio in enumerate(audio_batch):
            if audio is None:
                continue
            valid_audio.append(audio)
            index_map.append(idx)
        if not valid_audio:
            return None
        embeddings = clap_encoder(valid_audio)
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)
        embeddings = F.normalize(embeddings.float(), dim=-1)
        result: List[Optional[torch.Tensor]] = [None] * len(audio_batch)
        for pos, idx in enumerate(index_map):
            result[idx] = embeddings[pos]
        return result

    def _compute_clap_text_embedding(self, caption: str) -> Optional[torch.Tensor]:
        embeddings = self._compute_clap_text_embeddings([caption])
        if embeddings is None or embeddings.numel() == 0:
            return None
        return embeddings[0]

    def _compute_clap_text_embeddings(self, captions: Sequence[str]) -> Optional[torch.Tensor]:
        clap_encoder = self._resolve_clap_encoder()
        if clap_encoder is None:
            return None
        texts = [c for c in captions if c]
        if not texts:
            return None
        embeddings = clap_encoder.encode_text(texts)
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.from_numpy(embeddings)
        return F.normalize(embeddings.float(), dim=-1)

    def _load_tag_vocab(self) -> Optional[List[str]]:
        if self._tag_vocab_terms is not None:
            return self._tag_vocab_terms
        vocab_path = self.config.get("audio_rerank_tag_vocab")
        if not vocab_path:
            self._tag_vocab_terms = None
            return None
        path = Path(vocab_path).expanduser()
        if not path.exists():
            print(f"⚠️  Tag vocabulary file not found: {path}", flush=True)
            self._tag_vocab_terms = None
            return None
        with path.open("r", encoding="utf-8") as handle:
            terms = [line.strip() for line in handle if line.strip()]
        self._tag_vocab_terms = terms if terms else None
        if self._tag_vocab_terms:
            embeddings = self._compute_clap_text_embeddings(self._tag_vocab_terms)
            if embeddings is None:
                self._tag_vocab_terms = None
                self._tag_vocab_embeddings = None
            else:
                self._tag_vocab_embeddings = embeddings
        else:
            self._tag_vocab_embeddings = None
        return self._tag_vocab_terms

    def _predict_audio_tags(self, audio_embedding: torch.Tensor) -> List[str]:
        vocab = self._load_tag_vocab()
        if not vocab or self._tag_vocab_embeddings is None:
            return []
        if audio_embedding is None or audio_embedding.numel() == 0:
            return []
        tag_embeddings = self._tag_vocab_embeddings.to(audio_embedding.device)
        sims = torch.matmul(tag_embeddings, audio_embedding)
        topk = min(self.audio_rerank_num_tags, sims.size(0))
        if topk <= 0:
            return []
        values, indices = torch.topk(sims, topk)
        tags: List[str] = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            if score > 0:
                tags.append(vocab[idx])
        return tags

    def _score_tag_bonus(self, caption_text: str, tags: Sequence[str]) -> float:
        if not tags:
            return 0.0
        tokens = set(self._tokenize_caption(caption_text))
        if not tokens:
            return 0.0
        matches = 0
        for tag in tags:
            tag_tokens = self._tokenize_caption(tag)
            if tag_tokens and all(token in tokens for token in tag_tokens):
                matches += 1
        return matches / len(tags)

    def _ensure_cider_scorer(self):
        if self._cider_scorer is None:
            try:
                from pycocoevalcap.cider.cider import Cider

                self._cider_scorer = Cider()
            except Exception as exc:
                self._log_caption_metric_warning("CIDEr", exc)
                self._cider_scorer = False
        return self._cider_scorer if self._cider_scorer is not False else None

    def _ensure_spice_scorer(self):
        if self._spice_scorer is None:
            try:
                from pycocoevalcap.spice.spice import Spice

                self._spice_scorer = Spice()
            except Exception as exc:
                self._log_caption_metric_warning("SPICE", exc)
                self._spice_scorer = False
        return self._spice_scorer if self._spice_scorer is not False else None

    def _compute_cider_score(self, caption: str, references: Sequence[str]) -> float:
        if not caption or not references:
            return 0.0
        scorer = self._ensure_cider_scorer()
        if scorer is None:
            return 0.0
        try:
            score, _ = scorer.compute_score({"0": list(references)}, {"0": [caption]})
            if isinstance(score, (list, tuple)):
                score = score[0]
            return float(score)
        except Exception as exc:
            self._log_caption_metric_warning("CIDEr", exc)
            return 0.0

    def _compute_spice_score(self, caption: str, references: Sequence[str]) -> float:
        if not caption or not references:
            return 0.0
        scorer = self._ensure_spice_scorer()
        if scorer is None:
            return 0.0
        try:
            score, details = scorer.compute_score({"0": list(references)}, {"0": [caption]})
            if isinstance(score, (list, tuple)):
                score = score[0]
            return float(score)
        except Exception as exc:
            self._log_caption_metric_warning("SPICE", exc)
            return 0.0

    def _embed_texts(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        tokenizer = self.safe_model.base_vl.tokenizer
        embedding_layer = self.safe_model.base_vl.llm.get_input_embeddings()
        hidden_size = embedding_layer.weight.size(1)
        if not texts:
            return torch.empty(0, hidden_size, device=device)
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.audio_contrastive_answer_max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            text_embeds = embedding_layer(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (text_embeds * mask).sum(dim=1).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        return pooled / denom

    def _batch_contains_pixels(self, batch: Dict[str, Any], idx: int) -> bool:
        images = batch.get("images")
        if isinstance(images, (list, tuple)) and idx < len(images):
            if images[idx] is not None:
                return True

        pixel_values = batch.get("pixel_values")
        if isinstance(pixel_values, torch.Tensor) and pixel_values.numel() > 0:
            if pixel_values.ndim == 4:
                if idx < pixel_values.size(0):
                    return True
            else:
                return True

        return False

    def _generate_predictions(
        self,
        inputs,
        model_choice="safe",
        batch_audio: Optional[Sequence[Any]] = None,
        questions: Optional[Sequence[str]] = None,
        references: Optional[Sequence[Any]] = None,
    ):
        """Generate predictions using consistent SAFE/base pathways."""
        tok = self.safe_model.base_vl.tokenizer
        
        # Verify tokenizer is properly configured (should be done in BaseVL init)
        if getattr(tok, 'padding_side', None) != 'left':
            print(f"[GenWarning] Tokenizer padding_side is '{getattr(tok, 'padding_side', 'NOT_SET')}', should be 'left'", flush=True)
            # Fix it as fallback, but this shouldn't be needed
            tok.padding_side = 'left'
            
        if getattr(tok, "pad_token_id", None) is None:
            print(f"[GenWarning] Tokenizer pad_token_id is None, fixing...", flush=True)
            if getattr(tok, "eos_token_id", None) is not None:
                tok.pad_token = tok.eos_token

        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")
        audio_tokens = inputs.get("audio_tokens")

        if input_ids is None:
            batch_size = pixel_values.size(0) if isinstance(pixel_values, torch.Tensor) else 0
            return [torch.tensor([], dtype=torch.long) for _ in range(batch_size)]

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

        device = next(self.safe_model.parameters()).device

        # Removed verbose AUDIO_EVAL logging

        # Ensure model config has proper pad/eos tokens and respects max_new_tokens
        if hasattr(self.safe_model.base_vl.llm, 'config'):
            self.safe_model.base_vl.llm.config.pad_token_id = tok.pad_token_id
            self.safe_model.base_vl.llm.config.eos_token_id = getattr(tok, "eos_token_id", None)
        if hasattr(self.safe_model.base_vl.llm, 'generation_config'):
            self.safe_model.base_vl.llm.generation_config.pad_token_id = tok.pad_token_id
            self.safe_model.base_vl.llm.generation_config.eos_token_id = getattr(tok, "eos_token_id", None)
            # Override max_length to respect max_new_tokens limit
            self.safe_model.base_vl.llm.generation_config.max_length = None

        configured_max_new_tokens = int(self.config.get("generation_max_new_tokens", 32) or 32)
        configured_max_new_tokens = max(1, configured_max_new_tokens)

        # Determine if this is audio task (needs longer captions) or VL task (needs short answers)
        has_audio = audio_tokens is not None
        pixel_present = pixel_values is not None and (
            isinstance(pixel_values, torch.Tensor) or isinstance(pixel_values, (list, tuple))
        )

        # AudioCaps captions are longer (e.g., "A dog barks while birds chirp in the background")
        # VQA answers are short (e.g., "blue", "2", "yes")
        if has_audio and not pixel_present:
            # Pure audio task (AudioCaps) - allow longer generation budget
            audio_max_new_tokens = int(
                self.config.get("audio_generation_max_new_tokens", configured_max_new_tokens) or configured_max_new_tokens
            )
            audio_max_new_tokens = max(1, audio_max_new_tokens)
            max_new_tokens = audio_max_new_tokens
        elif pixel_present and not has_audio:
            # Pure VL task (VQA) - restrict to short answers
            vl_max = int(self.config.get("vl_max_new_tokens", 4) or 4)
            max_new_tokens = max(1, min(configured_max_new_tokens, vl_max))
        else:
            # Audio-visual or ambiguous - use configured value
            max_new_tokens = configured_max_new_tokens

        # Modality-specific repetition penalty
        vl_rep = float(self.config.get("vl_repetition_penalty", 1.0))
        audio_rep = float(self.config.get("audio_repetition_penalty", 1.1))
        if has_audio and not pixel_present:
            repetition_penalty = audio_rep
        elif pixel_present and not has_audio:
            repetition_penalty = vl_rep
        else:
            repetition_penalty = audio_rep

        base_gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,  # CRITICAL FIX: Force at least 1 token to prevent empty generation
            do_sample=False,    # Greedy decoding for deterministic results
            temperature=None,
            top_p=None,
            pad_token_id=tok.pad_token_id,
            eos_token_id=getattr(tok, "eos_token_id", None),
            repetition_penalty=repetition_penalty,
            # Removed no_repeat_ngram_size - was blocking fluent generation
            # Removed encoder_repetition_penalty - was making it worse
            output_scores=False,
            return_dict_in_generate=False
        )

        is_audio_only = has_audio and not pixel_present
        audio_num_beams = max(1, int(self.config.get("audio_num_beams", 1) or 1))
        audio_num_return = self.config.get("audio_num_return_sequences")
        audio_num_return = (
            int(audio_num_return)
            if audio_num_return is not None
            else audio_num_beams
        )

        beam_kwargs = dict(base_gen_kwargs)
        sample_kwargs: Optional[Dict[str, Any]] = None

        if is_audio_only:
            beam_kwargs["num_beams"] = audio_num_beams
            if audio_num_return > 1:
                beam_kwargs["num_return_sequences"] = audio_num_return
            elif "num_return_sequences" in beam_kwargs:
                beam_kwargs.pop("num_return_sequences")
            beam_kwargs["length_penalty"] = float(self.config.get("audio_length_penalty", 1.0))

            audio_num_samples = max(0, int(self.config.get("audio_num_samples", 0) or 0))
            if audio_num_samples > 0:
                sample_kwargs = dict(base_gen_kwargs)
                sample_kwargs["do_sample"] = True
                temperature_val = float(self.config.get("audio_sample_temperature", 1.0))
                sample_kwargs["temperature"] = temperature_val if temperature_val > 0 else None
                top_p_val = float(self.config.get("audio_sample_top_p", 0.9))
                if 0.0 < top_p_val < 1.0:
                    sample_kwargs["top_p"] = top_p_val
                else:
                    sample_kwargs["top_p"] = None
                sample_kwargs["num_beams"] = 1
                sample_kwargs["num_return_sequences"] = audio_num_samples
                sample_kwargs["length_penalty"] = float(self.config.get("audio_length_penalty", 1.0))
        else:
            beam_kwargs["num_beams"] = 1
            beam_kwargs.pop("num_return_sequences", None)
            beam_kwargs.pop("length_penalty", None)
            sample_kwargs = None

        # EARLY TRAINING FIX: Suppress EOS in first 500 steps to force output
        # The untrained audio features may confuse the model into immediate EOS
        early_audio_suppress = bool(self.config.get("suppress_eos_for_audio_early_steps", True))
        if early_audio_suppress and has_audio and self.global_step < 500 and getattr(tok, "eos_token_id", None) is not None:
            suppress_list = [tok.eos_token_id]
            # Also suppress pad token if different from EOS
            if getattr(tok, "pad_token_id", None) is not None and tok.pad_token_id != tok.eos_token_id:
                suppress_list.append(tok.pad_token_id)
            beam_kwargs["suppress_tokens"] = suppress_list
            if sample_kwargs is not None:
                sample_kwargs["suppress_tokens"] = suppress_list


        input_ids_device = input_ids.to(device)
        attention_mask_device = attention_mask.to(device) if isinstance(attention_mask, torch.Tensor) else None
        pixel_values_device = pixel_values.to(device) if isinstance(pixel_values, torch.Tensor) else None
        audio_tokens_device = audio_tokens.to(device) if isinstance(audio_tokens, torch.Tensor) else None
        audio_attention_mask = inputs.get("audio_attention_mask")
        if isinstance(audio_attention_mask, torch.Tensor):
            audio_attention_mask = audio_attention_mask.to(device)

        sanitized_ids: Optional[torch.Tensor] = None
        base_input_ids_device: Optional[torch.Tensor] = None
        if model_choice != "safe":
            has_audio_tokens = (
                audio_tokens is not None
                and isinstance(audio_tokens, torch.Tensor)
                and audio_tokens.numel() > 0
            )
            if has_audio_tokens:
                sanitized_ids = self._sanitize_input_ids_batch(input_ids)
                if sanitized_ids is None:
                    sanitized_ids = input_ids
            else:
                sanitized_ids = input_ids
            base_input_ids_device = sanitized_ids.to(device)

        def _run_generate(kwargs: Dict[str, Any]) -> torch.Tensor:
            if model_choice == "safe":
                output = self.safe_model.generate(
                    input_ids=input_ids_device,
                    attention_mask=attention_mask_device,
                    pixel_values=pixel_values_device,
                    audio_tokens=audio_tokens_device,
                    audio_attention_mask=audio_attention_mask,
                    **kwargs,
                )
            else:
                output = self.safe_model.base_vl.llm.generate(
                    input_ids=base_input_ids_device,
                    attention_mask=attention_mask_device,
                    pixel_values=pixel_values_device,
                    **kwargs,
                )
            if not isinstance(output, torch.Tensor):
                output = torch.as_tensor(output)
            return output.to(device)

        batch_size = input_ids.size(0)
        beam_output = _run_generate(beam_kwargs)
        beam_return = int(beam_kwargs.get("num_return_sequences", 1) or 1)
        beam_output = beam_output.view(batch_size, beam_return, beam_output.size(-1))

        candidate_groups: List[torch.Tensor] = [beam_output]

        if sample_kwargs is not None:
            sample_output = _run_generate(sample_kwargs)
            sample_return = int(sample_kwargs.get("num_return_sequences", 1) or 1)
            sample_output = sample_output.view(batch_size, sample_return, sample_output.size(-1))
            candidate_groups.append(sample_output)

        max_seq_len = max(group.size(-1) for group in candidate_groups)
        if any(group.size(-1) != max_seq_len for group in candidate_groups):
            candidate_groups = [
                F.pad(group, (0, max_seq_len - group.size(-1))) if group.size(-1) != max_seq_len else group
                for group in candidate_groups
            ]

        generated = torch.cat(candidate_groups, dim=1)
        total_candidates = generated.size(1)

        # Calculate per-sample prompt lengths using attention mask to handle left padding
        attention_mask_tensor = inputs.get("attention_mask", None)
        if isinstance(attention_mask_tensor, torch.Tensor):
            prompt_lengths = attention_mask_tensor.to(device).sum(dim=1)
        else:
            prompt_source = sanitized_ids if (model_choice == "base" and sanitized_ids is not None) else input_ids
            prompt_lengths = torch.full((generated.size(0),), prompt_source.shape[1], dtype=torch.long, device=generated.device)

        has_audio_mask: torch.Tensor
        if batch_audio is not None:
            has_audio_mask = torch.tensor(
                [audio is not None for audio in batch_audio],
                dtype=torch.bool,
                device=device,
            )
        elif has_audio:
            has_audio_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            has_audio_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        rerank_enabled = (
            model_choice == "safe"
            and is_audio_only
            and total_candidates > 1
            and bool(self.config.get("audio_rerank_with_clap", False))
        )

        references = list(references or [])
        questions = list(questions or [])
        if rerank_enabled:
            best_sequences = self._rerank_audio_candidates(
                generated,
                prompt_lengths,
                has_audio_mask,
                batch_audio,
                questions,
                references,
                device,
            )
        else:
            best_sequences = generated[:, 0, :]

        # CRITICAL FIX: Check if generation returned truncated output (only new tokens)
        # This happens when using inputs_embeds with some HuggingFace configs
        if best_sequences.shape[1] < input_ids.shape[1]:
            extracted_tokens = [best_sequences[i] for i in range(best_sequences.size(0))]
        else:
            extracted_tokens = []
            for i in range(best_sequences.size(0)):
                sample_prompt_len = int(prompt_lengths[i].item())
                start_idx = sample_prompt_len
                if start_idx < best_sequences.shape[1]:
                    new_tokens = best_sequences[i, start_idx:]
                    extracted_tokens.append(new_tokens)
                else:
                    extracted_tokens.append(torch.tensor([], dtype=torch.long, device=best_sequences.device))
        
        return extracted_tokens

    def _normalize_answer(self, s: str) -> str:
        """Normalize answer text for consistent comparison."""
        if not s:
            return ""
            
        s = s.strip().lower()
        
        # Remove common answer prefixes
        if "answer:" in s:
            s = s.split("answer:")[-1].strip()
        
        # Map common words to digits
        word_to_digit = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", 
            "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", 
            "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
            "eighteen": "18", "nineteen": "19", "twenty": "20",
            "yes": "yes", "no": "no"
        }
        
        # Check if answer starts with a known word
        for word, digit in word_to_digit.items():
            if s.startswith(word):
                return digit
        
        # Extract first number if present
        import re
        number_match = re.search(r'-?\d+', s)
        if number_match:
            return number_match.group(0)
        
        # Return first word if no number found
        first_word = s.split()[0] if s.split() else ""
        return first_word

    def _clean_answer(self, s: str) -> str:
        """Clean and normalize answer text (legacy method)."""
        return self._normalize_answer(s)
    
    def _extract_answer(self, generated_text, *, mode: str = "vl"):
        """Extract answer text, supporting both VQA-style and audio caption modes."""
        if not generated_text:
            return ""

        import re

        # Normalise whitespace early so downstream parsing is easier.
        answer = str(generated_text).strip()
        answer = re.sub(r"[\r\n]+", " ", answer)
        answer = re.sub(r"\s+", " ", answer).strip()
        if not answer:
            return ""

        lower_answer = answer.lower()

        # First, try to extract text after "ASSISTANT:" marker (case-insensitive)
        # This handles cases like "reaching for? ASSISTANT: Ball"
        assistant_match = re.search(r'(?:assistant|ssistant)\s*[:\-]\s*(.+)', lower_answer, re.IGNORECASE)
        if assistant_match:
            # Extract everything after ASSISTANT:
            # Need to extract from original answer to preserve case
            match_in_original = re.search(r'(?:assistant|ssistant)\s*[:\-]\s*(.+)', answer, re.IGNORECASE)
            if match_in_original:
                answer = match_in_original.group(1).strip()
                lower_answer = answer.lower()

        # Remove obvious assistant-style prefixes (including truncated variants).
        prefix_patterns = [
            r"^assistant\s*[:\-]\s*",
            r"^ssistant\s*[:\-]\s*",  # Handles truncated "Assistant"
            r"^ans(?:wer)?\s*[:\-]\s*",
            r"^ant\s*[:\-]\s*",
            r"^response\s*[:\-]\s*",
            r"^reply\s*[:\-]\s*",
        ]
        for pattern in prefix_patterns:
            if re.match(pattern, lower_answer):
                answer = re.sub(pattern, "", answer, count=1, flags=re.IGNORECASE).strip()
                lower_answer = answer.lower()
                break

        # Generic "prefix: value" handling for other short alphabetic prefixes.
        if ":" in answer:
            prefix, remainder = answer.split(":", 1)
            prefix_clean = prefix.strip().lower()
            if (
                prefix_clean
                and len(prefix_clean.split()) <= 3
                and all(ch.isalpha() for ch in prefix_clean.replace(" ", ""))
            ):
                answer = remainder.strip()
                lower_answer = answer.lower()

        # Remove leading markdown bullets or numbering artefacts.
        answer = re.sub(r"^(?:[\-\*\u2022]+|\d+\.)\s*", "", answer)

        if mode == "audio":
            return answer.strip()

        # Trim trailing sentence once we hit strong punctuation to keep VQA answers short.
        sentence_chunks = re.split(r"[\.?!]", answer)
        if sentence_chunks:
            answer = sentence_chunks[0].strip()

        if not answer:
            return ""

        # Split into tokens, strip punctuation per token, and cap at 3 tokens.
        tokens = []
        for raw_token in answer.split():
            token = re.sub(r"^[^a-z0-9]+", "", raw_token, flags=re.IGNORECASE)
            token = re.sub(r"[^a-z0-9]+$", "", token, flags=re.IGNORECASE)
            if token:
                tokens.append(token)
            if len(tokens) >= 3:
                break

        return " ".join(tokens)

    # ------------------------------------------------------------------
    # SCST Finetuning helpers
    # ------------------------------------------------------------------

    def _run_scst_finetune(self) -> Optional[Dict[str, float]]:
        if not self.scst_enabled:
            return None

        print(
            f"\n🚀 Starting SCST fine-tune for {self.config.get('variant', 'unknown')} "
            f"(epochs_since_improvement={self._epochs_since_improvement})",
            flush=True,
        )

        scst_epochs = max(1, int(self.config.get("scst_epochs", 1)))
        scst_lr = float(self.config.get("scst_learning_rate", 5e-6))
        scst_num_samples = max(1, int(self.config.get("scst_num_samples", 1)))
        scst_top_p = float(self.config.get("scst_sample_top_p", 0.9))
        scst_temperature = float(self.config.get("scst_sample_temperature", 0.9))
        scst_reward_metric = str(self.config.get("scst_reward_metric", "cider")).lower()
        max_grad_norm = float(self.config.get("max_grad_norm", 1.0))

        device = next(self.safe_model.parameters()).device

        # Preserve optimizer state (learning rate) and disable scheduler for SCST stage
        original_lrs = [group.get("lr", scst_lr) for group in self.optimizer.param_groups]
        for group in self.optimizer.param_groups:
            group["lr"] = scst_lr

        original_scheduler = self.scheduler
        self.scheduler = None

        self.safe_model.enable_audio_training()
        self.safe_model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_steps = 0
        accum_loss = []

        for epoch in range(scst_epochs):
            print(f"[SCST] Epoch {epoch + 1}/{scst_epochs}", flush=True)
            for batch in self.train_dataloader:
                # Move tensors to device
                for key in list(batch.keys()):
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                has_audio = batch.get(
                    "has_audio",
                    torch.zeros(len(batch.get("questions", [])), dtype=torch.bool, device=device),
                )
                if isinstance(has_audio, torch.Tensor):
                    has_audio = has_audio.to(device=device, dtype=torch.bool)
                else:
                    has_audio = torch.tensor(has_audio, dtype=torch.bool, device=device)

                if self.config.get("filter_silent_audio", True):
                    filtered = self._check_silent_audio(batch)
                    if isinstance(filtered, torch.Tensor) and filtered.numel() == has_audio.numel():
                        has_audio = filtered.to(device=device, dtype=torch.bool)

                audio_indices = torch.where(has_audio)[0]
                if audio_indices.numel() == 0:
                    continue

                audio_subset = self._select_batch_indices(batch, audio_indices, clone=True)
                scst_loss = self._compute_scst_loss(
                    audio_subset,
                    scst_top_p=scst_top_p,
                    scst_temperature=scst_temperature,
                    scst_num_samples=scst_num_samples,
                    reward_metric=scst_reward_metric,
                )

                if scst_loss is None:
                    continue

                (scst_loss).backward()
                self._sanitize_audio_gradients()
                if self.trainable_param_list:
                    torch.nn.utils.clip_grad_norm_(self.trainable_param_list, max_grad_norm)

                self.optimizer.step()
                self._sanitize_audio_parameters()
                self.optimizer.zero_grad(set_to_none=True)

                total_steps += 1
                accum_loss.append(float(scst_loss.detach().cpu().item()))

            if accum_loss:
                print(
                    f"[SCST] Epoch {epoch + 1} average loss: {np.mean(accum_loss[-len(self.train_dataloader):]):.4f}",
                    flush=True,
                )

        # Restore optimizer learning rates
        for group, lr in zip(self.optimizer.param_groups, original_lrs):
            group["lr"] = lr

        self.scheduler = original_scheduler
        self._scst_triggered = True

        print("[SCST] Fine-tune complete. Running final evaluation...", flush=True)
        scst_metrics = self.evaluate(max_batches=self.config.get("max_eval_batches", None))

        self.save_checkpoint(scst_metrics, is_best=False, suffix="scst")
        self._export_scst_checkpoint()

        print(
            f"[SCST] Final Audio Accuracy: {scst_metrics.get('audio_accuracy', 0.0):.4f}",
            flush=True,
        )
        return scst_metrics

    def _compute_scst_loss(
        self,
        batch: Dict[str, Any],
        *,
        scst_top_p: float,
        scst_temperature: float,
        scst_num_samples: int,
        reward_metric: str,
    ) -> Optional[torch.Tensor]:
        device = next(self.safe_model.parameters()).device

        questions = batch.get("questions", [])
        images = batch.get("images")
        audio = batch.get("audio")
        answers = batch.get("answers")
        batch_size = len(questions)
        if batch_size == 0:
            return None

        reference_answers = self._build_training_answer_list(answers, batch_size)
        reference_texts: List[List[str]] = []
        for raw in reference_answers:
            refs = [
                self._normalize_audio_caption(ans)
                for ans in self._prepare_gt_answers(raw)
                if ans
            ]
            reference_texts.append([ref for ref in refs if ref])

        inputs = self.safe_model.prepare_multimodal_inputs(
            text=questions,
            images=images,
            audio=audio,
            answers=None,
            device=device,
            training_mode=False,
        )

        baseline_texts, _ = self._scst_generate_texts(inputs, do_sample=False)
        baseline_rewards = self._scst_compute_rewards(baseline_texts, reference_texts, reward_metric)

        losses: List[torch.Tensor] = []
        for _ in range(scst_num_samples):
            sample_texts, _ = self._scst_generate_texts(
                inputs,
                do_sample=True,
                top_p=scst_top_p,
                temperature=scst_temperature,
            )

            log_probs, valid_mask = self._scst_sequence_log_probs(
                questions,
                images,
                audio,
                sample_texts,
            )

            if log_probs is None:
                continue

            sample_rewards = self._scst_compute_rewards(sample_texts, reference_texts, reward_metric)
            rewards_tensor = torch.tensor(sample_rewards, dtype=torch.float32, device=device)
            baseline_tensor = torch.tensor(baseline_rewards, dtype=torch.float32, device=device)
            advantage = rewards_tensor - baseline_tensor
            if valid_mask is not None:
                log_probs = log_probs * valid_mask
                advantage = advantage * valid_mask
                effective = valid_mask.sum().item()
                if effective == 0:
                    continue

            advantage = advantage.detach()
            loss = -(advantage * log_probs).mean()
            losses.append(loss)

        if not losses:
            return None

        return torch.stack(losses).mean()

    def _scst_generate_texts(
        self,
        inputs: Dict[str, torch.Tensor],
        *,
        do_sample: bool,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[List[str], torch.Tensor]:
        tok = self.safe_model.base_vl.tokenizer
        gen_kwargs = dict(
            max_new_tokens=int(self.config.get("audio_generation_max_new_tokens", self.config.get("generation_max_new_tokens", 32))),
            min_new_tokens=1,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample and top_p and 0.0 < top_p < 1.0 else None,
            num_beams=1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=getattr(tok, "eos_token_id", None),
            repetition_penalty=float(self.config.get("audio_repetition_penalty", 1.1)),
            output_scores=False,
            return_dict_in_generate=False,
        )

        generated = self.safe_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            audio_tokens=inputs.get("audio_tokens"),
            audio_attention_mask=inputs.get("audio_attention_mask"),
            **gen_kwargs,
        )

        if not isinstance(generated, torch.Tensor):
            generated = torch.as_tensor(generated)
        generated = generated.to(inputs["input_ids"].device)

        if generated.dim() == 2:
            generated = generated.unsqueeze(1)

        attention_mask = inputs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            prompt_lengths = attention_mask.sum(dim=1)
        else:
            prompt_lengths = torch.full(
                (generated.size(0),),
                inputs["input_ids"].shape[1],
                dtype=torch.long,
                device=generated.device,
            )

        captions: List[str] = []
        sequences: List[torch.Tensor] = []
        for batch_idx in range(generated.size(0)):
            sample_prompt_len = int(prompt_lengths[batch_idx].item())
            tokens = generated[batch_idx, 0, sample_prompt_len:]
            sequences.append(tokens)
            text = tok.decode(tokens, skip_special_tokens=True).strip()
            captions.append(text)

        return captions, generated[:, 0, :]

    def _scst_sequence_log_probs(
        self,
        questions: Sequence[str],
        images: Optional[Sequence[Any]],
        audio: Optional[Sequence[Any]],
        captions: Sequence[str],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        device = next(self.safe_model.parameters()).device
        inputs = self.safe_model.prepare_multimodal_inputs(
            text=questions,
            images=images,
            audio=audio,
            answers=captions,
            device=device,
            training_mode=True,
        )

        logits = self.safe_model(**inputs)
        logits = logits.get("logits") if isinstance(logits, dict) else getattr(logits, "logits", None)
        if logits is None:
            return None, None

        labels = inputs.get("labels")
        if labels is None:
            return None, None

        mask = (labels != -100)
        if mask.sum() == 0:
            return None, None

        log_probs = F.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        gathered = gathered * mask
        seq_log_probs = gathered.sum(dim=1)
        valid_mask = (mask.sum(dim=1) > 0)
        return seq_log_probs, valid_mask.to(logits.dtype)

    def _scst_compute_rewards(
        self,
        captions: Sequence[str],
        references: Sequence[Sequence[str]],
        reward_metric: str,
    ) -> List[float]:
        rewards: List[float] = []
        for caption, refs in zip(captions, references):
            normalized_caption = self._normalize_audio_caption(caption)
            if not normalized_caption or not refs:
                rewards.append(0.0)
                continue

            if reward_metric == "spider":
                cider = self._compute_cider_score(normalized_caption, refs)
                spice = self._compute_spice_score(normalized_caption, refs)
                rewards.append((cider + spice) / 2.0)
            elif reward_metric == "spice":
                rewards.append(self._compute_spice_score(normalized_caption, refs))
            else:
                rewards.append(self._compute_cider_score(normalized_caption, refs))
        return rewards

    def _export_scst_checkpoint(self) -> None:
        output_dir = Path(self.config.get("output_dir", ".")).resolve()
        scst_checkpoint = output_dir / f"checkpoint_scst_epoch_{self.epoch}_step_{self.global_step}.pt"
        if not scst_checkpoint.exists():
            return

        try:
            run_dir = output_dir.parent
            runs_dir = run_dir.parent
            if runs_dir.name == "runs":
                experiments_root = runs_dir.parent
            else:
                experiments_root = runs_dir
            finetune_dir = experiments_root / "finetuned"
            finetune_dir.mkdir(parents=True, exist_ok=True)

            run_stamp = run_dir.name
            variant = self.config.get("variant", "variant")
            target_name = f"{run_stamp}_{variant}_scst.pt"
            target_path = finetune_dir / target_name
            shutil.copy2(scst_checkpoint, target_path)
            print(f"[SCST] Checkpoint copied to {target_path}", flush=True)
        except Exception as exc:
            print(f"[SCST] Failed to export checkpoint: {exc}", flush=True)

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False, suffix: str = ""):
        """Save model checkpoint with error handling."""
        try:
            checkpoint = {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "model_state_dict": self.safe_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "config": self.config,
                "metrics": metrics,
                "training_stats": self.training_stats,
                "curriculum_state": self.curriculum_manager.get_progress_summary() if self.use_curriculum else None,
                "model_class": "SAFEModel"  # Explicitly store model class name
            }
        
            # Save regular checkpoint
            if suffix:
                checkpoint_path = os.path.join(
                    self.config["output_dir"], 
                    f"checkpoint_{suffix}_epoch_{self.epoch}_step_{self.global_step}.pt"
                )
            else:
                checkpoint_path = os.path.join(
                    self.config["output_dir"],
                    "last_checkpoint.pt"
                )

            print(f"Saving checkpoint to: {checkpoint_path}", flush=True)
            torch.save(checkpoint, checkpoint_path)
            print(f"Successfully saved checkpoint: {checkpoint_path}", flush=True)
            
            # Save best checkpoint
            if is_best:
                best_path = os.path.join(self.config["output_dir"], "best_checkpoint.pt")
                print(f"Saving best checkpoint to: {best_path}", flush=True)
                torch.save(checkpoint, best_path)
                print(f"Successfully saved best checkpoint: {best_path}", flush=True)
                
        except Exception as e:
            print(f"ERROR saving checkpoint: {e}", flush=True)
            print(f"Model type: {type(self.safe_model)}", flush=True)
            print(f"Model class name: {self.safe_model.__class__.__name__}", flush=True)
            raise
            
        # Save curriculum checkpoint if using curriculum
        if self.use_curriculum:
            curriculum_checkpoint_path = os.path.join(
                self.config["output_dir"], 
                "curriculum_checkpoint.pt"
            )
            self.curriculum_manager.save_checkpoint(Path(curriculum_checkpoint_path))
    
    def train(self):
        """
        Main training loop for Stage A with curriculum learning support.
        """
        if self.use_curriculum:
            return self._train_with_curriculum()
        else:
            return self._train_traditional()

    def _maybe_log_progress(
        self,
        *,
        label: str,
        step_index: int,
        total_batches: int,
        epoch_losses: Dict[str, List[float]],
    ) -> None:
        """Emit periodic training progress logs irrespective of accumulation settings."""

        total_batches = max(1, total_batches)
        now = time.time()

        should_log_step = (step_index + 1) % self.log_interval == 0
        should_log_time = (now - self._last_progress_log_time) >= self.progress_log_timeout

        if not should_log_step and not should_log_time:
            return

        losses = epoch_losses.get("total_loss") if epoch_losses else None
        if not losses:
            if should_log_time:
                print(
                    f"[Train] {label} Batch {step_index + 1}/{total_batches}: waiting for loss statistics"
                    f" (optimizer_step={self.global_step})",
                    flush=True,
                )
                self._last_progress_log_time = now
            return

        window_size = min(len(losses), self.log_interval)
        avg_loss = float(np.mean(losses[-window_size:])) if window_size else float("nan")

        accum_position = self._micro_step if self._micro_step > 0 else self.grad_accum_steps
        message = (
            f"[Train] {label} Batch {step_index + 1}/{total_batches} | "
            f"optimizer_step={self.global_step} | accum={accum_position}/{self.grad_accum_steps} | "
            f"avg_total_loss={avg_loss:.4f}"
        )
        print(message, flush=True)
        self._last_progress_log_time = now

    def _train_with_curriculum(self):
        """Training loop with curriculum learning."""
        print("🎓 Starting Stage A training with curriculum learning...", flush=True)
        print(f"Curriculum has {self.curriculum_manager.config.get_num_stages()} stages", flush=True)

        # Compute baseline metrics before training (quick evaluation)
        print("Computing baseline metrics (quick evaluation)...", flush=True)
        baseline_metrics = self.evaluate(max_batches=50)  # Only evaluate first 50 batches
        self.baseline_metrics = baseline_metrics
        self.curriculum_manager.set_baseline_metrics({
            "vl_retention": baseline_metrics["retention_score"],
            "audio_accuracy": baseline_metrics["audio_accuracy"]
        })
        print(f"Baseline metrics computed:", flush=True)
        print(f"  Retention Score: {baseline_metrics['retention_score']:.4f}", flush=True)
        print(f"  Audio Accuracy: {baseline_metrics['audio_accuracy']:.4f}", flush=True)
        print(f"  VL Safe Accuracy: {baseline_metrics['vl_safe_accuracy']:.4f}", flush=True)
        print(f"  VL Base Accuracy: {baseline_metrics['vl_base_accuracy']:.4f}", flush=True)
        print(f"  Total Loss: {baseline_metrics['total_loss']:.4f}", flush=True)
        print(f"", flush=True)
        print(f"🔍 Model Comparison:", flush=True)
        print(f"  SAFE Model VL Accuracy: {baseline_metrics['vl_safe_accuracy']:.4f}", flush=True)
        print(f"  Base VL Model Accuracy: {baseline_metrics['vl_base_accuracy']:.4f}", flush=True)
        if baseline_metrics['vl_base_accuracy'] > 0.01:  # If base model has decent accuracy
            if baseline_metrics['vl_safe_accuracy'] < baseline_metrics['vl_base_accuracy'] * 0.5:
                print(f"  ⚠️  WARNING: SAFE model significantly underperforming base VL model!", flush=True)
                print(f"     This suggests an integration issue in the SAFE architecture.", flush=True)
        else:
            print(f"  ℹ️  Both models show low accuracy - may be due to dataset/task mismatch.", flush=True)
        
        # Initialize first stage
        self.update_curriculum_config()
        
        epoch_in_stage = 0
        samples_in_stage = 0
        
        while not self.curriculum_manager.is_completed:
            self.epoch += 1
            epoch_in_stage += 1
            epoch_losses = defaultdict(list)
            
            # Get current stage info
            stage_name = self.current_stage_config["stage_name"] if self.current_stage_config else "unknown"
            stage_idx = self.current_stage_config["stage_idx"] if self.current_stage_config else 0
            
            progress_bar = self.train_dataloader
            for step, batch in enumerate(progress_bar):
                # Training step
                step_losses, optimizer_stepped = self.train_step(batch)
                batch_size = len(batch["questions"])
                samples_in_stage += batch_size

                # Accumulate losses
                for key, value in step_losses.items():
                    epoch_losses[key].append(float(value))

                if optimizer_stepped:
                    self.global_step += 1

                self._maybe_log_progress(
                    label=f"Stage {stage_name} Epoch {epoch_in_stage}",
                    step_index=step,
                    total_batches=len(self.train_dataloader),
                    epoch_losses=epoch_losses,
                )

                if optimizer_stepped and self.global_step % self.log_interval == 0:
                    avg_losses = {k: np.mean(v[-self.log_interval:]) for k, v in epoch_losses.items() if v}
                    if avg_losses:
                        summary_parts = [f"{k}={val:.4f}" for k, val in avg_losses.items()]
                        print(
                            f"[Train|Stage {stage_name}] Global step {self.global_step}: " + ", ".join(summary_parts),
                            flush=True,
                        )
                    else:
                        print(
                            f"[Train|Stage {stage_name}] Global step {self.global_step}: metrics pending",
                            flush=True,
                        )

                    try:
                        log_dict = {"train/" + k: v for k, v in avg_losses.items()}
                        log_dict.update({
                            "curriculum/stage": stage_idx,
                            "curriculum/stage_name": stage_name,
                            "curriculum/samples_in_stage": samples_in_stage,
                        })
                        wandb.log(log_dict, step=self.global_step)
                    except Exception:
                        pass
            
            # End of epoch evaluation
            print(f"\n📊 End of Epoch {self.epoch} (Stage {stage_name}, Epoch {epoch_in_stage})", flush=True)
            max_eval_batches = self.config.get("max_eval_batches", None)

            # eval every n epochs
            if self.epoch % 10 == 0:
                eval_metrics = self.evaluate(max_batches=max_eval_batches)

                # Save CSV if accuracy exceeds threshold (check current epoch, not previous)
                if (self.config.get("save_audio_csv", False) and
                    eval_metrics.get('audio_accuracy', 0.0) >= self.config.get("csv_min_accuracy", 0.45)):
                    csv_path = f"{self.config['output_dir']}/audio_eval_epoch{self.epoch}.csv"
                    print(f"🔍 Audio accuracy {eval_metrics['audio_accuracy']:.3f} >= {self.config.get('csv_min_accuracy', 0.45):.3f}, saving CSV to {csv_path}", flush=True)
                    # Re-run evaluation to collect CSV samples
                    self.evaluate(
                        max_batches=max_eval_batches,
                        save_audio_samples_csv=csv_path,
                        max_csv_samples=self.config.get("csv_max_samples", 500)
                    )

                # Store accuracy for tracking
                self._last_audio_accuracy = eval_metrics.get('audio_accuracy', 0.0)
            
            # Calculate curriculum-specific metrics
            curriculum_metrics = {
                "audio_accuracy": eval_metrics["audio_accuracy"],
                "vl_retention": eval_metrics["retention_score"], 
                "efficiency_rate": 0.0,  # TODO: Implement efficiency calculation
                "cross_modal_alignment": 0.0  # TODO: Implement alignment calculation
            }
            
            # Update curriculum manager
            self.curriculum_manager.update_metrics(curriculum_metrics, samples_processed=samples_in_stage)
            
            print(f"  Metrics:", flush=True)
            print(f"    Audio Accuracy: {eval_metrics['audio_accuracy']:.4f}", flush=True)
            print(f"    VL Retention: {eval_metrics['retention_score']:.4f}", flush=True)
            print(f"    Samples in Stage: {samples_in_stage}", flush=True)
            
            # Check curriculum progression
            if self.epoch % self.config.get("validation_frequency", 5) == 0:
                progression_status = self.curriculum_manager.advance_epoch()
                
                if progression_status == ProgressionStatus.ADVANCE:
                    print(f"🎯 Advanced to next curriculum stage!", flush=True)
                    
                    # Save checkpoint before advancing
                    self.save_checkpoint(eval_metrics, is_best=True, 
                                       suffix=f"stage_{stage_name}_completed")
                    
                    # Update configuration for new stage
                    self.update_curriculum_config()
                    epoch_in_stage = 0
                    samples_in_stage = 0
                    
                elif progression_status == ProgressionStatus.EXTEND:
                    print(f"⏳ Extended current stage due to unmet criteria", flush=True)
                    
                elif progression_status == ProgressionStatus.FAILED:
                    print(f"❌ Curriculum failed - criteria not met after extensions", flush=True)
                    break
                    
                else:  # CONTINUE
                    print(f"📈 Continuing current stage...", flush=True)
            else:
                # Still advance curriculum epoch counter for timing
                self.curriculum_manager.advance_epoch()
            
            # Save regular checkpoint
            if self.epoch % max(1, self.config.get("save_steps", 5000) // len(self.train_dataloader)) == 0:
                self.save_checkpoint(eval_metrics, is_best=False)
            
            # Log curriculum progress
            try:
                progress_summary = self.curriculum_manager.get_progress_summary()
                log_dict = {
                    "curriculum/total_epochs": self.epoch,
                    "curriculum/current_stage_idx": progress_summary["current_stage_idx"],
                    "curriculum/is_completed": progress_summary["is_completed"]
                }
                log_dict.update({"eval/" + k: v for k, v in eval_metrics.items()})
                wandb.log(log_dict, step=self.global_step)
            except:
                pass
        
            print("🎓 Curriculum learning completed!", flush=True)
        
        # Final evaluation and checkpoint
        max_eval_batches = self.config.get("max_eval_batches", None)
        final_metrics = self.evaluate(max_batches=max_eval_batches)
        self.save_checkpoint(final_metrics, is_best=False, suffix="final")
        
        # Print curriculum summary
        summary = self.curriculum_manager.get_progress_summary()
        print(f"\n📋 Curriculum Summary:", flush=True)
        print(f"  Total Epochs: {summary['total_epochs']}", flush=True)
        print(f"  Stages Completed: {summary['current_stage_idx']}/{summary['total_stages']}", flush=True)
        print(f"  Final Audio Accuracy: {summary['current_metrics']['audio_accuracy']:.4f}", flush=True)
        print(f"  Final VL Retention: {summary['current_metrics']['vl_retention']:.4f}", flush=True)
        
        return final_metrics
    
    def _train_traditional(self):
        """Traditional fixed-epoch training loop."""
        print(f"Starting Stage A training for {self.config['num_epochs']} epochs...", flush=True)
        print(f"Total training steps: {len(self.train_dataloader) * self.config['num_epochs']}", flush=True)

        max_eval_batches = self.config.get("max_eval_batches", None)
        
        for epoch in range(self.config["num_epochs"]):
            self.epoch = epoch
            epoch_losses = defaultdict(list)
            
            # Training loop
            progress_bar = self.train_dataloader
            print(f"Starting epoch {epoch+1}/{self.config['num_epochs']} with {len(self.train_dataloader)} batches...", flush=True)
            
            for step, batch in enumerate(progress_bar):
                if step == 0:
                    print(f"Processing first batch of epoch {epoch+1}...", flush=True)
                
                # Training step
                step_losses, optimizer_stepped = self.train_step(batch)
                
                if step == 0:
                    first_loss = step_losses.get("total_loss")
                    if isinstance(first_loss, (int, float)):
                        loss_str = f"{float(first_loss):.4f}"
                    elif first_loss is not None:
                        try:
                            loss_str = f"{float(first_loss):.4f}"
                        except (TypeError, ValueError):
                            loss_str = str(first_loss)
                    else:
                        loss_str = "N/A"
                    print(
                        f"First batch completed successfully! Loss: {loss_str}",
                        flush=True,
                    )

                # Accumulate losses
                for key, value in step_losses.items():
                    epoch_losses[key].append(float(value))

                if optimizer_stepped:
                    self.global_step += 1

                self._maybe_log_progress(
                    label=f"Epoch {epoch + 1}/{self.config['num_epochs']}",
                    step_index=step,
                    total_batches=len(self.train_dataloader),
                    epoch_losses=epoch_losses,
                )

                if optimizer_stepped and self.global_step % self.log_interval == 0:
                    avg_losses = {k: np.mean(v[-self.log_interval:]) for k, v in epoch_losses.items() if v}
                    if avg_losses:
                        summary_parts = [f"{k}={val:.4f}" for k, val in avg_losses.items()]
                        print(
                            f"[Train] Global step {self.global_step}: " + ", ".join(summary_parts),
                            flush=True,
                        )
                    else:
                        print(
                            f"[Train] Global step {self.global_step}: metrics pending",
                            flush=True,
                        )

                    try:
                        wandb.log({
                            "train/" + k: v for k, v in avg_losses.items()
                        }, step=self.global_step)
                    except Exception:
                        pass

                # Evaluation (only trigger on actual optimizer steps)
                if optimizer_stepped and self.global_step % self.eval_interval == 0:
                    max_eval_batches = self.config.get("max_eval_batches", None)
                    eval_metrics = self.evaluate(max_batches=max_eval_batches)
                    
                    print(f"\nStep {self.global_step} Evaluation:", flush=True)
                    print(f"  Audio Accuracy: {eval_metrics['audio_accuracy']:.4f}", flush=True)
                    print(f"  Total Loss: {eval_metrics['total_loss']:.4f}", flush=True)
                    self.logger.info(
                        "[Eval][step=%s] audio_acc=%.4f total_loss=%.4f",
                        self.global_step,
                        eval_metrics['audio_accuracy'],
                        eval_metrics['total_loss'],
                    )

                    if self.global_step % self.config["save_steps"] == 0:
                        self.save_checkpoint(eval_metrics, is_best=False)
                    
                    # Log to wandb
                    try:
                        wandb.log({
                            "eval/" + k: v for k, v in eval_metrics.items()
                        }, step=self.global_step)
                    except:
                        pass
            
            # End of epoch evaluation
            print(f"\nEnd of Epoch {epoch+1}", flush=True)
            max_eval_batches = self.config.get("max_eval_batches", None)

            epoch_metrics = self.evaluate(max_batches=max_eval_batches)

            # Save CSV if accuracy exceeds threshold (check current epoch, not previous)
            if (self.config.get("save_audio_csv", False) and
                epoch_metrics.get('audio_accuracy', 0.0) >= self.config.get("csv_min_accuracy", 0.45)):
                csv_path = f"{self.config['output_dir']}/audio_eval_epoch{epoch+1}.csv"
                print(f"🔍 Audio accuracy {epoch_metrics['audio_accuracy']:.3f} >= {self.config.get('csv_min_accuracy', 0.45):.3f}, saving CSV to {csv_path}", flush=True)
                # Re-run evaluation to collect CSV samples
                self.evaluate(
                    max_batches=max_eval_batches,
                    save_audio_samples_csv=csv_path,
                    max_csv_samples=self.config.get("csv_max_samples", 500)
                )

            # Store accuracy for next epoch check
            self._last_audio_accuracy = epoch_metrics.get('audio_accuracy', 0.0)

            print(f"Epoch {epoch+1} Results:", flush=True)
            print(f"  Audio Accuracy: {epoch_metrics['audio_accuracy']:.4f}", flush=True)
            self.logger.info(
                "[Eval][epoch=%s] audio_acc=%.4f total_loss=%.4f",
                epoch + 1,
                epoch_metrics['audio_accuracy'],
                epoch_metrics['total_loss'],
            )

            current_acc = epoch_metrics.get('audio_accuracy', 0.0)
            improvement = current_acc - self._best_audio_accuracy
            if improvement > self.scst_min_delta:
                self._best_audio_accuracy = current_acc
                self._epochs_since_improvement = 0
            else:
                self._epochs_since_improvement += 1

        print("Stage A training completed!", flush=True)

        # Final checkpoint
        max_eval_batches = self.config.get("max_eval_batches", None)
        final_metrics = self.evaluate(max_batches=max_eval_batches)
        self.save_checkpoint(final_metrics, is_best=False)

        if (
            self.scst_enabled
            and not self._scst_triggered
            and self._epochs_since_improvement >= self.scst_patience_epochs
        ):
            scst_metrics = self._run_scst_finetune()
            if scst_metrics is not None:
                final_metrics = scst_metrics

        return final_metrics
