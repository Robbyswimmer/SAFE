import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
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

# Set CPU threads for predictable performance
torch.set_num_threads(min(8, os.cpu_count() or 2))

from ..models.safe_model import SAFEModel
from ..models.base_vl import BaseVLModel
from ..data.datasets import create_safe_dataloader
from ..data.curriculum import CurriculumManager, CurriculumConfig, ProgressionStatus
from .losses import RetentionLoss, AudioTaskLoss, CombinedStageLoss
from .null_space import NullSpaceProjector, NullSpaceConfig


@dataclass
class AccuracyResult:
    """Container for per-sample accuracy with auxiliary metrics."""

    score: float
    metrics: Dict[str, float]


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
            "retention_loss_weight": 1.0,
            "distillation_weight": 1.0,
            "distillation_temperature": 3.0,
            "fisher_weight": 0.1,
            "compute_fisher_at_start": True,  # Compute Fisher information before training starts
            "fisher_num_samples": 1000,  # Number of samples for Fisher computation
            "save_steps": 5000,
            "eval_steps": 1000,
            "logging_steps": 100,
            "output_dir": "./checkpoints/stage_a",
            "early_stopping_patience": 3,
            "retention_tolerance": 0.005,  # 0.5% tolerance for VL degradation
            "validation_frequency": 5 if self.use_curriculum else 1000,  # More frequent validation for curriculum
            "sample_log_interval": 500,
            "sample_log_limit": 5,
            "sample_log_examples": 3,
            "enable_null_space": False,
            "null_space_rank": 8,
            "null_space_min_samples": 32,
            "null_space_max_samples": 128,
            "null_space_audio_threshold": 0.25,
            "null_space_refresh_interval": 2000,
            "null_space_verbose": False,
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
            "gradient_accumulation_steps": 1,
            "audio_bertscore_threshold": 0.7,
            "gate_warmup_steps": 2000,
        }

        if config:
            self.config.update(config)

        self.debug_logging = bool(self.config.get("debug_logging", False))
        self.grad_accum_steps = max(1, int(self.config.get("gradient_accumulation_steps", 1)))
        self._micro_step = 0

        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        # Create output directory
        Path(self.config["output_dir"]).mkdir(parents=True, exist_ok=True)

        # Initialize loss functions (will be updated by curriculum)
        self._setup_loss_functions()
        
        # Setup optimizer
        self._setup_optimizer()

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
        self.best_retention_score = 0.0
        self.patience_counter = 0
        self.sample_log_interval = max(1, int(self.config.get("sample_log_interval", 500)))
        self.sample_log_limit = int(self.config.get("sample_log_limit", 5))
        self.sample_log_examples = max(1, int(self.config.get("sample_log_examples", 3)))
        self.sample_logs_emitted = 0
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

        if hasattr(self.safe_model, "set_debug_logging"):
            self.safe_model.set_debug_logging(self.debug_logging)
        else:
            setattr(self.safe_model, "debug_logging", self.debug_logging)

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
            "retention_loss": [],
            "distillation_loss": [],
            "fisher_loss": [],
            "vl_retention_score": [],
            "audio_gain_score": []
        }

        # Initialize BERTScore for audio caption evaluation (lazy load to avoid startup delays)
        self._bertscore_metric = None
        self._bertscore_failed = False

        # Curriculum state
        self.current_stage_config = None
        self.baseline_metrics = None

        # Null-space projector (optional hard retention guard)
        self.null_space_projector = None
        if self.config.get("enable_null_space", False):
            if not self.trainable_params:
                raise ValueError("Null-space editing enabled but no trainable parameters found")

            ns_config = NullSpaceConfig(
                max_rank=self.config.get("null_space_rank", 8),
                min_samples=self.config.get("null_space_min_samples", 32),
                max_samples=self.config.get("null_space_max_samples", 128),
                audio_ratio_threshold=self.config.get("null_space_audio_threshold", 0.25),
                refresh_interval=self.config.get("null_space_refresh_interval", 2000),
                verbose=self.config.get("null_space_verbose", False)
            )

            self.null_space_projector = NullSpaceProjector(
                params=self.trainable_params,
                config=ns_config
            )
    
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
            if self.debug_logging and not getattr(self, "_teacher_subset_debugged", False):
                print(
                    f"[TeacherDebug] subset indices={indices.tolist()} logits_shape={tuple(logits_subset.shape)}",
                    flush=True
                )
            for slot, tensor in zip(indices.tolist(), logits_subset):
                outputs_ordered[slot] = tensor.unsqueeze(0)
            loss_tensor = result.get("loss") if isinstance(result, dict) else getattr(result, "loss", None)
            if loss_tensor is not None:
                loss_values.append(loss_tensor)
            self._teacher_subset_debugged = True

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
            
        dist_weight = loss_weights.get("distillation_loss", self.config.get("distillation_weight", 1.0))
        fisher_weight = loss_weights.get("fisher_loss", self.config["fisher_weight"])
        use_fisher = fisher_weight > 0

        self.retention_loss = RetentionLoss(
            distillation_weight=dist_weight,
            fisher_weight=fisher_weight,
            temperature=self.config["distillation_temperature"],
            use_fisher_information=use_fisher
        )

        self.audio_task_loss = AudioTaskLoss(
            task_type="qa",
            label_smoothing=float(self.config.get("audio_label_smoothing", 0.1))
        )

        # Compute actual weights being used
        audio_weight_used = loss_weights.get("audio_task_loss", self.config["audio_loss_weight"])
        retention_weight_used = loss_weights.get("retention_loss", self.config["retention_loss_weight"])

        self.combined_loss = CombinedStageLoss(
            retention_loss=self.retention_loss,
            audio_task_loss=self.audio_task_loss,
            audio_weight=audio_weight_used,
            retention_weight=retention_weight_used
        )

        # ALWAYS log loss weights for visibility (critical for debugging)
        print(
            f"[LossSetup] ✓ audio_weight={audio_weight_used}, retention_weight={retention_weight_used}, distillation_weight={dist_weight}",
            flush=True,
        )
        print(
            f"[LossSetup] ✓ retention_enabled={self.combined_loss.retention_enabled}",
            flush=True,
        )
    
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

    def _run_audio_sanity_checks(self, inputs: Dict[str, Any], has_audio: torch.Tensor) -> None:
        if self._sanity_checks_completed:
            return
        if not self.config.get("enable_audio_sanity_checks", False):
            self._sanity_checks_completed = True
            return

        if has_audio.numel() == 0 or not torch.any(has_audio):
            self._sanity_checks_completed = True
            return

        audio_indices = torch.nonzero(has_audio, as_tuple=False).flatten()
        audio_inputs = self._select_batch_indices(inputs, audio_indices)
        if not audio_inputs or audio_inputs.get("audio_tokens") is None:
            self._sanity_checks_completed = True
            return


        baseline_loss = self._compute_audio_loss(self._clone_inputs(audio_inputs), gate=1.0)
        gate_off_loss = self._compute_audio_loss(self._clone_inputs(audio_inputs), gate=0.0)

        shuffled_inputs = self._clone_inputs(audio_inputs)
        audio_tokens = shuffled_inputs.get("audio_tokens")
        if isinstance(audio_tokens, torch.Tensor):
            perm = torch.randperm(audio_tokens.size(0)).to(audio_tokens.device)
            shuffled_inputs["audio_tokens"] = audio_tokens.index_select(0, perm)
        shuffled_loss = self._compute_audio_loss(shuffled_inputs, gate=1.0)

        zero_inputs = self._clone_inputs(audio_inputs)
        if isinstance(zero_inputs.get("audio_tokens"), torch.Tensor):
            zero_inputs["audio_tokens"] = torch.zeros_like(zero_inputs["audio_tokens"])
        zero_loss = self._compute_audio_loss(zero_inputs, gate=1.0)

        def fmt(value: Optional[float]) -> str:
            return "N/A" if value is None else f"{value:.6f}"

        # Only log if audio fusion is significantly hurting performance
        if gate_off_loss is not None and baseline_loss is not None and gate_off_loss < baseline_loss - 0.1:
            print(f"WARNING: Audio fusion may be hurting (gate=1.0: {baseline_loss:.4f}, gate=0.0: {gate_off_loss:.4f})", flush=True)
        print(f"  shuffled audio: {fmt(shuffled_loss)}", flush=True)
        print(f"  zeroed audio: {fmt(zero_loss)}", flush=True)

        self._sanity_checks_completed = True

    def _log_attention_probe(self) -> None:
        if not self.config.get("log_attention_probe", False):
            return
        log_limit = int(self.config.get("attention_probe_log_limit", 0))
        if log_limit and self._attention_logs_emitted >= log_limit:
            return

        summary = None
        if hasattr(self.safe_model, "get_last_attention_summary"):
            summary = self.safe_model.get_last_attention_summary()
        elif hasattr(self.safe_model, "fusion_adapter"):
            summary = getattr(self.safe_model.fusion_adapter, "last_attention_summary", None)

        if not summary:
            return

        msg = f"[AttentionProbe][step {self.global_step}]"
        if "overall_mean" in summary:
            msg += f" overall_mean={summary['overall_mean']:.6f}"
        if "overall_max" in summary:
            msg += f" overall_max={summary['overall_max']:.6f}"
        if "supervised_mean" in summary:
            msg += f" supervised_mean={summary['supervised_mean']:.6f}"
        print(msg, flush=True)
        self._attention_logs_emitted += 1

    def _log_grad_norms(self) -> None:
        """Log gradient norms by component with warnings for issues."""
        if not self.config.get("log_grad_norms", False):
            return
        log_limit = int(self.config.get("grad_log_limit", 0))
        should_print = not (log_limit and self._grad_logs_emitted >= log_limit)
        interval = max(1, int(self.config.get("grad_log_interval", 1)))
        if self.global_step % interval != 0:
            return

        projector_sq = 0.0
        fusion_sq = 0.0
        token_sq = 0.0
        base_vl_sq = 0.0  # NEW: Track base VL gradients
        overall_sq = 0.0
        projector_params = fusion_params = token_params = base_vl_params = overall_params = 0

        for name, param in self.safe_model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad = param.grad.detach()
            norm = grad.norm(2).item()
            overall_sq += norm ** 2
            overall_params += 1
            lower = name.lower()
            if "audio_projector" in lower:
                projector_sq += norm ** 2
                projector_params += 1
            elif "fusion_adapter" in lower or "lora" in lower:
                fusion_sq += norm ** 2
                fusion_params += 1
            elif "audio_token_embeddings" in lower:
                token_sq += norm ** 2
                token_params += 1
            elif "base_vl" in lower or "vision_encoder" in lower or "language_model" in lower:
                # NEW: Track base VL model gradients (should be ~0 if frozen)
                base_vl_sq += norm ** 2
                base_vl_params += 1

        def safe_sqrt(value: float) -> float:
            return float(math.sqrt(value)) if value > 0 else 0.0

        projector_norm = safe_sqrt(projector_sq)
        fusion_norm = safe_sqrt(fusion_sq)
        token_norm = safe_sqrt(token_sq)
        base_vl_norm = safe_sqrt(base_vl_sq)
        overall_norm = safe_sqrt(overall_sq)

        # Store in metrics for logging to WandB/TensorBoard
        if hasattr(self, 'metrics'):
            self.metrics['grad_norm/projector'] = projector_norm
            self.metrics['grad_norm/fusion'] = fusion_norm
            self.metrics['grad_norm/audio_tokens'] = token_norm
            self.metrics['grad_norm/base_vl'] = base_vl_norm
            self.metrics['grad_norm/overall'] = overall_norm

        # WARNING: Base VL should not have gradients if frozen
        if base_vl_norm > 1e-6 and should_print:
            print(
                f"⚠️ [GradWarning] step {self.global_step}: Base VL has gradients "
                f"(norm={base_vl_norm:.6f}, n={base_vl_params}) - should be frozen!",
                flush=True
            )

        # WARNING: Vanishing gradients
        if projector_norm < 1e-6 and projector_params > 0 and self.global_step > 10:
            print(
                f"⚠️ [GradWarning] step {self.global_step}: Projector gradients vanishing "
                f"(norm={projector_norm:.6e})",
                flush=True
            )
        if fusion_norm < 1e-6 and fusion_params > 0 and self.global_step > 10:
            print(
                f"⚠️ [GradWarning] step {self.global_step}: Fusion gradients vanishing "
                f"(norm={fusion_norm:.6e})",
                flush=True
            )

        if should_print:
            print(
                "[GradNorm] step {}: projector={:.6f} (n={}) fusion={:.6f} (n={}) "
                "audio_token={:.6f} (n={}) base_vl={:.6f} (n={}) overall={:.6f} (n={})".format(
                    self.global_step,
                    projector_norm,
                    projector_params,
                    fusion_norm,
                    fusion_params,
                    token_norm,
                    token_params,
                    base_vl_norm,
                    base_vl_params,
                    overall_norm,
                    overall_params,
                ),
                flush=True,
            )
            self._grad_logs_emitted += 1

    def _sanitize_audio_gradients(self) -> None:
        """Replace NaN/Inf grads on audio modules to keep optimization stable."""
        if not self.config.get("sanitize_nan_gradients", True):
            return

        clip_value = float(self.config.get("grad_sanitize_clip", 1e3))
        sanitized_any = False

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
                sanitized_any = True

        if sanitized_any and self.debug_logging:
            print(
                f"[GradSanitize] step {self.global_step}: sanitized NaN/Inf gradients in audio pathway",
                flush=True,
            )

    def _sanitize_audio_parameters(self) -> None:
        """Clamp/sanitize audio pathway parameters if they ever contain NaNs/Infs."""
        clip_value = float(self.config.get("param_sanitize_clip", 1e3))
        sanitized_any = False

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
                sanitized_any = True

        if sanitized_any and self.debug_logging:
            print(
                f"[ParamSanitize] step {self.global_step}: repaired audio parameters containing NaN/Inf",
                flush=True,
            )

    def _run_periodic_ablation_check(self, inputs: Dict, has_audio: torch.Tensor) -> None:
        """Run periodic ablation check to monitor gate=1.0 vs gate=0.0 performance."""
        audio_indices = torch.nonzero(has_audio, as_tuple=False).flatten()
        if len(audio_indices) == 0:
            return
            
        audio_inputs = self._select_batch_indices(inputs, audio_indices)
        if not audio_inputs or audio_inputs.get("audio_tokens") is None:
            return

        print(f"\n[PeriodicAblation] step {self.global_step}: Checking gate=1.0 vs gate=0.0 performance...", flush=True)
        
        baseline_loss = self._compute_audio_loss(self._clone_inputs(audio_inputs), gate=1.0)
        gate_off_loss = self._compute_audio_loss(self._clone_inputs(audio_inputs), gate=0.0)
        
        improvement = gate_off_loss - baseline_loss if (baseline_loss is not None and gate_off_loss is not None) else None
        
        if improvement is not None:
            print(f"[PeriodicAblation] gate=1.0: {baseline_loss:.4f}, gate=0.0: {gate_off_loss:.4f}, improvement: {improvement:+.4f}", flush=True)
            if improvement < 0:
                print(f"[PeriodicAblation] ✓ Audio fusion is helping! (lower loss with gate=1.0)", flush=True)
            else:
                print(f"[PeriodicAblation] ⚠ Audio fusion still hurting performance", flush=True)
        else:
            print(f"[PeriodicAblation] Could not compute comparison", flush=True)

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

    def _log_update_norms(self, pre_update_params: Dict[str, torch.Tensor]) -> None:
        """Log parameter update norms to track learning progress."""
        projector_sq = 0.0
        fusion_sq = 0.0
        token_sq = 0.0
        overall_sq = 0.0
        projector_params = fusion_params = token_params = overall_params = 0

        for name, param in self.safe_model.named_parameters():
            if not param.requires_grad or name not in pre_update_params:
                continue
            
            # Compute update norm: ||θ_t - θ_{t-1}||
            update = param.data - pre_update_params[name]
            update_norm = update.norm(2).item()
            overall_sq += update_norm ** 2
            overall_params += 1
            
            lower = name.lower()
            if "audio_projector" in lower:
                projector_sq += update_norm ** 2
                projector_params += 1
            elif "fusion_adapter" in lower or "lora" in lower:
                fusion_sq += update_norm ** 2
                fusion_params += 1
            elif "audio_token_embeddings" in lower:
                token_sq += update_norm ** 2
                token_params += 1

        def safe_sqrt(value: float) -> float:
            return float(math.sqrt(value)) if value > 0 else 0.0

        # Removed verbose UpdateDebug logging

    def _verify_trainable_parameters(self) -> None:
        """Verify which parameters are trainable to debug gradient flow issues."""
        audio_trainable = 0
        audio_total = 0
        lora_trainable = 0
        lora_total = 0
        other_trainable = 0
        
        for name, param in self.safe_model.named_parameters():
            lower_name = name.lower()
            
            # Count audio-related parameters
            if any(x in lower_name for x in ["audio_projector", "fusion_adapter", "audio_token"]):
                audio_total += 1
                if param.requires_grad:
                    audio_trainable += 1
                    
            # Count LoRA parameters specifically
            if "lora" in lower_name:
                lora_total += 1
                if param.requires_grad:
                    lora_trainable += 1
                    
            # Count other trainable parameters
            elif param.requires_grad:
                other_trainable += 1
        
        # Only warn about critical issues
        if audio_trainable == 0 and audio_total > 0:
            print(f"CRITICAL: NO AUDIO PARAMETERS TRAINABLE! ({audio_trainable}/{audio_total})", flush=True)
        elif lora_trainable == 0 and lora_total > 0:
            print(f"CRITICAL: NO LORA PARAMETERS TRAINABLE! ({lora_trainable}/{lora_total})", flush=True)

    def _check_gradient_flow(self) -> None:
        """Check if gradients are actually flowing to key parameters."""
        audio_with_grads = 0
        lora_with_grads = 0
        total_audio = 0
        total_lora = 0
        audio_grad_norms = []
        lora_grad_norms = []

        for name, param in self.safe_model.named_parameters():
            if not param.requires_grad:
                continue

            lower_name = name.lower()
            has_grad = param.grad is not None and torch.any(param.grad != 0)

            if any(x in lower_name for x in ["audio_projector", "fusion_adapter"]):
                total_audio += 1
                if has_grad:
                    audio_with_grads += 1
                    grad_norm = param.grad.norm().item()
                    audio_grad_norms.append(grad_norm)

            if "lora" in lower_name:
                total_lora += 1
                if has_grad:
                    lora_with_grads += 1
                    grad_norm = param.grad.norm().item()
                    lora_grad_norms.append(grad_norm)

        # ENHANCED: Always log gradient flow status at critical steps
        if self.global_step < 10 or audio_with_grads == 0 or lora_with_grads == 0:
            avg_audio_norm = sum(audio_grad_norms) / len(audio_grad_norms) if audio_grad_norms else 0.0
            avg_lora_norm = sum(lora_grad_norms) / len(lora_grad_norms) if lora_grad_norms else 0.0
            print(f"[GradFlow] step {self.global_step}: audio={audio_with_grads}/{total_audio} (norm={avg_audio_norm:.2e}), lora={lora_with_grads}/{total_lora} (norm={avg_lora_norm:.2e})", flush=True)

        # CRITICAL: Warn if gradients are completely missing
        if audio_with_grads == 0 and total_audio > 0:
            print(f"❌ CRITICAL: No audio gradients flowing ({audio_with_grads}/{total_audio}) - audio won't learn!", flush=True)
        elif lora_with_grads == 0 and total_lora > 0:
            print(f"❌ WARNING: No LoRA gradients flowing ({lora_with_grads}/{total_lora})", flush=True)

    def _prepare_fisher_batch(self, batch: Dict[str, Any], device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenize a raw curriculum batch for Fisher information computation."""

        questions = batch.get("questions")
        if not questions:
            return None

        inputs = self.safe_model.prepare_multimodal_inputs(
            text=questions,
            images=batch.get("images"),
            audio=None,
            answers=batch.get("answers"),
            device=device,
            include_audio_tokens=False,
            training_mode=True,
        )

        fisher_inputs: Dict[str, torch.Tensor] = {}
        for key in ("input_ids", "attention_mask", "labels", "pixel_values"):
            value = inputs.get(key)
            if isinstance(value, torch.Tensor):
                fisher_inputs[key] = value.detach()

        if "input_ids" not in fisher_inputs:
            return None

        return fisher_inputs

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

        # Verify trainable parameters (debug gradient flow issues)
        if self.global_step % 10 == 0 or self.global_step < 5:  # Check every 10 steps or first 5 steps
            self._verify_trainable_parameters()

        if self._micro_step == 0:
            self.optimizer.zero_grad(set_to_none=True)

        step_timer_start = None
        if self.debug_logging:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_timer_start = time.time()
            batch_size_dbg = len(batch.get("questions", [])) if isinstance(batch, dict) else None
            print(
                f"[TrainDebug] step {self.global_step}: received batch size={batch_size_dbg}",
                flush=True,
            )
        
        # Move batch to device
        device = next(self.safe_model.parameters()).device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Validate training batch composition
        batch_size = len(batch.get("questions", []))
        
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

        # Create input tensors for training - apply answers for supervised learning
        inputs = self.safe_model.prepare_multimodal_inputs(
            text=batch["questions"],
            images=batch.get("images", None),
            audio=batch.get("audio", None),
            answers=batch.get("answers", None),
            device=device,
            training_mode=True  # Apply answers during training
        )

        # Get processed inputs
        audio_tokens = inputs.get("audio_tokens", None)
        labels = inputs.get("labels", None)

        # Drive fusion gate warmup using micro-step aware schedule
        if hasattr(self.safe_model, "set_gate_warmup"):
            warmup_steps = int(self.config.get("gate_warmup_steps", self.warmup_steps))
            warmup_steps = max(1, warmup_steps)
            effective_step = self.global_step * self.grad_accum_steps + self._micro_step
            self.safe_model.set_gate_warmup(effective_step, warmup_steps)

        if self.debug_logging:
            input_shape = None
            if isinstance(inputs.get("input_ids"), torch.Tensor):
                input_shape = tuple(inputs["input_ids"].shape)
            print(
                f"[TrainDebug] step {self.global_step}: prepared inputs input_ids={input_shape} has_audio={int(has_audio.sum().item())}",
                flush=True,
            )

        self._run_audio_sanity_checks(inputs, has_audio)

        # Optional debug: Check multimodal input preparation (disabled for clean logs)
        # print(f"DEBUG: After prepare_multimodal_inputs, keys: {inputs.keys()}")
        # print(f"DEBUG: Has pixel_values: {'pixel_values' in inputs}")

        # Forward pass through SAFE model
        safe_outputs = self.safe_model(**inputs)
        self._log_attention_probe()
        if self.debug_logging:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_safe = time.time() - step_timer_start if step_timer_start is not None else 0.0
            print(
                f"[TrainDebug] step {self.global_step}: SAFE forward took {elapsed_safe:.2f}s",
                flush=True,
            )
        
        safe_logits = safe_outputs.get("logits") if isinstance(safe_outputs, dict) else getattr(safe_outputs, "logits", None)
        base_outputs = None

        # Forward pass through base VL model (for retention loss)
        if self.combined_loss.retention_enabled:
            with torch.no_grad():
                sanitized_ids = self._sanitize_input_ids_batch(inputs.get("input_ids"))
                sanitized_labels = self.safe_model.sanitize_labels_for_base(
                    inputs.get("labels")
                )
                base_inputs = {
                    "attention_mask": inputs["attention_mask"],
                }
                if sanitized_ids is not None:
                    base_inputs["input_ids"] = sanitized_ids
                elif "inputs_embeds" in inputs:
                    base_inputs["inputs_embeds"] = inputs["inputs_embeds"]
                if sanitized_labels is not None:
                    base_inputs["labels"] = sanitized_labels
                if self.safe_model.base_vl.model_type in ["llava", "blip2"] and "pixel_values" in inputs:
                    base_inputs["pixel_values"] = inputs["pixel_values"]

                if self.safe_model.base_vl.model_type == "llava":
                    # Debug: Log input shapes before calling teacher
                    if not getattr(self, "_train_debug_logged", False):
                        print(f"[TrainDebug] Base inputs keys: {list(base_inputs.keys())}", flush=True)
                        if "input_ids" in base_inputs:
                            print(f"[TrainDebug] Base input_ids shape: {base_inputs['input_ids'].shape}", flush=True)
                        if "pixel_values" in base_inputs:
                            print(f"[TrainDebug] Base pixel_values shape: {base_inputs['pixel_values'].shape}", flush=True)
                        self._train_debug_logged = True

                    base_outputs = self._forward_llava_teacher(base_inputs)

                    # Debug: Log output shapes
                    if not getattr(self, "_train_output_logged", False):
                        print(f"[TrainDebug] Teacher output logits shape: {base_outputs['logits'].shape}", flush=True)
                        self._train_output_logged = True

                elif self.safe_model.base_vl.model_type == "blip2":
                    # BLIP-2: use directly without splitting
                    base_outputs = self.safe_model.base_vl(**base_inputs)
                else:
                    # For custom models, use vision features
                    base_inputs["vision_features"] = inputs.get("vision_features")
                    base_outputs = self.safe_model.base_vl(**base_inputs)

                if not self._shape_debug_once:
                    input_shape = None if inputs.get("input_ids") is None else tuple(inputs["input_ids"].shape)
                    sanitized_shape = None if base_inputs.get("input_ids") is None else tuple(base_inputs["input_ids"].shape)
                    print(f"[ShapeDebug] SAFE input_ids: {input_shape}; teacher input_ids: {sanitized_shape}", flush=True)
                    self._shape_debug_once = True

                teacher_logits = base_outputs.get("logits") if isinstance(base_outputs, dict) else getattr(base_outputs, "logits", None)
                if safe_logits is not None and teacher_logits is not None:
                    trim_len = min(safe_logits.size(1), teacher_logits.size(1))
                    if safe_logits.size(1) != trim_len:
                        trimmed_student = safe_logits[:, -trim_len:, :]
                        if isinstance(safe_outputs, dict):
                            safe_outputs["logits"] = trimmed_student
                        else:
                            safe_outputs.logits = trimmed_student
                        safe_logits = trimmed_student
                        if inputs.get("labels") is not None and inputs["labels"].size(1) != trim_len:
                            inputs["labels"] = inputs["labels"][:, -trim_len:]
                        if inputs.get("attention_mask") is not None and inputs["attention_mask"].size(1) != trim_len:
                            inputs["attention_mask"] = inputs["attention_mask"][:, -trim_len:]
                    if teacher_logits.size(1) != trim_len:
                        trimmed_teacher = teacher_logits[:, -trim_len:, :]
                        if isinstance(base_outputs, dict):
                            base_outputs["logits"] = trimmed_teacher
                        else:
                            base_outputs.logits = trimmed_teacher
                        teacher_logits = trimmed_teacher

                if safe_logits is not None and teacher_logits is not None and teacher_logits.size(0) != safe_logits.size(0):
                    if teacher_logits.size(0) == 1:
                        if not getattr(self, "_teacher_broadcast_warned", False):
                            print(
                                f"[Warn] Teacher batch={teacher_logits.size(0)} but student batch={safe_logits.size(0)}; expanding teacher logits",
                                flush=True
                            )
                            self._teacher_broadcast_warned = True
                        teacher_logits = teacher_logits.expand_as(safe_logits)
                        if isinstance(base_outputs, dict):
                            base_outputs["logits"] = teacher_logits
                    else:
                        raise RuntimeError(
                            f"Teacher/student batch mismatch: {teacher_logits.size(0)} vs {safe_logits.size(0)}"
                        )
        else:
            base_outputs = {"logits": safe_logits.detach() if safe_logits is not None else None}
        if self.debug_logging and self.combined_loss.retention_enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_teacher = time.time() - step_timer_start if step_timer_start is not None else 0.0
            print(
                f"[TrainDebug] step {self.global_step}: teacher forward cumulative {elapsed_teacher:.2f}s",
                flush=True,
            )

        # Extract current parameters for Fisher regularization (if enabled)
        safe_params = None
        base_params = None
        if self.config.get("fisher_weight", 0.0) > 0 and hasattr(self, 'initial_params_snapshot'):
            safe_params = {
                name: param.data
                for name, param in self.trainable_params.items()
            }
            base_params = self.initial_params_snapshot

        # Compute combined loss
        loss_dict = self.combined_loss(
            safe_outputs=safe_outputs,
            base_outputs=base_outputs,
            batch=inputs,
            has_audio=has_audio,
            safe_model_params=safe_params,
            base_model_params=base_params
        )

        log_metrics: Dict[str, float] = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                log_metrics[key] = float(value.detach().cpu().item())
            else:
                log_metrics[key] = float(value)

        train_metrics: Dict[str, float] = {}
        interval = int(self.config.get("train_accuracy_interval", 0) or 0)
        warmup = int(self.config.get("train_accuracy_warmup", 0) or 0)
        step_index = self.global_step + 1
        should_compute = interval > 0 and (step_index <= max(1, warmup) or step_index % interval == 0)
        
        if self.debug_logging and interval > 0:
            print(f"[TrainAccDebug] step {step_index}: interval={interval}, warmup={warmup}, should_compute={should_compute}", flush=True)
        
        if should_compute:
            try:
                prev_mode = self.safe_model.training
                with torch.no_grad():
                    self.safe_model.eval()
                    safe_results, _ = self._compute_robust_accuracy(
                        safe_outputs=safe_outputs,
                        base_outputs=base_outputs,
                        inputs=inputs,
                        has_audio=has_audio,
                        batch=batch,
                    )
                if prev_mode:
                    self.safe_model.train()
                    self.safe_model.enable_audio_training()

                has_audio_cpu = has_audio.detach().to(device="cpu")
                audio_indices = has_audio_cpu.nonzero(as_tuple=False).view(-1).tolist()
                vl_indices = (~has_audio_cpu).nonzero(as_tuple=False).view(-1).tolist()

                if audio_indices:
                    audio_scores = [safe_results[i].score for i in audio_indices]
                    train_metrics["train_audio_accuracy"] = float(np.mean(audio_scores)) if audio_scores else 0.0
                    token_f1_scores = [safe_results[i].metrics.get("token_f1", 0.0) for i in audio_indices]
                    if token_f1_scores:
                        train_metrics["train_audio_token_f1"] = float(np.mean(token_f1_scores))
                    bert_scores = [safe_results[i].metrics.get("bertscore", 0.0) for i in audio_indices]
                    if bert_scores:
                        train_metrics["train_audio_bertscore"] = float(np.mean(bert_scores))
                if vl_indices:
                    vl_scores = [safe_results[i].score for i in vl_indices]
                    train_metrics["train_vl_accuracy"] = float(np.mean(vl_scores)) if vl_scores else 0.0

                if safe_results:
                    train_metrics["train_overall_accuracy"] = float(
                        np.mean([res.score for res in safe_results])
                    )

                if train_metrics:
                    metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in train_metrics.items())
                    print(f"[TrainAcc] step {step_index}: {metrics_str}", flush=True)
            except Exception as exc:  # pragma: no cover - best effort logging
                print(f"[TrainAcc] Warning: failed to compute training accuracy ({exc})", flush=True)

        log_metrics.update(train_metrics)

        # Debug: Log loss components
        audio_loss = loss_dict.get('audio_task_loss', 0.0)
        retention_loss = loss_dict.get('retention_loss', 0.0)
        total_loss = loss_dict.get('total_loss', 0.0)
        audio_val = audio_loss.item() if isinstance(audio_loss, torch.Tensor) else audio_loss
        retention_val = retention_loss.item() if isinstance(retention_loss, torch.Tensor) else retention_loss
        total_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        # Update EMA for audio loss learning curves
        if audio_val > 0:  # Only track when we have audio loss
            if self.audio_loss_ema is None:
                self.audio_loss_ema = audio_val
            else:
                self.audio_loss_ema = self.ema_decay * self.audio_loss_ema + (1 - self.ema_decay) * audio_val
            
            # Run ablation every 100 steps
            
            # Run ablation check periodically
            if (self.global_step - self.last_ablation_step >= self.ablation_check_interval and 
                self.global_step > 0 and torch.any(has_audio)):
                self._run_periodic_ablation_check(inputs, has_audio)
                self.last_ablation_step = self.global_step
        if self.debug_logging:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(
                f"[TrainDebug] step {self.global_step}: combined loss total={loss_dict['total_loss'].item():.4f}",
                flush=True,
            )
            if 'audio_task_loss' in loss_dict:
                audio_val = loss_dict['audio_task_loss']
                audio_val = audio_val.item() if isinstance(audio_val, torch.Tensor) else audio_val
            else:
                audio_val = None
            if 'retention_loss' in loss_dict:
                retention_val = loss_dict['retention_loss']
                retention_val = retention_val.item() if isinstance(retention_val, torch.Tensor) else retention_val
            else:
                retention_val = None
            print(
                f"[TrainDebug] step {self.global_step}: components audio={audio_val} retention={retention_val}",
                flush=True,
            )

        if self._should_log_sample():
            try:
                self._log_sample_predictions(
                    safe_outputs=safe_outputs,
                    base_outputs=base_outputs,
                    inputs=inputs,
                    batch=batch,
                    context="train"
                )
            except Exception as exc:
                self.logger.debug(f"Sample logging skipped due to error: {exc}")
        
        # DEBUG: Print detailed loss information (enabled for plateau investigation)
        total_loss = loss_dict["total_loss"]
        if self.global_step % 50 == 0:  # Log every 50 steps
            print(f"\n=== LOSS DEBUG STEP {self.global_step} ===", flush=True)
            print(f"Total Loss: {total_loss.item():.6f}", flush=True)
            print(f"Loss breakdown:", flush=True)
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.item():.6f}", flush=True)
                else:
                    print(f"  {key}: {value:.6f}", flush=True)
            print(f"Has audio samples: {torch.sum(has_audio).item()}/{len(has_audio)}", flush=True)
            print(f"Batch size: {len(inputs.get('input_ids', []))}", flush=True)
            print("="*30, flush=True)
        
        # Backward pass with gradient accumulation support
        loss_scale = 1.0 / float(self.grad_accum_steps)
        (total_loss * loss_scale).backward()

        # Clean up NaN/Inf gradients from audio pathway before diagnostics
        self._sanitize_audio_gradients()

        self._micro_step += 1
        ready_to_step = self._micro_step >= self.grad_accum_steps

        if ready_to_step:
            # Optional additional parameter sanitization before inspection
            self._sanitize_audio_parameters()

            # Verify gradients are flowing (debug)
            if self.global_step % 5 == 0 or self.global_step < 3:
                self._check_gradient_flow()

            self._log_grad_norms()

            if self.null_space_projector is not None:
                self.null_space_projector.observe(step=self.global_step, has_audio=has_audio)
                self.null_space_projector.project()

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

            # Log parameter update norms
            if self.config.get("log_update_norms", True) and self.global_step % self.config.get("grad_log_interval", 1) == 0:
                self._log_update_norms(pre_update_params)

            self._micro_step = 0

            # Learning rate scheduling (fixed: use local_step, assign don't multiply)
            local_step = self.global_step + 1  # caller increments after returning

            if local_step <= self.warmup_steps:
                self._apply_warmup(local_step)
            elif self.scheduler is not None:
                self.scheduler.step()

        if self.debug_logging:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_total = time.time() - step_timer_start if step_timer_start is not None else 0.0
            print(
                f"[TrainDebug] step {self.global_step}: total_loss={log_metrics.get('total_loss')} elapsed={elapsed_total:.2f}s",
                flush=True,
            )

        return log_metrics, ready_to_step
    
    def evaluate(
        self,
        max_batches: Optional[int] = None,
        dataloader: Optional[DataLoader] = None,
        description: str = "Validation",
        split_batches: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a specified dataset.

        Args:
            max_batches: Optional limit on number of batches to evaluate
            dataloader: Optional dataloader to evaluate. Defaults to validation loader
            description: Human readable description used in logs
            split_batches: Override whether to perform audio/VL split evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        previous_mode = self.safe_model.training
        self.safe_model.eval()

        data_source = dataloader if dataloader is not None else self.val_dataloader
        if data_source is None:
            raise ValueError("No dataloader available for evaluation")

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
        
        # Metrics for VL retention
        vl_safe_correct = 0
        vl_base_correct = 0
        vl_total = 0
        
        eval_losses = {
            "total_loss": 0.0,
            "audio_task_loss": 0.0,
            "retention_loss": 0.0,
            "distillation_loss": 0.0,
            "fisher_loss": 0.0
        }
        
        device = next(self.safe_model.parameters()).device
        eval_logging_steps = self.config.get("eval_logging_steps", 20)
        
        # Get split batch evaluation parameters
        max_audio_eval_batches = self.config.get("max_audio_eval_batches", 4)
        max_vl_eval_batches = self.config.get("max_vl_eval_batches", 4)
        if split_batches is None:
            split_batches = (max_audio_eval_batches > 0) or (max_vl_eval_batches > 0)
        if not split_batches:
            max_audio_eval_batches = 0
            max_vl_eval_batches = 0

        eval_with_audio_gate = self.config.get("eval_with_audio_gate", True)
        eval_audio_gate_comparison = self.config.get("eval_audio_gate_comparison", False)
        
        # Save current gate state to restore after evaluation
        saved_gate = None
        if hasattr(self.safe_model, 'get_gate'):
            try:
                saved_gate = self.safe_model.get_gate()
                if self.debug_logging:
                    print(f"[EvalGate] Saved current gate state: {saved_gate}", flush=True)
            except:
                saved_gate = 1.0  # Default fallback
                if self.debug_logging:
                    print(f"[EvalGate] Could not read gate, using fallback: {saved_gate}", flush=True)
        elif hasattr(self.safe_model, '_default_gate'):
            saved_gate = self.safe_model._default_gate
            if self.debug_logging:
                print(f"[EvalGate] Saved _default_gate: {saved_gate}", flush=True)
        else:
            saved_gate = 1.0  # Safe default
            if self.debug_logging:
                print(f"[EvalGate] No gate found, using default: {saved_gate}", flush=True)
        
        try:
            with torch.no_grad():
                # Separate audio and VL batches if split evaluation is configured
                if split_batches and (max_audio_eval_batches > 0 or max_vl_eval_batches > 0):
                    print(
                        f"Using split evaluation on {description} set: {max_audio_eval_batches} audio + {max_vl_eval_batches} VL batches",
                        flush=True,
                    )
                    audio_batches = []
                    vl_batches = []

                    # Collect batches and separate them
                    batch_count = 0
                    for batch in data_source:
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
                        audio_count = int(has_audio.sum().item())
                        total_count = int(has_audio.numel())
                        has_audio_any = bool(has_audio.any().item()) if has_audio.numel() > 0 else False
                        print(
                            f"[BatchDebug] Batch {batch_count}: {audio_count}/{total_count} samples have audio, has_audio.any()={has_audio_any}",
                            flush=True,
                        )

                        audio_indices = torch.nonzero(has_audio, as_tuple=False).flatten()
                        vl_indices = torch.nonzero(~has_audio, as_tuple=False).flatten()

                        # Separate into audio vs VL subsets within the mixed batch
                        if audio_indices.numel() > 0 and len(audio_batches) < max_audio_eval_batches:
                            audio_subset = self._select_batch_indices(batch, audio_indices, clone=True)
                            audio_batches.append(audio_subset)
                            print(
                                f"[EvalSplit] Added audio subset ({audio_indices.numel()} samples) {len(audio_batches)}/{max_audio_eval_batches}",
                                flush=True,
                            )
                        elif audio_indices.numel() > 0:
                            print(
                                f"[BatchDebug] Skipping audio samples in batch {batch_count}: already have {len(audio_batches)}/{max_audio_eval_batches} audio batches",
                                flush=True,
                            )

                        if vl_indices.numel() > 0 and len(vl_batches) < max_vl_eval_batches:
                            vl_subset = self._select_batch_indices(batch, vl_indices, clone=True)
                            vl_batches.append(vl_subset)
                            print(
                                f"[EvalSplit] Added VL subset ({vl_indices.numel()} samples) {len(vl_batches)}/{max_vl_eval_batches}",
                                flush=True,
                            )
                        elif vl_indices.numel() > 0 and max_vl_eval_batches > 0:
                            print(
                                f"[BatchDebug] Skipping VL samples in batch {batch_count}: already have {len(vl_batches)}/{max_vl_eval_batches} VL batches",
                                flush=True,
                            )

                        batch_count += 1
                        if len(audio_batches) >= max_audio_eval_batches and len(vl_batches) >= max_vl_eval_batches:
                            break

                    # Create combined evaluation list
                    eval_batch_list = []
                    for i, batch in enumerate(audio_batches):
                        eval_batch_list.append((f"AUDIO-{i+1}", batch))
                    for i, batch in enumerate(vl_batches):
                        eval_batch_list.append((f"VL-{i+1}", batch))

                    batch_iterable = eval_batch_list
                    total_eval_batches = len(eval_batch_list)
                    print(
                        f"Split evaluation prepared {len(audio_batches)} audio + {len(vl_batches)} VL batches (total={total_eval_batches})",
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
                    batch_t0 = time.time()
                    
                    # Handle both split evaluation format (batch_label, batch) and standard format
                    if isinstance(batch_data, tuple) and len(batch_data) == 2:
                        batch_label, batch = batch_data
                        if self.debug_logging:
                            print(
                                f"[EvalDebug] {batch_label} batch {batch_idx}/{total_eval_batches}: processing",
                                flush=True,
                            )
                    else:
                        batch = batch_data
                        batch_label = f"batch"
                        if self.debug_logging:
                            print(
                                f"[EvalDebug] batch {batch_idx}/{total_eval_batches}: loading batch",
                                flush=True,
                            )
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
                    if self.debug_logging:
                        print(
                            f"[EvalDebug] batch {batch_idx}: preparing multimodal inputs (has_audio={int(has_audio.sum().item())})",
                            flush=True,
                        )
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
                    
                    # Debug multimodal input preparation
                    if "attention_mask" in inputs:
                        attn_shape = inputs["attention_mask"].shape
                        attn_sum_per_sample = inputs["attention_mask"].sum(dim=1) if len(attn_shape) > 1 else [inputs["attention_mask"].sum()]
                        # Removed: print(f"[AttentionDebug] attention_mask shape: {attn_shape}, sum per sample: {attn_sum_per_sample.tolist()}", flush=True)
                    if "input_ids" in inputs:
                        input_shape = inputs["input_ids"].shape  
                        # Removed: print(f"[AttentionDebug] input_ids shape: {input_shape}", flush=True)
                    if "audio_tokens" in inputs:
                        audio_shape = inputs["audio_tokens"].shape
                        # Removed: print(f"[AttentionDebug] audio_tokens shape: {audio_shape}", flush=True)
                    
                    # Apply audio gate control if configured
                    if eval_audio_gate_comparison and hasattr(self.safe_model, 'set_gate'):
                        # Run evaluation with both audio gate on/off to measure VL drift
                        if batch_label.startswith("AUDIO"):
                            # For audio batches, use gate=1.0 (with audio)
                            self.safe_model.set_gate(1.0)
                            if self.debug_logging:
                                print(f"[EvalGate] {batch_label}: Using gate=1.0 (audio enabled)", flush=True)
                        else:
                            # For VL batches, use gate=0.0 (without audio) to measure drift
                            self.safe_model.set_gate(0.0)
                            if self.debug_logging:
                                print(f"[EvalGate] {batch_label}: Using gate=0.0 (audio disabled)", flush=True)
                    elif not eval_with_audio_gate and hasattr(self.safe_model, 'set_gate'):
                        # Disable audio entirely if configured
                        self.safe_model.set_gate(0.0)
                        if self.debug_logging:
                            print(f"[EvalGate] {batch_label}: Audio disabled (gate=0.0)", flush=True)
                    
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
                                print(f"[EvalDebug] Removing pixel_values for batch without image tokens: {batch_label}", flush=True)
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
                    if self.debug_logging:
                        print(
                            f"[EvalDebug] {batch_label} batch {batch_idx}: SAFE forward start",
                            flush=True,
                        )
                    safe_outputs = self.safe_model(**inputs)
                    if self.debug_logging:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        print(
                            f"[EvalDebug] batch {batch_idx}: SAFE forward done in {time.time() - batch_t0:.2f}s",
                            flush=True,
                        )
                    
                    # Base VL model forward pass (skip when retention disabled)
                    if self.combined_loss.retention_enabled:
                        teacher_t0 = time.time()
                        sanitized_ids = self._sanitize_input_ids_batch(inputs.get("input_ids"))
                        sanitized_labels = self.safe_model.sanitize_labels_for_base(inputs.get("labels"))
                        base_inputs = {
                            "attention_mask": inputs["attention_mask"],
                        }
                        if sanitized_ids is not None:
                            base_inputs["input_ids"] = sanitized_ids
                        elif "inputs_embeds" in inputs:
                            base_inputs["inputs_embeds"] = inputs["inputs_embeds"]
                        if sanitized_labels is not None:
                            base_inputs["labels"] = sanitized_labels
                        if self.safe_model.base_vl.model_type in ["llava", "blip2"] and "pixel_values" in inputs:
                            base_inputs["pixel_values"] = inputs["pixel_values"]
    
                        if self.safe_model.base_vl.model_type == "llava":
                            base_outputs = self._forward_llava_teacher(base_inputs)
                        elif self.safe_model.base_vl.model_type == "blip2":
                            # BLIP-2: use directly without splitting
                            base_outputs = self.safe_model.base_vl(**base_inputs)
                        else:
                            # For custom models, use vision features
                            base_inputs["vision_features"] = inputs.get("vision_features")
                            base_outputs = self.safe_model.base_vl(**base_inputs)
                        if self.debug_logging:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            print(
                                f"[EvalDebug] batch {batch_idx}: teacher forward done in {time.time() - teacher_t0:.2f}s",
                                flush=True,
                            )
                    else:
                        base_inputs = {}
                        if isinstance(safe_outputs, dict):
                            safe_logits = safe_outputs.get("logits")
                        else:
                            safe_logits = getattr(safe_outputs, "logits", None)
                        base_outputs = {
                            "logits": safe_logits.detach() if safe_logits is not None else None
                        }
    
                    if not getattr(self, "_eval_shape_debug_once", False):
                        input_shape = None if inputs.get("input_ids") is None else tuple(inputs["input_ids"].shape)
                        sanitized_shape = None if base_inputs.get("input_ids") is None else tuple(base_inputs["input_ids"].shape)
                        print(f"[EvalShapeDebug] SAFE input_ids: {input_shape}; teacher input_ids: {sanitized_shape}", flush=True)
                        self._eval_shape_debug_once = True
    
                    student_logits = safe_outputs.get("logits") if isinstance(safe_outputs, dict) else getattr(safe_outputs, "logits", None)
                    teacher_logits = base_outputs.get("logits") if isinstance(base_outputs, dict) else getattr(base_outputs, "logits", None)
                    if student_logits is not None and teacher_logits is not None:
                        trim_len = min(student_logits.size(1), teacher_logits.size(1))
                        if student_logits.size(1) != trim_len:
                            trimmed_student = student_logits[:, -trim_len:, :]
                            if isinstance(safe_outputs, dict):
                                safe_outputs["logits"] = trimmed_student
                            else:
                                safe_outputs.logits = trimmed_student
                            student_logits = trimmed_student
                            if inputs.get("labels") is not None and inputs["labels"].size(1) != trim_len:
                                inputs["labels"] = inputs["labels"][:, -trim_len:]
                            if inputs.get("attention_mask") is not None and inputs["attention_mask"].size(1) != trim_len:
                                inputs["attention_mask"] = inputs["attention_mask"][:, -trim_len:]
                        if teacher_logits.size(1) != trim_len:
                            trimmed_teacher = teacher_logits[:, -trim_len:, :]
                            if isinstance(base_outputs, dict):
                                base_outputs["logits"] = trimmed_teacher
                            else:
                                base_outputs.logits = trimmed_teacher
                            teacher_logits = trimmed_teacher
    
                    if student_logits is not None and teacher_logits is not None and teacher_logits.size(0) != student_logits.size(0):
                        if teacher_logits.size(0) == 1:
                            if not getattr(self, "_teacher_broadcast_warned", False):
                                print(
                                    f"[Warn] Teacher batch={teacher_logits.size(0)} but student batch={student_logits.size(0)}; expanding teacher logits",
                                    flush=True
                                )
                                self._teacher_broadcast_warned = True
                            teacher_logits = teacher_logits.expand_as(student_logits)
                            if isinstance(base_outputs, dict):
                                base_outputs["logits"] = teacher_logits
                        else:
                            raise RuntimeError(
                                f"Teacher/student batch mismatch: {teacher_logits.size(0)} vs {student_logits.size(0)}"
                            )
    
                    # Extract current parameters for Fisher regularization (if enabled)
                    safe_params = None
                    base_params = None
                    if self.config.get("fisher_weight", 0.0) > 0 and hasattr(self, 'initial_params_snapshot'):
                        safe_params = {
                            name: param.data
                            for name, param in self.trainable_params.items()
                        }
                        base_params = self.initial_params_snapshot

                    # Compute losses
                    batch_losses = self.combined_loss(
                        safe_outputs=safe_outputs,
                        base_outputs=base_outputs,
                        batch=inputs,
                        has_audio=has_audio,
                        safe_model_params=safe_params,
                        base_model_params=base_params
                    )
                    
                    # Accumulate losses
                    batch_size = len(batch["questions"])
                    for key in eval_losses:
                        if key in batch_losses:
                            eval_losses[key] += batch_losses[key].item() * batch_size
                    
                    # Compute accuracy metrics using robust methods
                    # Generate predictions and compute answer-level accuracy
                    # Removed verbose progress logging
                    if self.debug_logging:
                        print(
                            f"[EvalDebug] batch {batch_idx}: computing robust accuracy",
                            flush=True,
                        )
                    robust_t0 = time.time()
                    safe_results_batch, base_results_batch = self._compute_robust_accuracy(
                        safe_outputs, base_outputs, inputs, has_audio, batch
                    )
                    # Removed verbose progress logging
                    if self.debug_logging:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        print(
                            f"[EvalDebug] batch {batch_idx}: robust accuracy done in {time.time() - robust_t0:.2f}s",
                            flush=True,
                        )
                        if safe_results_batch:
                            print(
                                f"[EvalDebug] batch {batch_idx}: sample SAFE acc={safe_results_batch[0].score:.3f}, base acc={base_results_batch[0].score:.3f}",
                                flush=True,
                            )

                    if self._should_log_sample():
                        try:
                            self._log_sample_predictions(
                                safe_outputs=safe_outputs,
                                base_outputs=base_outputs,
                                inputs=inputs,
                                batch=batch,
                                safe_accuracies=[res.score for res in safe_results_batch],
                                base_accuracies=[res.score for res in base_results_batch],
                                context="eval"
                            )
                        except Exception as exc:
                            self.logger.debug(f"Sample logging skipped during eval due to error: {exc}")

                    # Audio-dependent accuracy
                    if torch.any(has_audio):
                        audio_indices = torch.where(has_audio)[0]
                        if len(audio_indices) > 0:
                            audio_correct += sum(safe_results_batch[int(i)].score for i in audio_indices)
                            audio_exact += sum(safe_results_batch[int(i)].metrics.get("exact", 0.0) for i in audio_indices)
                            audio_token_f1 += sum(safe_results_batch[int(i)].metrics.get("token_f1", 0.0) for i in audio_indices)
                            audio_bertscore += sum(safe_results_batch[int(i)].metrics.get("bertscore", 0.0) for i in audio_indices)
                            audio_total += len(audio_indices)
                            audio_samples += len(audio_indices)

                    # VL retention accuracy  
                    vl_indices = torch.where(~has_audio)[0] if torch.any(~has_audio) else torch.arange(len(has_audio))
                    if len(vl_indices) > 0:
                        vl_safe_correct += sum(safe_results_batch[int(i)].score for i in vl_indices)
                        vl_base_correct += sum(base_results_batch[int(i)].score for i in vl_indices)
                        vl_total += len(vl_indices)
                        vl_samples += len(vl_indices)
                    
                    total_samples += batch_size
    
                    if eval_logging_steps and (
                        batch_idx % eval_logging_steps == 0
                        or (isinstance(total_eval_batches, int) and batch_idx == total_eval_batches)
                    ):
                        print(
                            f"[Eval] Processed {batch_idx}/{total_eval_batches if isinstance(total_eval_batches, int) else '?'} batches",
                            flush=True
                        )
                    if self.debug_logging:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        print(
                            f"[EvalDebug] batch {batch_idx}: total time {time.time() - batch_t0:.2f}s",
                            flush=True,
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
                    if self.debug_logging:
                        print(f"[EvalGate] Restored gate state to {saved_gate}", flush=True)
                    else:
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
        
        # Retention score (how well SAFE preserves base VL performance)
        retention_score = vl_safe_accuracy / max(vl_base_accuracy, 1e-8)
        
        eval_metrics = {
            **eval_losses,
            "audio_accuracy": audio_accuracy,
            "audio_exact_match": audio_exact_avg,
            "audio_token_f1": audio_token_f1_avg,
            "audio_bertscore": audio_bertscore_avg,
            "vl_safe_accuracy": vl_safe_accuracy,
            "vl_base_accuracy": vl_base_accuracy,
            "retention_score": retention_score,
            "audio_gain": audio_accuracy,  # Simplified audio gain metric
            "audio_samples": audio_samples,
            "vl_samples": vl_samples,
            "total_samples": total_samples
        }
        
        # Report cumulative evaluation results
        print(f"\n=== {description.upper()} EVALUATION COMPLETE ===", flush=True)
        if split_batches and (max_audio_eval_batches > 0 or max_vl_eval_batches > 0):
            print(f"Split Evaluation Summary:", flush=True)
            print(f"  Audio batches: {max_audio_eval_batches}, samples: {audio_samples}", flush=True)
            print(f"  VL batches: {max_vl_eval_batches}, samples: {vl_samples}", flush=True)
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
            print(f"  📈 Retention Score: {retention_score:.3f} (SAFE/Base ratio)", flush=True)
            
            # Calculate VL drift if we have both audio and VL evaluation
            if eval_audio_gate_comparison and audio_total > 0 and vl_total > 0:
                vl_drift = (1.0 - retention_score) * 100
                print(f"  ⚠️  VL Drift: {vl_drift:+.1f}% (performance change with audio enabled)", flush=True)
        
        print(f"==============================\n", flush=True)

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
                # Debug: Check what we're getting from the preserved answers
                if self.debug_logging and len(gt_answers) > 0:
                    print(f"[AnswerDebug] Batch has {len(gt_answers)} answers", flush=True)
                    print(f"[AnswerDebug] First answer type: {type(gt_answers[0])}", flush=True)
                    print(f"[AnswerDebug] First answer value: '{gt_answers[0]}'", flush=True)
            elif "answers" in batch:
                # Fallback for backwards compatibility
                gt_answers = batch["answers"]
                if self.debug_logging and len(gt_answers) > 0:
                    print(f"[AnswerDebug] Using batch['answers'] (fallback)", flush=True)
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
            safe_pred_tokens = self._generate_predictions(gen_inputs, "safe")
            if self.combined_loss.retention_enabled:
                base_pred_tokens = self._generate_predictions(gen_inputs, "base")
            else:
                base_pred_tokens = [
                    pred.clone() if isinstance(pred, torch.Tensor) else torch.as_tensor(pred)
                    for pred in safe_pred_tokens
                ]
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
                        print(f"  GT: '{gt_display}'", flush=True)
                        print(f"  SAFE_full: '{safe_pred_full}'", flush=True)
                        print(f"  SAFE_pred: '{safe_display}'", flush=True)
                        print(f"  BASE_full: '{base_pred_full}'", flush=True)
                        print(f"  BASE_pred: '{base_display}'", flush=True)
                        print(f"  SAFE_acc: {safe_acc_value}, BASE_acc: {base_acc_value}", flush=True)

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

        normalized = str(text)
        import re
        import unicodedata

        normalized = unicodedata.normalize("NFKC", normalized)
        normalized = normalized.strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.lower()

    def _get_bertscore_metric(self):
        """Lazy load BERTScore metric."""
        if self._bertscore_metric is None and not self._bertscore_failed:
            try:
                import evaluate
                self._bertscore_metric = evaluate.load("bertscore")
                print("✅ BERTScore metric loaded successfully", flush=True)
            except Exception as e:
                self._bertscore_failed = True
                print(f"⚠️  BERTScore unavailable (will use token F1 only): {e}", flush=True)
        return self._bertscore_metric

    def _tokenize_caption(self, text: str) -> List[str]:
        """Tokenize caption into words (keeps duplicates for F1)."""
        if not text:
            return []
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [token for token in text.split() if token]

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
            results = bertscore.compute(
                predictions=predictions,
                references=valid_refs,
                lang="en",
                model_type="distilbert-base-uncased",  # Faster than BERT
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            if not results or "f1" not in results:
                return 0.0
            scores = results["f1"]
            return max(scores) if scores else 0.0
        except Exception as e:
            if not self._bertscore_failed:
                print(f"⚠️  BERTScore computation failed: {e}", flush=True)
                self._bertscore_failed = True
            return 0.0

    def _compute_audio_caption_metrics(self, pred_caption: Any, gt_caption: Any, debug: bool = False) -> Dict[str, float]:
        """Compute granular audio caption metrics and a composite score."""

        pred_norm = self._normalize_audio_caption(pred_caption)
        if not pred_norm:
            return {"composite": 0.0, "exact": 0.0, "token_f1": 0.0, "bertscore": 0.0}

        gt_norms = [self._normalize_audio_caption(ans) for ans in self._prepare_gt_answers(gt_caption)]
        gt_norms = [ans for ans in gt_norms if ans]
        if not gt_norms:
            return {"composite": 0.0, "exact": 0.0, "token_f1": 0.0, "bertscore": 0.0}

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
        }

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

    def _generate_predictions(self, inputs, model_choice="safe"):
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

        # Debug logging with audio token verification
        if self.debug_logging:
            print(f"[GenDebug] Input shape: {input_ids.shape}", flush=True)
            sample_input_text = tok.decode(input_ids[0], skip_special_tokens=True)
            print(f"[GenDebug] Sample input text: '{sample_input_text[:200]}...'", flush=True)
            
        # Removed verbose AUDIO_EVAL logging
        pass

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
            # Pure audio task (AudioCaps) - allow longer generation
            max_new_tokens = configured_max_new_tokens
        elif pixel_present and not has_audio:
            # Pure VL task (VQA) - restrict to short answers
            max_new_tokens = min(configured_max_new_tokens, 10)
        else:
            # Audio-visual or ambiguous - use configured value
            max_new_tokens = configured_max_new_tokens

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,  # CRITICAL FIX: Force at least 1 token to prevent empty generation
            do_sample=False,    # Greedy decoding for deterministic results
            temperature=None,
            top_p=None,
            num_beams=1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=getattr(tok, "eos_token_id", None),
            repetition_penalty=1.1,  # Reduced from 1.8 - was too aggressive
            # Removed no_repeat_ngram_size - was blocking fluent generation
            # Removed encoder_repetition_penalty - was making it worse
            output_scores=False,
            return_dict_in_generate=False
        )

        # EARLY TRAINING FIX: Suppress EOS in first 500 steps to force output
        # The untrained audio features may confuse the model into immediate EOS
        if self.global_step < 500 and getattr(tok, "eos_token_id", None) is not None:
            suppress_list = [tok.eos_token_id]
            # Also suppress pad token if different from EOS
            if getattr(tok, "pad_token_id", None) is not None and tok.pad_token_id != tok.eos_token_id:
                suppress_list.append(tok.pad_token_id)
            gen_kwargs["suppress_tokens"] = suppress_list
            print(f"[GenDebug] Early training (step {self.global_step}): Suppressing EOS tokens {suppress_list}", flush=True)

        # Debug: Verify tokenizer EOS setup
        if self.debug_logging:
            print(f"[GenDebug] Tokenizer EOS: token='{tok.eos_token}', id={tok.eos_token_id}", flush=True)
            print(f"[GenDebug] Tokenizer PAD: token='{tok.pad_token}', id={tok.pad_token_id}", flush=True)
            print(f"[GenDebug] Generation kwargs: {gen_kwargs}", flush=True)

        if model_choice == "safe":
            # Debug: Log input before generation
            if self.debug_logging:
                print(f"[GenDebug] input_ids before generation: {input_ids[0, :20].tolist()}", flush=True)
                print(f"[GenDebug] max_new_tokens: {gen_kwargs.get('max_new_tokens')}", flush=True)

            # CRITICAL DEBUG: Decode the actual prompt being sent to the model
            decoded_prompt = tok.decode(input_ids[0], skip_special_tokens=False)
            print(f"[PromptDebug] First sample prompt (raw): '{decoded_prompt}'", flush=True)
            print(f"[PromptDebug] Input length: {input_ids.shape[1]} tokens", flush=True)

            generated = self.safe_model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device) if isinstance(attention_mask, torch.Tensor) else None,
                pixel_values=pixel_values.to(device) if isinstance(pixel_values, torch.Tensor) else None,
                audio_tokens=audio_tokens.to(device) if isinstance(audio_tokens, torch.Tensor) else None,
                audio_attention_mask=inputs.get("audio_attention_mask"),
                **gen_kwargs,
            )

            # CRITICAL DEBUG: Log generation results to diagnose empty strings
            print(f"[GenDebug] Input shape: {input_ids.shape} → Generated shape: {generated.shape}", flush=True)
            if generated.shape[1] <= input_ids.shape[1]:
                print(f"[GenDebug] ⚠️  WARNING: No new tokens generated! (gen_len={generated.shape[1]} vs input_len={input_ids.shape[1]})", flush=True)
            print(f"[GenDebug] First 10 generated token IDs: {generated[0, :10].tolist()}", flush=True)
            decoded_sample = tok.decode(generated[0], skip_special_tokens=True)
            print(f"[GenDebug] First decoded output: '{decoded_sample}'", flush=True)
        else:
            # BASE model: Use raw base_vl directly to avoid audio_token_embeddings contamination
            # Even with gate=0, SAFE's get_input_embeddings() uses trained audio_token_embeddings
            # which causes BASE to improve over time. True baseline must use frozen base_vl.
            print(f"[GenDebug] BASE generation using raw base_vl (completely frozen)", flush=True)

            # Sanitize input_ids to remove audio tokens
            sanitized_ids = self._sanitize_input_ids_batch(input_ids)
            if sanitized_ids is None:
                sanitized_ids = input_ids

            generated = self.safe_model.base_vl.llm.generate(
                input_ids=sanitized_ids.to(device),
                attention_mask=attention_mask.to(device) if isinstance(attention_mask, torch.Tensor) else None,
                pixel_values=pixel_values.to(device) if isinstance(pixel_values, torch.Tensor) else None,
                **gen_kwargs,
            )

        if not isinstance(generated, torch.Tensor):
            generated = torch.as_tensor(generated)
        generated = generated.to(device)

        # CRITICAL DEBUG: Log raw generation output to diagnose truncation
        print(f"[GenDebug] RAW generated shape: {generated.shape}", flush=True)
        print(f"[GenDebug] RAW first sample all IDs: {generated[0].tolist()}", flush=True)
        print(f"[GenDebug] RAW decoded first sample: '{tok.decode(generated[0])}'", flush=True)
        print(f"[GenDebug] Expected: prompt_len={input_ids.shape[1]} + new_tokens, Got: {generated.shape[1]}", flush=True)

        # Calculate per-sample prompt lengths using attention mask to handle left padding
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            # For left-padded sequences, count actual tokens per sample
            prompt_lengths = attention_mask.sum(dim=1)  # Per-sample actual lengths
        else:
            # Fallback: assume no padding
            prompt_source = sanitized_ids if model_choice == "base" and 'sanitized_ids' in locals() and sanitized_ids is not None else input_ids
            prompt_lengths = torch.full((generated.size(0),), prompt_source.shape[1], 
                                       dtype=torch.long, device=generated.device)

        audio_prefix = audio_tokens.size(1) if isinstance(audio_tokens, torch.Tensor) else 0

        if self.debug_logging:
            print(f"[GenDebug] Generated shape: {generated.shape}", flush=True)
            print(f"[GenDebug] Audio prefix: {audio_prefix}", flush=True)
            print(f"[GenDebug] Per-sample prompt lengths: {prompt_lengths.tolist()}", flush=True)

        # CRITICAL FIX: Check if generation returned truncated output (only new tokens)
        # This happens when using inputs_embeds with some HuggingFace configs
        if generated.shape[1] < input_ids.shape[1]:
            print(f"[GenDebug] ⚠️  UNEXPECTED: Generated sequence is SHORTER than input!", flush=True)
            print(f"[GenDebug] ⚠️  Input: {input_ids.shape}, Generated: {generated.shape}", flush=True)
            print(f"[GenDebug] ⚠️  Treating generated output as NEW TOKENS ONLY (not full sequence)", flush=True)
            # Generation returned ONLY new tokens (unusual but handle it)
            extracted_tokens = [generated[i] for i in range(generated.size(0))]
        else:
            # Normal case: Extract only the newly generated tokens (after prompt) using per-sample lengths
            extracted_tokens = []
            for i in range(generated.size(0)):
                # Use per-sample prompt length to handle left padding correctly
                sample_prompt_len = prompt_lengths[i].item()

                # The attention mask length already includes any audio placeholder tokens
                # that were tokenized as part of the text input. We don't need to adjust
                # for audio_tokens here since they are injected via inputs_embeds and
                # don't affect the text token count.
                start_idx = sample_prompt_len

                if start_idx < generated.shape[1]:
                    new_tokens = generated[i, start_idx:]
                    decoded_new = self.safe_model.base_vl.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    if len(new_tokens) == 0 or not decoded_new.strip():
                        print(f"[GenDebug] ⚠️  Sample {i}: Extracted {len(new_tokens)} tokens but decoded to empty: '{decoded_new}'", flush=True)
                    extracted_tokens.append(new_tokens)
                else:
                    print(f"[GenDebug] ❌ Sample {i}: start_idx={start_idx} >= gen_len={generated.shape[1]}, returning empty!", flush=True)
                    print(f"[GenDebug] ❌ This means no new tokens were generated (model hit EOS or generation failed)", flush=True)
                    extracted_tokens.append(torch.tensor([], dtype=torch.long, device=generated.device))
        
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
    
    def _should_log_sample(self) -> bool:
        if self.sample_logs_emitted >= self.sample_log_limit:
            return False
        if self.global_step == 0 and self.sample_logs_emitted == 0:
            return True
        return (self.global_step % self.sample_log_interval) == 0

    def _log_sample_predictions(
        self,
        safe_outputs,
        base_outputs,
        inputs,
        batch,
        safe_accuracies: Optional[List[float]] = None,
        base_accuracies: Optional[List[float]] = None,
        context: str = "eval",
    ) -> None:
        if self.sample_logs_emitted >= self.sample_log_limit:
            return

        try:
            batch_size = len(batch.get("questions", inputs.get("input_ids", [])))
            questions = batch.get("questions", ["<no question>"] * batch_size)

            gt_answers: List[str] = []
            if "answers" in batch and batch["answers"] is not None:
                gt_answers = [self._select_training_answer(ans) for ans in batch["answers"]]
            elif "labels" in inputs:
                labels = inputs["labels"]
                tokenizer = self.safe_model.base_vl.tokenizer
                for i in range(batch_size):
                    label_tokens = labels[i][labels[i] != -100]
                    if len(label_tokens) > 0:
                        try:
                            gt_answers.append(tokenizer.decode(label_tokens, skip_special_tokens=True).strip())
                        except Exception:
                            gt_answers.append("<decode_error>")
                    else:
                        gt_answers.append("")

            gen_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            try:
                safe_sequences = self._generate_predictions(gen_inputs, "safe")
                if self.combined_loss.retention_enabled:
                    base_sequences = self._generate_predictions(gen_inputs, "base")
                else:
                    base_sequences = [
                        seq.clone() if isinstance(seq, torch.Tensor) else torch.as_tensor(seq)
                        for seq in safe_sequences
                    ]
            except Exception as exc:
                self.logger.debug(f"Generation for logging failed: {exc}")
                safe_sequences = torch.argmax(safe_outputs["logits"], dim=-1)
                base_sequences = torch.argmax(base_outputs["logits"], dim=-1)

            tokenizer = self.safe_model.base_vl.tokenizer

            self.logger.info("[SampleLog][%s] step=%s epoch=%s batch_size=%s", context, self.global_step, self.epoch, batch_size)

            num_examples = min(batch_size, self.sample_log_examples)
            for idx in range(num_examples):
                question = textwrap.shorten(str(questions[idx]), width=160, placeholder="...")
                gt = textwrap.shorten(gt_answers[idx] if idx < len(gt_answers) else "", width=120, placeholder="...")

                safe_text = self._decode_token_sequence(safe_sequences, idx, tokenizer)
                base_text = self._decode_token_sequence(base_sequences, idx, tokenizer)

                safe_acc = safe_accuracies[idx] if safe_accuracies and idx < len(safe_accuracies) else None
                base_acc = base_accuracies[idx] if base_accuracies and idx < len(base_accuracies) else None

                self.logger.info(
                    "[SampleLog][%s] #%d Q: %s | GT: %s | SAFE: %s%s | BASE: %s%s",
                    context,
                    idx + 1,
                    question,
                    gt,
                    textwrap.shorten(safe_text, width=120, placeholder="..."),
                    f" (acc={safe_acc:.2f})" if safe_acc is not None else "",
                    textwrap.shorten(base_text, width=120, placeholder="..."),
                    f" (acc={base_acc:.2f})" if base_acc is not None else "",
                )

            self.sample_logs_emitted += 1
        except Exception as exc:
            self.logger.debug(f"Sample logging error: {exc}")

    def _decode_token_sequence(self, sequences, index: int, tokenizer) -> str:
        try:
            if isinstance(sequences, list) and index < len(sequences):
                return tokenizer.decode(sequences[index], skip_special_tokens=True).strip()
            if isinstance(sequences, torch.Tensor):
                tokens = sequences[index]
                if tokens.dim() > 1:
                    tokens = tokens.argmax(dim=-1)
                return tokenizer.decode(tokens, skip_special_tokens=True).strip()
        except Exception:
            pass
        return ""
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
                    f"checkpoint_epoch_{self.epoch}_step_{self.global_step}.pt"
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
    
    def _save_initial_parameters(self):
        """
        Save initial snapshot of trainable parameters for Fisher regularization.

        Fisher regularization penalizes changes to parameters, so we need a reference
        point. In SAFE, we track the fusion adapter and audio projector parameters
        to prevent them from degrading VL performance.
        """
        if self.config.get("fisher_weight", 0.0) <= 0:
            return

        self.initial_params_snapshot = {}
        for name, param in self.trainable_params.items():
            self.initial_params_snapshot[name] = param.data.clone().detach()

        print(f"✓ Saved initial parameter snapshot: {len(self.initial_params_snapshot)} parameters for Fisher regularization", flush=True)

    def _compute_fisher_information(self):
        """
        Compute Fisher information matrix for trainable parameters.

        Fisher information measures parameter importance based on gradient variance
        on VL-only tasks. This helps constrain fusion adapter changes that might
        hurt VL performance.
        """
        if not self.config.get("compute_fisher_at_start", False):
            return
        if self.config.get("fisher_weight", 0.0) <= 0:
            print("Skipping Fisher computation (fisher_weight=0)", flush=True)
            return

        if not self.trainable_params:
            print("Skipping Fisher computation (no trainable parameters)", flush=True)
            return

        print("🔍 Computing Fisher information matrix for trainable parameters (fusion adapter, projector, etc.)...", flush=True)
        target_samples = int(self.config.get("fisher_num_samples", 1000))
        device = next(self.safe_model.parameters()).device
        prepared_batches: List[Dict[str, torch.Tensor]] = []
        collected = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                fisher_batch = self._prepare_fisher_batch(batch, device)
                if fisher_batch is None:
                    continue
                prepared_batches.append(fisher_batch)
                collected += fisher_batch["input_ids"].size(0)
                if collected >= target_samples:
                    break

        if not prepared_batches:
            print("⚠️ Fisher computation skipped: unable to prepare validation batches", flush=True)
            return

        try:
            fisher_info = self.retention_loss.compute_fisher_information(
                self.safe_model,
                prepared_batches,
                num_samples=collected,
                param_names=list(self.trainable_params.keys())
            )
            print(f"✓ Fisher information computed from {collected} samples for {len(self.trainable_params)} parameters", flush=True)
        except Exception as e:
            print(f"⚠️ Fisher computation failed: {e}", flush=True)
            print("   Continuing without Fisher information", flush=True)

    def _train_with_curriculum(self):
        """Training loop with curriculum learning."""
        print("🎓 Starting Stage A training with curriculum learning...", flush=True)
        print(f"Curriculum has {self.curriculum_manager.config.get_num_stages()} stages", flush=True)

        # Save initial parameter snapshot for Fisher regularization
        self._save_initial_parameters()

        # Compute Fisher information if enabled
        self._compute_fisher_information()

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

                    # Update progress logs using most recent window of metrics
                    if epoch_losses["total_loss"] and (step + 1) % self.config.get("logging_steps", 50) == 0:
                        avg_loss = np.mean(epoch_losses["total_loss"][-10:])
                        print(
                            f"Stage {stage_name} | Epoch {epoch_in_stage} Step {step+1}: loss={avg_loss:.4f}",
                            flush=True
                        )

                    if self.global_step % self.config["logging_steps"] == 0:
                        avg_losses = {k: np.mean(v[-self.config["logging_steps"]:])
                                      for k, v in epoch_losses.items() if v}
                        if avg_losses:
                            summary_parts = [f"{k}={val:.4f}" for k, val in avg_losses.items()]
                            print(
                                f"[Train|Stage {stage_name}] Global step {self.global_step}: " + ", ".join(summary_parts),
                                flush=True
                            )
                        else:
                            print(
                                f"[Train|Stage {stage_name}] Global step {self.global_step}: metrics pending",
                                flush=True
                            )

                        # Log to wandb if available
                        try:
                            log_dict = {"train/" + k: v for k, v in avg_losses.items()}
                            log_dict.update({
                                "curriculum/stage": stage_idx,
                                "curriculum/stage_name": stage_name,
                                "curriculum/samples_in_stage": samples_in_stage
                            })
                            wandb.log(log_dict, step=self.global_step)
                        except:
                            pass
            
            # End of epoch evaluation
            print(f"\n📊 End of Epoch {self.epoch} (Stage {stage_name}, Epoch {epoch_in_stage})", flush=True)
            max_eval_batches = self.config.get("max_eval_batches", None)

            # eval every n epochs
            if self.epoch % 10 == 0:
                eval_metrics = self.evaluate(max_batches=max_eval_batches)
            
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
            print(f"    Retention Loss: {eval_metrics['retention_loss']:.4f}", flush=True)
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

        # Save initial parameter snapshot for Fisher regularization
        self._save_initial_parameters()

        # Compute Fisher information if enabled
        self._compute_fisher_information()

        # Compute baseline retention score
        print("Computing baseline metrics...", flush=True)
        max_eval_batches = self.config.get("max_eval_batches", None)
        baseline_metrics = self.evaluate(max_batches=max_eval_batches)
        baseline_retention = baseline_metrics["retention_score"]
        print(f"Baseline retention score: {baseline_retention:.4f}", flush=True)
        if max_eval_batches:
            print(f"(Evaluation limited to {max_eval_batches} batches)", flush=True)
        
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
                    print(
                        f"First batch completed successfully! Loss: {step_losses.get('total_loss', 'N/A'):.4f}",
                        flush=True,
                    )

                # Accumulate losses
                for key, value in step_losses.items():
                    epoch_losses[key].append(float(value))

                if optimizer_stepped:
                    self.global_step += 1

                    # Update progress bar
                    if epoch_losses["total_loss"] and (step + 1) % self.config.get("logging_steps", 50) == 0:
                        avg_loss = np.mean(epoch_losses["total_loss"][-100:])
                        print(f"Epoch {epoch+1} Step {step+1}: loss={avg_loss:.4f}", flush=True)

                    # Logging
                    if self.global_step % self.config["logging_steps"] == 0:
                        avg_losses = {k: np.mean(v[-self.config["logging_steps"]:])
                                      for k, v in epoch_losses.items() if v}
                        if avg_losses:
                            summary_parts = [f"{k}={val:.4f}" for k, val in avg_losses.items()]
                            print(
                                f"[Train] Global step {self.global_step}: " + ", ".join(summary_parts),
                                flush=True,
                            )
                        else:
                            print(f"[Train] Global step {self.global_step}: metrics pending", flush=True)

                        # Log to wandb if available
                        try:
                            wandb.log({
                                "train/" + k: v for k, v in avg_losses.items()
                            }, step=self.global_step)
                        except:
                            pass

                # Evaluation (only trigger on actual optimizer steps)
                if optimizer_stepped and self.global_step % self.config["eval_steps"] == 0:
                    max_eval_batches = self.config.get("max_eval_batches", None)
                    eval_metrics = self.evaluate(max_batches=max_eval_batches)
                    
                    print(f"\nStep {self.global_step} Evaluation:", flush=True)
                    print(f"  Retention Score: {eval_metrics['retention_score']:.4f}", flush=True)
                    print(f"  Audio Accuracy: {eval_metrics['audio_accuracy']:.4f}", flush=True)
                    print(f"  VL Safe Accuracy: {eval_metrics['vl_safe_accuracy']:.4f}", flush=True)
                    print(f"  Total Loss: {eval_metrics['total_loss']:.4f}", flush=True)
                    self.logger.info(
                        "[Eval][step=%s] retention=%.4f audio_acc=%.4f vl_safe=%.4f total_loss=%.4f",
                        self.global_step,
                        eval_metrics['retention_score'],
                        eval_metrics['audio_accuracy'],
                        eval_metrics['vl_safe_accuracy'],
                        eval_metrics['total_loss'],
                    )
                    
                    # Check for improvement
                    current_retention = eval_metrics["retention_score"]
                    is_best = current_retention > self.best_retention_score
                    
                    if is_best:
                        self.best_retention_score = current_retention
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                    
                    # Save checkpoint
                    if self.global_step % self.config["save_steps"] == 0 or is_best:
                        self.save_checkpoint(eval_metrics, is_best)
                    
                    # Early stopping check
                    retention_degradation = baseline_retention - current_retention
                    if retention_degradation > self.config["retention_tolerance"]:
                        print(f"WARNING: Retention degradation ({retention_degradation:.4f}) exceeds tolerance!", flush=True)

                    if self.patience_counter >= self.config["early_stopping_patience"]:
                        print(f"Early stopping after {self.patience_counter} evaluations without improvement", flush=True)
                        return eval_metrics
                    
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
            print(f"Epoch {epoch+1} Results:", flush=True)
            print(f"  Retention Score: {epoch_metrics['retention_score']:.4f}", flush=True)
            print(f"  Audio Accuracy: {epoch_metrics['audio_accuracy']:.4f}", flush=True)
            print(f"  Retention Loss: {epoch_metrics['retention_loss']:.4f}", flush=True)
            self.logger.info(
                "[Eval][epoch=%s] retention=%.4f audio_acc=%.4f retention_loss=%.4f",
                epoch + 1,
                epoch_metrics['retention_score'],
                epoch_metrics['audio_accuracy'],
                epoch_metrics['retention_loss'],
            )
        
        print("Stage A training completed!", flush=True)
        print(f"Best retention score: {self.best_retention_score:.4f}", flush=True)
        
        # Final checkpoint
        max_eval_batches = self.config.get("max_eval_batches", None)
        final_metrics = self.evaluate(max_batches=max_eval_batches)
        self.save_checkpoint(final_metrics, is_best=False)
        return final_metrics
