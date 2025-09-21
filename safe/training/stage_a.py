import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import wandb
import os
from pathlib import Path
import logging
import textwrap

# Set CPU threads for predictable performance
torch.set_num_threads(min(8, os.cpu_count() or 2))

from ..models.safe_model import SAFEModel
from ..models.base_vl import BaseVLModel
from ..data.datasets import create_safe_dataloader
from ..data.curriculum import CurriculumManager, CurriculumConfig, ProgressionStatus
from .losses import RetentionLoss, AudioTaskLoss, CombinedStageLoss
from .null_space import NullSpaceProjector, NullSpaceConfig


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
            print(f"Curriculum learning enabled with {self.curriculum_manager.config.get_num_stages()} stages")
        else:
            self.curriculum_manager = None
            self.use_curriculum = False
            print("Using traditional fixed-epoch training")
        
        # Default configuration
        self.config = {
            "learning_rate_projector": 1e-4,
            "learning_rate_adapter": 5e-5,
            "weight_decay": 0.01,
            "num_epochs": 10 if not self.use_curriculum else None,  # Curriculum controls epochs
            "warmup_steps": 1000,
            "max_grad_norm": 1.0,
            "audio_loss_weight": 1.0,
            "retention_loss_weight": 1.0,
            "distillation_temperature": 3.0,
            "fisher_weight": 0.1,
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
            "null_space_verbose": False
        }

        if config:
            self.config.update(config)

        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        # Create output directory
        Path(self.config["output_dir"]).mkdir(parents=True, exist_ok=True)

        # Initialize loss functions (will be updated by curriculum)
        self._setup_loss_functions()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup scheduler (will be updated by curriculum if needed)
        if self.use_curriculum:
            # Curriculum will manage learning rate
            self.scheduler = None
        else:
            total_steps = len(train_dataloader) * self.config["num_epochs"]
            cold_steps = max(1, total_steps - self.config["warmup_steps"])
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

    def _split_mm_text_batches(self, inputs: dict):
        """
        Split a batch dict into (mm_inputs, text_inputs) where:
          - mm_inputs: rows that contain the <image> token in input_ids; must include pixel_values
          - text_inputs: rows that DO NOT contain the <image> token; must NOT include pixel_values
        Returns (mm_inputs, text_inputs, idx_mm, idx_text) with indices to preserve batch order.
        
        Note: Only applies to LLaVA models. BLIP-2 doesn't use <image> tokens.
        """
        if self.safe_model.base_vl.model_type != "llava":
            # Return "no split" + identity indices to simplify caller logic
            B = inputs["input_ids"].size(0) if "input_ids" in inputs else 0
            idx_full = torch.arange(B, device=inputs["input_ids"].device) if B else None
            return inputs, None, idx_full, None
            
        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            return None, None, None, None
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        image_token_id = self._get_image_token_id()
        has_img_tokens = (input_ids == image_token_id).any(dim=1)  # [B]
        idx_mm = torch.nonzero(has_img_tokens, as_tuple=False).flatten()
        idx_text = torch.nonzero(~has_img_tokens, as_tuple=False).flatten()

        def _subset(t, idx):
            if t is None or idx.numel() == 0:
                return None
            if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.size(0) == input_ids.size(0):
                return t.index_select(0, idx.to(t.device))
            return t

        keys = list(inputs.keys())
        
        def _build_subset(indices, allow_pixels):
            sub = {}
            for k in keys:
                if k == "pixel_values" and not allow_pixels:
                    continue
                v = inputs.get(k, None)
                if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.size(0) == input_ids.size(0):
                    v = _subset(v, indices)
                sub[k] = v
            if not allow_pixels and "pixel_values" in sub:
                sub.pop("pixel_values", None)
            return sub

        mm_inputs = _build_subset(idx_mm, allow_pixels=True) if idx_mm.numel() > 0 else None
        text_inputs = _build_subset(idx_text, allow_pixels=False) if idx_text.numel() > 0 else None

        if mm_inputs is not None and ("pixel_values" not in mm_inputs or mm_inputs["pixel_values"] is None):
            print("Warning: <image> tokens present but no pixel_values; skipping MM branch.")
            return None, inputs, None, torch.arange(input_ids.size(0), device=input_ids.device)

        return mm_inputs, text_inputs, idx_mm, idx_text
    
    def _setup_loss_functions(self):
        """Setup loss functions with curriculum-aware weights."""
        if self.use_curriculum and self.curriculum_manager.current_stage:
            stage_config = self.curriculum_manager.get_current_config()
            loss_weights = stage_config.get("loss_weights", {})
        else:
            loss_weights = {}
            
        self.retention_loss = RetentionLoss(
            temperature=self.config["distillation_temperature"],
            fisher_weight=loss_weights.get("fisher_loss", self.config["fisher_weight"])
        )
        
        self.audio_task_loss = AudioTaskLoss(task_type="qa")
        
        self.combined_loss = CombinedStageLoss(
            retention_loss=self.retention_loss,
            audio_task_loss=self.audio_task_loss,
            audio_weight=loss_weights.get("audio_task_loss", self.config["audio_loss_weight"]),
            retention_weight=loss_weights.get("retention_loss", self.config["retention_loss_weight"])
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
        
        print(f"Updated to curriculum stage: {stage_config['stage_name']}")
        print(f"  Audio ratio: {stage_config['audio_ratio']:.2f}")
        print(f"  Difficulty filter: {stage_config['difficulty_filter']}")
        print(f"  LR multiplier: {lr_multiplier:.2f}")
    
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
        
        if not param_groups:
            raise ValueError("No trainable parameters found!")
        
        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.config["weight_decay"],
            betas=(0.9, 0.999)
        )
        
        # base_lr is already set in parameter groups above
    
    def _apply_warmup(self, step: int):
        """Apply learning rate warmup (assign, don't multiply)."""
        factor = min(1.0, step / max(1, self.config["warmup_steps"]))
        for g in self.optimizer.param_groups:
            base_lr = g.get("base_lr", g["lr"])
            # Ensure base_lr is a number, not a sequence
            if isinstance(base_lr, (list, tuple)):
                print(f"Warning: base_lr is a sequence: {base_lr}, using first element")
                base_lr = base_lr[0]
            g["lr"] = float(base_lr) * factor
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary with loss values
        """
        self.safe_model.train()
        self.safe_model.enable_audio_training()  # Only train audio components
        
        # Move batch to device
        device = next(self.safe_model.parameters()).device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Prepare inputs for SAFE model (device-consistent masks)
        has_audio = batch.get("has_audio", torch.zeros(len(batch["questions"]), dtype=torch.bool, device=device))
        
        # Create input tensors - always pass images/audio if available, don't gate on flags
        inputs = self.safe_model.prepare_multimodal_inputs(
            text=batch["questions"],
            images=batch.get("images", None),
            audio=batch.get("audio", None),
            answers=batch.get("answers", None),
            device=device
        )
        
        # Optional debug: Check multimodal input preparation (disabled for clean logs)
        # print(f"DEBUG: After prepare_multimodal_inputs, keys: {inputs.keys()}")
        # print(f"DEBUG: Has pixel_values: {'pixel_values' in inputs}")
        
        # Forward pass through SAFE model
        safe_outputs = self.safe_model(**inputs)
        
        # Forward pass through base VL model (for retention loss)
        with torch.no_grad():
            base_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": inputs["labels"]
            }
            if self.safe_model.base_vl.model_type in ["llava", "blip2"] and "pixel_values" in inputs:
                base_inputs["pixel_values"] = inputs["pixel_values"]

            if self.safe_model.base_vl.model_type == "llava":
                # Use splitter for LLaVA mixed batches with order preservation
                mm_inputs, text_inputs, idx_mm, idx_text = self._split_mm_text_batches(base_inputs)
                
                def _forward_and_restore(inputs_a, inputs_b, idx_a, idx_b):
                    B = base_inputs["input_ids"].size(0)
                    out_a = self.safe_model.base_vl(**inputs_a) if inputs_a is not None else None
                    out_b = self.safe_model.base_vl(**inputs_b) if inputs_b is not None else None

                    # Pull logits
                    logits_a = (out_a["logits"] if isinstance(out_a, dict) else getattr(out_a, "logits", None)) if out_a else None
                    logits_b = (out_b["logits"] if isinstance(out_b, dict) else getattr(out_b, "logits", None)) if out_b else None

                    # Allocate and restore original order
                    if logits_a is not None or logits_b is not None:
                        ref_logits = logits_a or logits_b
                        T, V = ref_logits.shape[1], ref_logits.shape[2]
                        logits = torch.zeros(B, T, V, device=ref_logits.device, dtype=ref_logits.dtype)
                        if logits_a is not None and idx_a is not None:
                            logits[idx_a] = logits_a
                        if logits_b is not None and idx_b is not None:
                            logits[idx_b] = logits_b
                    else:
                        logits = None

                    # Stitch a minimal outputs object (dict-like so losses can index it)
                    class OutputsWrapper(dict):
                        pass
                    
                    out = OutputsWrapper()
                    out["logits"] = logits  # may be None
                    
                    # Sum losses if they exist
                    if out_a and (hasattr(out_a, "loss") or (isinstance(out_a, dict) and "loss" in out_a)):
                        la = out_a["loss"] if isinstance(out_a, dict) else out_a.loss
                        if la is not None:
                            out["loss"] = la if "loss" not in out else out["loss"] + la
                    
                    if out_b and (hasattr(out_b, "loss") or (isinstance(out_b, dict) and "loss" in out_b)):
                        lb = out_b["loss"] if isinstance(out_b, dict) else out_b.loss
                        if lb is not None:
                            out["loss"] = lb if "loss" not in out else out["loss"] + lb
                    
                    return out

                base_outputs = _forward_and_restore(mm_inputs, text_inputs, idx_mm, idx_text)
            elif self.safe_model.base_vl.model_type == "blip2":
                # BLIP-2: use directly without splitting
                base_outputs = self.safe_model.base_vl(**base_inputs)
            else:
                # For custom models, use vision features
                base_inputs["vision_features"] = inputs.get("vision_features")
                base_outputs = self.safe_model.base_vl(**base_inputs)
        
        # Compute combined loss
        loss_dict = self.combined_loss(
            safe_outputs=safe_outputs,
            base_outputs=base_outputs,
            batch=inputs,
            has_audio=has_audio
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
        
        # DEBUG: Print detailed loss information (disabled for cleaner training)
        total_loss = loss_dict["total_loss"]
        # print(f"\n=== DEBUG STEP {self.global_step} ===")
        # print(f"Total Loss: {total_loss.item():.6f}")
        # print(f"Loss breakdown:")
        # for key, value in loss_dict.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"  {key}: {value.item():.6f}")
        #     else:
        #         print(f"  {key}: {value:.6f}")
        # print(f"Has audio samples: {torch.sum(has_audio).item()}/{len(has_audio)}")
        # print(f"Batch size: {len(inputs.get('input_ids', []))}")
        # print("="*30)
        
        # Backward pass
        total_loss.backward()

        if self.null_space_projector is not None:
            self.null_space_projector.observe(step=self.global_step, has_audio=has_audio)
            self.null_space_projector.project()

        # Gradient clipping
        if self.trainable_param_list:
            torch.nn.utils.clip_grad_norm_(
                self.trainable_param_list,
                self.config["max_grad_norm"]
            )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Learning rate scheduling (fixed: use local_step, assign don't multiply)
        local_step = self.global_step + 1  # because caller increments after train_step returns
        
        if local_step <= self.config["warmup_steps"]:
            self._apply_warmup(local_step)
        elif self.scheduler is not None:
            self.scheduler.step()
        
        # Convert losses to float
        step_losses = {k: v.item() if isinstance(v, torch.Tensor) else v 
                      for k, v in loss_dict.items()}
        
        return step_losses
    
    def evaluate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.safe_model.eval()
        
        # Check if validation dataset has samples
        if len(self.val_dataloader.dataset) == 0:
            print("Warning: Validation dataset is empty!")
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
        
        with torch.no_grad():
            dataloader = self.val_dataloader
            if max_batches:
                from itertools import islice
                dataloader = islice(self.val_dataloader, max_batches)
                desc = f"Evaluating (first {max_batches} batches)"
            else:
                desc = "Evaluating"
            
            for batch in tqdm(dataloader, desc=desc, total=max_batches or len(self.val_dataloader)):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                has_audio = batch.get("has_audio", torch.zeros(len(batch["questions"]), dtype=torch.bool, device=device))
                
                # Prepare inputs - always pass images/audio if available, don't gate on flags
                inputs = self.safe_model.prepare_multimodal_inputs(
                    text=batch["questions"],
                    images=batch.get("images", None),
                    audio=batch.get("audio", None),
                    answers=batch.get("answers", None),
                    device=device
                )
                
                # Handle pixel_values based on model type
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
                            print(f"Warning: Found <image> tokens but no pixel_values. Skipping multimodal processing.")
                elif self.safe_model.base_vl.model_type == "blip2":
                    # BLIP-2 doesn't use <image> tokens, so keep pixel_values as-is if they exist
                    pass
                
                
                # SAFE model forward pass
                safe_outputs = self.safe_model(**inputs)
                
                # Base VL model forward pass
                sanitized_ids = self.safe_model.sanitize_input_ids_for_base(inputs.get("input_ids"))
                base_inputs = {
                    "attention_mask": inputs["attention_mask"],
                    "labels": inputs["labels"]
                }
                if sanitized_ids is not None:
                    base_inputs["input_ids"] = sanitized_ids
                elif "inputs_embeds" in inputs:
                    base_inputs["inputs_embeds"] = inputs["inputs_embeds"]
                if self.safe_model.base_vl.model_type in ["llava", "blip2"] and "pixel_values" in inputs:
                    base_inputs["pixel_values"] = inputs["pixel_values"]

                if self.safe_model.base_vl.model_type == "llava":
                    # Use splitter for LLaVA mixed batches with order preservation
                    mm_inputs, text_inputs, idx_mm, idx_text = self._split_mm_text_batches(base_inputs)
                    
                    def _forward_and_restore(inputs_a, inputs_b, idx_a, idx_b):
                        B = base_inputs["input_ids"].size(0)
                        out_a = self.safe_model.base_vl(**inputs_a) if inputs_a is not None else None
                        out_b = self.safe_model.base_vl(**inputs_b) if inputs_b is not None else None

                        # Pull logits
                        logits_a = (out_a["logits"] if isinstance(out_a, dict) else getattr(out_a, "logits", None)) if out_a else None
                        logits_b = (out_b["logits"] if isinstance(out_b, dict) else getattr(out_b, "logits", None)) if out_b else None

                        # Allocate and restore original order
                        if logits_a is not None or logits_b is not None:
                            ref_logits = logits_a if logits_a is not None else logits_b
                            T, V = ref_logits.shape[1], ref_logits.shape[2]
                            logits = torch.zeros(B, T, V, device=ref_logits.device, dtype=ref_logits.dtype)
                            if logits_a is not None and idx_a is not None:
                                logits[idx_a] = logits_a
                            if logits_b is not None and idx_b is not None:
                                logits[idx_b] = logits_b
                        else:
                            logits = None

                        # Stitch a minimal outputs object (dict-like so losses can index it)
                        class OutputsWrapper(dict):
                            pass
                        
                        out = OutputsWrapper()
                        out["logits"] = logits  # may be None
                        
                        # Sum losses if they exist
                        if out_a and (hasattr(out_a, "loss") or (isinstance(out_a, dict) and "loss" in out_a)):
                            la = out_a["loss"] if isinstance(out_a, dict) else out_a.loss
                            if la is not None:
                                out["loss"] = la if "loss" not in out else out["loss"] + la
                        
                        if out_b and (hasattr(out_b, "loss") or (isinstance(out_b, dict) and "loss" in out_b)):
                            lb = out_b["loss"] if isinstance(out_b, dict) else out_b.loss
                            if lb is not None:
                                out["loss"] = lb if "loss" not in out else out["loss"] + lb
                        
                        return out

                    base_outputs = _forward_and_restore(mm_inputs, text_inputs, idx_mm, idx_text)
                elif self.safe_model.base_vl.model_type == "blip2":
                    # BLIP-2: use directly without splitting
                    base_outputs = self.safe_model.base_vl(**base_inputs)
                else:
                    # For custom models, use vision features
                    base_inputs["vision_features"] = inputs.get("vision_features")
                    base_outputs = self.safe_model.base_vl(**base_inputs)
                
                # Compute losses
                batch_losses = self.combined_loss(
                    safe_outputs=safe_outputs,
                    base_outputs=base_outputs,
                    batch=inputs,
                    has_audio=has_audio
                )
                
                # Accumulate losses
                batch_size = len(batch["questions"])
                for key in eval_losses:
                    if key in batch_losses:
                        eval_losses[key] += batch_losses[key].item() * batch_size
                
                # Compute accuracy metrics using robust methods
                # Generate predictions and compute answer-level accuracy
                safe_accuracy_batch, base_accuracy_batch = self._compute_robust_accuracy(
                    safe_outputs, base_outputs, inputs, has_audio, batch
                )
                
                if self._should_log_sample():
                    try:
                        self._log_sample_predictions(
                            safe_outputs=safe_outputs,
                            base_outputs=base_outputs,
                            inputs=inputs,
                            batch=batch,
                            safe_accuracies=safe_accuracy_batch,
                            base_accuracies=base_accuracy_batch,
                            context="eval"
                        )
                    except Exception as exc:
                        self.logger.debug(f"Sample logging skipped during eval due to error: {exc}")
                
                # Audio-dependent accuracy
                if torch.any(has_audio):
                    audio_indices = torch.where(has_audio)[0]
                    if len(audio_indices) > 0:
                        # Use answer-level accuracy for audio samples (fix tensor indexing)
                        audio_correct += sum(safe_accuracy_batch[int(i)] for i in audio_indices)
                        audio_total += len(audio_indices)
                        audio_samples += len(audio_indices)
                
                # VL retention accuracy  
                vl_indices = torch.where(~has_audio)[0] if torch.any(~has_audio) else torch.arange(len(has_audio))
                if len(vl_indices) > 0:
                    # Use answer-level accuracy for VL samples (fix tensor indexing)
                    vl_safe_correct += sum(safe_accuracy_batch[int(i)] for i in vl_indices)
                    vl_base_correct += sum(base_accuracy_batch[int(i)] for i in vl_indices)
                    vl_total += len(vl_indices)
                    vl_samples += len(vl_indices)
                
                total_samples += batch_size
        
        # Normalize losses (avoid division by zero)
        if total_samples > 0:
            for key in eval_losses:
                eval_losses[key] /= total_samples
        else:
            print("Warning: No validation samples processed!")
            for key in eval_losses:
                eval_losses[key] = 0.0
        
        # Compute accuracy metrics
        audio_accuracy = audio_correct / max(audio_total, 1)
        vl_safe_accuracy = vl_safe_correct / max(vl_total, 1)
        vl_base_accuracy = vl_base_correct / max(vl_total, 1)
        
        # Retention score (how well SAFE preserves base VL performance)
        retention_score = vl_safe_accuracy / max(vl_base_accuracy, 1e-8)
        
        eval_metrics = {
            **eval_losses,
            "audio_accuracy": audio_accuracy,
            "vl_safe_accuracy": vl_safe_accuracy,
            "vl_base_accuracy": vl_base_accuracy,
            "retention_score": retention_score,
            "audio_gain": audio_accuracy,  # Simplified audio gain metric
            "audio_samples": audio_samples,
            "vl_samples": vl_samples,
            "total_samples": total_samples
        }
        
        return eval_metrics
    
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
        safe_accuracies = [0.0] * batch_size
        base_accuracies = [0.0] * batch_size
        
        try:
            # Get ground truth answers
            if "answers" in batch:
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
                return safe_accuracies, base_accuracies
            
            # Generate predictions using proper generation (without labels)
            gen_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            safe_pred_tokens = self._generate_predictions(gen_inputs, "safe")
            base_pred_tokens = self._generate_predictions(gen_inputs, "base")
            
            for i in range(batch_size):
                try:
                    # Check bounds before accessing prediction tokens
                    if i >= len(safe_pred_tokens) or i >= len(base_pred_tokens):
                        safe_accuracies[i] = 0.0
                        base_accuracies[i] = 0.0
                        continue
                    
                    # Decode SAFE model predictions and extract answer
                    safe_pred_full = self.safe_model.base_vl.tokenizer.decode(
                        safe_pred_tokens[i], skip_special_tokens=True
                    ).strip()
                    safe_pred = self._clean_answer(self._extract_answer(safe_pred_full))
                    
                    # Decode base model predictions and extract answer
                    base_pred_full = self.safe_model.base_vl.tokenizer.decode(
                        base_pred_tokens[i], skip_special_tokens=True
                    ).strip()
                    base_pred = self._clean_answer(self._extract_answer(base_pred_full))
                    
                    # Get ground truth with proper cleaning
                    gt_answer = self._clean_answer(gt_answers[i]) if i < len(gt_answers) else ""
                    
                    # Compute answer-level accuracy (exact match or fuzzy match)
                    safe_accuracies[i] = self._compute_answer_accuracy(safe_pred, gt_answer)
                    base_accuracies[i] = self._compute_answer_accuracy(base_pred, gt_answer)
                    
                except Exception as e:
                    # Handle decoding errors gracefully
                    safe_accuracies[i] = 0.0
                    base_accuracies[i] = 0.0
                    
        except Exception as e:
            # Handle any errors gracefully
            print(f"Warning: Error in robust accuracy computation: {e}")
            pass
        
        return safe_accuracies, base_accuracies
    
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
        """Compute VQA-style consensus accuracy for a prediction."""

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

        return min(1.0, matches / 3.0)
    
    def _generate_predictions(self, inputs, model_choice="safe"):
        """Generate predictions using consistent SAFE/base pathways."""
        tok = self.safe_model.base_vl.tokenizer
        if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
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

        gen_kwargs = dict(
            max_new_tokens=8,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=getattr(tok, "eos_token_id", None),
        )

        if model_choice == "safe":
            generated = self.safe_model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device) if isinstance(attention_mask, torch.Tensor) else None,
                pixel_values=pixel_values.to(device) if isinstance(pixel_values, torch.Tensor) else None,
                audio_tokens=audio_tokens.to(device) if isinstance(audio_tokens, torch.Tensor) else None,
                **gen_kwargs,
            )
        else:
            sanitized_ids = self.safe_model.sanitize_input_ids_for_base(input_ids)
            base_kwargs = {
                "attention_mask": attention_mask.to(device) if isinstance(attention_mask, torch.Tensor) else None,
                **gen_kwargs,
            }
            if sanitized_ids is not None:
                base_kwargs["input_ids"] = sanitized_ids.to(device)
            elif inputs.get("inputs_embeds") is not None:
                base_kwargs["inputs_embeds"] = inputs["inputs_embeds"].to(device)
            if isinstance(pixel_values, torch.Tensor):
                base_kwargs["pixel_values"] = pixel_values.to(device)

            generated = self.safe_model.base_vl.llm.generate(**base_kwargs)

        if not isinstance(generated, torch.Tensor):
            generated = torch.as_tensor(generated)

        audio_prefix = audio_tokens.size(1) if isinstance(audio_tokens, torch.Tensor) else 0
        prompt_source = sanitized_ids if model_choice == "base" and 'sanitized_ids' in locals() and sanitized_ids is not None else input_ids
        prompt_len = prompt_source.shape[1]
        return [generated[i, audio_prefix + prompt_len :] for i in range(generated.size(0))]
    
    def _clean_answer(self, s: str) -> str:
        """Clean and normalize answer text."""
        s = s.strip()
        s = s.split("Answer:")[-1].strip() if "Answer:" in s else s
        return " ".join(s.split())  # collapse spaces
    
    def _extract_answer(self, generated_text):
        """Extract answer from generated text after 'Answer:'"""
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text.strip()
        
        # Extract first few words as answer (VQA answers are usually short)
        answer_words = answer.split()[:5]  # Max 5 words
        return " ".join(answer_words).strip()
    
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
                base_sequences = self._generate_predictions(gen_inputs, "base")
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
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.safe_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
            "metrics": metrics,
            "training_stats": self.training_stats,
            "curriculum_state": self.curriculum_manager.get_progress_summary() if self.use_curriculum else None
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
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config["output_dir"], "best_checkpoint.pt")
            torch.save(checkpoint, best_path)
            
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
    
    def _train_with_curriculum(self):
        """Training loop with curriculum learning."""
        print("🎓 Starting Stage A training with curriculum learning...")
        print(f"Curriculum has {self.curriculum_manager.config.get_num_stages()} stages")
        
        # Compute baseline metrics before training (quick evaluation)
        print("Computing baseline metrics (quick evaluation)...")
        baseline_metrics = self.evaluate(max_batches=50)  # Only evaluate first 50 batches
        self.baseline_metrics = baseline_metrics
        self.curriculum_manager.set_baseline_metrics({
            "vl_retention": baseline_metrics["retention_score"],
            "audio_accuracy": baseline_metrics["audio_accuracy"]
        })
        print(f"Baseline metrics computed:")
        print(f"  Retention Score: {baseline_metrics['retention_score']:.4f}")
        print(f"  Audio Accuracy: {baseline_metrics['audio_accuracy']:.4f}")
        print(f"  VL Safe Accuracy: {baseline_metrics['vl_safe_accuracy']:.4f}")
        print(f"  VL Base Accuracy: {baseline_metrics['vl_base_accuracy']:.4f}")
        print(f"  Total Loss: {baseline_metrics['total_loss']:.4f}")
        print(f"")
        print(f"🔍 Model Comparison:")
        print(f"  SAFE Model VL Accuracy: {baseline_metrics['vl_safe_accuracy']:.4f}")
        print(f"  Base VL Model Accuracy: {baseline_metrics['vl_base_accuracy']:.4f}")
        if baseline_metrics['vl_base_accuracy'] > 0.01:  # If base model has decent accuracy
            if baseline_metrics['vl_safe_accuracy'] < baseline_metrics['vl_base_accuracy'] * 0.5:
                print(f"  ⚠️  WARNING: SAFE model significantly underperforming base VL model!")
                print(f"     This suggests an integration issue in the SAFE architecture.")
        else:
            print(f"  ℹ️  Both models show low accuracy - may be due to dataset/task mismatch.")
        
        # Initialize first stage
        self.update_curriculum_config()
        
        epoch_in_stage = 0
        samples_in_stage = 0
        
        while not self.curriculum_manager.is_completed:
            self.epoch += 1
            epoch_in_stage += 1
            epoch_losses = {key: [] for key in ["total_loss", "audio_task_loss", "retention_loss"]}
            
            # Get current stage info
            stage_name = self.current_stage_config["stage_name"] if self.current_stage_config else "unknown"
            stage_idx = self.current_stage_config["stage_idx"] if self.current_stage_config else 0
            
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Stage {stage_name} | Epoch {epoch_in_stage} | Global {self.epoch}"
            )
            
            for step, batch in enumerate(progress_bar):
                # Training step
                step_losses = self.train_step(batch)
                self.global_step += 1
                batch_size = len(batch["questions"])
                samples_in_stage += batch_size
                
                # Accumulate losses
                for key in epoch_losses:
                    if key in step_losses:
                        epoch_losses[key].append(step_losses[key])
                
                # Update progress bar
                if epoch_losses["total_loss"]:
                    avg_loss = np.mean(epoch_losses["total_loss"][-10:])
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "stage": stage_name
                    })
                
                # Logging
                if self.global_step % self.config["logging_steps"] == 0:
                    avg_losses = {k: np.mean(v[-self.config["logging_steps"]:]) 
                                 for k, v in epoch_losses.items() if v}
                    
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
            print(f"\n📊 End of Epoch {self.epoch} (Stage {stage_name}, Epoch {epoch_in_stage})")
            max_eval_batches = self.config.get("max_eval_batches", None)
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
            
            print(f"  Metrics:")
            print(f"    Audio Accuracy: {eval_metrics['audio_accuracy']:.4f}")
            print(f"    VL Retention: {eval_metrics['retention_score']:.4f}")
            print(f"    Retention Loss: {eval_metrics['retention_loss']:.4f}")
            print(f"    Samples in Stage: {samples_in_stage}")
            
            # Check curriculum progression
            if self.epoch % self.config.get("validation_frequency", 5) == 0:
                progression_status = self.curriculum_manager.advance_epoch()
                
                if progression_status == ProgressionStatus.ADVANCE:
                    print(f"🎯 Advanced to next curriculum stage!")
                    
                    # Save checkpoint before advancing
                    self.save_checkpoint(eval_metrics, is_best=True, 
                                       suffix=f"stage_{stage_name}_completed")
                    
                    # Update configuration for new stage
                    self.update_curriculum_config()
                    epoch_in_stage = 0
                    samples_in_stage = 0
                    
                elif progression_status == ProgressionStatus.EXTEND:
                    print(f"⏳ Extended current stage due to unmet criteria")
                    
                elif progression_status == ProgressionStatus.FAILED:
                    print(f"❌ Curriculum failed - criteria not met after extensions")
                    break
                    
                else:  # CONTINUE
                    print(f"📈 Continuing current stage...")
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
        
        print("🎓 Curriculum learning completed!")
        
        # Final evaluation and checkpoint
        max_eval_batches = self.config.get("max_eval_batches", None)
        final_metrics = self.evaluate(max_batches=max_eval_batches)
        self.save_checkpoint(final_metrics, is_best=False, suffix="final")
        
        # Print curriculum summary
        summary = self.curriculum_manager.get_progress_summary()
        print(f"\n📋 Curriculum Summary:")
        print(f"  Total Epochs: {summary['total_epochs']}")
        print(f"  Stages Completed: {summary['current_stage_idx']}/{summary['total_stages']}")
        print(f"  Final Audio Accuracy: {summary['current_metrics']['audio_accuracy']:.4f}")
        print(f"  Final VL Retention: {summary['current_metrics']['vl_retention']:.4f}")
        
        return final_metrics
    
    def _train_traditional(self):
        """Traditional fixed-epoch training loop."""
        print(f"Starting Stage A training for {self.config['num_epochs']} epochs...")
        print(f"Total training steps: {len(self.train_dataloader) * self.config['num_epochs']}")
        
        # Compute baseline retention score
        print("Computing baseline metrics...")
        max_eval_batches = self.config.get("max_eval_batches", None)
        baseline_metrics = self.evaluate(max_batches=max_eval_batches)
        baseline_retention = baseline_metrics["retention_score"]
        print(f"Baseline retention score: {baseline_retention:.4f}")
        if max_eval_batches:
            print(f"(Evaluation limited to {max_eval_batches} batches)")
        
        for epoch in range(self.config["num_epochs"]):
            self.epoch = epoch
            epoch_losses = {key: [] for key in ["total_loss", "audio_task_loss", "retention_loss"]}
            
            # Training loop
            try:
                progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            except OSError:
                # Fallback for cluster environments with stale file handles
                progress_bar = self.train_dataloader
            
            for step, batch in enumerate(progress_bar):
                # Training step
                step_losses = self.train_step(batch)
                self.global_step += 1
                
                # Accumulate losses
                for key in epoch_losses:
                    if key in step_losses:
                        epoch_losses[key].append(step_losses[key])
                
                # Update progress bar
                if epoch_losses["total_loss"]:
                    avg_loss = np.mean(epoch_losses["total_loss"][-100:])
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                # Logging
                if self.global_step % self.config["logging_steps"] == 0:
                    avg_losses = {k: np.mean(v[-self.config["logging_steps"]:]) 
                                 for k, v in epoch_losses.items() if v}
                    
                    # Log to wandb if available
                    try:
                        wandb.log({
                            "train/" + k: v for k, v in avg_losses.items()
                        }, step=self.global_step)
                    except:
                        pass
                
                # Evaluation
                if self.global_step % self.config["eval_steps"] == 0:
                    max_eval_batches = self.config.get("max_eval_batches", None)
                    eval_metrics = self.evaluate(max_batches=max_eval_batches)
                    
                    print(f"\nStep {self.global_step} Evaluation:")
                    print(f"  Retention Score: {eval_metrics['retention_score']:.4f}")
                    print(f"  Audio Accuracy: {eval_metrics['audio_accuracy']:.4f}")
                    print(f"  VL Safe Accuracy: {eval_metrics['vl_safe_accuracy']:.4f}")
                    print(f"  Total Loss: {eval_metrics['total_loss']:.4f}")
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
                        print(f"WARNING: Retention degradation ({retention_degradation:.4f}) exceeds tolerance!")
                    
                    if self.patience_counter >= self.config["early_stopping_patience"]:
                        print(f"Early stopping after {self.patience_counter} evaluations without improvement")
                        return eval_metrics
                    
                    # Log to wandb
                    try:
                        wandb.log({
                            "eval/" + k: v for k, v in eval_metrics.items()
                        }, step=self.global_step)
                    except:
                        pass
            
            # End of epoch evaluation
            print(f"\nEnd of Epoch {epoch+1}")
            max_eval_batches = self.config.get("max_eval_batches", None)
            epoch_metrics = self.evaluate(max_batches=max_eval_batches)
            print(f"Epoch {epoch+1} Results:")
            print(f"  Retention Score: {epoch_metrics['retention_score']:.4f}")
            print(f"  Audio Accuracy: {epoch_metrics['audio_accuracy']:.4f}")
            print(f"  Retention Loss: {epoch_metrics['retention_loss']:.4f}")
            self.logger.info(
                "[Eval][epoch=%s] retention=%.4f audio_acc=%.4f retention_loss=%.4f",
                epoch + 1,
                epoch_metrics['retention_score'],
                epoch_metrics['audio_accuracy'],
                epoch_metrics['retention_loss'],
            )
        
        print("Stage A training completed!")
        print(f"Best retention score: {self.best_retention_score:.4f}")
        
        # Final checkpoint
        max_eval_batches = self.config.get("max_eval_batches", None)
        final_metrics = self.evaluate(max_batches=max_eval_batches)
        self.save_checkpoint(final_metrics, is_best=False)
        return final_metrics
