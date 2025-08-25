import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import wandb
import os
from pathlib import Path

from ..models.safe_model import SAFEModel
from ..models.base_vl import BaseVLModel
from ..data.datasets import create_safe_dataloader
from ..data.curriculum import CurriculumManager, CurriculumConfig, ProgressionStatus
from .losses import RetentionLoss, AudioTaskLoss, CombinedStageLoss


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
            "validation_frequency": 5 if self.use_curriculum else 1000  # More frequent validation for curriculum
        }
        
        if config:
            self.config.update(config)
        
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
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=total_steps - self.config["warmup_steps"]
            )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_retention_score = 0.0
        self.patience_counter = 0
        
        # Base VL model for retention comparison
        self.base_vl_model = self.safe_model.base_vl
        self.base_vl_model.eval()
        
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
        Returns (None, None) if batch is empty.
        """
        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            return None, None

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [B, T]

        image_token_id = self._get_image_token_id()
        has_img_tokens = (input_ids == image_token_id).any(dim=1)  # [B]

        if has_img_tokens.numel() == 0:
            return None, None

        idx_mm = torch.nonzero(has_img_tokens, as_tuple=False).flatten()
        idx_text = torch.nonzero(~has_img_tokens, as_tuple=False).flatten()

        def _subset(t, idx):
            if t is None or idx.numel() == 0:
                return None
            if isinstance(t, torch.Tensor):
                return t.index_select(0, idx.to(t.device))
            return t  # leave non-tensors unchanged (usually absent here)

        # Build mm/text dicts by slicing tensors on batch dimension
        def _build_subset(keys, allow_pixels: bool):
            sub = {}
            indices = idx_mm if allow_pixels else idx_text
            
            for k in keys:
                if k == "pixel_values" and not allow_pixels:
                    continue
                v = inputs.get(k, None)
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    # Special handling for pixel_values - they already match the number of MM samples
                    if k == "pixel_values" and allow_pixels:
                        # pixel_values already have the right samples, just use them directly
                        sub[k] = v
                    elif v.size(0) == input_ids.size(0):
                        # Regular tensors that match input_ids batch size
                        subset_tensor = _subset(v, indices)
                        if subset_tensor is not None:
                            sub[k] = subset_tensor
                    # else: skip tensors with mismatched batch sizes
                else:
                    # Non-batched entries or missing keys are passed as-is only if harmless
                    if k != "pixel_values":
                        sub[k] = v
            return sub

        keys = list(inputs.keys())

        mm_inputs = _build_subset(keys, allow_pixels=True) if idx_mm.numel() > 0 else None
        text_inputs = _build_subset(keys, allow_pixels=False) if idx_text.numel() > 0 else None

        # Ensure invariants:
        if mm_inputs is not None:
            
            # Check if we have image tokens but no pixel_values - this indicates a problem upstream
            if "pixel_values" not in mm_inputs or mm_inputs["pixel_values"] is None:
                # Log the issue and skip multimodal processing for this batch
                print(f"Warning: Found image tokens but no pixel_values in batch. Skipping MM processing.")
                print(f"Available keys: {list(inputs.keys())}")
                return None, inputs  # Return all as text batch
        if text_inputs is not None and "pixel_values" in text_inputs:
            text_inputs.pop("pixel_values", None)

        return mm_inputs, text_inputs
    
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
            param_groups.append({
                "params": projector_params,
                "lr": self.config["learning_rate_projector"],
                "name": "projector"
            })
        
        # Fusion adapter parameters (LoRA)
        adapter_params = list(self.safe_model.fusion_adapter.parameters())
        if adapter_params:
            param_groups.append({
                "params": adapter_params,
                "lr": self.config["learning_rate_adapter"],
                "name": "adapter"
            })
        
        if not param_groups:
            raise ValueError("No trainable parameters found!")
        
        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.config["weight_decay"],
            betas=(0.9, 0.999)
        )
    
    def _warmup_lr(self):
        """Apply learning rate warmup."""
        if self.global_step < self.config["warmup_steps"]:
            warmup_factor = self.global_step / self.config["warmup_steps"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= warmup_factor
    
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
        
        # Prepare inputs for SAFE model
        has_audio = batch.get("has_audio", torch.zeros(len(batch["questions"]), dtype=torch.bool))
        has_images = batch.get("has_images", torch.zeros(len(batch["questions"]), dtype=torch.bool))
        
        # Create input tensors
        inputs = self.safe_model.prepare_multimodal_inputs(
            text=batch["questions"],
            images=batch.get("images") if torch.any(has_images) else None,
            audio=batch.get("audio") if torch.any(has_audio) else None,
            device=device
        )
        
        # Forward pass through SAFE model
        safe_outputs = self.safe_model(**inputs)
        
        # Forward pass through base VL model (for retention loss)
        with torch.no_grad():
            base_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": inputs["labels"]
            }
            if self.base_vl_model.model_type in ["llava", "blip2"] and "pixel_values" in inputs:
                base_inputs["pixel_values"] = inputs["pixel_values"]

            if self.base_vl_model.model_type in ["llava", "blip2"]:
                # Use splitter for mixed batches
                mm_inputs, text_inputs = self._split_mm_text_batches(base_inputs)

                base_outputs = None

                # 1) multimodal sub-batch
                if mm_inputs is not None:
                    base_outputs = self.base_vl_model(**mm_inputs)

                # 2) text-only sub-batch
                if text_inputs is not None:
                    out_text = self.base_vl_model(**text_inputs)
                    if base_outputs is not None:
                        # Handle dict outputs - access logits as dict keys
                        mm_logits = base_outputs["logits"] if isinstance(base_outputs, dict) else base_outputs.logits
                        text_logits = out_text["logits"] if isinstance(out_text, dict) else out_text.logits
                        
                        # Combine logits from both sub-batches
                        combined_logits = torch.cat([mm_logits, text_logits], dim=0)
                        
                        if isinstance(base_outputs, dict):
                            base_outputs["logits"] = combined_logits
                            # Handle loss if present
                            if "loss" in base_outputs and "loss" in out_text:
                                base_outputs["loss"] = base_outputs["loss"] + out_text["loss"]
                        else:
                            base_outputs.logits = combined_logits
                            if hasattr(base_outputs, 'loss') and hasattr(out_text, 'loss'):
                                base_outputs.loss = base_outputs.loss + out_text.loss
                    else:
                        base_outputs = out_text

                # Safety: if somehow neither branch ran, fall back to original
                if base_outputs is None:
                    base_outputs = self.base_vl_model(**base_inputs)
            else:
                # For custom models, use vision features
                base_inputs["vision_features"] = inputs.get("vision_features")
                base_outputs = self.base_vl_model(**base_inputs)
        
        # Compute combined loss
        loss_dict = self.combined_loss(
            safe_outputs=safe_outputs,
            base_outputs=base_outputs,
            batch=inputs,
            has_audio=has_audio
        )
        
        # DEBUG: Print detailed loss information
        total_loss = loss_dict["total_loss"]
        print(f"\n=== DEBUG STEP {self.global_step} ===")
        print(f"Total Loss: {total_loss.item():.6f}")
        print(f"Loss breakdown:")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.6f}")
            else:
                print(f"  {key}: {value:.6f}")
        print(f"Has audio samples: {torch.sum(has_audio).item()}/{len(has_audio)}")
        print(f"Batch size: {len(inputs.get('input_ids', []))}")
        print("="*30)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.safe_model.get_trainable_parameters(),
            self.config["max_grad_norm"]
        )
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Learning rate scheduling
        if self.global_step >= self.config["warmup_steps"]:
            if self.scheduler is not None:
                self.scheduler.step()
        else:
            self._warmup_lr()
        
        # Convert losses to float
        step_losses = {k: v.item() if isinstance(v, torch.Tensor) else v 
                      for k, v in loss_dict.items()}
        
        return step_losses
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.safe_model.eval()
        
        # Check if validation dataset has samples
        if len(self.val_loader.dataset) == 0:
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
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                has_audio = batch.get("has_audio", torch.zeros(len(batch["questions"]), dtype=torch.bool))
                has_images = batch.get("has_images", torch.zeros(len(batch["questions"]), dtype=torch.bool))
                
                # Prepare inputs
                inputs = self.safe_model.prepare_multimodal_inputs(
                    text=batch["questions"],
                    images=batch.get("images") if torch.any(has_images) else None,
                    audio=batch.get("audio") if torch.any(has_audio) else None,
                    device=device
                )
                
                # For BLIP2/LLaVA models, ensure pixel_values match the presence of image tokens
                if self.safe_model.base_vl.model_type in ["blip2", "llava"]:
                    # Check which samples have image tokens
                    image_token_id = self._get_image_token_id()
                    input_ids = inputs.get("input_ids")
                    if input_ids is not None:
                        if input_ids.dim() == 1:
                            input_ids = input_ids.unsqueeze(0)
                        
                        has_img_tokens = (input_ids == image_token_id).any(dim=1)
                        
                        # If no samples have image tokens but we have pixel_values, remove them
                        if not torch.any(has_img_tokens) and "pixel_values" in inputs:
                            inputs.pop("pixel_values")
                        # If some samples have image tokens but we don't have pixel_values, create dummy ones
                        elif torch.any(has_img_tokens) and "pixel_values" not in inputs:
                            # Only create pixel_values for the batch size we actually have
                            batch_size = input_ids.size(0)
                            dummy_pixel_values = torch.zeros((batch_size, 3, 224, 224), 
                                                           dtype=torch.float32, device=device)
                            inputs["pixel_values"] = dummy_pixel_values
                
                
                # SAFE model forward pass
                safe_outputs = self.safe_model(**inputs)
                
                # Base VL model forward pass
                base_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "labels": inputs["labels"]
                }
                if self.base_vl_model.model_type in ["llava", "blip2"] and "pixel_values" in inputs:
                    base_inputs["pixel_values"] = inputs["pixel_values"]

                if self.base_vl_model.model_type in ["llava", "blip2"]:
                    # Use splitter for mixed batches
                    mm_inputs, text_inputs = self._split_mm_text_batches(base_inputs)

                    base_outputs = None

                    # 1) multimodal sub-batch
                    if mm_inputs is not None:
                        
                        base_outputs = self.base_vl_model(**mm_inputs)

                    # 2) text-only sub-batch
                    if text_inputs is not None:
                        out_text = self.base_vl_model(**text_inputs)
                        if base_outputs is not None:
                            # Handle dict outputs - access logits as dict keys
                            mm_logits = base_outputs["logits"] if isinstance(base_outputs, dict) else base_outputs.logits
                            text_logits = out_text["logits"] if isinstance(out_text, dict) else out_text.logits
                            
                            # Combine logits from both sub-batches
                            combined_logits = torch.cat([mm_logits, text_logits], dim=0)
                            
                            if isinstance(base_outputs, dict):
                                base_outputs["logits"] = combined_logits
                                # Handle loss if present
                                if "loss" in base_outputs and "loss" in out_text:
                                    base_outputs["loss"] = base_outputs["loss"] + out_text["loss"]
                            else:
                                base_outputs.logits = combined_logits
                                if hasattr(base_outputs, 'loss') and hasattr(out_text, 'loss'):
                                    base_outputs.loss = base_outputs.loss + out_text.loss
                        else:
                            base_outputs = out_text

                    # Safety: if somehow neither branch ran, fall back to original
                    if base_outputs is None:
                        base_outputs = self.base_vl_model(**base_inputs)
                else:
                    # For custom models, use vision features
                    base_inputs["vision_features"] = inputs.get("vision_features")
                    base_outputs = self.base_vl_model(**base_inputs)
                
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
                
                # Compute accuracy metrics
                safe_preds = torch.argmax(safe_outputs["logits"], dim=-1)
                base_preds = torch.argmax(base_outputs["logits"], dim=-1)
                targets = inputs["labels"]
                
                # Audio-dependent accuracy
                if torch.any(has_audio):
                    audio_indices = torch.where(has_audio)[0]
                    # Ensure indices are within bounds
                    audio_indices = audio_indices[audio_indices < len(safe_preds)]
                    if len(audio_indices) > 0:
                        audio_pred_sample = safe_preds[audio_indices]
                        audio_target_sample = targets[audio_indices]
                        
                        # Simple token-level accuracy (simplified)
                        audio_matches = (audio_pred_sample == audio_target_sample).float()
                        audio_correct += torch.sum(audio_matches).item()
                        audio_total += audio_matches.numel()
                        audio_samples += len(audio_indices)
                
                # VL retention accuracy
                vl_indices = torch.where(~has_audio)[0] if torch.any(~has_audio) else torch.arange(len(has_audio))
                if len(vl_indices) > 0:
                    # Ensure indices are within bounds
                    vl_indices = vl_indices[vl_indices < len(safe_preds)]
                    if len(vl_indices) > 0:
                        vl_safe_pred = safe_preds[vl_indices]
                        vl_base_pred = base_preds[vl_indices]
                        vl_target = targets[vl_indices]
                        
                        vl_safe_matches = (vl_safe_pred == vl_target).float()
                        vl_base_matches = (vl_base_pred == vl_target).float()
                        
                        vl_safe_correct += torch.sum(vl_safe_matches).item()
                        vl_base_correct += torch.sum(vl_base_matches).item()
                        vl_total += vl_safe_matches.numel()
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
        
        # Skip baseline evaluation for faster startup - will evaluate after first epoch
        print("Skipping baseline metrics computation for faster startup...")
        baseline_metrics = {
            "retention_score": 0.0,
            "audio_accuracy": 0.0,
            "vl_loss": 0.0,
            "audio_loss": 0.0
        }
        self.baseline_metrics = baseline_metrics
        self.curriculum_manager.set_baseline_metrics({
            "vl_retention": 0.0,
            "audio_accuracy": 0.0
        })
        print("Will compute metrics after first epoch...")
        
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
            eval_metrics = self.evaluate()
            
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
        final_metrics = self.evaluate()
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
        baseline_metrics = self.evaluate()
        baseline_retention = baseline_metrics["retention_score"]
        print(f"Baseline retention score: {baseline_retention:.4f}")
        
        for epoch in range(self.config["num_epochs"]):
            self.epoch = epoch
            epoch_losses = {key: [] for key in ["total_loss", "audio_task_loss", "retention_loss"]}
            
            # Training loop
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
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
                    eval_metrics = self.evaluate()
                    
                    print(f"\nStep {self.global_step} Evaluation:")
                    print(f"  Retention Score: {eval_metrics['retention_score']:.4f}")
                    print(f"  Audio Accuracy: {eval_metrics['audio_accuracy']:.4f}")
                    print(f"  VL Safe Accuracy: {eval_metrics['vl_safe_accuracy']:.4f}")
                    print(f"  Total Loss: {eval_metrics['total_loss']:.4f}")
                    
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
            epoch_metrics = self.evaluate()
            print(f"Epoch {epoch+1} Results:")
            print(f"  Retention Score: {epoch_metrics['retention_score']:.4f}")
            print(f"  Audio Accuracy: {epoch_metrics['audio_accuracy']:.4f}")
            print(f"  Retention Loss: {epoch_metrics['retention_loss']:.4f}")
        
        print("Stage A training completed!")
        print(f"Best retention score: {self.best_retention_score:.4f}")
        
        # Final checkpoint
        final_metrics = self.evaluate()
        self.save_checkpoint(final_metrics, is_best=False)
        return final_metrics