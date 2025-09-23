import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


class RetentionLoss(nn.Module):
    """
    Retention loss to preserve base VL performance during audio training.
    
    Combines:
    1. Logit distillation (KL divergence) from base VL model
    2. Fisher-weighted L2 regularization on base logits
    """
    
    def __init__(
        self,
        distillation_weight: float = 1.0,
        fisher_weight: float = 0.1,
        temperature: float = 2.0,
        use_fisher_information: bool = True
    ):
        super().__init__()
        
        self.distillation_weight = distillation_weight
        self.fisher_weight = fisher_weight
        self.temperature = temperature
        self.use_fisher_information = use_fisher_information
        
        # Fisher information matrix (computed during base model evaluation)
        self.fisher_information = None
        
    def set_fisher_information(self, fisher_info: torch.Tensor):
        """
        Set Fisher information matrix for regularization.
        
        Args:
            fisher_info: Precomputed Fisher information matrix
        """
        self.fisher_information = fisher_info
        
    def compute_fisher_information(
        self,
        base_model,
        dataloader,
        num_samples: int = 1000
    ) -> torch.Tensor:
        """
        Compute Fisher information matrix for important parameters.
        
        Args:
            base_model: Base VL model
            dataloader: Data loader for Fisher computation
            num_samples: Number of samples to use
            
        Returns:
            Fisher information matrix
        """
        base_model.eval()
        fisher_info = {}
        
        # Initialize Fisher information
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param)
        
        samples_processed = 0
        
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
                
            # Forward pass
            outputs = base_model(**batch)
            loss = outputs["loss"]
            
            # Backward pass
            base_model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher information)
            for name, param in base_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
                    
            samples_processed += batch["input_ids"].size(0)
        
        # Normalize by number of samples
        for name in fisher_info:
            fisher_info[name] /= samples_processed
            
        # Convert to single tensor (simplified)
        fisher_tensor = torch.cat([fisher_info[name].view(-1) for name in fisher_info])
        
        self.fisher_information = fisher_tensor
        return fisher_tensor
    
    def kl_divergence_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between student and teacher logits.
        
        Args:
            student_logits: SAFE model output logits
            teacher_logits: Base VL model output logits (detached)
            
        Returns:
            KL divergence loss
        """
        # Ensure teacher logits are detached
        teacher_logits = teacher_logits.detach()
        
        # Handle shape mismatches
        if student_logits.shape != teacher_logits.shape:
            print(f"KL divergence shape mismatch: student={student_logits.shape}, teacher={teacher_logits.shape}", flush=True)
            
            # For 3D tensors (batch, seq_len, vocab), handle all dimension mismatches
            if student_logits.dim() == 3 and teacher_logits.dim() == 3:
                # Handle batch size mismatch - expand teacher if needed
                if student_logits.shape[0] != teacher_logits.shape[0]:
                    if teacher_logits.shape[0] == 1:
                        teacher_logits = teacher_logits.expand(student_logits.shape[0], -1, -1)
                        print(f"KL divergence: expanded teacher from [1, ...] to {teacher_logits.shape}", flush=True)
                    elif student_logits.shape[0] == 1:
                        student_logits = student_logits.expand(teacher_logits.shape[0], -1, -1)
                        print(f"KL divergence: expanded student from [1, ...] to {student_logits.shape}", flush=True)
                    else:
                        # Take minimum batch size
                        min_batch_size = min(student_logits.shape[0], teacher_logits.shape[0])
                        student_logits = student_logits[:min_batch_size]
                        teacher_logits = teacher_logits[:min_batch_size]
                        print(f"KL divergence: truncated both to size {min_batch_size}", flush=True)
                
                # Handle sequence length mismatch
                if student_logits.shape[1] != teacher_logits.shape[1]:
                    min_seq_len = min(student_logits.shape[1], teacher_logits.shape[1])
                    student_logits = student_logits[:, :min_seq_len, :]
                    teacher_logits = teacher_logits[:, :min_seq_len, :]
            
            # For vocab size mismatches, use minimum vocab size
            if student_logits.shape[-1] != teacher_logits.shape[-1]:
                min_vocab_size = min(student_logits.shape[-1], teacher_logits.shape[-1])
                student_logits = student_logits[..., :min_vocab_size]
                teacher_logits = teacher_logits[..., :min_vocab_size]
        
        # Apply temperature scaling
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Compute KL divergence
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
            log_target=False
        )
        
        # Scale by temperature squared (standard distillation scaling)
        kl_loss *= (self.temperature ** 2)
        
        return kl_loss
    
    def fisher_regularization_loss(
        self,
        current_params: Dict[str, torch.Tensor],
        base_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute Fisher-weighted L2 regularization loss.
        
        Args:
            current_params: Current model parameters
            base_params: Base model parameters (reference)
            
        Returns:
            Fisher-weighted regularization loss
        """
        if not current_params:
            return torch.tensor(0.0)
            
        device = next(iter(current_params.values())).device
        
        if self.fisher_information is None:
            # Fallback to uniform L2 regularization if Fisher information not available
            reg_loss = torch.tensor(0.0, device=device)
            for name in current_params:
                if name in base_params:
                    param_diff = current_params[name] - base_params[name]
                    reg_loss = reg_loss + torch.sum(param_diff ** 2)
            return reg_loss
        
        reg_loss = torch.tensor(0.0, device=device)
        param_idx = 0
        
        for name in sorted(current_params.keys()):  # Ensure consistent ordering
            if name in base_params:
                param_diff = current_params[name] - base_params[name]
                param_size = param_diff.numel()
                
                # Check if we have enough Fisher information
                if param_idx + param_size <= len(self.fisher_information):
                    # Get corresponding Fisher information
                    fisher_slice = self.fisher_information[param_idx:param_idx + param_size]
                    fisher_slice = fisher_slice.view_as(param_diff)
                    
                    # Weighted L2 loss
                    weighted_loss = torch.sum(fisher_slice * (param_diff ** 2))
                    reg_loss = reg_loss + weighted_loss
                else:
                    # Fallback to unweighted L2 for remaining parameters
                    reg_loss = reg_loss + torch.sum(param_diff ** 2)
                
                param_idx += param_size
        
        return reg_loss
    
    def forward(
        self,
        safe_logits: torch.Tensor,
        base_logits: torch.Tensor,
        safe_model_params: Optional[Dict[str, torch.Tensor]] = None,
        base_model_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute retention loss.
        
        Args:
            safe_logits: Output logits from SAFE model
            base_logits: Output logits from base VL model (detached)
            safe_model_params: Current SAFE model parameters
            base_model_params: Base VL model parameters
            
        Returns:
            Dictionary with loss components
        """
        # Logit distillation loss
        distillation_loss = self.kl_divergence_loss(safe_logits, base_logits.detach())
        
        # Fisher regularization loss
        fisher_loss = torch.tensor(0.0, device=safe_logits.device)
        if (self.use_fisher_information and 
            safe_model_params is not None and 
            base_model_params is not None):
            fisher_loss = self.fisher_regularization_loss(
                safe_model_params, 
                base_model_params
            )
        
        # Total retention loss
        total_loss = (
            self.distillation_weight * distillation_loss +
            self.fisher_weight * fisher_loss
        )
        
        return {
            "retention_loss": total_loss,
            "distillation_loss": distillation_loss,
            "fisher_loss": fisher_loss
        }


class AudioTaskLoss(nn.Module):
    """
    Loss for audio-dependent tasks (QA accuracy, caption CIDEr).
    """
    
    def __init__(
        self,
        task_type: str = "qa",  # "qa" or "caption"
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        self.task_type = task_type
        self.label_smoothing = label_smoothing
        
        if task_type == "qa":
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        elif task_type == "caption":
            # For caption tasks, use cross-entropy with CIDEr-optimized weights
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute audio task loss.
        
        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            labels: Target labels (batch_size, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Task loss
        """
        # Flatten for loss computation
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        seq_len = min(shift_logits.size(-2), shift_labels.size(-1))
        shift_logits = shift_logits[..., :seq_len, :].contiguous()
        shift_labels = shift_labels[..., :seq_len].contiguous()

        if attention_mask is not None:
            shift_mask = attention_mask[..., 1:]
            shift_mask = shift_mask[..., :seq_len].contiguous()

            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.reshape(-1)
            flat_mask = shift_mask.reshape(-1).eq(1)
            
            # Also ignore label positions explicitly marked as -100
            valid = flat_mask & (flat_labels != -100)
            if not torch.any(valid):
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            # Ensure labels fall inside the model vocabulary to avoid NaNs from
            # cross-entropy when label smoothing is enabled.
            vocab_size = flat_logits.size(-1)
            in_vocab = (flat_labels >= 0) & (flat_labels < vocab_size)
            invalid_positions = valid & ~in_vocab
            if torch.any(invalid_positions):
                invalid_count = invalid_positions.sum().item()
                LOGGER.warning(
                    "AudioTaskLoss: filtering %d labels outside vocab range [0, %d). "
                    "This may indicate tokenization issues.",
                    invalid_count, vocab_size
                )
                valid &= in_vocab

            if not torch.any(valid):
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            # Debug: Log supervised token count and label distribution
            valid_count = valid.sum().item()
            unique_labels = flat_labels[valid].unique() if torch.any(valid) else torch.tensor([])
            LOGGER.info(f"AudioTaskLoss: valid_tokens={valid_count}, unique_labels={len(unique_labels)}")

            loss = self.loss_fn(flat_logits[valid].float(), flat_labels[valid])
        else:
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.reshape(-1)
            valid = flat_labels != -100
            if not torch.any(valid):
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            # Ensure labels fall inside the model vocabulary to avoid NaNs from
            # cross-entropy when label smoothing is enabled.
            vocab_size = flat_logits.size(-1)
            in_vocab = (flat_labels >= 0) & (flat_labels < vocab_size)
            invalid_positions = valid & ~in_vocab
            if torch.any(invalid_positions):
                invalid_count = invalid_positions.sum().item()
                LOGGER.warning(
                    "AudioTaskLoss: filtering %d labels outside vocab range [0, %d). "
                    "This may indicate tokenization issues.",
                    invalid_count, vocab_size
                )
                valid &= in_vocab

            if not torch.any(valid):
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            # Debug: Log supervised token count and label distribution
            valid_count = valid.sum().item()
            unique_labels = flat_labels[valid].unique() if torch.any(valid) else torch.tensor([])
            LOGGER.info(f"AudioTaskLoss: valid_tokens={valid_count}, unique_labels={len(unique_labels)}")

            loss = self.loss_fn(flat_logits[valid].float(), flat_labels[valid])
        
        return loss


class CombinedStageLoss(nn.Module):
    """
    Combined loss for Stage A training with balanced audio and VL batches.
    """
    
    def __init__(
        self,
        retention_loss: RetentionLoss,
        audio_task_loss: AudioTaskLoss,
        audio_weight: float = 1.0,
        retention_weight: float = 1.0
    ):
        super().__init__()
        
        self.retention_loss = retention_loss
        self.audio_task_loss = audio_task_loss
        self.audio_weight = audio_weight
        self.retention_weight = retention_weight
        self.retention_enabled = (
            retention_weight > 0.0
            and getattr(retention_loss, "distillation_weight", 0.0) > 0.0
        )

    def forward(
        self,
        safe_outputs: Dict[str, torch.Tensor],
        base_outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        has_audio: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined Stage A loss.
        
        Args:
            safe_outputs: SAFE model outputs
            base_outputs: Base VL model outputs  
            batch: Training batch
            has_audio: Boolean mask indicating which samples have audio
            
        Returns:
            Dictionary with loss components
        """
        safe_logits = safe_outputs["logits"] if isinstance(safe_outputs, dict) else getattr(safe_outputs, "logits")
        if safe_logits is None:
            raise ValueError("SAFE outputs must contain logits for loss computation")

        base_logits = None
        if base_outputs is not None:
            if isinstance(base_outputs, dict):
                base_logits = base_outputs.get("logits")
            else:
                base_logits = getattr(base_outputs, "logits", None)

        device = safe_logits.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_dict = {}
        
        # DEBUG: Print input information (disabled for cleaner training)
        # print(f"\n--- LOSS COMPUTATION DEBUG ---")
        # print(f"Safe logits shape: {safe_outputs['logits'].shape}")
        # print(f"Base logits shape: {base_outputs['logits'].shape if 'logits' in base_outputs else 'No base logits'}")
        # print(f"Has audio: {torch.sum(has_audio)}/{len(has_audio)} samples")
        # print(f"Device: {device}")
        
        # Audio task loss (for samples with audio)
        if torch.any(has_audio):
            audio_indices = torch.where(has_audio)[0]
            # Ensure indices are within bounds of both logits and labels tensors
            max_safe = safe_logits.size(0) if safe_logits is not None else 0
            labels = batch.get("labels")
            max_labels = labels.size(0) if labels is not None else 0
            max_bound = min(max_safe, max_labels)
            audio_indices = audio_indices[audio_indices < max_bound]
            if len(audio_indices) > 0:
                audio_logits = safe_logits[audio_indices]
                audio_labels = batch["labels"][audio_indices]
                audio_mask = batch.get("attention_mask", None)
                if audio_mask is not None:
                    audio_mask = audio_mask[audio_indices]
                
                # print(f"Computing audio loss for {len(audio_indices)} samples...")
                audio_loss = self.audio_task_loss(
                    logits=audio_logits,
                    labels=audio_labels,
                    attention_mask=audio_mask
                )
                # print(f"Audio loss computed: {audio_loss.item():.6f}")
                
                total_loss = total_loss + self.audio_weight * audio_loss
                loss_dict["audio_task_loss"] = audio_loss
            else:
                # print("No valid audio indices after bounds check")
                loss_dict["audio_task_loss"] = torch.tensor(0.0, device=device)
        else:
            # print("No audio samples in batch")
            loss_dict["audio_task_loss"] = torch.tensor(0.0, device=device)
        
        # Retention loss (for all samples, especially VL-only)
        retention_active = self.retention_enabled and base_logits is not None

        if retention_active:
            retention_losses = self.retention_loss(
                safe_logits=safe_logits,
                base_logits=base_logits
            )
            retention_loss = retention_losses["retention_loss"]
        else:
            zero = torch.tensor(0.0, device=device)
            retention_losses = {
                "retention_loss": zero,
                "distillation_loss": zero,
                "fisher_loss": zero,
            }
            retention_loss = zero
        # print(f"Retention loss computed: {retention_loss.item():.6f}")
        # print(f"Distillation loss: {retention_losses['distillation_loss'].item():.6f}")
        # print(f"Fisher loss: {retention_losses['fisher_loss'].item():.6f}")
        
        if retention_active:
            total_loss = total_loss + self.retention_weight * retention_loss

        loss_dict.update({
            "retention_loss": retention_loss,
            "distillation_loss": retention_losses["distillation_loss"],
            "fisher_loss": retention_losses["fisher_loss"]
        })
        
        loss_dict["total_loss"] = total_loss
        # print(f"Final total loss: {total_loss.item():.6f}")
        # print("--- END LOSS DEBUG ---\n")
        
        return loss_dict


class RewardFunction:
    """
    Reward function for Stage B RL training.
    
    r = Score - α * LatencyCost - γ * IrrelevancePenalty
    """
    
    def __init__(
        self,
        alpha: float = 0.3,  # Latency cost weight
        gamma: float = 0.5,  # Irrelevance penalty weight
        token_cost: float = 0.01,  # Cost per audio token
        latency_cost: float = 0.001  # Cost per ms
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.token_cost = token_cost
        self.latency_cost = latency_cost
    
    def compute_score(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        task_type: str = "qa"
    ) -> torch.Tensor:
        """
        Compute task performance score.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            task_type: Type of task ("qa" or "caption")
            
        Returns:
            Scores (0-1 for QA accuracy, normalized CIDEr for captions)
        """
        if task_type == "qa":
            # Simple accuracy for QA
            if predictions.dim() > 1:
                predictions = torch.argmax(predictions, dim=-1)
            correct = (predictions == targets).float()
            return correct
        elif task_type == "caption":
            # Placeholder for caption metrics (would need proper CIDEr implementation)
            # For now, use token-level accuracy as proxy
            if predictions.dim() > 2:
                predictions = torch.argmax(predictions, dim=-1)
            
            # Compute token-level accuracy and normalize
            correct_tokens = (predictions == targets).float()
            accuracy = torch.mean(correct_tokens, dim=-1)  # Average over sequence
            
            # Scale to approximate CIDEr range (0-100) and normalize
            normalized_score = accuracy  # Simplified
            return normalized_score
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def compute_latency_cost(
        self,
        num_tokens: torch.Tensor,
        processing_time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute latency cost based on token count and processing time.
        
        Args:
            num_tokens: Number of audio tokens used
            processing_time: Optional processing time in ms
            
        Returns:
            Latency costs
        """
        # Token-based cost
        token_costs = num_tokens.float() * self.token_cost
        
        # Time-based cost
        if processing_time is not None:
            time_costs = processing_time * self.latency_cost
            return token_costs + time_costs
        else:
            return token_costs
    
    def compute_irrelevance_penalty(
        self,
        used_audio: torch.Tensor,
        is_audio_irrelevant: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute penalty for using audio on irrelevant examples.
        
        Args:
            used_audio: Boolean tensor indicating if audio was used
            is_audio_irrelevant: Boolean tensor indicating if audio is irrelevant
            
        Returns:
            Irrelevance penalties
        """
        # Penalty only when audio is used but irrelevant
        penalties = used_audio.float() * is_audio_irrelevant.float()
        return penalties
    
    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_tokens: torch.Tensor,
        used_audio: torch.Tensor,
        is_audio_irrelevant: torch.Tensor,
        task_type: str = "qa",
        processing_time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute total reward.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            num_tokens: Number of audio tokens used
            used_audio: Whether audio was consulted
            is_audio_irrelevant: Whether audio is irrelevant for this sample
            task_type: Task type
            processing_time: Optional processing time
            
        Returns:
            Total rewards for each sample
        """
        # Task performance score
        scores = self.compute_score(predictions, targets, task_type)
        
        # Latency cost
        latency_costs = self.compute_latency_cost(num_tokens, processing_time)
        
        # Irrelevance penalty
        irrelevance_penalties = self.compute_irrelevance_penalty(
            used_audio, 
            is_audio_irrelevant
        )
        
        # Total reward
        rewards = (
            scores - 
            self.alpha * latency_costs - 
            self.gamma * irrelevance_penalties
        )
        
        return rewards


class ConstrainedRetentionLoss:
    """
    Lagrangian-constrained loss to maintain VL performance during RL training.
    
    Adds -λ * max(0, τ - J_VL) to the batch return to enforce retention constraint.
    """
    
    def __init__(
        self,
        baseline_score: float,
        tolerance: float = 0.003,  # 0.3% tolerance
        lambda_lr: float = 0.01,
        lambda_max: float = 10.0
    ):
        self.baseline_score = baseline_score
        self.tolerance = tolerance
        self.threshold = baseline_score - tolerance
        
        self.lambda_multiplier = torch.tensor(0.0)
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max
        
        # Running estimate of VL performance
        self.vl_score_ema = baseline_score
        self.ema_alpha = 0.1
    
    def update_vl_estimate(self, current_vl_score: float):
        """Update running estimate of VL performance."""
        self.vl_score_ema = (
            self.ema_alpha * current_vl_score + 
            (1 - self.ema_alpha) * self.vl_score_ema
        )
    
    def compute_constraint_penalty(self) -> torch.Tensor:
        """Compute constraint penalty based on current VL performance."""
        violation = max(0.0, self.threshold - self.vl_score_ema)
        penalty = self.lambda_multiplier * violation
        return torch.tensor(penalty)
    
    def update_lambda(self):
        """Update Lagrangian multiplier using dual ascent."""
        violation = max(0.0, self.threshold - self.vl_score_ema)
        
        if violation > 0:
            # Increase lambda if constraint is violated
            self.lambda_multiplier += self.lambda_lr * violation
        else:
            # Decrease lambda if constraint is satisfied
            self.lambda_multiplier = max(0.0, self.lambda_multiplier - self.lambda_lr * 0.1)
        
        # Clip lambda to reasonable range
        self.lambda_multiplier = min(self.lambda_multiplier, self.lambda_max)
    
    def __call__(self, batch_returns: torch.Tensor) -> torch.Tensor:
        """
        Apply constraint penalty to batch returns.
        
        Args:
            batch_returns: Original batch returns
            
        Returns:
            Constrained returns with penalty applied
        """
        penalty = self.compute_constraint_penalty()
        constrained_returns = batch_returns - penalty
        
        return constrained_returns
