import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, NamedTuple
import numpy as np
from .state_features import StateFeatureExtractor


class PolicyAction(NamedTuple):
    """Structure for policy actions."""
    consult_audio: bool           # Whether to use audio
    num_tokens: int              # Token budget k ∈ {0, 4, 8, 12}
    fusion_layer: int            # Which layer to apply fusion
    temporal_crop: Optional[Tuple[float, float]] = None  # (start, end) for long audio


class AudioPolicyNetwork(nn.Module):
    """
    RL controller policy π_θ that decides when and how to use audio.
    
    Outputs:
    - Consult decision (binary)
    - Token budget k ∈ {0, 4, 8, 12}
    - Fusion layer selection (1 of 2 candidate mid layers)
    - Optional temporal crop for long audio
    """
    
    def __init__(
        self,
        state_feature_dim: int = 256,
        hidden_dim: int = 512,
        token_budgets: List[int] = None,
        fusion_layers: List[int] = None,
        use_temporal_crop: bool = False,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.state_feature_dim = state_feature_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.use_temporal_crop = use_temporal_crop
        
        # Token budget options
        self.token_budgets = token_budgets or [0, 4, 8, 12]
        self.num_token_options = len(self.token_budgets)
        
        # Fusion layer options (typically 2 mid-layers)
        self.fusion_layers = fusion_layers or [6, 12]  # Example layer indices
        self.num_layer_options = len(self.fusion_layers)
        
        # Shared feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Decision heads
        self.consult_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.token_budget_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_token_options)
        )
        
        self.fusion_layer_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_layer_options)
        )
        
        # Optional temporal cropping head
        if use_temporal_crop:
            self.temporal_crop_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2)  # (start, width) as fractions
            )
    
    def forward(
        self, 
        state_features: torch.Tensor,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through policy network.
        
        Args:
            state_features: (batch_size, state_feature_dim) state representation
            training: Whether in training mode (affects sampling)
            
        Returns:
            Dictionary with action logits and sampled actions
        """
        batch_size = state_features.shape[0]
        device = state_features.device
        
        # Process state features
        features = self.feature_processor(state_features)
        
        # Compute action logits
        consult_logits = self.consult_head(features)  # (batch_size, 1)
        token_logits = self.token_budget_head(features)  # (batch_size, num_token_options)
        layer_logits = self.fusion_layer_head(features)  # (batch_size, num_layer_options)
        
        # Apply temperature scaling
        consult_logits = consult_logits / self.temperature
        token_logits = token_logits / self.temperature
        layer_logits = layer_logits / self.temperature
        
        # Sample actions
        if training:
            # Use Gumbel-Softmax for differentiable sampling
            consult_probs = torch.sigmoid(consult_logits)
            consult_actions = torch.bernoulli(consult_probs)
            
            token_probs = F.softmax(token_logits, dim=-1)
            token_actions = torch.multinomial(token_probs, 1).squeeze(-1)
            
            layer_probs = F.softmax(layer_logits, dim=-1)
            layer_actions = torch.multinomial(layer_probs, 1).squeeze(-1)
        else:
            # Deterministic actions for evaluation
            consult_actions = (torch.sigmoid(consult_logits) > 0.5).float()
            token_actions = torch.argmax(token_logits, dim=-1)
            layer_actions = torch.argmax(layer_logits, dim=-1)
        
        results = {
            "consult_logits": consult_logits,
            "token_logits": token_logits,
            "layer_logits": layer_logits,
            "consult_actions": consult_actions,
            "token_actions": token_actions,
            "layer_actions": layer_actions,
            "consult_probs": torch.sigmoid(consult_logits),
            "token_probs": F.softmax(token_logits, dim=-1),
            "layer_probs": F.softmax(layer_logits, dim=-1)
        }
        
        # Optional temporal cropping
        if self.use_temporal_crop:
            crop_params = self.temporal_crop_head(features)  # (batch_size, 2)
            crop_start = torch.sigmoid(crop_params[:, 0])  # [0, 1]
            crop_width = torch.sigmoid(crop_params[:, 1])  # [0, 1]
            
            results.update({
                "crop_start": crop_start,
                "crop_width": crop_width
            })
        
        return results
    
    def get_actions(
        self, 
        state_features: torch.Tensor,
        deterministic: bool = False
    ) -> List[PolicyAction]:
        """
        Get policy actions from state features.
        
        Args:
            state_features: State representation
            deterministic: Whether to use deterministic actions
            
        Returns:
            List of PolicyAction objects
        """
        with torch.no_grad():
            outputs = self.forward(state_features, training=not deterministic)
            
            batch_size = state_features.shape[0]
            actions = []
            
            for i in range(batch_size):
                consult = bool(outputs["consult_actions"][i].item())
                token_idx = int(outputs["token_actions"][i].item())
                layer_idx = int(outputs["layer_actions"][i].item())
                
                num_tokens = self.token_budgets[token_idx] if consult else 0
                fusion_layer = self.fusion_layers[layer_idx]
                
                temporal_crop = None
                if self.use_temporal_crop and "crop_start" in outputs:
                    start = outputs["crop_start"][i].item()
                    width = outputs["crop_width"][i].item()
                    end = min(start + width, 1.0)
                    temporal_crop = (start, end)
                
                action = PolicyAction(
                    consult_audio=consult,
                    num_tokens=num_tokens,
                    fusion_layer=fusion_layer,
                    temporal_crop=temporal_crop
                )
                actions.append(action)
                
            return actions
    
    def compute_log_probs(
        self, 
        state_features: torch.Tensor,
        actions: List[PolicyAction]
    ) -> torch.Tensor:
        """
        Compute log probabilities of given actions.
        
        Args:
            state_features: State representation
            actions: List of actions taken
            
        Returns:
            log_probs: (batch_size,) log probabilities
        """
        outputs = self.forward(state_features, training=True)
        
        batch_size = len(actions)
        log_probs = torch.zeros(batch_size, device=state_features.device)
        
        for i, action in enumerate(actions):
            # Consult decision log prob
            consult_logits = outputs["consult_logits"][i]
            consult_prob = torch.sigmoid(consult_logits)
            if action.consult_audio:
                consult_log_prob = torch.log(consult_prob + 1e-8)
            else:
                consult_log_prob = torch.log(1 - consult_prob + 1e-8)
            
            # Token budget log prob
            token_idx = self.token_budgets.index(action.num_tokens)
            token_log_probs = F.log_softmax(outputs["token_logits"][i], dim=-1)
            token_log_prob = token_log_probs[token_idx]
            
            # Layer selection log prob
            layer_idx = self.fusion_layers.index(action.fusion_layer)
            layer_log_probs = F.log_softmax(outputs["layer_logits"][i], dim=-1)
            layer_log_prob = layer_log_probs[layer_idx]
            
            # Combine log probabilities
            total_log_prob = consult_log_prob + token_log_prob + layer_log_prob
            
            # Add temporal crop log prob if used
            if self.use_temporal_crop and action.temporal_crop is not None:
                # Simplified: assume Gaussian distribution around predicted values
                crop_start_pred = outputs["crop_start"][i]
                crop_width_pred = outputs["crop_width"][i]
                
                start_target, end_target = action.temporal_crop
                width_target = end_target - start_target
                
                # Simple L2 loss converted to log prob (this is a simplification)
                start_loss = (crop_start_pred - start_target) ** 2
                width_loss = (crop_width_pred - width_target) ** 2
                crop_log_prob = -(start_loss + width_loss)  # Negative for log prob
                
                total_log_prob += crop_log_prob
            
            log_probs[i] = total_log_prob
        
        return log_probs
    
    def entropy(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the policy for regularization.
        
        Args:
            state_features: State representation
            
        Returns:
            entropy: (batch_size,) policy entropy
        """
        outputs = self.forward(state_features, training=True)
        
        # Consult entropy
        consult_probs = outputs["consult_probs"]
        consult_entropy = -(consult_probs * torch.log(consult_probs + 1e-8) + 
                          (1 - consult_probs) * torch.log(1 - consult_probs + 1e-8))
        
        # Token budget entropy
        token_probs = outputs["token_probs"]
        token_entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-8), dim=-1)
        
        # Layer selection entropy
        layer_probs = outputs["layer_probs"]
        layer_entropy = -torch.sum(layer_probs * torch.log(layer_probs + 1e-8), dim=-1)
        
        total_entropy = consult_entropy.squeeze(-1) + token_entropy + layer_entropy
        
        return total_entropy


class ValueNetwork(nn.Module):
    """
    Value network for PPO-style training.
    Estimates the expected return from a given state.
    """
    
    def __init__(
        self,
        state_feature_dim: int = 256,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.value_network = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Compute state values.
        
        Args:
            state_features: (batch_size, state_feature_dim)
            
        Returns:
            values: (batch_size, 1) estimated values
        """
        return self.value_network(state_features)


class PolicyGradientAgent:
    """
    Complete RL agent combining policy and value networks for SAFE training.
    """
    
    def __init__(
        self,
        state_feature_extractor: StateFeatureExtractor,
        policy_config: dict = None,
        value_config: dict = None,
        device: str = "cuda"
    ):
        self.device = device
        self.state_feature_extractor = state_feature_extractor.to(device)
        
        policy_config = policy_config or {}
        value_config = value_config or {}
        
        # Initialize networks
        self.policy_network = AudioPolicyNetwork(**policy_config).to(device)
        self.value_network = ValueNetwork(**value_config).to(device)
        
        # Training statistics
        self.training_stats = {
            "episodes": 0,
            "total_reward": 0.0,
            "audio_usage_rate": 0.0,
            "average_tokens": 0.0
        }
    
    def get_actions(
        self,
        questions: List[str],
        question_ids: torch.Tensor,
        question_embeddings: torch.Tensor,
        clip_features: torch.Tensor,
        vl_logits: torch.Tensor,
        clap_features: torch.Tensor,
        audio: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[List[PolicyAction], torch.Tensor, torch.Tensor]:
        """
        Get actions from current policy.
        
        Returns:
            Tuple of (actions, state_features, values)
        """
        # Extract state features
        state_features = self.state_feature_extractor(
            questions=questions,
            question_ids=question_ids,
            question_embeddings=question_embeddings,
            clip_features=clip_features,
            vl_logits=vl_logits,
            clap_features=clap_features,
            audio=audio
        )
        
        # Get policy actions
        actions = self.policy_network.get_actions(state_features, deterministic)
        
        # Get state values
        values = self.value_network(state_features)
        
        return actions, state_features, values
    
    def compute_action_log_probs(
        self,
        state_features: torch.Tensor,
        actions: List[PolicyAction]
    ) -> torch.Tensor:
        """Compute log probabilities of actions."""
        return self.policy_network.compute_log_probs(state_features, actions)
    
    def compute_entropy(self, state_features: torch.Tensor) -> torch.Tensor:
        """Compute policy entropy."""
        return self.policy_network.entropy(state_features)
    
    def update_stats(self, rewards: torch.Tensor, actions: List[PolicyAction]):
        """Update training statistics."""
        self.training_stats["episodes"] += len(rewards)
        self.training_stats["total_reward"] += torch.sum(rewards).item()
        
        audio_usage = sum(1 for action in actions if action.consult_audio)
        self.training_stats["audio_usage_rate"] = audio_usage / len(actions)
        
        avg_tokens = np.mean([action.num_tokens for action in actions])
        self.training_stats["average_tokens"] = avg_tokens
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics."""
        if self.training_stats["episodes"] > 0:
            stats = self.training_stats.copy()
            stats["average_reward"] = stats["total_reward"] / stats["episodes"]
            return stats
        return self.training_stats