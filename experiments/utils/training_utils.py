"""
Training Utilities

Shared utilities for training experiments including learning curve analysis,
overfitting detection, and training progress monitoring.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from collections import deque
import time

from .validation_metrics import ValidationResult

class LearningCurveAnalyzer:
    """Analyzes and tracks learning curves during training."""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.metrics_history = {}
        self.smoothed_history = {}
        
    def add_metrics(self, step: int, metrics: Dict[str, float]):
        """Add metrics for a training step."""
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
                self.smoothed_history[metric_name] = deque(maxlen=self.window_size)
            
            self.metrics_history[metric_name].append((step, value))
            self.smoothed_history[metric_name].append(value)
    
    def get_smoothed_value(self, metric_name: str) -> Optional[float]:
        """Get smoothed value for a metric."""
        if metric_name in self.smoothed_history:
            values = list(self.smoothed_history[metric_name])
            return np.mean(values) if values else None
        return None
    
    def get_trend(self, metric_name: str, min_points: int = 10) -> Optional[str]:
        """Determine trend for a metric (improving, degrading, stable)."""
        if metric_name not in self.metrics_history:
            return None
            
        history = self.metrics_history[metric_name]
        if len(history) < min_points:
            return None
        
        # Look at recent points vs earlier points
        recent_points = history[-min_points//2:]
        earlier_points = history[-min_points:-min_points//2]
        
        recent_avg = np.mean([val for _, val in recent_points])
        earlier_avg = np.mean([val for _, val in earlier_points])
        
        # For loss metrics, decreasing is improving
        if 'loss' in metric_name.lower():
            if recent_avg < earlier_avg - 0.01:  # Significant decrease
                return 'improving'
            elif recent_avg > earlier_avg + 0.01:  # Significant increase
                return 'degrading'
            else:
                return 'stable'
        else:
            # For accuracy metrics, increasing is improving
            if recent_avg > earlier_avg + 0.01:  # Significant increase
                return 'improving'
            elif recent_avg < earlier_avg - 0.01:  # Significant decrease
                return 'degrading'
            else:
                return 'stable'
    
    def detect_overfitting(self, train_loss_name: str = 'train_loss', 
                          val_loss_name: str = 'val_loss',
                          min_points: int = 20,
                          divergence_threshold: float = 0.1) -> Dict[str, Any]:
        """Detect overfitting by comparing train and validation loss trends."""
        result = {
            'overfitting_detected': False,
            'confidence': 0.0,
            'train_trend': None,
            'val_trend': None,
            'divergence': 0.0
        }
        
        if (train_loss_name not in self.metrics_history or 
            val_loss_name not in self.metrics_history):
            return result
        
        train_history = self.metrics_history[train_loss_name]
        val_history = self.metrics_history[val_loss_name]
        
        if len(train_history) < min_points or len(val_history) < min_points:
            return result
        
        # Get trends
        train_trend = self.get_trend(train_loss_name, min_points)
        val_trend = self.get_trend(val_loss_name, min_points)
        
        result['train_trend'] = train_trend
        result['val_trend'] = val_trend
        
        # Calculate divergence (recent validation vs training loss difference)
        recent_train = np.mean([val for _, val in train_history[-10:]])
        recent_val = np.mean([val for _, val in val_history[-10:]])
        
        result['divergence'] = recent_val - recent_train
        
        # Overfitting indicators:
        # 1. Training loss improving but validation loss degrading/stable
        # 2. Significant divergence between train and val loss
        overfitting_signals = []
        
        if train_trend == 'improving' and val_trend in ['degrading', 'stable']:
            overfitting_signals.append(0.6)
        
        if result['divergence'] > divergence_threshold:
            overfitting_signals.append(0.4)
        
        if overfitting_signals:
            result['overfitting_detected'] = True
            result['confidence'] = sum(overfitting_signals)
        
        return result
    
    def plot_learning_curves(self, output_path: Path, title: str = "Learning Curves"):
        """Generate learning curve plots."""
        if not self.metrics_history:
            return
        
        # Separate metrics by type
        loss_metrics = {k: v for k, v in self.metrics_history.items() if 'loss' in k.lower()}
        accuracy_metrics = {k: v for k, v in self.metrics_history.items() if 'accuracy' in k.lower() or 'acc' in k.lower()}
        other_metrics = {k: v for k, v in self.metrics_history.items() 
                        if k not in loss_metrics and k not in accuracy_metrics}
        
        # Determine subplot layout
        num_plots = sum([len(loss_metrics) > 0, len(accuracy_metrics) > 0, len(other_metrics) > 0])
        if num_plots == 0:
            return
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot loss metrics
        if loss_metrics:
            ax = axes[plot_idx]
            for metric_name, history in loss_metrics.items():
                steps, values = zip(*history)
                ax.plot(steps, values, label=metric_name, alpha=0.7)
                
                # Add smoothed line
                if len(values) > 5:
                    smoothed = np.convolve(values, np.ones(min(5, len(values)))/min(5, len(values)), mode='valid')
                    smoothed_steps = steps[len(steps)-len(smoothed):]
                    ax.plot(smoothed_steps, smoothed, '--', alpha=0.9, linewidth=2)
            
            ax.set_title('Loss Metrics')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot accuracy metrics
        if accuracy_metrics:
            ax = axes[plot_idx]
            for metric_name, history in accuracy_metrics.items():
                steps, values = zip(*history)
                ax.plot(steps, values, label=metric_name, alpha=0.7)
                
                # Add smoothed line
                if len(values) > 5:
                    smoothed = np.convolve(values, np.ones(min(5, len(values)))/min(5, len(values)), mode='valid')
                    smoothed_steps = steps[len(steps)-len(smoothed):]
                    ax.plot(smoothed_steps, smoothed, '--', alpha=0.9, linewidth=2)
            
            ax.set_title('Accuracy Metrics')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot other metrics
        if other_metrics:
            ax = axes[plot_idx]
            for metric_name, history in other_metrics.items():
                steps, values = zip(*history)
                ax.plot(steps, values, label=metric_name, alpha=0.7)
            
            ax.set_title('Other Metrics')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

class TrainingProgressMonitor:
    """Monitors training progress and provides real-time feedback."""
    
    def __init__(self, target_metrics: Dict[str, float] = None):
        self.target_metrics = target_metrics or {}
        self.start_time = None
        self.step_times = []
        self.current_metrics = {}
        
    def start_training(self):
        """Mark start of training."""
        self.start_time = time.time()
        
    def step_completed(self, step: int, metrics: Dict[str, float], step_time: float = None):
        """Record completion of a training step."""
        if step_time is not None:
            self.step_times.append(step_time)
        
        self.current_metrics = metrics.copy()
        
    def get_progress_stats(self) -> Dict[str, Any]:
        """Get current training progress statistics."""
        if not self.start_time:
            return {}
        
        elapsed_time = time.time() - self.start_time
        avg_step_time = np.mean(self.step_times) if self.step_times else 0
        
        stats = {
            'elapsed_time': elapsed_time,
            'avg_step_time': avg_step_time,
            'steps_completed': len(self.step_times),
            'current_metrics': self.current_metrics
        }
        
        # Check target achievement
        target_achievement = {}
        for metric_name, target_value in self.target_metrics.items():
            if metric_name in self.current_metrics:
                current_value = self.current_metrics[metric_name]
                
                # For loss metrics, we want lower values
                if 'loss' in metric_name.lower():
                    achieved = current_value <= target_value
                    progress = max(0, 1 - (current_value / target_value)) if target_value > 0 else 0
                else:
                    # For accuracy metrics, we want higher values
                    achieved = current_value >= target_value
                    progress = min(1, current_value / target_value) if target_value > 0 else 0
                
                target_achievement[metric_name] = {
                    'achieved': achieved,
                    'progress': progress,
                    'current': current_value,
                    'target': target_value
                }
        
        stats['target_achievement'] = target_achievement
        return stats
    
    def check_early_stopping(self, patience_metrics: Dict[str, Tuple[int, str]] = None) -> bool:
        """Check if early stopping criteria are met."""
        # This is a placeholder for more sophisticated early stopping logic
        # Could be extended based on specific requirements
        return False

class OverfittingDetector:
    """Specialized detector for overfitting in small-scale training."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def update(self, val_loss: float) -> Dict[str, Any]:
        """Update with new validation loss and check for overfitting."""
        result = {
            'overfitting_detected': False,
            'patience_counter': self.patience_counter,
            'best_val_loss': self.best_val_loss
        }
        
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        result['patience_counter'] = self.patience_counter
        result['best_val_loss'] = self.best_val_loss
        
        if self.patience_counter >= self.patience:
            result['overfitting_detected'] = True
        
        return result

def calculate_training_metrics(predictions: torch.Tensor, 
                             targets: torch.Tensor,
                             task_type: str = "classification") -> Dict[str, float]:
    """Calculate standard training metrics."""
    metrics = {}
    
    if task_type == "classification":
        # Accuracy
        if predictions.dim() > 1 and predictions.size(1) > 1:
            pred_classes = torch.argmax(predictions, dim=1)
            target_classes = targets if targets.dim() == 1 else torch.argmax(targets, dim=1)
            accuracy = (pred_classes == target_classes).float().mean().item()
            metrics['accuracy'] = accuracy
        
        # Cross-entropy loss
        if predictions.dim() > 1 and targets.dim() == 1:
            loss = torch.nn.functional.cross_entropy(predictions, targets).item()
            metrics['cross_entropy_loss'] = loss
    
    elif task_type == "regression":
        # MSE loss
        mse_loss = torch.nn.functional.mse_loss(predictions, targets).item()
        metrics['mse_loss'] = mse_loss
        
        # MAE
        mae = torch.abs(predictions - targets).mean().item()
        metrics['mae'] = mae
    
    return metrics

def setup_tiny_training_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup configuration optimized for tiny-scale training."""
    tiny_config = base_config.copy()
    
    # Tiny-scale optimizations
    tiny_config.update({
        'batch_size': 4,  # Small batch size
        'learning_rate': 5e-4,  # Higher learning rate for fast convergence
        'num_epochs': 50,  # More epochs for overfitting
        'warmup_steps': 10,  # Minimal warmup
        'eval_steps': 10,  # Frequent evaluation
        'logging_steps': 5,  # Frequent logging
        'save_steps': 50,  # Less frequent saving
        'gradient_accumulation_steps': 1,  # No accumulation for tiny batches
        'max_grad_norm': 1.0,  # Gradient clipping
        'weight_decay': 0.01,  # Minimal regularization
    })
    
    return tiny_config