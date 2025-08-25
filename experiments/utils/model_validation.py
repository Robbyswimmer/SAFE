"""
Model Validation Utilities

Shared validation functions for model safety checks and parameter analysis.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from collections import defaultdict

from .validation_metrics import ValidationResult

class ModelParameterAnalyzer:
    """Analyzes model parameter structure and trainability."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.parameter_info = None
    
    def analyze_parameters(self) -> Dict[str, Any]:
        """Analyze all model parameters and their trainability."""
        param_info = {
            'components': {},
            'totals': {
                'total_params': 0,
                'trainable_params': 0,
                'frozen_params': 0
            },
            'detailed_breakdown': []
        }
        
        # Track parameters by component
        for name, param in self.model.named_parameters():
            # Determine component based on parameter name
            component = self._classify_parameter(name)
            
            param_count = param.numel()
            is_trainable = param.requires_grad
            
            # Add to component tracking
            if component not in param_info['components']:
                param_info['components'][component] = {
                    'total_params': 0,
                    'trainable_params': 0,
                    'frozen_params': 0,
                    'parameters': []
                }
            
            param_info['components'][component]['total_params'] += param_count
            if is_trainable:
                param_info['components'][component]['trainable_params'] += param_count
            else:
                param_info['components'][component]['frozen_params'] += param_count
            
            # Detailed parameter info
            param_detail = {
                'name': name,
                'component': component,
                'shape': list(param.shape),
                'param_count': param_count,
                'requires_grad': is_trainable,
                'dtype': str(param.dtype)
            }
            
            param_info['components'][component]['parameters'].append(param_detail)
            param_info['detailed_breakdown'].append(param_detail)
            
            # Update totals
            param_info['totals']['total_params'] += param_count
            if is_trainable:
                param_info['totals']['trainable_params'] += param_count
            else:
                param_info['totals']['frozen_params'] += param_count
        
        # Calculate ratios
        total = param_info['totals']['total_params']
        if total > 0:
            param_info['totals']['trainable_ratio'] = param_info['totals']['trainable_params'] / total
            param_info['totals']['frozen_ratio'] = param_info['totals']['frozen_params'] / total
        
        # Add component ratios
        for component_data in param_info['components'].values():
            comp_total = component_data['total_params']
            if comp_total > 0:
                component_data['trainable_ratio'] = component_data['trainable_params'] / comp_total
                component_data['frozen_ratio'] = component_data['frozen_params'] / comp_total
        
        self.parameter_info = param_info
        return param_info
    
    def _classify_parameter(self, param_name: str) -> str:
        """Classify parameter by model component."""
        name_lower = param_name.lower()
        
        # Base VL model components
        if any(keyword in name_lower for keyword in ['base_vl', 'llm', 'language_model', 'transformer']):
            if 'vision' in name_lower or 'image' in name_lower:
                return 'base_vl_vision'
            else:
                return 'base_vl_language'
        
        # Vision encoder
        if any(keyword in name_lower for keyword in ['vision_encoder', 'image_encoder', 'clip', 'vit']):
            return 'vision_encoder'
        
        # Audio components
        if any(keyword in name_lower for keyword in ['audio_encoder', 'clap', 'whisper']):
            return 'audio_encoder'
        
        if any(keyword in name_lower for keyword in ['audio_projector', 'projector']):
            return 'audio_projector'
        
        # Fusion components
        if any(keyword in name_lower for keyword in ['fusion', 'lora', 'adapter']):
            return 'fusion_adapter'
        
        # Token embeddings
        if 'embed' in name_lower or 'token' in name_lower:
            return 'embeddings'
        
        # Default
        return 'other'
    
    def validate_freeze_requirements(self, expected_frozen: List[str], expected_trainable: List[str]) -> List[ValidationResult]:
        """Validate that specific components are properly frozen/trainable."""
        if not self.parameter_info:
            self.analyze_parameters()
        
        validations = []
        components = self.parameter_info['components']
        
        # Check frozen components
        for component in expected_frozen:
            if component in components:
                frozen_ratio = components[component].get('frozen_ratio', 0)
                is_fully_frozen = frozen_ratio == 1.0
                
                validation = ValidationResult(
                    f"{component}_freeze_check",
                    is_fully_frozen,
                    f"{component}: {frozen_ratio:.1%} frozen ({components[component]['frozen_params']:,} / {components[component]['total_params']:,} params)",
                    frozen_ratio, 1.0
                )
                validations.append(validation)
        
        # Check trainable components
        for component in expected_trainable:
            if component in components:
                trainable_ratio = components[component].get('trainable_ratio', 0)
                has_trainable = trainable_ratio > 0
                
                validation = ValidationResult(
                    f"{component}_trainable_check",
                    has_trainable,
                    f"{component}: {trainable_ratio:.1%} trainable ({components[component]['trainable_params']:,} / {components[component]['total_params']:,} params)",
                    trainable_ratio, 0.0
                )
                validations.append(validation)
        
        return validations
    
    def validate_trainable_ratio(self, min_ratio: float = 0.001, max_ratio: float = 0.005) -> ValidationResult:
        """Validate that total trainable ratio is within expected bounds."""
        if not self.parameter_info:
            self.analyze_parameters()
        
        trainable_ratio = self.parameter_info['totals']['trainable_ratio']
        in_range = min_ratio <= trainable_ratio <= max_ratio
        
        return ValidationResult(
            "trainable_parameter_ratio",
            in_range,
            f"Trainable ratio: {trainable_ratio:.3%} ({self.parameter_info['totals']['trainable_params']:,} / {self.parameter_info['totals']['total_params']:,})",
            trainable_ratio, (min_ratio, max_ratio)
        )

class ModelEquivalenceTester:
    """Tests equivalence between different models or configurations."""
    
    @staticmethod
    def compare_model_outputs(output1: Dict[str, torch.Tensor], 
                            output2: Dict[str, torch.Tensor],
                            tolerance: float = 1e-5) -> Dict[str, Any]:
        """Compare outputs from two models."""
        comparison_results = {
            'keys_match': set(output1.keys()) == set(output2.keys()),
            'common_keys': list(set(output1.keys()) & set(output2.keys())),
            'differences': {},
            'max_difference': 0.0,
            'mean_difference': 0.0,
            'within_tolerance': True
        }
        
        total_diff = 0.0
        total_elements = 0
        max_diff = 0.0
        
        for key in comparison_results['common_keys']:
            if key in output1 and key in output2:
                tensor1, tensor2 = output1[key], output2[key]
                
                if isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
                    if tensor1.shape == tensor2.shape:
                        diff = torch.abs(tensor1 - tensor2)
                        key_max_diff = torch.max(diff).item()
                        key_mean_diff = torch.mean(diff).item()
                        
                        comparison_results['differences'][key] = {
                            'shape': list(tensor1.shape),
                            'max_diff': key_max_diff,
                            'mean_diff': key_mean_diff,
                            'within_tolerance': key_max_diff < tolerance
                        }
                        
                        max_diff = max(max_diff, key_max_diff)
                        total_diff += key_mean_diff * tensor1.numel()
                        total_elements += tensor1.numel()
                        
                        if key_max_diff >= tolerance:
                            comparison_results['within_tolerance'] = False
                    else:
                        comparison_results['differences'][key] = {
                            'shape_mismatch': True,
                            'shape1': list(tensor1.shape),
                            'shape2': list(tensor2.shape)
                        }
                        comparison_results['within_tolerance'] = False
        
        comparison_results['max_difference'] = max_diff
        comparison_results['mean_difference'] = total_diff / total_elements if total_elements > 0 else 0.0
        
        return comparison_results
    
    @staticmethod
    def compare_accuracy_metrics(metrics1: Dict[str, float], 
                               metrics2: Dict[str, float],
                               tolerance: float = 1e-6) -> ValidationResult:
        """Compare accuracy metrics between two models."""
        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        
        if not common_metrics:
            return ValidationResult(
                "accuracy_comparison",
                False,
                "No common metrics found between models",
                0, tolerance
            )
        
        max_diff = 0.0
        differences = {}
        
        for metric in common_metrics:
            diff = abs(metrics1[metric] - metrics2[metric])
            differences[metric] = diff
            max_diff = max(max_diff, diff)
        
        within_tolerance = max_diff < tolerance
        
        return ValidationResult(
            "accuracy_comparison",
            within_tolerance,
            f"Max accuracy difference: {max_diff:.2e} across {len(common_metrics)} metrics",
            max_diff, tolerance,
            metadata={'differences': differences}
        )
    
    @staticmethod
    def measure_computational_overhead(time_baseline: float,
                                     time_test: float,
                                     max_overhead: float = 0.05) -> ValidationResult:
        """Measure computational overhead between two model configurations."""
        overhead = (time_test - time_baseline) / time_baseline if time_baseline > 0 else float('inf')
        
        within_limit = overhead <= max_overhead
        
        return ValidationResult(
            "computational_overhead",
            within_limit,
            f"Computational overhead: {overhead:.1%} (baseline: {time_baseline:.3f}s, test: {time_test:.3f}s)",
            overhead, max_overhead
        )

def count_model_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params
    }

def format_parameter_count(count: int) -> str:
    """Format parameter count with appropriate units."""
    if count >= 1e9:
        return f"{count / 1e9:.1f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.1f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.1f}K"
    else:
        return str(count)