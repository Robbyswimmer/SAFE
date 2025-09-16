"""
Configuration Management System for SAFE Experiments

This module provides utilities for loading, composing, and validating
experiment configurations from YAML files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from .model_configs import get_config as get_model_config

@dataclass
class ConfigPaths:
    """Paths to different configuration directories."""
    base_dir: Path = Path(__file__).parent
    experiments: Path = None
    training: Path = None
    datasets: Path = None
    curriculum: Path = None
    
    def __post_init__(self):
        if self.experiments is None:
            self.experiments = self.base_dir / "experiments"
        if self.training is None:
            self.training = self.base_dir / "training"
        if self.datasets is None:
            self.datasets = self.base_dir / "datasets"
        if self.curriculum is None:
            self.curriculum = self.base_dir / "curriculum"


class ConfigurationError(Exception):
    """Raised when there's an error in configuration loading or validation."""
    pass


class ConfigManager:
    """Manages loading and composition of SAFE experiment configurations."""
    
    def __init__(self, config_paths: Optional[ConfigPaths] = None):
        self.paths = config_paths or ConfigPaths()
        
    def load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
            
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            return config or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML file {file_path}: {e}")
    
    def load_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """Load an experiment configuration file."""
        file_path = self.paths.experiments / f"{experiment_name}.yaml"
        return self.load_yaml(file_path)
    
    def load_training_config(self, training_name: str) -> Dict[str, Any]:
        """Load a training configuration file."""
        file_path = self.paths.training / f"{training_name}.yaml"
        return self.load_yaml(file_path)
        
    def load_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """Load a dataset configuration file."""
        file_path = self.paths.datasets / f"{dataset_name}.yaml"
        return self.load_yaml(file_path)
    
    def compose_config(self, experiment_name: str) -> Dict[str, Any]:
        """
        Compose a full configuration by combining experiment, training, dataset, and model configs.
        
        Args:
            experiment_name: Name of the experiment configuration
            
        Returns:
            Complete composed configuration dictionary
        """
        # Load experiment config
        experiment_config = self.load_experiment_config(experiment_name)
        
        # Get base config names
        base_configs = experiment_config.get("base_configs", {})
        training_name = base_configs.get("training", "micro")
        dataset_name = base_configs.get("dataset", "debug")
        model_name = base_configs.get("model", "demo")
        
        # Load base configurations
        training_config = self.load_training_config(training_name)
        dataset_config = self.load_dataset_config(dataset_name)
        model_config = get_model_config(model_name)
        
        # Apply overrides from experiment config
        overrides = experiment_config.get("overrides", {})
        
        # Override training config
        if "training" in overrides:
            training_config.update(overrides["training"])
            
        # Override dataset config  
        if "dataset" in overrides:
            dataset_config.update(overrides["dataset"])
            
        # Override model config
        if "model" in overrides:
            model_config.update(overrides["model"])
        
        # Compose final configuration
        composed_config = {
            "experiment": {
                "name": experiment_config["name"],
                "description": experiment_config["description"],
                "experiment_type": experiment_config.get("experiment_type", "general"),
                **{k: v for k, v in experiment_config.items() 
                   if k not in ["name", "description", "base_configs", "overrides"]}
            },
            "training": training_config,
            "dataset": dataset_config,
            "model": model_config
        }
        
        return composed_config
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate a composed configuration and return any warnings.
        
        Args:
            config: Composed configuration dictionary
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Check resource requirements
        training_config = config.get("training", {})
        model_config = config.get("model", {})
        dataset_config = config.get("dataset", {})
        
        # Memory validation
        expected_memory = model_config.get("expected_vram_gb", 8)
        batch_size = training_config.get("batch_size", 4)
        dataset_size = dataset_config.get("train_size", 100)
        
        if expected_memory > 16 and batch_size > 4:
            warnings.append(f"High memory usage: {expected_memory}GB with batch size {batch_size}")
            
        # Runtime validation
        expected_runtime = config.get("experiment", {}).get("expected_runtime_minutes", 30)
        num_epochs = training_config.get("num_epochs", 5)
        
        estimated_runtime = (dataset_size / batch_size) * num_epochs * 0.5  # rough estimate
        if estimated_runtime > expected_runtime * 2:
            warnings.append(f"Estimated runtime ({estimated_runtime:.1f}min) much longer than expected ({expected_runtime}min)")
            
        # Dataset size validation
        train_size = dataset_config.get("train_size", 100)
        val_size = dataset_config.get("val_size", 50)
        max_eval_batches = training_config.get("max_eval_batches", 10)
        
        val_batches = val_size / batch_size
        if max_eval_batches > val_batches:
            warnings.append(f"max_eval_batches ({max_eval_batches}) > validation batches ({val_batches:.1f})")
            
        return warnings
    
    def list_available_configs(self) -> Dict[str, List[str]]:
        """List all available configuration files."""
        configs = {
            "experiments": [],
            "training": [],
            "datasets": [],
        }
        
        for config_type, path in [
            ("experiments", self.paths.experiments),
            ("training", self.paths.training), 
            ("datasets", self.paths.datasets)
        ]:
            if path.exists():
                configs[config_type] = [
                    f.stem for f in path.glob("*.yaml")
                ]
                
        return configs
    
    def print_config_summary(self, config: Dict[str, Any]):
        """Print a human-readable summary of a configuration."""
        experiment = config.get("experiment", {})
        training = config.get("training", {})
        dataset = config.get("dataset", {})
        model = config.get("model", {})
        
        print(f"\n📋 Experiment Configuration: {experiment.get('name', 'Unknown')}")
        print("=" * 60)
        print(f"Description: {experiment.get('description', 'No description')}")
        print(f"Type: {experiment.get('experiment_type', 'general')}")
        
        print(f"\n🎯 Training Settings:")
        print(f"  Epochs: {training.get('num_epochs', 'N/A')}")
        print(f"  Batch Size: {training.get('batch_size', 'N/A')}")
        print(f"  Max Eval Batches: {training.get('max_eval_batches', 'N/A')}")
        print(f"  Learning Rates: proj={training.get('learning_rate_projector', 'N/A')}, adapter={training.get('learning_rate_adapter', 'N/A')}")
        
        print(f"\n📊 Dataset Settings:")
        print(f"  Train Size: {dataset.get('train_size', 'N/A')}")
        print(f"  Val Size: {dataset.get('val_size', 'N/A')}")
        print(f"  Use Dummy Data: {dataset.get('use_dummy_data', True)}")
        
        print(f"\n🤖 Model Settings:")
        print(f"  LLM: {model.get('llm_model_name', 'N/A')}")
        print(f"  Vision: {model.get('vision_model_name', 'N/A')}")
        print(f"  Audio: {model.get('audio_encoder_type', 'N/A')}")
        print(f"  Expected VRAM: {model.get('expected_vram_gb', 'N/A')}GB")
        
        print(f"\n⏱️  Performance Estimates:")
        print(f"  Runtime: {experiment.get('expected_runtime_minutes', 'N/A')} minutes")
        print(f"  Memory: {experiment.get('expected_memory_usage_gb', model.get('expected_vram_gb', 'N/A'))}GB")
        print(f"  Compute: {experiment.get('compute_requirements', 'N/A')}")


def load_experiment(experiment_name: str) -> Dict[str, Any]:
    """
    Load and compose a complete experiment configuration.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Complete composed configuration
    """
    manager = ConfigManager()
    return manager.compose_config(experiment_name)


def validate_experiment(experiment_name: str) -> List[str]:
    """
    Validate an experiment configuration.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        List of validation warnings
    """
    manager = ConfigManager()
    config = manager.compose_config(experiment_name)
    return manager.validate_config(config)


def print_experiment_summary(experiment_name: str):
    """Print a summary of an experiment configuration."""
    manager = ConfigManager()
    config = manager.compose_config(experiment_name)
    warnings = manager.validate_config(config)
    
    manager.print_config_summary(config)
    
    if warnings:
        print(f"\n⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
            
    print()


if __name__ == "__main__":
    # Example usage and testing
    manager = ConfigManager()
    available = manager.list_available_configs()
    
    print("Available Configurations:")
    for config_type, names in available.items():
        print(f"  {config_type}: {', '.join(names)}")
    
    # Test loading debug experiment
    try:
        print_experiment_summary("debug")
    except Exception as e:
        print(f"Error loading debug experiment: {e}")