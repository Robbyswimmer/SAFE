"""
Ablation Experiment Configurations for SAFE Research Plan

This file defines all ablation experiments needed to validate SAFE as
a scalable incremental learning framework.

Usage:
    from configs.ablation_experiments import ABLATION_CONFIGS, get_experiment_configs
    configs = get_experiment_configs("projector_depth")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from itertools import product


# =============================================================================
# Phase 1: Audio Baseline Ablations
# =============================================================================

PROJECTOR_ABLATIONS = {
    "depth": {
        "description": "Projector layer depth study",
        "base_config": {
            "bottleneck_dim": 1024,
            "num_audio_tokens": 8,
            "activation": "gelu",
            "dropout": 0.1,
        },
        "sweep": {
            "num_layers": [1, 2, 3, 4],
        },
        "seeds": [42, 123, 456],
        "priority": "P0",
    },

    "width": {
        "description": "Projector hidden dimension study",
        "base_config": {
            "num_layers": 2,
            "num_audio_tokens": 8,
            "activation": "gelu",
            "dropout": 0.1,
        },
        "sweep": {
            "bottleneck_dim": [512, 1024, 2048, 4096],
        },
        "seeds": [42, 123, 456],
        "priority": "P0",
    },

    "tokens": {
        "description": "Number of audio tokens study",
        "base_config": {
            "num_layers": 2,
            "bottleneck_dim": 1024,
            "activation": "gelu",
            "dropout": 0.1,
        },
        "sweep": {
            "num_audio_tokens": [4, 8, 16, 32],
        },
        "seeds": [42, 123, 456],
        "priority": "P0",
    },

    "activation": {
        "description": "Activation function comparison",
        "base_config": {
            "num_layers": 2,
            "bottleneck_dim": 1024,
            "num_audio_tokens": 8,
            "dropout": 0.1,
        },
        "sweep": {
            "activation": ["gelu", "relu", "silu", "tanh"],
        },
        "seeds": [42, 123, 456],
        "priority": "P1",
    },
}


FUSION_ABLATIONS = {
    "lora_rank": {
        "description": "LoRA rank vs performance",
        "base_config": {
            "lora_alpha": 16.0,
            "target_modules": ["query", "value"],
            "residual_scale": 0.05,
        },
        "sweep": {
            "lora_rank": [4, 8, 16, 32, 64],
        },
        "seeds": [42, 123, 456],
        "priority": "P0",
    },

    "lora_alpha": {
        "description": "LoRA alpha scaling study",
        "base_config": {
            "lora_rank": 8,
            "target_modules": ["query", "value"],
            "residual_scale": 0.05,
        },
        "sweep": {
            "lora_alpha": [8.0, 16.0, 32.0, 64.0],
        },
        "seeds": [42, 123, 456],
        "priority": "P1",
    },

    "residual_scale": {
        "description": "Residual connection scaling",
        "base_config": {
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "target_modules": ["query", "value"],
        },
        "sweep": {
            "residual_scale": [0.01, 0.05, 0.1, 0.2, 0.5],
        },
        "seeds": [42, 123, 456],
        "priority": "P0",
    },

    "target_modules": {
        "description": "Which modules to apply LoRA to",
        "base_config": {
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "residual_scale": 0.05,
        },
        "sweep": {
            "target_modules": [
                ["query", "value"],
                ["query", "key", "value"],
                ["query", "key", "value", "output"],
            ],
        },
        "seeds": [42, 123, 456],
        "priority": "P1",
    },

    "fusion_layers": {
        "description": "Which LLM layers to fuse at",
        "base_config": {
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "residual_scale": 0.05,
            "target_modules": ["query", "value"],
        },
        "sweep": {
            "fusion_layer_indices": [
                [0],  # Embed only
                [6, 12],  # Early + mid
                [6, 12, 24],  # Distributed
                list(range(0, 32, 4)),  # Every 4th layer
            ],
        },
        "seeds": [42, 123, 456],
        "priority": "P1",
    },
}


RETENTION_ABLATIONS = {
    "strategy": {
        "description": "Retention strategy comparison",
        "base_config": {
            "kl_temperature": 2.0,
        },
        "sweep": {
            "retention_strategy": [
                "no_retention",
                "soft_retention",
                "fisher_retention",
                "nullspace_retention",
                "full_retention",
            ],
        },
        "seeds": [42, 123, 456],
        "priority": "P0",
    },

    "retention_weight": {
        "description": "Retention loss weight",
        "base_config": {
            "retention_strategy": "fisher_retention",
            "kl_temperature": 2.0,
        },
        "sweep": {
            "retention_weight": [0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
        },
        "seeds": [42, 123, 456],
        "priority": "P0",
    },

    "kl_temperature": {
        "description": "KL divergence temperature",
        "base_config": {
            "retention_strategy": "soft_retention",
            "retention_weight": 0.1,
        },
        "sweep": {
            "kl_temperature": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        "seeds": [42, 123, 456],
        "priority": "P1",
    },
}


TRAINING_ABLATIONS = {
    "gate_warmup": {
        "description": "Gate warmup schedule",
        "base_config": {
            "warmup_steps": 2000,
        },
        "sweep": {
            "gate_warmup_schedule": ["linear", "cosine", "step", "none"],
        },
        "seeds": [42, 123, 456],
        "priority": "P1",
    },

    "learning_rate": {
        "description": "Learning rate study",
        "base_config": {},
        "sweep": {
            "projector_lr": [1e-4, 2e-4, 5e-4, 1e-3],
            "fusion_lr": [5e-5, 1e-4, 2e-4],
        },
        "seeds": [42, 123, 456],
        "priority": "P1",
    },

    "batch_size": {
        "description": "Batch size study",
        "base_config": {},
        "sweep": {
            "batch_size": [8, 16, 32, 64],
        },
        "seeds": [42, 123, 456],
        "priority": "P2",
    },
}


# =============================================================================
# Phase 2: Multi-Modal Expansion
# =============================================================================

VIDEO_EXPERIMENTS = {
    "encoder_comparison": {
        "description": "Video encoder selection",
        "base_config": {
            "projector_type": "mlp",
            "num_layers": 2,
            "num_video_tokens": 8,
        },
        "sweep": {
            "video_encoder": [
                "clip_vit_base",
                "clip_vit_large",
                "videomae_base",
                "internvideo",
            ],
        },
        "seeds": [42, 123, 456],
        "priority": "P0",
    },

    "transfer_from_audio": {
        "description": "Transfer audio projector to video",
        "base_config": {
            "video_encoder": "clip_vit_base",
            "init_from": "best_audio_projector",
        },
        "sweep": {
            "transfer_mode": ["full_transfer", "partial_transfer", "random_init"],
        },
        "seeds": [42, 123, 456],
        "priority": "P0",
    },

    "temporal_modeling": {
        "description": "Temporal aggregation for video",
        "base_config": {
            "video_encoder": "clip_vit_base",
            "num_frames": 8,
        },
        "sweep": {
            "temporal_aggregation": ["mean", "max", "attention", "transformer"],
            "num_frames": [4, 8, 16, 32],
        },
        "seeds": [42, 123, 456],
        "priority": "P1",
    },
}


DEPTH_EXPERIMENTS = {
    "encoder_comparison": {
        "description": "Depth encoder selection",
        "base_config": {
            "projector_type": "spatial_pooling",
            "num_depth_tokens": 8,
        },
        "sweep": {
            "depth_encoder": ["depth_anything", "midas", "dpt"],
        },
        "seeds": [42, 123, 456],
        "priority": "P0",
    },

    "spatial_reduction": {
        "description": "Spatial reduction strategy",
        "base_config": {
            "depth_encoder": "depth_anything",
            "num_depth_tokens": 8,
        },
        "sweep": {
            "spatial_reduction": ["adaptive_avg", "adaptive_max", "conv", "attention"],
        },
        "seeds": [42, 123, 456],
        "priority": "P1",
    },
}


TACTILE_EXPERIMENTS = {
    "encoder_comparison": {
        "description": "Tactile encoder selection",
        "base_config": {
            "projector_type": "mlp",
            "num_tactile_tokens": 4,
        },
        "sweep": {
            "tactile_encoder": ["t3", "touch_and_go", "clip_tactile"],
        },
        "seeds": [42, 123, 456],
        "priority": "P1",
    },
}


MULTIMODAL_EXPERIMENTS = {
    "sequential_addition": {
        "description": "Add modalities one at a time",
        "base_config": {
            "freeze_previous": True,
        },
        "sweep": {
            "modality_order": [
                ["audio"],
                ["audio", "video"],
                ["audio", "video", "depth"],
                ["audio", "video", "depth", "tactile"],
            ],
        },
        "seeds": [42, 123, 456],
        "priority": "P0",
    },

    "joint_training": {
        "description": "Train all modalities jointly",
        "base_config": {
            "training_strategy": "joint",
        },
        "sweep": {
            "modalities": [
                ["audio", "video"],
                ["audio", "video", "depth"],
                ["audio", "video", "depth", "tactile"],
            ],
        },
        "seeds": [42, 123, 456],
        "priority": "P1",
    },

    "curriculum": {
        "description": "Curriculum-based multi-modal training",
        "base_config": {
            "training_strategy": "curriculum",
        },
        "sweep": {
            "curriculum_schedule": [
                "linear",  # Equal time per modality
                "inverse_difficulty",  # More time on harder modalities
                "performance_based",  # Adaptive based on metrics
            ],
        },
        "seeds": [42, 123, 456],
        "priority": "P2",
    },
}


# =============================================================================
# Phase 3: Scaling Experiments
# =============================================================================

SCALING_EXPERIMENTS = {
    "model_size": {
        "description": "Scale with base model size",
        "base_config": {
            "modality": "audio",
            "projector_config": "best_from_ablation",
        },
        "sweep": {
            "base_model": ["llava_7b", "llava_13b", "llava_34b"],
        },
        "seeds": [42, 123, 456],
        "priority": "P1",
    },

    "projector_size": {
        "description": "Scale projector parameters",
        "base_config": {
            "modality": "audio",
            "base_model": "llava_7b",
        },
        "sweep": {
            "projector_params": ["0.5M", "1M", "2M", "4M", "8M"],
        },
        "seeds": [42, 123, 456],
        "priority": "P1",
    },

    "data_size": {
        "description": "Scale with training data",
        "base_config": {
            "modality": "audio",
            "projector_config": "best_from_ablation",
        },
        "sweep": {
            "data_fraction": [0.1, 0.25, 0.5, 0.75, 1.0],
        },
        "seeds": [42, 123, 456],
        "priority": "P0",
    },
}


# =============================================================================
# Baselines
# =============================================================================

BASELINE_EXPERIMENTS = {
    "original_vl": {
        "description": "Original VL model (no audio)",
        "config": {
            "audio_enabled": False,
        },
        "datasets": ["vqa_v2", "gqa", "textvqa"],
        "priority": "P0",
    },

    "full_finetune": {
        "description": "Full fine-tuning on audio",
        "config": {
            "freeze_base": False,
            "train_all_params": True,
        },
        "seeds": [42, 123, 456],
        "priority": "P0",
    },

    "lora_only": {
        "description": "LoRA without gating",
        "config": {
            "use_gate": False,
            "lora_rank": 8,
        },
        "seeds": [42, 123, 456],
        "priority": "P0",
    },

    "linear_probe": {
        "description": "Simple linear projection",
        "config": {
            "projector_type": "linear",
            "num_layers": 1,
        },
        "seeds": [42, 123, 456],
        "priority": "P0",
    },

    "concatenation": {
        "description": "Concatenate audio tokens (no cross-attention)",
        "config": {
            "fusion_type": "concatenation",
        },
        "seeds": [42, 123, 456],
        "priority": "P1",
    },
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_experiment_configs(experiment_name: str) -> List[Dict[str, Any]]:
    """Generate all configurations for a given experiment."""

    all_ablations = {
        **PROJECTOR_ABLATIONS,
        **FUSION_ABLATIONS,
        **RETENTION_ABLATIONS,
        **TRAINING_ABLATIONS,
        **VIDEO_EXPERIMENTS,
        **DEPTH_EXPERIMENTS,
        **TACTILE_EXPERIMENTS,
        **MULTIMODAL_EXPERIMENTS,
        **SCALING_EXPERIMENTS,
    }

    if experiment_name not in all_ablations:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    exp = all_ablations[experiment_name]
    base = exp["base_config"]
    sweep = exp["sweep"]
    seeds = exp.get("seeds", [42])

    configs = []

    # Generate all combinations
    sweep_keys = list(sweep.keys())
    sweep_values = [sweep[k] for k in sweep_keys]

    for values in product(*sweep_values):
        for seed in seeds:
            config = base.copy()
            for key, value in zip(sweep_keys, values):
                config[key] = value
            config["seed"] = seed
            config["experiment_name"] = experiment_name
            configs.append(config)

    return configs


def get_experiment_count(experiment_name: str) -> int:
    """Get the number of runs for an experiment."""
    return len(get_experiment_configs(experiment_name))


def get_priority_experiments(priority: str = "P0") -> List[str]:
    """Get all experiments with given priority."""
    all_ablations = {
        **PROJECTOR_ABLATIONS,
        **FUSION_ABLATIONS,
        **RETENTION_ABLATIONS,
        **TRAINING_ABLATIONS,
        **VIDEO_EXPERIMENTS,
        **DEPTH_EXPERIMENTS,
        **TACTILE_EXPERIMENTS,
        **MULTIMODAL_EXPERIMENTS,
        **SCALING_EXPERIMENTS,
    }

    return [
        name for name, exp in all_ablations.items()
        if exp.get("priority", "P2") == priority
    ]


def estimate_compute_hours(experiment_name: str, hours_per_run: float = 4.0) -> float:
    """Estimate total compute hours for an experiment."""
    num_runs = get_experiment_count(experiment_name)
    return num_runs * hours_per_run


def generate_experiment_summary() -> str:
    """Generate a summary of all experiments."""
    all_ablations = {
        "Projector": PROJECTOR_ABLATIONS,
        "Fusion": FUSION_ABLATIONS,
        "Retention": RETENTION_ABLATIONS,
        "Training": TRAINING_ABLATIONS,
        "Video": VIDEO_EXPERIMENTS,
        "Depth": DEPTH_EXPERIMENTS,
        "Tactile": TACTILE_EXPERIMENTS,
        "Multimodal": MULTIMODAL_EXPERIMENTS,
        "Scaling": SCALING_EXPERIMENTS,
    }

    lines = ["# Experiment Summary\n"]
    total_runs = 0

    for category, experiments in all_ablations.items():
        lines.append(f"\n## {category} Experiments\n")
        lines.append("| Experiment | Description | # Runs | Priority |")
        lines.append("|------------|-------------|--------|----------|")

        for name, exp in experiments.items():
            num_runs = get_experiment_count(name)
            total_runs += num_runs
            priority = exp.get("priority", "P2")
            lines.append(f"| {name} | {exp['description']} | {num_runs} | {priority} |")

    lines.append(f"\n**Total Runs: {total_runs}**")
    lines.append(f"**Estimated Hours (4h/run): {total_runs * 4}**")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Print experiment summary
    print(generate_experiment_summary())

    # Example: Get configs for a specific ablation
    print("\n\nExample configs for 'depth' ablation:")
    for config in get_experiment_configs("depth")[:3]:
        print(f"  {config}")
