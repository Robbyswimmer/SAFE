"""
Retention variant configurations for SAFE experiments.

Each variant represents a different combination of retention mechanisms:
- KL Divergence: Distillation from base VL model outputs
- Fisher Information: Weight importance-based regularization
- Null-Space Projection: Gradient projection away from VL-critical directions

Use these configs to ablate each retention component independently.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RetentionVariantConfig:
    """Configuration for a specific retention variant."""

    name: str
    description: str
    retention_loss_weight: float
    distillation_weight: float
    fisher_weight: float
    enable_null_space: bool
    null_space_rank: int = 8
    null_space_min_samples: int = 128
    null_space_refresh_interval: int = 4000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for trainer config."""
        return {
            "retention_loss_weight": self.retention_loss_weight,
            "distillation_weight": self.distillation_weight,
            "fisher_weight": self.fisher_weight,
            "enable_null_space": self.enable_null_space,
            "null_space_rank": self.null_space_rank,
            "null_space_min_samples": self.null_space_min_samples,
            "null_space_refresh_interval": self.null_space_refresh_interval,
            "compute_fisher_at_start": self.fisher_weight > 0,
        }


# ============================================================================
# Variant Definitions
# ============================================================================

NO_RETENTION = RetentionVariantConfig(
    name="no_retention",
    description="Baseline: No retention mechanisms (pure audio training)",
    retention_loss_weight=0.0,
    distillation_weight=0.0,
    fisher_weight=0.0,
    enable_null_space=False,
)

KL_ONLY = RetentionVariantConfig(
    name="soft_retention",  # Keep existing name for compatibility
    description="KL divergence distillation only",
    retention_loss_weight=0.05,  # Reduced from 0.2 - was overwhelming audio learning
    distillation_weight=0.05,    # Reduced from 0.2 - distillation loss was too high
    fisher_weight=0.0,
    enable_null_space=False,
)

FISHER_RETENTION = RetentionVariantConfig(
    name="fisher_retention",
    description="KL divergence + Fisher information weighting",
    retention_loss_weight=0.2,
    distillation_weight=0.2,
    fisher_weight=0.1,
    enable_null_space=False,
)

NULLSPACE_RETENTION = RetentionVariantConfig(
    name="nullspace_retention",
    description="KL divergence + Null-space gradient projection",
    retention_loss_weight=0.2,
    distillation_weight=0.2,
    fisher_weight=0.0,
    enable_null_space=True,
    null_space_rank=8,
    null_space_min_samples=128,
    null_space_refresh_interval=4000,
)

FULL_RETENTION = RetentionVariantConfig(
    name="full_retention",
    description="All retention mechanisms: KL + Fisher + Null-space",
    retention_loss_weight=0.2,
    distillation_weight=0.2,
    fisher_weight=0.1,
    enable_null_space=True,
    null_space_rank=8,
    null_space_min_samples=128,
    null_space_refresh_interval=4000,
)


# ============================================================================
# Variant Registry
# ============================================================================

RETENTION_VARIANTS = {
    "no_retention": NO_RETENTION,
    "soft_retention": KL_ONLY,
    "fisher_retention": FISHER_RETENTION,
    "nullspace_retention": NULLSPACE_RETENTION,
    "full_retention": FULL_RETENTION,
}


def get_variant_config(variant_name: str) -> RetentionVariantConfig:
    """
    Get retention variant configuration by name.

    Args:
        variant_name: Name of the retention variant

    Returns:
        RetentionVariantConfig for the specified variant

    Raises:
        ValueError: If variant name is not recognized
    """
    if variant_name not in RETENTION_VARIANTS:
        available = ", ".join(RETENTION_VARIANTS.keys())
        raise ValueError(
            f"Unknown retention variant: '{variant_name}'. "
            f"Available variants: {available}"
        )
    return RETENTION_VARIANTS[variant_name]


def list_variants() -> None:
    """Print all available retention variants with descriptions."""
    print("\n=== Available Retention Variants ===\n")
    for variant_name, config in RETENTION_VARIANTS.items():
        mechanisms = []
        if config.distillation_weight > 0:
            mechanisms.append("KL")
        if config.fisher_weight > 0:
            mechanisms.append("Fisher")
        if config.enable_null_space:
            mechanisms.append("Null-space")

        mechanisms_str = " + ".join(mechanisms) if mechanisms else "None"

        print(f"{variant_name}:")
        print(f"  Description: {config.description}")
        print(f"  Mechanisms: {mechanisms_str}")
        print(f"  Retention Weight: {config.retention_loss_weight}")
        print()


if __name__ == "__main__":
    list_variants()