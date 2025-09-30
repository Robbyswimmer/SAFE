# SAFE Configuration Files

This directory contains configuration files for SAFE experiments.

## Retention Variants (`retention_variants.py`)

Defines 5 retention variants for ablation studies. Each variant uses a different combination of retention mechanisms to preserve base VL performance.

### Available Variants

#### 1. `no_retention` - Baseline
**No retention mechanisms**
- Use case: Baseline to show what happens without any retention
- Expected behavior: Audio training may degrade VL performance
- Mechanisms: None

#### 2. `soft_retention` - KL Only
**KL divergence distillation only**
- Use case: Standard knowledge distillation approach
- Expected behavior: Maintains VL performance through output matching
- Mechanisms:
  - KL Divergence: ✓ (weight=0.2)
  - Fisher Information: ✗
  - Null-Space Projection: ✗

#### 3. `fisher_retention` - KL + Fisher
**KL divergence + Fisher information weighting**
- Use case: Add parameter-importance weighting to KL distillation
- Expected behavior: Better retention by protecting important parameters
- Mechanisms:
  - KL Divergence: ✓ (weight=0.2)
  - Fisher Information: ✓ (weight=0.1)
  - Null-Space Projection: ✗

#### 4. `nullspace_retention` - KL + Null-Space
**KL divergence + Null-space gradient projection**
- Use case: Add gradient-level protection for VL directions
- Expected behavior: Prevents gradients from updating VL-critical directions
- Mechanisms:
  - KL Divergence: ✓ (weight=0.2)
  - Fisher Information: ✗
  - Null-Space Projection: ✓ (rank=8, samples=128)

#### 5. `full_retention` - All Mechanisms
**All retention mechanisms enabled**
- Use case: Maximum retention protection
- Expected behavior: Strongest VL preservation, may limit audio learning
- Mechanisms:
  - KL Divergence: ✓ (weight=0.2)
  - Fisher Information: ✓ (weight=0.1)
  - Null-Space Projection: ✓ (rank=8, samples=128)

### Usage

```python
from configs.retention_variants import get_variant_config, list_variants

# List all variants
list_variants()

# Get specific variant config
config = get_variant_config("fisher_retention")
print(config.description)

# Apply to trainer
trainer_config = config.to_dict()
```

### Running Experiments

To run all variants:
```bash
sbatch scripts/full_training.sh
```

To run specific variants:
```bash
VARIANT_ORDER="no_retention fisher_retention full_retention" sbatch scripts/full_training.sh
```

### Configuration Parameters

Each variant specifies:
- **retention_loss_weight**: Overall retention loss weight (0.0-1.0)
- **distillation_weight**: KL divergence component weight (0.0-1.0)
- **fisher_weight**: Fisher information regularization weight (0.0-1.0)
- **enable_null_space**: Whether to enable null-space projection (bool)
- **null_space_rank**: Rank of protected subspace (int, typically 8)
- **null_space_min_samples**: Samples before activating projection (int, typically 128)
- **null_space_refresh_interval**: Steps between subspace recomputation (int, typically 4000)

### Adding New Variants

To add a new retention variant:

1. Define the config in `retention_variants.py`:
```python
MY_VARIANT = RetentionVariantConfig(
    name="my_variant",
    description="Description of what this variant does",
    retention_loss_weight=0.3,
    distillation_weight=0.3,
    fisher_weight=0.05,
    enable_null_space=True,
    null_space_rank=16,
)
```

2. Add to the registry:
```python
RETENTION_VARIANTS = {
    # ... existing variants ...
    "my_variant": MY_VARIANT,
}
```

3. Update SLURM script if you want it in the default sweep:
```bash
VARIANT_ORDER="no_retention soft_retention my_variant"
```

## Model Configs (`model_configs.py`)

Defines model architecture configurations (DEMO, FULL, MULTIMODAL).

See individual files for detailed configuration options.

---

**For questions or issues, see `SAFE_improvements.md` for the full roadmap.**