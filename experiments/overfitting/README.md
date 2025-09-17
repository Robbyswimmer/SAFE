# Overfitting Ablation Experiments

This directory tracks controlled overfitting studies that stress-test SAFE's
retention mechanisms. The primary goal is to overfit the projector + fusion
layers on a tiny audio-heavy slice (≈300–500 samples) and measure how much
vision-language (VL) performance regresses under different safety settings:

1. **no_retention** – disable KL, Fisher, and null-space editing
2. **soft_retention** – enable KL + Fisher regularisation only
3. **full_safe** – enable KL + Fisher + gradient null-space projection

Each run uses the same audio subset (documented in
`subset_manifest.json`) and seed so the only difference between the runs is the
retention strategy.

## How to run locally

```bash
python -m experiments.overfitting.run_overfitting \
  --variant full_safe \
  --train-source pack \
  --val-source pack_vl \
  --subset-size 400 \
  --output-root experiments/overfitting/runs
```

Default settings rely on the packaged subset, making the experiment portable.
You can still point to the full datasets with `--train-source audiocaps` or
`--train-source mock` if you want to regenerate the subset from scratch.

### Building the packaged subset

1. Ensure the full AudioCaps data is available under `data/audiocaps/` and the
   VQA resources (annotations **and** `COCO_val2014_*.jpg` images) under
   `data/vqa/`.
2. (Optional) Regenerate the validation manifest if you need a fresh subset:
   ```bash
   python -m experiments.overfitting.create_val_manifest
   ```
3. Run
   ```bash
   python -m experiments.overfitting.package_subset \
     --source-root data/audiocaps \
     --dest-root experiments/overfitting/data_pack
   ```
   This copies the 400 clips listed in `subset_manifest.json` into
   `experiments/overfitting/data_pack/` and includes the validation manifest.
4. Optionally archive the pack for transfer:
   ```bash
   tar czf safe_overfit_pack.tgz -C experiments/overfitting data_pack
   ```
5. On the cluster, extract the archive inside the repo and set
   `DATA_ROOT`/`PACK_ROOT` to that location before launching jobs.
   Confirm `data_pack/images/` contains the expected COCO files; if not,
   re-run `package_subset` after placing the images locally.

## Outputs

Each run creates a timestamped directory containing:

- `config.json` – the exact Stage A trainer configuration
- `metrics.json` – final eval metrics (audio accuracy, VL retention, etc.)
- `subset_indices.json` – the dataset indices selected for overfitting
- `checkpoints/` – trainer checkpoints (Stage A default behaviour)

These artefacts make it straightforward to reproduce or compare runs later.

## SLURM submission

Use the `scripts/overfitting_test.sh` helper to launch the three variants on a
cluster. Set environment variables before `sbatch` to point at the packaged
subset and manifests, for example:

```bash
export DATA_ROOT=$PWD/experiments/overfitting/data_pack
export PACK_ROOT=$PWD/experiments/overfitting/data_pack
sbatch scripts/overfitting_test.sh
```
