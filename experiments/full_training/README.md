# Full-Scale Stage A Training

This directory captures the configuration and utilities for the "100x data"
follow-up to the original overfitting ablation.  Instead of the packaged
400-clip subset we now rely on the full AudioCaps training split for Stage A and
use VQA v2 for the vision-language retention evaluation.

## Contents

- `download_data.py` – convenience script for fetching the experiment data pack
  onto a cluster node.
- `run_full_training.py` – Python entry-point that wires datasets, model, and
  retention configuration before delegating to the Stage A trainer.
- `README.md` – this file.

The helper SBATCH script `scripts/full_training.sh` mirrors
`scripts/overfitting_test.sh` but targets this experiment.  It loops over the two
supported retention configurations (`no_retention` and `soft_retention`) so the
resulting directory contains one run per setting.

## Expected data layout

`download_data.py` extracts the archives into `experiments/full_training/data/`
by default, producing the following structure that matches the dataset helper
classes in `safe.data.datasets`:

```
experiments/full_training/data/
├── audiocaps/
│   ├── audiocaps_train.jsonl
│   ├── audiocaps_val.jsonl
│   └── audio/...
├── wavcaps/                      # Optional: additional audio data
│   ├── wavcaps_train.jsonl
│   └── audio/...
└── vqa/
    ├── vqa_train.jsonl
    ├── vqa_val.jsonl
    └── images/val2014/...
```

You can override the download destination with `--destination` and point the
runner at any directory that matches this layout via `--data-root`.

### WavCaps Setup (Optional)

WavCaps provides 400K+ additional audio-caption pairs to augment AudioCaps training.
If you have WavCaps data with FLAC files in non-standard directories, use the
repair script to generate correct metadata:

```bash
# 1. Fix metadata paths to match actual audio files
python scripts/fix_wavcaps_paths.py \
    --wavcaps-dir experiments/full_training/data/wavcaps

# 2. Validate the setup
python scripts/validate_wavcaps.py \
    --wavcaps-dir experiments/full_training/data/wavcaps \
    --test-loading --num-samples 10
```

**Common Issues:**
- **"No samples were processed"**: Audio file paths in JSONL don't match actual files.
  Run `fix_wavcaps_paths.py` to scan and regenerate metadata.
- **FLAC vs WAV**: The dataset loader supports both formats via torchaudio.
- **Symlinked directories**: The repair script handles complex directory structures
  including symlinks.

**Training with WavCaps:**
```bash
python experiments/full_training/run_full_training.py \
    --variant soft_retention \
    --use-wavcaps \
    --wavcaps-ratio 0.5  # Use 50% of WavCaps samples
```

## Running the experiment

1. Download or stage the required data pack:
   ```bash
   python -m experiments.full_training.download_data --audiocaps-url <URL> --vqa-url <URL>
   ```
2. Submit the training job (customise SBATCH resources as needed):
   ```bash
   sbatch scripts/full_training.sh
   ```
3. Results are written to `experiments/full_training/runs/<timestamp>_<variant>/`.

Each run serialises the exact trainer configuration (`config.json`) and the
final evaluation metrics (`metrics.json`) to help with downstream analysis.
