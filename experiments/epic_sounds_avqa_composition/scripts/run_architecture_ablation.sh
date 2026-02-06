#!/bin/bash
set -euo pipefail

# Example:
#   DATA_ROOT=experiments/epic_sounds_avqa_composition/data \
#   bash experiments/epic_sounds_avqa_composition/scripts/run_architecture_ablation.sh

DATA_ROOT=${DATA_ROOT:-experiments/epic_sounds_avqa_composition/data}
BASE_OUT=${BASE_OUT:-checkpoints/epic_sounds_avqa}

sbatch --gres=gpu:1 \
  --export=ALL,DATA_ROOT="$DATA_ROOT",OUTPUT_DIR="$BASE_OUT/preffn" \
  experiments/epic_sounds_avqa_composition/scripts/train_preffn.sh

sbatch --gres=gpu:1 \
  --export=ALL,DATA_ROOT="$DATA_ROOT",OUTPUT_DIR="$BASE_OUT/kv_augment" \
  experiments/epic_sounds_avqa_composition/scripts/train_kvaugment.sh
