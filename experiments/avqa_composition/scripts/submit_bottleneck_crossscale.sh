#!/bin/bash
# Cross-scale bottleneck verification (supports Table 4 in paper).
# Tests that the bottleneck ratio matters at scales OTHER than 8B.
#
# 4B (d=2560):
#   bn=128 → 5.0% ratio (should DEGRADE — below threshold)
#   bn=256 → 10.0% ratio (known good baseline from prior runs)
#
# This confirms the ratio law isn't an 8B artifact.
set -euo pipefail

if [[ -z "${SAFE_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    SAFE_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SAFE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
  fi
fi
cd "$SAFE_ROOT"

MODEL_CONFIG=${MODEL_CONFIG:-composition_4b_staggered}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-5}
LR=${LR:-5e-5}

echo "[crossscale] SAFE_ROOT=$SAFE_ROOT"
echo "[crossscale] MODEL_CONFIG=$MODEL_CONFIG (4B, d=2560)"
echo ""

# bn=128 on 4B → 5.0% ratio (expect degradation)
J1=$(sbatch --parsable --gres=gpu:1 \
  --job-name="bn128-4b" \
  --export=ALL,\
SAFE_ROOT="$SAFE_ROOT",\
MODEL_CONFIG="$MODEL_CONFIG",\
OUTPUT_DIR=checkpoints/bn_crossscale/4b_bn128_ratio5pct,\
SEED="$SEED",\
EPOCHS="$EPOCHS",\
LR="$LR",\
BOTTLENECK_DIM=128,\
SLIM_PROJECTOR=1,\
TRAIN_MODALITY=interleaved,\
EVAL_MODALITIES=text,audio,image,both,\
WANDB_RUN_NAME="crossscale_4b_bn128_s${SEED}",\
WANDB_TAGS="bottleneck_crossscale,4b,bn128,ratio5pct,paper_table4" \
  experiments/avqa_composition/scripts/train_composition_interleaved.sh)

echo "[crossscale] 4B bn=128 (5.0%) → job $J1"

# bn=256 on 4B → 10.0% ratio (control — should match prior results)
J2=$(sbatch --parsable --gres=gpu:1 \
  --job-name="bn256-4b" \
  --export=ALL,\
SAFE_ROOT="$SAFE_ROOT",\
MODEL_CONFIG="$MODEL_CONFIG",\
OUTPUT_DIR=checkpoints/bn_crossscale/4b_bn256_ratio10pct,\
SEED="$SEED",\
EPOCHS="$EPOCHS",\
LR="$LR",\
BOTTLENECK_DIM=256,\
SLIM_PROJECTOR=1,\
TRAIN_MODALITY=interleaved,\
EVAL_MODALITIES=text,audio,image,both,\
WANDB_RUN_NAME="crossscale_4b_bn256_s${SEED}",\
WANDB_TAGS="bottleneck_crossscale,4b,bn256,ratio10pct,paper_table4" \
  experiments/avqa_composition/scripts/train_composition_interleaved.sh)

echo "[crossscale] 4B bn=256 (10.0%) → job $J2"
echo ""
echo "[crossscale] Monitor: squeue -j $J1,$J2"
