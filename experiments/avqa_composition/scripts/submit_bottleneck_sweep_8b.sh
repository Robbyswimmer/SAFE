#!/bin/bash
# Bottleneck ratio ablation on InternVL 8B (Table 4 in paper).
# Sweep bn ∈ {128, 256, 410, 512, 756} to characterize the capacity scaling law.
# Key hypothesis: composition gain requires b/d ≥ ~0.10.
#
# Ratios:  128/4096=3.1%  256/4096=6.25%  410/4096=10.0%  512/4096=12.5%  756/4096=18.5%
#
# Uses internvl_binding config (InternVL 8B, audio-only training, vision at eval).
# SLIM_PROJECTOR=1 so projector output_dim = bottleneck_dim.
set -euo pipefail

if [[ -z "${SAFE_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    SAFE_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SAFE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
  fi
fi
cd "$SAFE_ROOT"

MODEL_CONFIG=${MODEL_CONFIG:-internvl_binding}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-5}
LR=${LR:-5e-5}

BN_VALUES=(128 256 410 512 756)

echo "[bn-sweep] SAFE_ROOT=$SAFE_ROOT"
echo "[bn-sweep] MODEL_CONFIG=$MODEL_CONFIG"
echo "[bn-sweep] EPOCHS=$EPOCHS  LR=$LR  SEED=$SEED"
echo "[bn-sweep] Bottleneck values: ${BN_VALUES[*]}"
echo ""

JOBS=()
for BN in "${BN_VALUES[@]}"; do
  RATIO=$(python3 -c "print(f'{${BN}/4096*100:.1f}')")
  OUTPUT_DIR="checkpoints/bn_sweep_8b/bn${BN}_ratio${RATIO}pct"

  JOB=$(sbatch --parsable --gres=gpu:1 \
    --job-name="bn${BN}" \
    --export=ALL,\
SAFE_ROOT="$SAFE_ROOT",\
MODEL_CONFIG="$MODEL_CONFIG",\
OUTPUT_DIR="$OUTPUT_DIR",\
SEED="$SEED",\
EPOCHS="$EPOCHS",\
LR="$LR",\
BOTTLENECK_DIM="$BN",\
SLIM_PROJECTOR=1,\
TRAIN_MODALITY=interleaved,\
EVAL_MODALITIES=text,audio,image,both,\
WANDB_RUN_NAME="bn_sweep_8b_bn${BN}_s${SEED}",\
WANDB_TAGS="bottleneck_sweep,8b,bn${BN},ratio${RATIO}pct,paper_table4" \
    experiments/avqa_composition/scripts/train_composition_interleaved.sh)

  JOBS+=("$JOB")
  echo "[bn-sweep] bn=$BN (${RATIO}% ratio) → job $JOB  output=$OUTPUT_DIR"
done

echo ""
echo "[bn-sweep] All jobs: ${JOBS[*]}"
JOB_LIST=$(IFS=,; echo "${JOBS[*]}")
echo "[bn-sweep] Monitor: squeue -j $JOB_LIST"
