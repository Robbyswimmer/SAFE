#!/bin/bash
#SBATCH --job-name=a2avqa-rkca
#SBATCH --output=logs/audiocaps_to_avqa_%j.out
#SBATCH --error=logs/audiocaps_to_avqa_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SAFE_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

cd "$SAFE_ROOT"

export MODEL_CONFIG=${MODEL_CONFIG:-rkca_joint_caption16}
export STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/audiocaps_align_rkca_joint_caption16}
export STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/vision_audio_from_audiocaps_align_rkca_joint_caption16}

echo "========================================"
echo "E2E Pipeline: AudioCaps -> MUSIC-AVQA"
echo "========================================"
echo "Model config: $MODEL_CONFIG"
echo "Stage 1 output: $STAGE1_OUTPUT_DIR"
echo "Stage 2 output: $STAGE2_OUTPUT_DIR"
echo "========================================"

OUTPUT_DIR="$STAGE1_OUTPUT_DIR" bash experiments/internvl_benchmark/scripts/train_audiocaps_prealign_rkca_joint.sh

INIT_AUDIO_CKPT="$STAGE1_OUTPUT_DIR/audio_aligned_best.pt" \
OUTPUT_DIR="$STAGE2_OUTPUT_DIR" \
bash experiments/internvl_benchmark/scripts/train_music_avqa_from_audiocaps_align_rkca_joint.sh

