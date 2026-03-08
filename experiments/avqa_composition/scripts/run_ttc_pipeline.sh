#!/bin/bash
#SBATCH --job-name=ttc-pipeline
#SBATCH --output=logs/ttc_pipeline_%j.out
#SBATCH --error=logs/ttc_pipeline_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu

set -euo pipefail

if [[ -z "${SAFE_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    SAFE_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SAFE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
  fi
fi

CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

AUDIO_OUT="${AUDIO_OUT:-checkpoints/composition_ttc_audio}"
VISION_OUT="${VISION_OUT:-checkpoints/composition_ttc_vision}"
BASELINE_OUT="${BASELINE_OUT:-checkpoints/composition_ttc_eval_baseline}"
SIMPLE_CAND_OUT="${SIMPLE_CAND_OUT:-checkpoints/composition_ttc_eval_simple_candidate}"
SIMPLE_HYBRID_OUT="${SIMPLE_HYBRID_OUT:-checkpoints/composition_ttc_eval_simple_hybrid}"
SIMPLE_INTER_OUT="${SIMPLE_INTER_OUT:-checkpoints/composition_ttc_eval_simple_interaction}"
COMPLEX_CAND_OUT="${COMPLEX_CAND_OUT:-checkpoints/composition_ttc_eval_complex_candidate}"
COMPLEX_HYBRID_OUT="${COMPLEX_HYBRID_OUT:-checkpoints/composition_ttc_eval_complex_hybrid}"
COMPLEX_INTER_OUT="${COMPLEX_INTER_OUT:-checkpoints/composition_ttc_eval_complex_interaction}"

mkdir -p "$SAFE_ROOT/logs"
cd "$SAFE_ROOT"

OUTPUT_DIR="$AUDIO_OUT" "$SAFE_ROOT/experiments/avqa_composition/scripts/train_composition_audio_ttc.sh"
OUTPUT_DIR="$VISION_OUT" "$SAFE_ROOT/experiments/avqa_composition/scripts/train_composition_vision_ttc.sh"
COMPOSE_AUDIO_CKPT="$AUDIO_OUT/best_model.pt" \
COMPOSE_VISION_CKPT="$VISION_OUT/best_model.pt" \
OUTPUT_DIR="$BASELINE_OUT" \
  "$SAFE_ROOT/experiments/avqa_composition/scripts/eval_composition_ttc_baseline.sh"
COMPOSE_AUDIO_CKPT="$AUDIO_OUT/best_model.pt" \
COMPOSE_VISION_CKPT="$VISION_OUT/best_model.pt" \
OUTPUT_DIR="$SIMPLE_CAND_OUT" \
TTC_SEARCH_MODE=candidate \
TTC_STABILITY_ENABLE=1 \
  "$SAFE_ROOT/experiments/avqa_composition/scripts/eval_composition_ttc_simple.sh"
COMPOSE_AUDIO_CKPT="$AUDIO_OUT/best_model.pt" \
COMPOSE_VISION_CKPT="$VISION_OUT/best_model.pt" \
OUTPUT_DIR="$SIMPLE_HYBRID_OUT" \
TTC_SEARCH_MODE=hybrid \
TTC_STABILITY_ENABLE=1 \
  "$SAFE_ROOT/experiments/avqa_composition/scripts/eval_composition_ttc_simple.sh"
COMPOSE_AUDIO_CKPT="$AUDIO_OUT/best_model.pt" \
COMPOSE_VISION_CKPT="$VISION_OUT/best_model.pt" \
OUTPUT_DIR="$SIMPLE_INTER_OUT" \
TTC_SEARCH_MODE=hybrid \
TTC_STABILITY_ENABLE=1 \
TTC_INTERACTION_ENABLE=1 \
  "$SAFE_ROOT/experiments/avqa_composition/scripts/eval_composition_ttc_simple.sh"
COMPOSE_AUDIO_CKPT="$AUDIO_OUT/best_model.pt" \
COMPOSE_VISION_CKPT="$VISION_OUT/best_model.pt" \
OUTPUT_DIR="$COMPLEX_CAND_OUT" \
TTC_SEARCH_MODE=candidate \
TTC_STABILITY_ENABLE=1 \
  "$SAFE_ROOT/experiments/avqa_composition/scripts/eval_composition_ttc_complex.sh"
COMPOSE_AUDIO_CKPT="$AUDIO_OUT/best_model.pt" \
COMPOSE_VISION_CKPT="$VISION_OUT/best_model.pt" \
OUTPUT_DIR="$COMPLEX_HYBRID_OUT" \
TTC_SEARCH_MODE=hybrid \
TTC_STABILITY_ENABLE=1 \
  "$SAFE_ROOT/experiments/avqa_composition/scripts/eval_composition_ttc_complex.sh"
COMPOSE_AUDIO_CKPT="$AUDIO_OUT/best_model.pt" \
COMPOSE_VISION_CKPT="$VISION_OUT/best_model.pt" \
OUTPUT_DIR="$COMPLEX_INTER_OUT" \
TTC_SEARCH_MODE=hybrid \
TTC_STABILITY_ENABLE=1 \
TTC_INTERACTION_ENABLE=1 \
  "$SAFE_ROOT/experiments/avqa_composition/scripts/eval_composition_ttc_complex.sh"
