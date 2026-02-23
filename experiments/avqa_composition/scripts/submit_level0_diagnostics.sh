#!/bin/bash
#SBATCH --job-name=L0-diag
#SBATCH --output=logs/level0_diag_%j.out
#SBATCH --error=logs/level0_diag_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# Level 0: Full composability diagnostics (zero training)
# Outputs: m*, rho_l, shift norms, subspace overlap, per-layer additivity probe
# Uses pre-trained adapters from comp_indep_additive_v1
# ~2 hours on 1 GPU

export MODEL_CONFIG=${MODEL_CONFIG:-composition_independent}
export COMPOSE_AUDIO_CKPT=${COMPOSE_AUDIO_CKPT:-checkpoints/comp_indep_additive_v1/best_model.pt}
export COMPOSE_VISION_CKPT=${COMPOSE_VISION_CKPT:-checkpoints/comp_indep_additive_v1/best_model.pt}
export OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/level0_diagnostics}
export MAX_SAMPLES=${MAX_SAMPLES:-1000}
export PROBE_SAMPLES=${PROBE_SAMPLES:-512}
export SUBSPACE_RANK=${SUBSPACE_RANK:-8}
export FUSION_GATE=${FUSION_GATE:-0.2}
export SLIM_PROJECTOR=${SLIM_PROJECTOR:-0}

source "$(dirname "$0")/run_composability_diagnostics.sh"
