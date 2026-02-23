#!/bin/bash
#SBATCH --job-name=L1-gates
#SBATCH --output=logs/level1_gates_%j.out
#SBATCH --error=logs/level1_gates_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# Level 1: Gates-only calibration (~100 paired samples, 2 epochs)
# Freezes all adapter params, learns only per-layer gate scalars
# Uses pre-trained adapters from comp_indep_additive_v1
# ~30 min on 1 GPU

export MODEL_CONFIG=${MODEL_CONFIG:-composition_independent}
export INIT_AUDIO_CKPT=${INIT_AUDIO_CKPT:-checkpoints/comp_indep_additive_v1/best_model.pt}
export INIT_VISION_CKPT=${INIT_VISION_CKPT:-checkpoints/comp_indep_additive_v1/best_model.pt}
export OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/level1_gates_only}

# Gate settings
export LEARNED_GATE=1
export LEARNED_GATE_INIT=2.0
export TRAIN_GATES_ONLY=1

# Training settings — small and fast
export EPOCHS=2
export LR=1e-3
export MAX_SAMPLES=${MAX_SAMPLES:-100}
export BATCH_SIZE=${BATCH_SIZE:-1}

# Eval all 4 conditions
export EVAL_MODALITIES="text,audio,image,both"
export LAYER_ADDITIVITY_PROBE=1
export LAYER_PROBE_SAMPLES=256

# No heavy regularizers — just gate learning
export COMPAT_REG_ENABLE=0
export COMPAT_ADD_REG_ENABLE=0
export COMPAT_TRANSPORT_ENABLE=0
export COMPAT_GATE_ADD_ENABLE=0
export COMPAT_NOHARM_ENABLE=0
export COMPAT_POE_ENABLE=0
export COMPAT_ROUTING_ENABLE=0
export COMPAT_LOGIT_FUSION_ENABLE=0

# Projector
export SLIM_PROJECTOR=${SLIM_PROJECTOR:-0}

# Wandb
export WANDB=${WANDB:-1}
export WANDB_PROJECT=${WANDB_PROJECT:-SAFE-Composition}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-level1_gates_only_${SLURM_JOB_ID:-local}}
export WANDB_TAGS=${WANDB_TAGS:-composition,level1,gates_only,calibration}

source "$(dirname "$0")/train_composition_interleaved.sh"
