#!/bin/bash
#SBATCH --job-name=hm-stagger
#SBATCH --output=logs/hailmary_staggered_%j.out
#SBATCH --error=logs/hailmary_staggered_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu

# Hail Mary Run 2: Tightened Staggered on Qwen3-4B
# V={9,15,21,27} A={11,17,23,29} — best layout + transport reg + from scratch

export MODEL_CONFIG=composition_4b_staggered
export AUDIO_FUSION_LAYERS="11,17,23,29"
export VISION_FUSION_LAYERS="9,15,21,27"
export SLIM_PROJECTOR=1
export INIT_AUDIO_CKPT=""
export INIT_VISION_CKPT=""
export LEARNED_GATE=1
export LEARNED_GATE_INIT=2.0
export DELTA_NORM_CAP_RATIO=0.0
export AUDIO_GATE_DEPTH_DECAY=1.0
export VISION_GATE_DEPTH_DECAY=1.0
export COMPAT_REG_ENABLE=0
export COMPAT_ADD_REG_ENABLE=1
export COMPAT_ADD_REG_LAMBDA=0.01
export COMPAT_ADD_REG_EVERY=100
export COMPAT_TRANSPORT_ENABLE=1
export COMPAT_TRANSPORT_LAMBDA=0.03
export COMPAT_TRANSPORT_NORMALIZE=1
export COMPAT_GATE_ADD_ENABLE=0
export COMPAT_NOHARM_ENABLE=0
export COMPAT_LOGIT_FUSION_ENABLE=0
export COMPAT_POE_ENABLE=0
export COMPAT_ROUTING_ENABLE=0
export EPOCHS=15
export LR=5e-5
export OUTPUT_DIR=checkpoints/comp_4b_staggered_transport
export WANDB_RUN_NAME=comp_4b_staggered_transport_s42
export WANDB_TAGS="composition,4b,staggered,transport,from_scratch,hailmary"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/train_composition_interleaved.sh"
