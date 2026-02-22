#!/bin/bash
# Hail Mary Run 1: Vision-First Disjoint on Qwen3-4B
# V={8,14,20} A={26,30,34} — structural transport elimination
set -euo pipefail

export MODEL_CONFIG=composition_4b_vfirst
export AUDIO_FUSION_LAYERS="26,30,34"
export VISION_FUSION_LAYERS="8,14,20"
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
export COMPAT_TRANSPORT_ENABLE=0
export COMPAT_GATE_ADD_ENABLE=0
export COMPAT_NOHARM_ENABLE=0
export COMPAT_LOGIT_FUSION_ENABLE=0
export COMPAT_POE_ENABLE=0
export COMPAT_ROUTING_ENABLE=0
export EPOCHS=15
export LR=5e-5
export OUTPUT_DIR=checkpoints/comp_4b_vfirst_disjoint
export WANDB_RUN_NAME=comp_4b_vfirst_disjoint_s42
export WANDB_TAGS="composition,4b,vfirst,disjoint,from_scratch,hailmary"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/submit_level2_full_config.sh"
