#!/bin/bash
#SBATCH --job-name=nuscenes_qa_composition
#SBATCH --output=experiments/nuscenes_qa_composition/logs/%x_%j.out
#SBATCH --error=experiments/nuscenes_qa_composition/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# NuScenes-QA Composition Training Script
# Dataset downloads automatically from Hugging Face - no registration required!

set -e

# Configuration
MODALITY=${MODALITY:-"both"}  # "pointcloud", "image", or "both"
SCENE_TYPE=${SCENE_TYPE:-"day"}  # "day", "night", or "both"
CAMERA_VIEW=${CAMERA_VIEW:-"CAM_FRONT"}

# Model settings
LLM_MODEL=${LLM_MODEL:-"llava-hf/llava-1.5-7b-hf"}
NUM_PC_TOKENS=${NUM_PC_TOKENS:-8}
FUSION_LAYERS=${FUSION_LAYERS:-"1,5,9,13,17,21"}

# Training settings
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}  # Effective batch = 16
NUM_EPOCHS=${NUM_EPOCHS:-20}
SAFE_LR=${SAFE_LR:-1e-5}
MAX_ANSWER_TOKENS=${MAX_ANSWER_TOKENS:-32}
NUM_POINTS=${NUM_POINTS:-8192}

# Output
OUTPUT_DIR=${OUTPUT_DIR:-"experiments/nuscenes_qa_composition/outputs/${MODALITY}_${SCENE_TYPE}"}
CACHE_DIR=${CACHE_DIR:-"experiments/full_training/data/cache"}

# W&B
WANDB_PROJECT=${WANDB_PROJECT:-"NuScenes-QA-Composition"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${MODALITY}_${SCENE_TYPE}_$(date +%Y%m%d_%H%M%S)"}

echo "=============================================="
echo "NuScenes-QA Composition Training"
echo "=============================================="
echo "Modality: ${MODALITY}"
echo "Scene type: ${SCENE_TYPE}"
echo "Camera view: ${CAMERA_VIEW}"
echo "LLM: ${LLM_MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p experiments/nuscenes_qa_composition/logs
mkdir -p "${CACHE_DIR}"

# Activate environment (adjust as needed)
source ~/.bashrc
conda activate safe 2>/dev/null || true

# Run training
python train_nuscenes_qa_composition.py \
    --modality "${MODALITY}" \
    --scene-type "${SCENE_TYPE}" \
    --camera-view "${CAMERA_VIEW}" \
    --cache-dir "${CACHE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --llm-model "${LLM_MODEL}" \
    --num-pointcloud-tokens "${NUM_PC_TOKENS}" \
    --fusion-layer-indices "${FUSION_LAYERS}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --num-epochs "${NUM_EPOCHS}" \
    --safe-lr "${SAFE_LR}" \
    --num-points "${NUM_POINTS}" \
    --max-answer-tokens "${MAX_ANSWER_TOKENS}" \
    --fp16 \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${WANDB_RUN_NAME}" \
    --wandb-tags "nuscenes_qa,${MODALITY},${SCENE_TYPE}"

echo "Training complete!"
