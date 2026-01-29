#!/bin/bash
#SBATCH --job-name=composition-captioning
#SBATCH --output=logs/composition_captioning_%j.log
#SBATCH --error=logs/composition_captioning_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

set -e

# Composition Ablation Study with Captioning Training
#
# Unlike classification training, captioning teaches the adapters to:
# 1. Generate language about the modality content
# 2. Produce tokens the LLM can understand for QA tasks
#
# Training Plan:
#   Audio: 2-stage alignment on AudioCaps (Stage A: contrastive, Stage B: captioning)
#   Point Cloud: Captioning on Cap3D
#
# Usage:
#   # Full pipeline
#   ./scripts/run_composition_captioning.sh
#
#   # Specific phases
#   PHASE=train_audio ./scripts/run_composition_captioning.sh
#   PHASE=train_pc ./scripts/run_composition_captioning.sh
#   PHASE=eval ./scripts/run_composition_captioning.sh
#
#   # Specific architecture
#   ARCH=preffn ./scripts/run_composition_captioning.sh
#   ARCH=kvaug ./scripts/run_composition_captioning.sh

# ============================================================================
# Configuration
# ============================================================================

PHASE=${PHASE:-"all"}  # all, train_audio, train_pc, eval
ARCH=${ARCH:-"both"}   # both, preffn, kvaug

# Data paths
# AudioCaps for audio captioning training
AUDIOCAPS_PATH=${AUDIOCAPS_PATH:-"/data/SalmanAsif/shared/SAFECaptions/audiocaps"}
# Cap3D for point cloud captioning (uses local data/cap3d with downloaded shards)
CAP3D_PATH=${CAP3D_PATH:-"data/cap3d"}
# MCUB for evaluation
MCUB_DATA_PATH=${MCUB_DATA_PATH:-"/data/SalmanAsif/Kaykobad-Reza/Model-Merging/data/test"}

# Output directories
OUTPUT_BASE=${OUTPUT_BASE:-"outputs/composition_captioning"}
OUTPUT_AUDIO_PREFFN="${OUTPUT_BASE}/audio_preffn"
OUTPUT_AUDIO_KVAUG="${OUTPUT_BASE}/audio_kvaug"
OUTPUT_PC_PREFFN="${OUTPUT_BASE}/pc_preffn"
OUTPUT_PC_KVAUG="${OUTPUT_BASE}/pc_kvaug"
OUTPUT_EVAL_PREFFN="${OUTPUT_BASE}/eval_preffn"
OUTPUT_EVAL_KVAUG="${OUTPUT_BASE}/eval_kvaug"

# Training settings
BATCH_SIZE=${BATCH_SIZE:-2}
AUDIO_EPOCHS=${AUDIO_EPOCHS:-10}
PC_EPOCHS=${PC_EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
WANDB_PROJECT=${WANDB_PROJECT:-"COMPOSITION-CAPTIONING"}

# Fusion layers (same as classification ablation)
FUSION_LAYERS="1,5,9,13,17,21"

mkdir -p logs
mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "Composition Captioning Ablation Study"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Phase: $PHASE"
echo "Architecture: $ARCH"
echo "Started: $(date)"
echo "========================================"
echo "AudioCaps data: $AUDIOCAPS_PATH"
echo "Cap3D data: $CAP3D_PATH"
echo "MCUB data: $MCUB_DATA_PATH"
echo "Output base: $OUTPUT_BASE"
echo "Fusion layers: $FUSION_LAYERS"
echo "========================================"

# Activate conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate safe-env
elif [ -f ~/.bashrc ]; then
  source ~/.bashrc
  conda activate safe-env 2>/dev/null || true
fi

echo "Python: $(which python)"

# ============================================================================
# Audio Training (2-Stage Alignment)
# ============================================================================

train_audio_preffn() {
    echo ""
    echo "========================================"
    echo "Training: Audio Adapter (Pre-FFN) - Captioning"
    echo "========================================"
    echo "Using AudioCaps for caption generation training"
    echo "========================================"

    # Use train_safe.py for captioning training
    # phase1 config uses pre-FFN residual fusion by default
    python train_safe.py \
        --model-config phase1 \
        --data-path "$AUDIOCAPS_PATH" \
        --output-dir "$OUTPUT_AUDIO_PREFFN" \
        --batch-size "$BATCH_SIZE" \
        --num-epochs "$AUDIO_EPOCHS" \
        --learning-rate-projector 1e-3 \
        --learning-rate-adapter "$LEARNING_RATE" \
        --fusion-layer-indices "$FUSION_LAYERS" \
        --gradient-accumulation-steps 16 \
        --fp16 \
        --wandb \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "captioning-audio-preffn"

    echo "✓ Audio pre-FFN captioning adapter saved to: $OUTPUT_AUDIO_PREFFN"
}

train_audio_kvaug() {
    echo ""
    echo "========================================"
    echo "Training: Audio Adapter (KV-Aug) - Captioning"
    echo "========================================"

    # kv_augment config uses KV augmentation fusion
    python train_safe.py \
        --model-config kv_augment \
        --data-path "$AUDIOCAPS_PATH" \
        --output-dir "$OUTPUT_AUDIO_KVAUG" \
        --batch-size "$BATCH_SIZE" \
        --num-epochs "$AUDIO_EPOCHS" \
        --learning-rate-projector 1e-3 \
        --learning-rate-adapter "$LEARNING_RATE" \
        --fusion-layer-indices "$FUSION_LAYERS" \
        --gradient-accumulation-steps 16 \
        --fp16 \
        --wandb \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "captioning-audio-kvaug"

    echo "✓ Audio KV-aug captioning adapter saved to: $OUTPUT_AUDIO_KVAUG"
}

# ============================================================================
# Point Cloud Training (Cap3D Captioning)
# ============================================================================

train_pc_preffn() {
    echo ""
    echo "========================================"
    echo "Training: Point Cloud Adapter (Pre-FFN) - Cap3D Captioning"
    echo "========================================"

    python train_pointcloud.py \
        --config cap3d \
        --phase captioning \
        --data-path "$CAP3D_PATH" \
        --output-dir "$OUTPUT_PC_PREFFN" \
        --batch-size "$BATCH_SIZE" \
        --num-epochs "$PC_EPOCHS" \
        --lr "$LEARNING_RATE" \
        --fusion-layer-indices "$FUSION_LAYERS" \
        --fusion-injection-point pre_ffn \
        --gradient-accumulation 16 \
        --fp16 \
        --wandb \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "captioning-pc-preffn"

    echo "✓ Point cloud pre-FFN captioning adapter saved to: $OUTPUT_PC_PREFFN"
}

train_pc_kvaug() {
    echo ""
    echo "========================================"
    echo "Training: Point Cloud Adapter (KV-Aug) - Cap3D Captioning"
    echo "========================================"

    python train_pointcloud.py \
        --config cap3d \
        --phase captioning \
        --data-path "$CAP3D_PATH" \
        --output-dir "$OUTPUT_PC_KVAUG" \
        --batch-size "$BATCH_SIZE" \
        --num-epochs "$PC_EPOCHS" \
        --lr "$LEARNING_RATE" \
        --fusion-layer-indices "$FUSION_LAYERS" \
        --fusion-mode kv_augment \
        --gradient-accumulation 16 \
        --fp16 \
        --wandb \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "captioning-pc-kvaug"

    echo "✓ Point cloud KV-aug captioning adapter saved to: $OUTPUT_PC_KVAUG"
}

# ============================================================================
# MCUB Evaluation
# ============================================================================

eval_preffn() {
    echo ""
    echo "========================================"
    echo "Evaluating: Pre-FFN Composition (Captioning-trained)"
    echo "========================================"

    # Find best checkpoints
    AUDIO_CKPT=$(find "$OUTPUT_AUDIO_PREFFN" -name "best_model.pt" -o -name "checkpoint_best.pt" 2>/dev/null | head -1)
    PC_CKPT=$(find "$OUTPUT_PC_PREFFN" -name "best_model.pt" -o -name "checkpoint_best.pt" 2>/dev/null | head -1)

    if [ -z "$AUDIO_CKPT" ]; then
        echo "Warning: No audio pre-FFN checkpoint found. Using latest."
        AUDIO_CKPT=$(find "$OUTPUT_AUDIO_PREFFN" -name "*.pt" 2>/dev/null | sort | tail -1)
    fi

    if [ -z "$PC_CKPT" ]; then
        echo "Warning: No PC pre-FFN checkpoint found. Using latest."
        PC_CKPT=$(find "$OUTPUT_PC_PREFFN" -name "*.pt" 2>/dev/null | sort | tail -1)
    fi

    echo "Audio checkpoint: $AUDIO_CKPT"
    echo "PC checkpoint: $PC_CKPT"

    python eval_mcub.py \
        --config composition_preffn \
        --audio-adapter "$AUDIO_CKPT" \
        --pc-adapter "$PC_CKPT" \
        --data-path "$MCUB_DATA_PATH" \
        --output-dir "$OUTPUT_EVAL_PREFFN" \
        --conditions audio_only pc_only both

    echo "✓ Pre-FFN evaluation saved to: $OUTPUT_EVAL_PREFFN"
}

eval_kvaug() {
    echo ""
    echo "========================================"
    echo "Evaluating: KV Augmentation Composition (Captioning-trained)"
    echo "========================================"

    # Find best checkpoints
    AUDIO_CKPT=$(find "$OUTPUT_AUDIO_KVAUG" -name "best_model.pt" -o -name "checkpoint_best.pt" 2>/dev/null | head -1)
    PC_CKPT=$(find "$OUTPUT_PC_KVAUG" -name "best_model.pt" -o -name "checkpoint_best.pt" 2>/dev/null | head -1)

    if [ -z "$AUDIO_CKPT" ]; then
        echo "Warning: No audio KV-aug checkpoint found. Using latest."
        AUDIO_CKPT=$(find "$OUTPUT_AUDIO_KVAUG" -name "*.pt" 2>/dev/null | sort | tail -1)
    fi

    if [ -z "$PC_CKPT" ]; then
        echo "Warning: No PC KV-aug checkpoint found. Using latest."
        PC_CKPT=$(find "$OUTPUT_PC_KVAUG" -name "*.pt" 2>/dev/null | sort | tail -1)
    fi

    echo "Audio checkpoint: $AUDIO_CKPT"
    echo "PC checkpoint: $PC_CKPT"

    python eval_mcub.py \
        --config composition_kvaug \
        --audio-adapter "$AUDIO_CKPT" \
        --pc-adapter "$PC_CKPT" \
        --data-path "$MCUB_DATA_PATH" \
        --output-dir "$OUTPUT_EVAL_KVAUG" \
        --conditions audio_only pc_only both

    echo "✓ KV-aug evaluation saved to: $OUTPUT_EVAL_KVAUG"
}

# ============================================================================
# Main Execution
# ============================================================================

run_all() {
    echo ""
    echo "========================================"
    echo "Running Full Captioning Composition Ablation"
    echo "========================================"

    # Phase 1: Train adapters with captioning
    if [ "$ARCH" = "both" ] || [ "$ARCH" = "preffn" ]; then
        train_audio_preffn
        train_pc_preffn
    fi

    if [ "$ARCH" = "both" ] || [ "$ARCH" = "kvaug" ]; then
        train_audio_kvaug
        train_pc_kvaug
    fi

    # Phase 2: Evaluate composition
    if [ "$ARCH" = "both" ] || [ "$ARCH" = "preffn" ]; then
        eval_preffn
    fi

    if [ "$ARCH" = "both" ] || [ "$ARCH" = "kvaug" ]; then
        eval_kvaug
    fi
}

# Execute based on PHASE
case "$PHASE" in
    "all")
        run_all
        ;;
    "train_audio")
        if [ "$ARCH" = "both" ] || [ "$ARCH" = "preffn" ]; then
            train_audio_preffn
        fi
        if [ "$ARCH" = "both" ] || [ "$ARCH" = "kvaug" ]; then
            train_audio_kvaug
        fi
        ;;
    "train_pc")
        if [ "$ARCH" = "both" ] || [ "$ARCH" = "preffn" ]; then
            train_pc_preffn
        fi
        if [ "$ARCH" = "both" ] || [ "$ARCH" = "kvaug" ]; then
            train_pc_kvaug
        fi
        ;;
    "eval")
        if [ "$ARCH" = "both" ] || [ "$ARCH" = "preffn" ]; then
            eval_preffn
        fi
        if [ "$ARCH" = "both" ] || [ "$ARCH" = "kvaug" ]; then
            eval_kvaug
        fi
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: all, train_audio, train_pc, eval"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Captioning Composition Ablation Complete"
echo "========================================"
echo "Finished: $(date)"
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Compare results:"
echo "  cat $OUTPUT_EVAL_PREFFN/results_composition_preffn.json"
echo "  cat $OUTPUT_EVAL_KVAUG/results_composition_kvaug.json"
echo ""
echo "Expected improvements over classification training:"
echo "  - Adapters learned to generate language about content"
echo "  - Better transfer to MCUB VQA task"
echo "  - More meaningful composition effects"
echo "========================================"
