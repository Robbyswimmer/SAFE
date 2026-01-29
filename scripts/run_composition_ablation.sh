#!/bin/bash
#SBATCH --job-name=composition-ablation
#SBATCH --output=logs/composition_ablation_%j.log
#SBATCH --error=logs/composition_ablation_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

set -e

# Composition Ablation Study: Pre-FFN vs KV Augmentation
# Tests whether multi-modal composition (audio + point cloud) behaves differently
# under pre-FFN residual fusion vs KV augmentation architectures.
#
# Phase 1: Train individual adapters
#   - Audio adapter (pre-FFN) on AVE classification
#   - Audio adapter (KV-aug) on AVE classification
#   - Point cloud adapter (pre-FFN) on ModelNet40
#   - Point cloud adapter (KV-aug) on ModelNet40
#
# Phase 2: Evaluate composition on MCUB
#   For each architecture, test 3 conditions:
#   - Audio only: both adapters loaded, only audio input
#   - PC only: both adapters loaded, only point cloud input
#   - Both: both adapters loaded, both modalities input
#
# Usage:
#   # Run full ablation
#   ./scripts/run_composition_ablation.sh
#
#   # Run specific phase
#   PHASE=train_audio ./scripts/run_composition_ablation.sh
#   PHASE=train_pc ./scripts/run_composition_ablation.sh
#   PHASE=eval ./scripts/run_composition_ablation.sh
#
#   # Run specific architecture
#   ARCH=preffn ./scripts/run_composition_ablation.sh
#   ARCH=kvaug ./scripts/run_composition_ablation.sh

# ============================================================================
# Configuration
# ============================================================================

PHASE=${PHASE:-"all"}  # all, train_audio, train_pc, eval
ARCH=${ARCH:-"both"}   # both, preffn, kvaug

# Data paths
AVE_DATA_PATH=${AVE_DATA_PATH:-"/data/SalmanAsif/AVE_Dataset"}
MODELNET_DATA_PATH=${MODELNET_DATA_PATH:-"./data"}
MCUB_DATA_PATH=${MCUB_DATA_PATH:-"/data/SalmanAsif/Kaykobad-Reza/Model-Merging/data/test"}

# Output directories
OUTPUT_BASE=${OUTPUT_BASE:-"outputs/composition_ablation"}
OUTPUT_AUDIO_PREFFN="${OUTPUT_BASE}/audio_preffn"
OUTPUT_AUDIO_KVAUG="${OUTPUT_BASE}/audio_kvaug"
OUTPUT_PC_PREFFN="${OUTPUT_BASE}/pc_preffn"
OUTPUT_PC_KVAUG="${OUTPUT_BASE}/pc_kvaug"
OUTPUT_EVAL_PREFFN="${OUTPUT_BASE}/eval_preffn"
OUTPUT_EVAL_KVAUG="${OUTPUT_BASE}/eval_kvaug"

# Training settings
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_EPOCHS=${NUM_EPOCHS:-20}
LEARNING_RATE=${LEARNING_RATE:-6e-5}
WANDB_PROJECT=${WANDB_PROJECT:-"COMPOSITION"}

# Fusion layers (same as layer ablation optimal: 6 layers starting at 1, stride 4)
FUSION_LAYERS="1,5,9,13,17,21"

mkdir -p logs
mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "Composition Ablation Study"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Phase: $PHASE"
echo "Architecture: $ARCH"
echo "Started: $(date)"
echo "========================================"
echo "AVE data: $AVE_DATA_PATH"
echo "ModelNet data: $MODELNET_DATA_PATH"
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
# Training Functions
# ============================================================================

train_audio_preffn() {
    echo ""
    echo "========================================"
    echo "Training: Audio Adapter (Pre-FFN)"
    echo "========================================"

    python train_audio_llm_probe.py \
        --model-config audio_preffn \
        --data-path "$AVE_DATA_PATH" \
        --output-dir "$OUTPUT_AUDIO_PREFFN" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$NUM_EPOCHS" \
        --lr "$LEARNING_RATE" \
        --fusion-layer-indices "$FUSION_LAYERS" \
        --fusion-injection-point pre_ffn \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "composition-audio-preffn"

    echo "✓ Audio pre-FFN adapter saved to: $OUTPUT_AUDIO_PREFFN"
}

train_audio_kvaug() {
    echo ""
    echo "========================================"
    echo "Training: Audio Adapter (KV Augmentation)"
    echo "========================================"

    python train_audio_llm_probe.py \
        --model-config audio_kvaug \
        --data-path "$AVE_DATA_PATH" \
        --output-dir "$OUTPUT_AUDIO_KVAUG" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$NUM_EPOCHS" \
        --lr "$LEARNING_RATE" \
        --fusion-layer-indices "$FUSION_LAYERS" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "composition-audio-kvaug"

    echo "✓ Audio KV-aug adapter saved to: $OUTPUT_AUDIO_KVAUG"
}

train_pc_preffn() {
    echo ""
    echo "========================================"
    echo "Training: Point Cloud Adapter (Pre-FFN)"
    echo "========================================"

    python train_pointcloud.py \
        --config modelnet40 \
        --phase classification \
        --llm-probe-head \
        --data-path "$MODELNET_DATA_PATH" \
        --output-dir "$OUTPUT_PC_PREFFN" \
        --batch-size "$BATCH_SIZE" \
        --num-epochs "$NUM_EPOCHS" \
        --learning-rate "$LEARNING_RATE" \
        --fusion-layer-indices "$FUSION_LAYERS" \
        --fusion-injection-point pre_ffn \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run "composition-pc-preffn"

    echo "✓ Point cloud pre-FFN adapter saved to: $OUTPUT_PC_PREFFN"
}

train_pc_kvaug() {
    echo ""
    echo "========================================"
    echo "Training: Point Cloud Adapter (KV Augmentation)"
    echo "========================================"

    python train_pointcloud.py \
        --config modelnet40 \
        --phase classification \
        --llm-probe-head \
        --data-path "$MODELNET_DATA_PATH" \
        --output-dir "$OUTPUT_PC_KVAUG" \
        --batch-size "$BATCH_SIZE" \
        --num-epochs "$NUM_EPOCHS" \
        --learning-rate "$LEARNING_RATE" \
        --fusion-layer-indices "$FUSION_LAYERS" \
        --fusion-mode kv_augment \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run "composition-pc-kvaug"

    echo "✓ Point cloud KV-aug adapter saved to: $OUTPUT_PC_KVAUG"
}

# ============================================================================
# Evaluation Functions
# ============================================================================

eval_preffn() {
    echo ""
    echo "========================================"
    echo "Evaluating: Pre-FFN Composition"
    echo "========================================"

    # Find best checkpoints
    AUDIO_CKPT=$(find "$OUTPUT_AUDIO_PREFFN" -name "best_model.pt" -o -name "checkpoint_best.pt" | head -1)
    PC_CKPT=$(find "$OUTPUT_PC_PREFFN" -name "best_model.pt" -o -name "checkpoint_best.pt" | head -1)

    if [ -z "$AUDIO_CKPT" ]; then
        echo "Warning: No audio pre-FFN checkpoint found. Using latest."
        AUDIO_CKPT=$(find "$OUTPUT_AUDIO_PREFFN" -name "*.pt" | sort | tail -1)
    fi

    if [ -z "$PC_CKPT" ]; then
        echo "Warning: No PC pre-FFN checkpoint found. Using latest."
        PC_CKPT=$(find "$OUTPUT_PC_PREFFN" -name "*.pt" | sort | tail -1)
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
    echo "Evaluating: KV Augmentation Composition"
    echo "========================================"

    # Find best checkpoints
    AUDIO_CKPT=$(find "$OUTPUT_AUDIO_KVAUG" -name "best_model.pt" -o -name "checkpoint_best.pt" | head -1)
    PC_CKPT=$(find "$OUTPUT_PC_KVAUG" -name "best_model.pt" -o -name "checkpoint_best.pt" | head -1)

    if [ -z "$AUDIO_CKPT" ]; then
        echo "Warning: No audio KV-aug checkpoint found. Using latest."
        AUDIO_CKPT=$(find "$OUTPUT_AUDIO_KVAUG" -name "*.pt" | sort | tail -1)
    fi

    if [ -z "$PC_CKPT" ]; then
        echo "Warning: No PC KV-aug checkpoint found. Using latest."
        PC_CKPT=$(find "$OUTPUT_PC_KVAUG" -name "*.pt" | sort | tail -1)
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

eval_synthetic() {
    echo ""
    echo "========================================"
    echo "Testing: Synthetic Data Evaluation"
    echo "========================================"

    echo "Testing Pre-FFN with synthetic data..."
    python eval_mcub.py \
        --config composition_preffn \
        --synthetic \
        --num-synthetic-samples 50 \
        --output-dir "${OUTPUT_BASE}/test_synthetic_preffn"

    echo "Testing KV-Aug with synthetic data..."
    python eval_mcub.py \
        --config composition_kvaug \
        --synthetic \
        --num-synthetic-samples 50 \
        --output-dir "${OUTPUT_BASE}/test_synthetic_kvaug"

    echo "✓ Synthetic evaluation tests complete"
}

# ============================================================================
# Main Execution
# ============================================================================

run_all() {
    echo ""
    echo "========================================"
    echo "Running Full Composition Ablation"
    echo "========================================"

    # Phase 1: Train adapters
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
    "test")
        eval_synthetic
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: all, train_audio, train_pc, eval, test"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Composition Ablation Complete"
echo "========================================"
echo "Finished: $(date)"
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Next steps:"
echo "1. Compare pre-FFN vs KV-aug results:"
echo "   cat $OUTPUT_EVAL_PREFFN/results_composition_preffn.json"
echo "   cat $OUTPUT_EVAL_KVAUG/results_composition_kvaug.json"
echo ""
echo "2. Key questions to analyze:"
echo "   - Does audio-only performance degrade when PC adapter is loaded?"
echo "   - Does PC-only performance degrade when audio adapter is loaded?"
echo "   - Does 'both' condition outperform single modality?"
echo "   - Is KV-aug better for composition than Pre-FFN?"
echo "========================================"
