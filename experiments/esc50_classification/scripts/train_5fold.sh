#!/bin/bash
#SBATCH --job-name=esc50-5fold
#SBATCH --output=logs/esc50_5fold_%j.log
#SBATCH --error=logs/esc50_5fold_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

# ESC-50 Audio Classification - 5-Fold Cross-Validation
#
# Standard ESC-50 evaluation protocol:
# - Train on 4 folds, validate on 1 fold
# - Repeat for all 5 folds
# - Report: mean accuracy ± std
#
# SOTA targets:
#   BEATs: 98.1%
#   CLAP: 96.7%
#   AST: 95.7%
#   Human: 81.3%
#   Our target: 90%+
#
# Usage:
#   sbatch experiments/esc50_classification/scripts/train_5fold.sh
#   # Or with custom config:
#   EXPERIMENT_NAME=ablation_more_layers sbatch experiments/esc50_classification/scripts/train_5fold.sh

set -e

# Configuration
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"baseline"}

# Get SAFE root - use SLURM_SUBMIT_DIR if available (submitted from SAFE root)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SAFE_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

# Data and output paths
DATA_PATH="${DATA_PATH:-$SAFE_ROOT/experiments/full_training/data}"
OUTPUT_BASE="${OUTPUT_DIR:-$SAFE_ROOT/experiments/esc50_classification/outputs/${EXPERIMENT_NAME}}"

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_EPOCHS=${NUM_EPOCHS:-50}
SAFE_LR=${SAFE_LR:-6e-5}
HEAD_LR=${HEAD_LR:-1e-3}
FUSION_LAYERS=${FUSION_LAYERS:-"1,5,9,13,17,21"}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}

# Regularization / Augmentation (set to 0 to disable)
MIXUP_ALPHA=${MIXUP_ALPHA:-0.0}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.0}
UNFREEZE_CLAP=${UNFREEZE_CLAP:-0}

# W&B settings
WANDB_PROJECT=${WANDB_PROJECT:-"ESC50-Classification"}
WANDB_TAGS=${WANDB_TAGS:-"esc50,5fold"}

echo "========================================"
echo "ESC-50 Classification - 5-Fold CV"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Experiment: $EXPERIMENT_NAME"
echo "SAFE root: $SAFE_ROOT"
echo "Data path: $DATA_PATH"
echo "Output base: $OUTPUT_BASE"
echo "========================================"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "SAFE LR (projector+fusion): $SAFE_LR"
echo "Head LR: $HEAD_LR"
echo "Fusion layers: $FUSION_LAYERS"
echo "Num audio tokens: $NUM_AUDIO_TOKENS"
echo "Mixup alpha: $MIXUP_ALPHA"
echo "Label smoothing: $LABEL_SMOOTHING"
echo "Unfreeze CLAP layers: $UNFREEZE_CLAP"
echo "W&B tags: $WANDB_TAGS"
echo "========================================"

# Activate conda environment
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate safe-env 2>/dev/null || conda activate safe 2>/dev/null || true
elif [ -f ~/.bashrc ]; then
    source ~/.bashrc
    conda activate safe-env 2>/dev/null || conda activate safe 2>/dev/null || true
fi

echo "Python: $(which python)"
echo "========================================"

cd "$SAFE_ROOT"

# Create output directories
mkdir -p "$OUTPUT_BASE"
mkdir -p "$SAFE_ROOT/logs"

# Array to store fold accuracies
declare -a FOLD_ACCURACIES

# Train and evaluate each fold
for FOLD in 1 2 3 4 5; do
    echo ""
    echo "========================================"
    echo "FOLD $FOLD / 5"
    echo "========================================"
    echo "Training on folds: $(seq -s, 1 5 | sed "s/,$FOLD,/,/" | sed "s/^$FOLD,//" | sed "s/,$FOLD$//")"
    echo "Validating on fold: $FOLD"
    echo "========================================"

    FOLD_OUTPUT_DIR="${OUTPUT_BASE}/fold${FOLD}"
    mkdir -p "$FOLD_OUTPUT_DIR"

    # Run training for this fold and capture output
    FOLD_LOG="${FOLD_OUTPUT_DIR}/training.log"
    python train_audio_llm_probe.py \
        --dataset esc50 \
        --data-path "$DATA_PATH" \
        --fold "$FOLD" \
        --output-dir "$FOLD_OUTPUT_DIR" \
        --model-config phase1 \
        --fusion-layer-indices "$FUSION_LAYERS" \
        --num-audio-tokens "$NUM_AUDIO_TOKENS" \
        --batch-size "$BATCH_SIZE" \
        --num-epochs "$NUM_EPOCHS" \
        --safe-learning-rate "$SAFE_LR" \
        --head-learning-rate "$HEAD_LR" \
        --mixup-alpha "$MIXUP_ALPHA" \
        --label-smoothing "$LABEL_SMOOTHING" \
        --unfreeze-clap-layers "$UNFREEZE_CLAP" \
        --fp16 \
        --wandb \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "${EXPERIMENT_NAME}-fold${FOLD}" \
        --wandb-tags "${WANDB_TAGS},${EXPERIMENT_NAME},fold${FOLD}" \
        2>&1 | tee "$FOLD_LOG"

    # Extract best accuracy from the log
    FOLD_ACC=$(grep "Best Test Accuracy:" "$FOLD_LOG" | tail -1 | sed 's/.*: \([0-9.]*\).*/\1/')
    if [ -n "$FOLD_ACC" ]; then
        FOLD_ACCURACIES[$FOLD]=$FOLD_ACC
        echo ">>> Fold $FOLD Best Accuracy: $FOLD_ACC"
    else
        echo ">>> WARNING: Could not extract accuracy for fold $FOLD"
        FOLD_ACCURACIES[$FOLD]="0.0"
    fi

    echo ""
    echo "Fold $FOLD training complete. Output: $FOLD_OUTPUT_DIR"
    echo "========================================"
done

echo ""
echo "========================================"
echo "5-Fold Cross-Validation Complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""

# Calculate and display summary statistics
echo "========================================"
echo "5-FOLD RESULTS SUMMARY"
echo "========================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "========================================"
echo ""
echo "Per-Fold Accuracies:"
for FOLD in 1 2 3 4 5; do
    ACC=${FOLD_ACCURACIES[$FOLD]}
    ACC_PCT=$(echo "$ACC * 100" | bc -l 2>/dev/null || echo "N/A")
    if [ "$ACC_PCT" != "N/A" ]; then
        ACC_PCT=$(printf "%.2f" $ACC_PCT)
    fi
    echo "  Fold $FOLD: $ACC ($ACC_PCT%)"
done
echo ""

# Calculate mean and std using Python (more reliable than bc for this)
python3 << EOF
import sys
accs = []
for fold in range(1, 6):
    acc_str = "${FOLD_ACCURACIES[1]}" if fold == 1 else \
              "${FOLD_ACCURACIES[2]}" if fold == 2 else \
              "${FOLD_ACCURACIES[3]}" if fold == 3 else \
              "${FOLD_ACCURACIES[4]}" if fold == 4 else \
              "${FOLD_ACCURACIES[5]}"
    try:
        accs.append(float(acc_str))
    except:
        pass

if len(accs) == 5:
    import statistics
    mean = statistics.mean(accs)
    std = statistics.stdev(accs)
    print("=" * 40)
    print(f"MEAN ACCURACY: {mean:.4f} ({mean*100:.2f}%)")
    print(f"STD DEVIATION: {std:.4f} ({std*100:.2f}%)")
    print(f"RESULT: {mean*100:.2f}% ± {std*100:.2f}%")
    print("=" * 40)

    # SOTA comparison
    print("")
    print("SOTA Comparison:")
    print(f"  BEATs:      98.10%")
    print(f"  CLAP:       96.70%")
    print(f"  AST:        95.70%")
    print(f"  SAFE (ours): {mean*100:.2f}% ± {std*100:.2f}%")
    print("=" * 40)
else:
    print(f"WARNING: Only {len(accs)} valid fold accuracies found")
    print(f"Accuracies: {accs}")
EOF

echo ""
echo "========================================"

# Save summary to file
SUMMARY_FILE="${OUTPUT_BASE}/5fold_summary.txt"
cat << EOF > "$SUMMARY_FILE"
ESC-50 5-Fold Cross-Validation Summary
======================================
Experiment: $EXPERIMENT_NAME
Date: $(date)
Job ID: ${SLURM_JOB_ID:-local}

Configuration:
  Batch size: $BATCH_SIZE
  Epochs: $NUM_EPOCHS
  SAFE LR: $SAFE_LR
  Head LR: $HEAD_LR
  Fusion layers: $FUSION_LAYERS
  Num audio tokens: $NUM_AUDIO_TOKENS
  Mixup alpha: $MIXUP_ALPHA
  Label smoothing: $LABEL_SMOOTHING
  Unfreeze CLAP layers: $UNFREEZE_CLAP

Per-Fold Results:
  Fold 1: ${FOLD_ACCURACIES[1]}
  Fold 2: ${FOLD_ACCURACIES[2]}
  Fold 3: ${FOLD_ACCURACIES[3]}
  Fold 4: ${FOLD_ACCURACIES[4]}
  Fold 5: ${FOLD_ACCURACIES[5]}

EOF

# Append mean/std calculation to summary file
python3 << EOF >> "$SUMMARY_FILE"
accs = []
for acc_str in ["${FOLD_ACCURACIES[1]}", "${FOLD_ACCURACIES[2]}", "${FOLD_ACCURACIES[3]}", "${FOLD_ACCURACIES[4]}", "${FOLD_ACCURACIES[5]}"]:
    try:
        accs.append(float(acc_str))
    except:
        pass

if len(accs) == 5:
    import statistics
    mean = statistics.mean(accs)
    std = statistics.stdev(accs)
    print(f"Mean Accuracy: {mean:.4f} ({mean*100:.2f}%)")
    print(f"Std Deviation: {std:.4f} ({std*100:.2f}%)")
    print(f"")
    print(f"Final Result: {mean*100:.2f}% ± {std*100:.2f}%")
EOF

echo "Summary saved to: $SUMMARY_FILE"
echo "========================================"
