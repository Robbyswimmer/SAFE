#!/bin/bash
#SBATCH --job-name=layer-ablation
#SBATCH --output=logs/layer_ablation_%j.log
#SBATCH --error=logs/layer_ablation_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --export=ALL

set -e

# Layer ablation study for audio classification
# Tests: 1, 2, 4, 8, 12, 16 fusion layers with pre-FFN injection
# LLaVA 1.5 13B has 40 layers (0-39)

DATA_PATH=${DATA_PATH:-"/data/SalmanAsif/AVE_Dataset"}
OUTPUT_BASE=${OUTPUT_BASE:-"outputs/layer_ablation"}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_EPOCHS=${NUM_EPOCHS:-20}
LEARNING_RATE=${LEARNING_RATE:-6e-5}
WANDB_PROJECT=${WANDB_PROJECT:-"SAFE"}

mkdir -p logs
mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "Layer Ablation Study - Pre-FFN Fusion"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "Data: $DATA_PATH"
echo "Output base: $OUTPUT_BASE"
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

# Define layer configurations
# LLaVA has 40 layers, we pick evenly spaced layers
declare -A LAYER_CONFIGS
LAYER_CONFIGS["1"]="20"                                    # Middle layer
LAYER_CONFIGS["2"]="13,26"                                 # Thirds
LAYER_CONFIGS["4"]="8,16,24,32"                            # Fifths
LAYER_CONFIGS["8"]="4,9,14,19,24,29,34,39"                 # Every 5
LAYER_CONFIGS["12"]="3,6,10,13,17,20,23,27,30,33,36,39"    # Every ~3
LAYER_CONFIGS["16"]="2,4,7,9,12,14,17,19,22,24,27,29,32,34,37,39"  # Every ~2.5

run_experiment() {
    local num_layers=$1
    local layer_indices=$2
    local run_name="layer-ablation-${num_layers}L-preffn"
    local output_dir="${OUTPUT_BASE}/${num_layers}layers"

    echo ""
    echo "========================================"
    echo "Running: ${num_layers} layers"
    echo "Layers: ${layer_indices}"
    echo "WandB run: ${run_name}"
    echo "Output: ${output_dir}"
    echo "Started: $(date)"
    echo "========================================"

    mkdir -p "$output_dir"

    python train_audio_llm_probe.py \
        --data-path "$DATA_PATH" \
        --output-dir "$output_dir" \
        --batch-size "$BATCH_SIZE" \
        --num-epochs "$NUM_EPOCHS" \
        --safe-learning-rate "$LEARNING_RATE" \
        --head-learning-rate 1e-3 \
        --model-config "phase1" \
        --fusion-layer-indices "$layer_indices" \
        --fusion-injection-point "pre_ffn" \
        --pooling "last" \
        --num-workers 4 \
        --log-interval 10 \
        --fp16 \
        --wandb \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "$run_name"

    echo "Completed: ${num_layers} layers at $(date)"
    echo ""
}

# Run all configurations sequentially
for num_layers in 1 2 4 8 12 16; do
    layer_indices="${LAYER_CONFIGS[$num_layers]}"
    run_experiment "$num_layers" "$layer_indices"
done

echo "========================================"
echo "Layer Ablation Study Complete"
echo "Finished: $(date)"
echo "========================================"
