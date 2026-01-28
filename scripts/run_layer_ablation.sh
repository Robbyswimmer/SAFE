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

# Layer ablation study for audio/pointcloud classification
# Tests: 1, 2, 4, 8, 12, 16 fusion layers with pre-FFN injection
#
# Usage:
#   MODEL_TYPE=llava ./scripts/run_layer_ablation.sh       # LLaVA 1.5 13B audio (default)
#   MODEL_TYPE=llava_single ./scripts/run_layer_ablation.sh # LLaVA single layer: 1,3,5,...,39
#   MODEL_TYPE=llava_kv ./scripts/run_layer_ablation.sh    # LLaVA 1.5 13B audio with KV augmentation
#   MODEL_TYPE=qwen ./scripts/run_layer_ablation.sh        # Qwen3 8B audio
#   MODEL_TYPE=pointcloud ./scripts/run_layer_ablation.sh  # LLaVA 1.5 13B point cloud

MODEL_TYPE=${MODEL_TYPE:-"llava"}
DATA_PATH=${DATA_PATH:-"/data/SalmanAsif/AVE_Dataset"}
POINTCLOUD_DATA_PATH=${POINTCLOUD_DATA_PATH:-"./data"}
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
echo "Model: $MODEL_TYPE"
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

# Define layer configurations for LLaVA (40 layers)
# Strategy: Always start at layer 1, add layers with stride 4
# Hypothesis: Early layers (1-7) have more transformation impact than layer count
declare -A LLAVA_LAYERS
LLAVA_LAYERS["1"]="1"                                     # Just layer 1
LLAVA_LAYERS["2"]="1,5"                                   # 1, 5
LLAVA_LAYERS["3"]="1,5,9"                                 # 1, 5, 9
LLAVA_LAYERS["4"]="1,5,9,13"                              # 1, 5, 9, 13
LLAVA_LAYERS["5"]="1,5,9,13,17"                           # 1, 5, 9, 13, 17
LLAVA_LAYERS["6"]="1,5,9,13,17,21"                        # 1, 5, 9, 13, 17, 21
LLAVA_LAYERS["7"]="1,5,9,13,17,21,25"                     # 1, 5, 9, 13, 17, 21, 25
LLAVA_LAYERS["8"]="1,5,9,13,17,21,25,29"                  # 1, 5, 9, 13, 17, 21, 25, 29
LLAVA_LAYERS["9"]="1,5,9,13,17,21,25,29,33"               # 1, 5, 9, 13, 17, 21, 25, 29, 33
LLAVA_LAYERS["10"]="1,5,9,13,17,21,25,29,33,37"           # 1, 5, 9, 13, 17, 21, 25, 29, 33, 37
LLAVA_LAYERS["11"]="1,5,9,13,17,21,25,29,33,37,39"        # + layer 39
LLAVA_LAYERS["12"]="1,5,9,13,17,21,25,29,33,35,37,39"     # + layer 35

# Define layer configurations for Qwen 8B (32 layers)
# Same strategy: start at 1, stride 4
declare -A QWEN_LAYERS
QWEN_LAYERS["1"]="1"                                      # Just layer 1
QWEN_LAYERS["2"]="1,5"                                    # 1, 5
QWEN_LAYERS["3"]="1,5,9"                                  # 1, 5, 9
QWEN_LAYERS["4"]="1,5,9,13"                               # 1, 5, 9, 13
QWEN_LAYERS["5"]="1,5,9,13,17"                            # 1, 5, 9, 13, 17
QWEN_LAYERS["6"]="1,5,9,13,17,21"                         # 1, 5, 9, 13, 17, 21
QWEN_LAYERS["7"]="1,5,9,13,17,21,25"                      # 1, 5, 9, 13, 17, 21, 25
QWEN_LAYERS["8"]="1,5,9,13,17,21,25,29"                   # 1, 5, 9, 13, 17, 21, 25, 29 (max for 32 layers)

run_audio_experiment() {
    local num_layers=$1
    local layer_indices=$2
    local model_config=$3
    local model_name=$4
    local extra_args=$5

    local run_name="layer-ablation-${model_name}-${num_layers}L-preffn"
    local output_dir="${OUTPUT_BASE}/${model_name}_${num_layers}layers"

    echo ""
    echo "========================================"
    echo "Running ${model_name} (audio): ${num_layers} layers"
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
        --model-config "$model_config" \
        --fusion-layer-indices "$layer_indices" \
        --fusion-injection-point "pre_ffn" \
        --pooling "last" \
        --num-workers 4 \
        --log-interval 10 \
        --wandb \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "$run_name" \
        $extra_args

    echo "Completed ${model_name} (audio): ${num_layers} layers at $(date)"
    echo ""
}

run_pointcloud_experiment() {
    local num_layers=$1
    local layer_indices=$2

    local run_name="layer-ablation-pointcloud-${num_layers}L-preffn"
    local output_dir="${OUTPUT_BASE}/pointcloud_${num_layers}layers"

    echo ""
    echo "========================================"
    echo "Running Point Cloud (LLaVA): ${num_layers} layers"
    echo "Layers: ${layer_indices}"
    echo "WandB run: ${run_name}"
    echo "Output: ${output_dir}"
    echo "Started: $(date)"
    echo "========================================"

    mkdir -p "$output_dir"

    python train_pointcloud.py \
        --config modelnet40 \
        --phase classification \
        --llm-probe-head \
        --probe-pooling last \
        --data-path "$POINTCLOUD_DATA_PATH" \
        --output-dir "$output_dir" \
        --batch-size "$BATCH_SIZE" \
        --num-epochs "$NUM_EPOCHS" \
        --lr "$LEARNING_RATE" \
        --safe-lr "$LEARNING_RATE" \
        --head-lr 1e-3 \
        --fusion-layer-indices "$layer_indices" \
        --fusion-injection-point "pre_ffn" \
        --num-workers 4 \
        --log-every 10 \
        --fp16 \
        --wandb \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "$run_name"

    echo "Completed Point Cloud (LLaVA): ${num_layers} layers at $(date)"
    echo ""
}

# ============================================================
# Run experiments based on MODEL_TYPE
# ============================================================

if [ "$MODEL_TYPE" = "llava" ]; then
    echo ""
    echo "========================================"
    echo "LLaVA 1.5 13B Audio Layer Ablation"
    echo "Strategy: Start at layer 1, stride 4"
    echo "========================================"

    for num_layers in 1 2 3 4 5 6 7 8 9 10 11 12; do
        layer_indices="${LLAVA_LAYERS[$num_layers]}"
        run_audio_experiment "$num_layers" "$layer_indices" "phase1" "llava" "--fp16"
    done

elif [ "$MODEL_TYPE" = "llava_single" ]; then
    echo ""
    echo "========================================"
    echo "LLaVA 1.5 13B Single Layer Ablation"
    echo "Strategy: One layer per run, stride 2 (1,3,5,...,39)"
    echo "========================================"

    # Single layer runs: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39
    for layer in 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39; do
        run_name="layer-ablation-llava-L${layer}-preffn"
        output_dir="${OUTPUT_BASE}/llava_layer${layer}"

        echo ""
        echo "========================================"
        echo "Running LLaVA single layer: ${layer}"
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
            --fusion-layer-indices "$layer" \
            --fusion-injection-point "pre_ffn" \
            --pooling "last" \
            --num-workers 4 \
            --log-interval 10 \
            --fp16 \
            --wandb \
            --wandb-project "$WANDB_PROJECT" \
            --wandb-run-name "$run_name"

        echo "Completed LLaVA single layer ${layer} at $(date)"
    done

elif [ "$MODEL_TYPE" = "llava_kv" ]; then
    echo ""
    echo "========================================"
    echo "LLaVA 1.5 13B Audio Layer Ablation (KV Augmentation)"
    echo "Strategy: Start at layer 1, stride 4"
    echo "========================================"

    for num_layers in 1 2 3 4 5 6 7 8 9 10 11 12; do
        layer_indices="${LLAVA_LAYERS[$num_layers]}"
        run_audio_experiment "$num_layers" "$layer_indices" "kv_augment" "llava-kv" "--fp16"
    done

elif [ "$MODEL_TYPE" = "qwen" ]; then
    echo ""
    echo "========================================"
    echo "Qwen3 8B Audio Layer Ablation"
    echo "Strategy: Start at layer 1, stride 4"
    echo "========================================"

    # Qwen requires: no quantization, no gradient checkpointing, no fp16
    export SAFE_QWEN_QUANT=none
    export SAFE_GRAD_CKPT=0
    BATCH_SIZE=1  # Qwen needs batch size 1

    for num_layers in 1 2 3 4 5 6 7 8; do
        layer_indices="${QWEN_LAYERS[$num_layers]}"
        run_audio_experiment "$num_layers" "$layer_indices" "qwen3_8b" "qwen8b" ""
    done

elif [ "$MODEL_TYPE" = "pointcloud" ]; then
    echo ""
    echo "========================================"
    echo "LLaVA 1.5 13B Point Cloud Layer Ablation"
    echo "Strategy: Start at layer 1, stride 4"
    echo "========================================"

    # Point cloud uses LLaVA layer indices (40 layers)
    for num_layers in 1 2 3 4 5 6 7 8 9 10 11 12; do
        layer_indices="${LLAVA_LAYERS[$num_layers]}"
        run_pointcloud_experiment "$num_layers" "$layer_indices"
    done

else
    echo "ERROR: Unknown MODEL_TYPE '$MODEL_TYPE'"
    echo "Valid options: llava, llava_single, llava_kv, qwen, pointcloud"
    exit 1
fi

echo "========================================"
echo "${MODEL_TYPE} Layer Ablation Study Complete"
echo "Finished: $(date)"
echo "========================================"
