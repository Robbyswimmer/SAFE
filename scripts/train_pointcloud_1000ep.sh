#!/bin/bash
#SBATCH --job-name=PC-1000ep
#SBATCH --output=logs/pointcloud_1000ep_%j.txt
#SBATCH --error=logs/pointcloud_1000ep_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH -p gpu

set -euo pipefail

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# Conda setup
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe-env

echo "========================================"
echo "ModelNet40 Point Cloud - 1000 Epochs"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "========================================"

mkdir -p logs

python /data/SalmanAsif/RobbyMoseley/SAFE/SAFE/train_pointcloud.py \
    --config modelnet40 \
    --phase classification \
    --llm-probe-head \
    --probe-pooling last \
    --probe-head-type linear \
    --data-path /data/SalmanAsif/RobbyMoseley/SAFE/SAFE/data \
    --output-dir /data/SalmanAsif/RobbyMoseley/SAFE/SAFE/experiments/modelnet40_classification/outputs/kitchen_sink_1000ep \
    --fusion-layer-indices 1,5,9,13,17,21 \
    --num-pointcloud-tokens 16 \
    --batch-size 16 \
    --num-epochs 1000 \
    --safe-lr 6e-5 \
    --head-lr 1e-3 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.3 \
    --lr-scheduler constant \
    --min-lr 1e-6 \
    --unfreeze-encoder-last-n 8 \
    --encoder-checkpoint /data/SalmanAsif/RobbyMoseley/SAFE/SAFE/checkpoints/pointbert/pointbert_modelnet40_1024.pt \
    --max-eval-batches 999 \
    --eval-every 1 \
    --fp16 \
    --wandb \
    --wandb-project ModelNet40-Classification \
    --wandb-run-name kitchen_sink_1000ep \
    --wandb-tags modelnet40,best,combo,augment,kitchen_sink,1000ep

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
