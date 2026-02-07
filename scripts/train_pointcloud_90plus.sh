#!/bin/bash
#SBATCH --job-name=PC-90plus
#SBATCH --output=logs/pointcloud_90plus_%j.txt
#SBATCH --error=logs/pointcloud_90plus_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH -p gpu

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe-env

echo "========================================"
echo "ModelNet40 - Target 90%+ Accuracy"
echo "========================================"
echo "Key changes from 84% baseline:"
echo "  - MLP head (4-layer w/ GELU+dropout) vs linear"
echo "  - Mean pooling vs last-token"
echo "  - 32 tokens vs 16"
echo "  - Cosine annealing vs constant LR"
echo "  - 10 fusion layers (full network) vs 6 (first half)"
echo "  - Point dropout augmentation enabled"
echo "  - Head weight decay 0.01"
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
    --probe-pooling mean \
    --probe-head-type mlp \
    --data-path /data/SalmanAsif/RobbyMoseley/SAFE/SAFE/data \
    --output-dir /data/SalmanAsif/RobbyMoseley/SAFE/SAFE/experiments/modelnet40_classification/outputs/target_90plus \
    --fusion-layer-indices 1,5,9,13,17,21,25,29,33,37 \
    --num-pointcloud-tokens 32 \
    --batch-size 16 \
    --num-epochs 1000 \
    --safe-lr 6e-5 \
    --head-lr 1e-3 \
    --weight-decay 0.01 \
    --head-weight-decay 0.01 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.3 \
    --lr-scheduler cosine \
    --warmup-steps 200 \
    --min-lr 1e-6 \
    --unfreeze-encoder-last-n 8 \
    --encoder-checkpoint /data/SalmanAsif/RobbyMoseley/SAFE/SAFE/checkpoints/pointbert/pointbert_modelnet40_1024.pt \
    --aug-dropout \
    --max-eval-batches 999 \
    --eval-every 1 \
    --early-stopping-patience 150 \
    --fp16 \
    --wandb \
    --wandb-project ModelNet40-Classification \
    --wandb-run-name target_90plus_mlp_mean_32tok \
    --wandb-tags modelnet40,target90,mlp,mean,32tok,cosine,10layers

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
