#!/bin/bash
#SBATCH --job-name=PC-3000ep
#SBATCH --output=logs/pointcloud_3000ep_%j.txt
#SBATCH --error=logs/pointcloud_3000ep_%j.err
#SBATCH --time=168:00:00
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
echo "ModelNet40 - 3000 Epochs Long Run"
echo "========================================"
echo "Same config as target_90plus but 3000 epochs"
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
    --output-dir /data/SalmanAsif/RobbyMoseley/SAFE/SAFE/experiments/modelnet40_classification/outputs/target_90plus_3000ep \
    --fusion-layer-indices 1,5,9,13,17,21,25,29,33,37 \
    --num-pointcloud-tokens 32 \
    --batch-size 16 \
    --num-epochs 3000 \
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
    --early-stopping-patience 300 \
    --fp16 \
    --wandb \
    --wandb-project ModelNet40-Classification \
    --wandb-run-name target_90plus_3000ep \
    --wandb-tags modelnet40,target90,mlp,mean,32tok,cosine,10layers,3000ep

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
