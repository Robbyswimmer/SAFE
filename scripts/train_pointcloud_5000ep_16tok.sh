#!/bin/bash
#SBATCH --job-name=PC-5k-full
#SBATCH --output=logs/pointcloud_5000ep_full_%j.txt
#SBATCH --error=logs/pointcloud_5000ep_full_%j.err
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
echo "ModelNet40 - 5000 Epochs, Full Encoder Unfreeze"
echo "========================================"
echo "16 tokens, full encoder unfrozen, constant LR, 10 fusion layers"
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
    --output-dir /data/SalmanAsif/RobbyMoseley/SAFE/SAFE/experiments/modelnet40_classification/outputs/5000ep_full_unfreeze \
    --fusion-layer-indices 1,5,9,13,17,21,25,29,33,37 \
    --num-pointcloud-tokens 16 \
    --batch-size 16 \
    --num-epochs 5000 \
    --safe-lr 6e-5 \
    --head-lr 1e-3 \
    --weight-decay 0.01 \
    --head-weight-decay 0.01 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.3 \
    --lr-scheduler constant \
    --unfreeze-encoder-last-n 12 \
    --encoder-checkpoint /data/SalmanAsif/RobbyMoseley/SAFE/SAFE/checkpoints/pointbert/pointbert_modelnet40_1024.pt \
    --aug-dropout \
    --max-eval-batches 999 \
    --eval-every 1 \
    --early-stopping-patience 500 \
    --fp16 \
    --wandb \
    --wandb-project ModelNet40-Classification \
    --wandb-run-name 5000ep_full_unfreeze \
    --wandb-tags modelnet40,target90,mlp,mean,16tok,fixedlr,10layers,5000ep,full_unfreeze

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
