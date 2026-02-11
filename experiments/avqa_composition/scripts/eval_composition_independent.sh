#!/bin/bash
#SBATCH --job-name=comp-indep-eval
#SBATCH --output=logs/composition_independent_eval_%j.out
#SBATCH --error=logs/composition_independent_eval_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
# pass --gres=gpu:3 at submit time

set -euo pipefail

SAFE_ROOT="${SAFE_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"

CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

MODEL_CONFIG=${MODEL_CONFIG:-composition_independent}
DATA_ROOT=${DATA_ROOT:-data/music_avqa}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/composition_independent_eval}

COMPOSE_AUDIO_CKPT=${COMPOSE_AUDIO_CKPT:-checkpoints/composition_audio_study/best_model.pt}
COMPOSE_VISION_CKPT=${COMPOSE_VISION_CKPT:-checkpoints/composition_vision_study/best_model.pt}

BATCH_SIZE=${BATCH_SIZE:-1}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}
FUSION_GATE=${FUSION_GATE:-0.1}
EVAL_MODALITIES=${EVAL_MODALITIES:-text,audio,image,both}
MAX_ANSWER_TOKENS=${MAX_ANSWER_TOKENS:-16}
MAX_SAMPLES=${MAX_SAMPLES:-0}

LAYER_ADDITIVITY_PROBE=${LAYER_ADDITIVITY_PROBE:-1}
LAYER_PROBE_SAMPLES=${LAYER_PROBE_SAMPLES:-512}
LAYER_PROBE_EVERY=${LAYER_PROBE_EVERY:-1}

WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-SAFE-Composition}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-composition_independent_eval_${SLURM_JOB_ID:-local}}
WANDB_TAGS=${WANDB_TAGS:-composition,independent,staggered,eval}

export SAFE_QWEN_QUANT=${SAFE_QWEN_QUANT:-none}
export SAFE_GRAD_CKPT=${SAFE_GRAD_CKPT:-0}
export FP16=${FP16:-0}
export SAFE_DEVICE_MAP=${SAFE_DEVICE_MAP:-auto}
export SAFE_MAX_MEMORY=${SAFE_MAX_MEMORY:-0=46GiB,1=46GiB,2=46GiB,cpu=160GiB}
export SAFE_OFFLOAD_FOLDER=${SAFE_OFFLOAD_FOLDER:-$SAFE_ROOT/.hf_offload}

mkdir -p logs "$OUTPUT_DIR" "$SAFE_OFFLOAD_FOLDER"
cd "$SAFE_ROOT"

WANDB_ARGS=()
if [[ "$WANDB" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME" --wandb-tags "$WANDB_TAGS")
fi

MAX_SAMPLES_ARGS=()
if [[ "$MAX_SAMPLES" != "0" ]]; then
  MAX_SAMPLES_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

PROBE_ARGS=()
if [[ "$LAYER_ADDITIVITY_PROBE" == "1" ]]; then
  PROBE_ARGS+=(--layer-additivity-probe --layer-probe-samples "$LAYER_PROBE_SAMPLES" --layer-probe-every "$LAYER_PROBE_EVERY")
fi

python3 "$SAFE_ROOT/experiments/avqa_composition/train_avqa_composition.py" \
  --dataset music_avqa \
  --model-config "$MODEL_CONFIG" \
  --train-manifest "$DATA_ROOT/manifests/train.jsonl" \
  --val-manifest "$DATA_ROOT/manifests/validation.jsonl" \
  --media-root "$MEDIA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs 0 \
  --num-audio-tokens "$NUM_AUDIO_TOKENS" \
  --train-modality both \
  --eval-modalities "$EVAL_MODALITIES" \
  --fusion-gate "$FUSION_GATE" \
  --max-answer-tokens "$MAX_ANSWER_TOKENS" \
  --freeze-audio-encoder \
  --compose-audio-ckpt "$COMPOSE_AUDIO_CKPT" \
  --compose-vision-ckpt "$COMPOSE_VISION_CKPT" \
  "${MAX_SAMPLES_ARGS[@]}" \
  "${PROBE_ARGS[@]}" \
  "${WANDB_ARGS[@]}"

