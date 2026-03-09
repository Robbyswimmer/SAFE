#!/bin/bash
#SBATCH --job-name=proj-manifold
#SBATCH --output=logs/proj_manifold_%j.out
#SBATCH --error=logs/proj_manifold_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ -z "${SAFE_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    SAFE_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SAFE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
  fi
fi

CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

MODEL_CONFIG=${MODEL_CONFIG:-rkca}
CHECKPOINT=${CHECKPOINT:-outputs/rkca_prompted_avqa/best_model.pt}
DATA_ROOT=${DATA_ROOT:-data/music_avqa}
MEDIA_ROOT=${MEDIA_ROOT:-$SAFE_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/projector_manifold_rkca}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_SAMPLES=${MAX_SAMPLES:-512}
NUM_AUDIO_TOKENS=${NUM_AUDIO_TOKENS:-8}
FUSION_GATE=${FUSION_GATE:-1.0}
SEED=${SEED:-42}
NUM_WORKERS=${NUM_WORKERS:-2}
LLM_MODEL=${LLM_MODEL:-}
MODALITY_AWARE_PROMPTS=${MODALITY_AWARE_PROMPTS:-0}
TEXT_PROMPT_PREFIX=${TEXT_PROMPT_PREFIX:-}
AUDIO_PROMPT_PREFIX=${AUDIO_PROMPT_PREFIX:-}
IMAGE_PROMPT_PREFIX=${IMAGE_PROMPT_PREFIX:-}
BOTH_PROMPT_PREFIX=${BOTH_PROMPT_PREFIX:-}

export SAFE_GRAD_CKPT=${SAFE_GRAD_CKPT:-0}

mkdir -p logs "$OUTPUT_DIR"
cd "$SAFE_ROOT"

PROMPT_ARGS=()
if [[ "$MODALITY_AWARE_PROMPTS" == "1" ]]; then
  PROMPT_ARGS+=(--modality-aware-prompts)
  if [[ -n "$TEXT_PROMPT_PREFIX" ]]; then
    PROMPT_ARGS+=(--text-prompt-prefix "$TEXT_PROMPT_PREFIX")
  fi
  if [[ -n "$AUDIO_PROMPT_PREFIX" ]]; then
    PROMPT_ARGS+=(--audio-prompt-prefix "$AUDIO_PROMPT_PREFIX")
  fi
  if [[ -n "$IMAGE_PROMPT_PREFIX" ]]; then
    PROMPT_ARGS+=(--image-prompt-prefix "$IMAGE_PROMPT_PREFIX")
  fi
  if [[ -n "$BOTH_PROMPT_PREFIX" ]]; then
    PROMPT_ARGS+=(--both-prompt-prefix "$BOTH_PROMPT_PREFIX")
  fi
fi

LLM_ARGS=()
if [[ -n "$LLM_MODEL" ]]; then
  LLM_ARGS+=(--llm-model "$LLM_MODEL")
fi

python3 "$SAFE_ROOT/experiments/avqa_composition/analyze_projector_manifold.py" \
  --model-config "$MODEL_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --val-manifest "$DATA_ROOT/manifests/validation.jsonl" \
  --media-root "$MEDIA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --num-audio-tokens "$NUM_AUDIO_TOKENS" \
  --fusion-gate "$FUSION_GATE" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  "${LLM_ARGS[@]}" \
  "${PROMPT_ARGS[@]}"
