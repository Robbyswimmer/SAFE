#!/bin/bash
#SBATCH --job-name=avqa-semantic
#SBATCH --output=logs/avqa_semantic_%j.out
#SBATCH --error=logs/avqa_semantic_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -p gpu

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SAFE_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

CONDA_ENV=${CONDA_ENV:-safe-env}
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

INPUT_MANIFEST=${INPUT_MANIFEST:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/music_avqa_teacher_captions_validation.jsonl}
OUTPUT_MANIFEST=${OUTPUT_MANIFEST:-$SAFE_ROOT/experiments/internvl_benchmark/outputs/music_avqa_teacher_captions_validation_semantic.jsonl}
RAW_CAPTION_FIELD=${RAW_CAPTION_FIELD:-rich_audio_caption}

cd "$SAFE_ROOT"
mkdir -p logs

python3 experiments/internvl_benchmark/build_music_avqa_semantic_targets.py \
  --input-manifest "$INPUT_MANIFEST" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --raw-caption-field "$RAW_CAPTION_FIELD"
