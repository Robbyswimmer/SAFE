#!/bin/bash
set -euo pipefail

# Create a separate env for raw InternVL training/eval so we don't have to keep
# one environment compatible with both:
#   1) newer Qwen Omni captioner support, and
#   2) older trust_remote_code InternVL loading behavior.
#
# Usage:
#   bash experiments/internvl_benchmark/scripts/create_safe_internvl_env.sh
#   SRC_ENV=safe-env DST_ENV=safe-internvl bash ...
#   SRC_ENV=safe-env DST_ENV=safe-internvl TRANSFORMERS_VER=4.49.0 bash ...

SRC_ENV=${SRC_ENV:-safe-env}
DST_ENV=${DST_ENV:-safe-internvl}
TRANSFORMERS_VER=${TRANSFORMERS_VER:-}
TOKENIZERS_VER=${TOKENIZERS_VER:-}
ACCELERATE_VER=${ACCELERATE_VER:-}
HF_HUB_VER=${HF_HUB_VER:-}

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "Could not find conda.sh under ~/miniconda3" >&2
    exit 1
fi

echo "Cloning env '$SRC_ENV' -> '$DST_ENV'"
conda create -y -n "$DST_ENV" --clone "$SRC_ENV"

echo "Activating '$DST_ENV'"
conda activate "$DST_ENV"

if [[ -n "${TRANSFORMERS_VER}" || -n "${TOKENIZERS_VER}" || -n "${ACCELERATE_VER}" || -n "${HF_HUB_VER}" ]]; then
    echo "Applying explicit package overrides"
    PKGS=()
    if [[ -n "${TRANSFORMERS_VER}" ]]; then
        PKGS+=("transformers==${TRANSFORMERS_VER}")
    fi
    if [[ -n "${TOKENIZERS_VER}" ]]; then
        PKGS+=("tokenizers==${TOKENIZERS_VER}")
    fi
    if [[ -n "${ACCELERATE_VER}" ]]; then
        PKGS+=("accelerate==${ACCELERATE_VER}")
    fi
    if [[ -n "${HF_HUB_VER}" ]]; then
        PKGS+=("huggingface_hub==${HF_HUB_VER}")
    fi
    python -m pip install --upgrade "${PKGS[@]}"
else
    echo "No package overrides requested; cloned env versions are preserved."
fi

echo
echo "Done. Suggested smoke test:"
echo "  CONDA_ENV=${DST_ENV} sbatch experiments/internvl_benchmark/scripts/eval_music_avqa_text_caption_baseline.sh"
