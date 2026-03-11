#!/bin/bash
# Download a practical external audio captioning teacher model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$SAFE_ROOT"

python3 scripts/download_audio_caption_teacher.py \
  --model "${TEACHER_MODEL:-whisper-small-audio-captioning}" \
  --output-dir "${OUTPUT_DIR:-models}"
