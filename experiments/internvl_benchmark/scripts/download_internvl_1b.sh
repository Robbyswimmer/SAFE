#!/bin/bash
# Download InternVL 3.5-1B model weights (smallest 3.5 variant)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$SAFE_ROOT"

python3 scripts/download_internvl.py --model internvl3.5-1b --output-dir models
