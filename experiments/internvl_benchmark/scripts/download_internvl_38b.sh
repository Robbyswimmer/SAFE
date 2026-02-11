#!/bin/bash
# Download InternVL 3.5-38B model weights (~76GB)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$SAFE_ROOT"

python3 scripts/download_internvl.py --model internvl3.5-38b --output-dir models
