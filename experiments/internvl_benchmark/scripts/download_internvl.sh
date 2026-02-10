#!/bin/bash
# Download InternVL 3.5-8B model weights
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$SAFE_ROOT"

python3 scripts/download_internvl.py --model internvl3.5-8b --output-dir models
