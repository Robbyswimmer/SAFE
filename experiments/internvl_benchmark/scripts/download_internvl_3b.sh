#!/bin/bash
# Download InternVL "3B" variant for benchmark sweeps.
# Note: InternVL 3.5 has no official 3B checkpoint; this uses the 4B model alias.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$SAFE_ROOT"

python3 scripts/download_internvl.py --model internvl3.5-3b --output-dir models
