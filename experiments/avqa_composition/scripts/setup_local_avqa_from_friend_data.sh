#!/bin/bash
set -euo pipefail

# One-shot local setup for AVQA + MUSIC-AVQA using read-only source files
# from a friend's directory. All generated files are written to LOCAL_ROOT.
#
# Usage:
#   bash experiments/avqa_composition/scripts/setup_local_avqa_from_friend_data.sh
#
# Optional overrides:
#   LOCAL_ROOT=/data/.../SAFE/SAFE
#   FRIEND_ROOT=/data/.../Model-Merging
#   SPLIT_SEED=42
#   TRAIN_RATIO=0.9

LOCAL_ROOT=${LOCAL_ROOT:-/data/SalmanAsif/RobbyMoseley/SAFE/SAFE}
FRIEND_ROOT=${FRIEND_ROOT:-/data/SalmanAsif/Kaykobad-Reza/Model-Merging}
SPLIT_SEED=${SPLIT_SEED:-42}
TRAIN_RATIO=${TRAIN_RATIO:-0.9}

AVQA_SRC_JSON="$FRIEND_ROOT/data/test/avqa-test_mm_video+image+audio.json"
MUSIC_SRC_JSON="$FRIEND_ROOT/data/test/music-avqa-test_mm_video+image+audio.json"

AVQA_MEDIA_ROOT="$FRIEND_ROOT/data/evaluation_datasets/AVQA"
MUSIC_MEDIA_ROOT="$FRIEND_ROOT/data/evaluation_datasets/MUSIC-AVQA"

AVQA_LOCAL_ROOT="$LOCAL_ROOT/data/avqa_local"
MUSIC_LOCAL_ROOT="$LOCAL_ROOT/data/music_avqa_local"

PREP_SCRIPT="$LOCAL_ROOT/experiments/avqa_composition/scripts/prepare_avqa_manifests.py"
AVQA_TRAIN_SCRIPT="$LOCAL_ROOT/experiments/avqa_composition/scripts/train_preffn_avqa.sh"
MUSIC_TRAIN_SCRIPT="$LOCAL_ROOT/experiments/avqa_composition/scripts/train_preffn_music_avqa.sh"

echo "[info] LOCAL_ROOT=$LOCAL_ROOT"
echo "[info] FRIEND_ROOT=$FRIEND_ROOT"

for req in "$AVQA_SRC_JSON" "$MUSIC_SRC_JSON" "$PREP_SCRIPT" "$AVQA_TRAIN_SCRIPT" "$MUSIC_TRAIN_SCRIPT"; do
  if [[ ! -f "$req" ]]; then
    echo "[error] missing required file: $req" >&2
    exit 1
  fi
done

mkdir -p "$AVQA_LOCAL_ROOT" "$MUSIC_LOCAL_ROOT"

echo "[step] Creating pseudo train/val splits in local directory..."
python3 - "$AVQA_SRC_JSON" "$MUSIC_SRC_JSON" "$AVQA_LOCAL_ROOT" "$MUSIC_LOCAL_ROOT" "$SPLIT_SEED" "$TRAIN_RATIO" <<'PY'
import json
import random
import sys
from pathlib import Path

avqa_src = Path(sys.argv[1])
music_src = Path(sys.argv[2])
avqa_out = Path(sys.argv[3])
music_out = Path(sys.argv[4])
seed = int(sys.argv[5])
ratio = float(sys.argv[6])

if not (0.0 < ratio < 1.0):
    raise ValueError(f"TRAIN_RATIO must be between 0 and 1, got {ratio}")

random.seed(seed)

jobs = [
    (avqa_src, avqa_out, "avqa_pseudo"),
    (music_src, music_out, "music_avqa_pseudo"),
]

for src, out_dir, prefix in jobs:
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list) or not data:
        raise ValueError(f"Unexpected/empty data in {src}")
    random.shuffle(data)
    k = max(1, int(ratio * len(data)))
    train = data[:k]
    val = data[k:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{prefix}_train.json"
    val_path = out_dir / f"{prefix}_val.json"
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train, f, indent=2)
    with val_path.open("w", encoding="utf-8") as f:
        json.dump(val, f, indent=2)

    print(f"[split] {prefix}: total={len(data)} train={len(train)} val={len(val)}")
    print(f"[split] wrote: {train_path}")
    print(f"[split] wrote: {val_path}")
PY

echo "[step] Building local AVQA manifest..."
python3 "$PREP_SCRIPT" \
  --dataset avqa \
  --train-json "$AVQA_LOCAL_ROOT/avqa_pseudo_train.json" \
  --val-json "$AVQA_LOCAL_ROOT/avqa_pseudo_val.json" \
  --output-root "$AVQA_LOCAL_ROOT" \
  --media-root "$AVQA_MEDIA_ROOT" \
  --audio-root "$AVQA_MEDIA_ROOT/audio" \
  --image-root "$AVQA_MEDIA_ROOT/frames" \
  --require-both

echo "[step] Building local MUSIC-AVQA manifest..."
python3 "$PREP_SCRIPT" \
  --dataset music_avqa \
  --train-json "$MUSIC_LOCAL_ROOT/music_avqa_pseudo_train.json" \
  --val-json "$MUSIC_LOCAL_ROOT/music_avqa_pseudo_val.json" \
  --output-root "$MUSIC_LOCAL_ROOT" \
  --media-root "$MUSIC_MEDIA_ROOT" \
  --audio-root "$MUSIC_MEDIA_ROOT/audio" \
  --image-root "$MUSIC_MEDIA_ROOT/frames" \
  --require-both

echo
echo "[done] Local setup complete."
echo "[verify]"
echo "  ls -lh $AVQA_LOCAL_ROOT/manifests"
echo "  ls -lh $MUSIC_LOCAL_ROOT/manifests"
echo
echo "[submit AVQA]"
echo "  sbatch --gres=gpu:1 --export=ALL,DATA_ROOT=$AVQA_LOCAL_ROOT,MEDIA_ROOT=/,OUTPUT_DIR=$LOCAL_ROOT/checkpoints/avqa_composition/preffn_local $AVQA_TRAIN_SCRIPT"
echo
echo "[submit MUSIC-AVQA]"
echo "  sbatch --gres=gpu:1 --export=ALL,DATA_ROOT=$MUSIC_LOCAL_ROOT,MEDIA_ROOT=/,OUTPUT_DIR=$LOCAL_ROOT/checkpoints/avqa_composition/music_preffn_local $MUSIC_TRAIN_SCRIPT"

