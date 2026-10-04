#!/usr/bin/env bash
# Encode DIS5K with the frozen FLUX VAE, T5, and CLIP into one HDF5 file per split.
# A finished file is what training reads when use_cache is true.
#
#   flowdis/scripts/cache_latents.sh              # DIS-TR and DIS-VD
#   flowdis/scripts/cache_latents.sh DIS-TR
#   LIMIT=2 flowdis/scripts/cache_latents.sh DIS-VD

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/flowdis/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/jeremy/dataset/DIS5K_extracted}"
SCHNELL_DIR="${SCHNELL_DIR:-$ROOT/checkpoints/FLUX.1-schnell}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"
RESOLUTION="${RESOLUTION:-1024}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-0}"

if [[ $# -eq 0 ]]; then
  splits=(DIS-TR DIS-VD)
else
  splits=("$@")
fi

mkdir -p "$CACHE_DIR"
export PYTHONPATH="$ROOT"

for split in "${splits[@]}"; do
  echo "caching ${split} -> ${CACHE_DIR}/${split}.h5"
  "$PYTHON" -m flowdis.data.cache \
    --data-root "$DATA_ROOT" \
    --schnell-dir "$SCHNELL_DIR" \
    --split "$split" \
    --output "$CACHE_DIR/${split}.h5" \
    --resolution "$RESOLUTION" \
    --device "$DEVICE" \
    --limit "$LIMIT"
done
