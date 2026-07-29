#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run python data/prepare.py --vocab 16384
uv run python src/train.py \
    --arm ple \
    --vocab 16384 \
    --d-model 96 \
    --n-layers 6 \
    --n-heads 4 \
    --ple-dim 96 \
    --target-core 500000 \
    --batch-size 16 \
    --seq-len 256 \
    --steps 5000 \
    --seed 0 \
    --tag xiao
uv run python src/export.py ple-xiao-s0
uv run python src/gen_assets.py --vocab 16384

model_bytes=$(wc -c < firmware/model/model.bin | tr -d ' ')
partition_bytes=$((0x6E0000))
if ((model_bytes > partition_bytes)); then
    echo "model.bin is ${model_bytes} bytes; partition limit is ${partition_bytes}" >&2
    exit 1
fi

echo "model.bin: ${model_bytes} / ${partition_bytes} bytes"
