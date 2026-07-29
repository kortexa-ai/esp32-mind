#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

fqbn='esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=8M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info'
port=${1:-}

if [[ -z "$port" ]]; then
    port=$(find /dev -maxdepth 1 -name 'cu.usbmodem*' -print | head -n 1)
fi
if [[ -z "$port" ]]; then
    echo "No /dev/cu.usbmodem* device found" >&2
    exit 1
fi
if [[ ! -f .build/esp32-mind/esp32_llm.ino.bin ]]; then
    echo "Build missing; run ./build.sh first" >&2
    exit 1
fi
if [[ ! -f firmware/model/model.bin ]]; then
    echo "Model missing; run ./experiments/train_xiao.sh first" >&2
    exit 1
fi

arduino-cli upload \
    --port "$port" \
    --fqbn "$fqbn" \
    --input-dir .build/esp32-mind \
    firmware/esp32_llm

for _ in {1..40}; do
    next_port=$(find /dev -maxdepth 1 -name 'cu.usbmodem*' -print | head -n 1)
    if [[ -n "$next_port" ]]; then
        port=$next_port
        break
    fi
    sleep 0.25
done

esptool \
    --chip esp32s3 \
    --port "$port" \
    --baud 921600 \
    write-flash 0x110000 firmware/model/model.bin
