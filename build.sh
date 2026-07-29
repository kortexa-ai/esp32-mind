#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

fqbn='esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=8M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info'

arduino-cli compile \
    --fqbn "$fqbn" \
    --build-property compiler.optimization_flags=-O3 \
    --build-path .build/esp32-mind \
    firmware/esp32_llm
