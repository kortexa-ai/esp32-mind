# esp32-mind

A small, entirely on-device TinyStories language model for the Seeed Studio
XIAO ESP32-S3. Send a text seed from a computer over USB serial; the XIAO
generates and streams a continuation back over the same cable.

The target board has 8 MB flash and 8 MB PSRAM. Its model is an 11.5M-parameter
Per-Layer Embedding (PLE) transformer compressed to 4-bit weights:

| | XIAO model |
| --- | ---: |
| Vocabulary | 16,384 |
| Core / table / total parameters | 0.50M / 9.44M / 11.51M |
| Model image | 5.69 MiB |
| Context | 256 tokens |
| Model flash partition | 6.88 MiB |

This is a story continuation model, not an assistant. It can complete “Once
upon a time…”; it cannot reliably answer questions or follow instructions.
The interesting hamster is the on-chip runtime.

## Use the connected XIAO

Requirements:

- Seeed Studio XIAO ESP32-S3 with 8 MB PSRAM enabled
- Arduino CLI with Espressif Arduino core 3.3.7
- `esptool`, `uv`, and Python 3.12+
- a data-capable USB cable, apparently the rarest mineral on Earth

Train and export the model:

```bash
./experiments/train_xiao.sh
```

Verify the exported model with the portable C runtime:

```bash
cc -O3 -o /tmp/esp32-mind-verify firmware/host_verify/verify.c -lm
/tmp/esp32-mind-verify firmware/model/model.bin firmware/model/golden.txt
```

Build and flash:

```bash
./build.sh
./flash.sh /dev/cu.usbmodem31114301
```

Send a prompt:

```bash
./chat.sh "Once upon a time" --max-tokens 80
```

`chat.sh` tokenizes on the host, sends token IDs over USB, and prints the raw
UTF-8 continuation emitted by the XIAO. The serial protocol is deliberately
small: `PING`, `INFO`, and `GENERATE <count> <token_id>...`.

## Layout

- `firmware/esp32_llm/` — XIAO firmware and 8 MB partition table
- `firmware/common/llm.h` — portable 4-bit PLE inference runtime
- `tools/mind.py` — USB prompt client
- `src/` — model, training, quantization, and export code
- `experiments/train_xiao.sh` — reproducible XIAO training configuration

Training data, checkpoints, tokenizers, and model binaries are intentionally
gitignored. See [`UPSTREAM.md`](UPSTREAM.md) for origin and licensing details
and [`RESULTS.md`](RESULTS.md) for the original architecture experiments.

## Origin

This project is adapted from
[`slvDev/esp32-ai`](https://github.com/slvDev/esp32-ai), an MIT-licensed
28.9M-parameter ESP32-S3 PLE demonstration targeting a 16 MB board. The
upstream source remains untouched; this repository adds the 8 MB XIAO target
and USB prompt/response path.
