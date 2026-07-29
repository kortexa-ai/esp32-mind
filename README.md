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
| Validation perplexity | 11.59 |
| Measured XIAO throughput | 13.6–14.2 tokens/s |

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

Or fetch the tested model, tokenizer, and golden output from the private
[`xiao-model-v1` release](https://github.com/kortexa-ai/esp32-mind/releases/tag/xiao-model-v1):

```bash
mkdir -p firmware/model
gh release download xiao-model-v1 --pattern model.bin --pattern golden.txt \
  --dir firmware/model
gh release download xiao-model-v1 --pattern vocab.h --dir firmware/esp32_llm
gh release download xiao-model-v1 --pattern bpe16384.json --dir data
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

## Validated run

The `xiao-model-v1` artifact was trained on smarty’s RTX PRO 6000 in 35.24
seconds over 20.48M TinyStories tokens. Its 4-bit image has SHA-256
`3b1a1cf0438f5efe63b95451b13aaa5af5da5fa96985099ef6a8616a9027f26b`.
The portable C runtime matches the exported PyTorch golden with maximum
absolute logit error below `1e-5`.

On the physical XIAO (`74:4D:BD:95:BB:88`):

```text
prompt: Once upon a time
mind> , there was a little girl named Lily. She loved to play outside and
      explore the world around her. One day, she found a big, shiny rock...
MIND END tokens=40 seconds=2.81 tok_s=14.22
```

After model staging and scratch allocation, 5,294 KiB of PSRAM remains free.

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
