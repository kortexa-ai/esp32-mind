# XIAO ESP32-S3 inference firmware

The sketch runs the quantized TinyStories PLE model from a memory-mapped
partition on an 8 MB XIAO ESP32-S3. The host performs BPE tokenization; the
device performs every model operation and streams generated UTF-8 over native
USB serial.

The custom flash layout is:

| Offset | Size | Contents |
| ---: | ---: | --- |
| `0x10000` | 1 MiB | firmware |
| `0x110000` | 6.88 MiB | 4-bit model |
| `0x7F0000` | 64 KiB | core dump |

From the repository root:

```bash
./experiments/train_xiao.sh
./build.sh
./flash.sh /dev/cu.usbmodem31114301
./chat.sh "Once upon a time" --max-tokens 80
```

`flash.sh` writes the Arduino image and then writes
`firmware/model/model.bin` at `0x110000`. The model only needs to be rewritten
after a new export.

The line-oriented protocol is:

```text
PING
INFO
GENERATE <max_new_tokens> <token_id> [<token_id> ...]
```

Generation starts after a `MIND BEGIN` line and ends with a `MIND END` line
containing token count, wall time, and throughput. Prompts are limited to 96
tokens, generation to 128 tokens, and total context to 256 tokens.
