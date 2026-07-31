# Multi-XIAO Plan

Ideas captured 2026-07-30 so they don't evaporate. Nothing here is committed
work; it is the feasibility analysis for scaling past one board, plus one
single-board sidequest.

Baseline for all numbers: the validated single-XIAO model — 11.5M stored
params, d_model=96, 6 layers, 4-bit weights, ~70 ms per token, 13.6–14.2
tok/s, 8 MB flash / 8 MB PSRAM per board.

## Why splitting across boards is unusually feasible here

The usual killer for distributed inference on microcontrollers is moving
activations. Our hidden state is 96 values: **192 bytes in fp16, 384 in
fp32**. That fits in a single ESP-NOW frame, or ~1.5 ms over a 2 Mbps UART
link. Against ~70 ms of compute per board per token, inter-board traffic is
roughly 2–5% overhead. For most models a mesh is a bandwidth fantasy; for
this one it is a rounding error.

PLE actively helps:

- The table is per-layer, so in a layer-wise split each board carries only
  the PLE rows for its own layers. A stage needs just the incoming hidden
  state plus the token id (2 bytes riding in the same packet).
- Each board holds only the KV cache for its own layers.

The memory-hierarchy split the project is built on extends across boards
almost embarrassingly cleanly.

## Options for three boards

### Pipeline split (the obvious one)

Board 1 = USB + tokenizer + layers 0–5, board 2 = layers 6–11, board 3 =
layers 12–17 + output head; the sampled token rings back to board 1.

- Capacity: ~3x flash and PSRAM → roughly a **34M stored-param model**
  (more layers, or push d_model instead).
- Latency: autoregressive generation cannot pipeline — token t+1 needs
  token t — so stages run sequentially per token. 3x the model at ~1/3 the
  speed: **~4–5 tok/s**.
- Prefill *does* pipeline, so long prompts ingest at near-aggregate speed.

### Tensor parallelism

No. At d_model=96 the matmuls are tiny and every layer costs two
all-reduces of radio latency. This is how you turn three computers into
half a computer.

### Speculative decoding (the interesting one)

Keep the current 11.5M model on board 1 as the draft; use the 3-board
pipeline as the verifier. Verifying K drafted tokens pipelines exactly like
prefill, so most of the lost speed comes back. TinyStories is a narrow
domain, so draft acceptance should be high. This is the configuration where
three XIAOs run a ~34M model at close to the current 14 tok/s. A real
project, but *the* project.

### Cheap wins, no mesh protocol needed

- **Best-of-3 sampling.** Same model on all three boards, different
  temperatures/seeds, pick the continuation with the best total logprob.
  Only token ids and scores cross the wire.
- **Three stories at once.** Trivial 3x aggregate throughput; best
  demo-per-effort ratio available.
- **Exquisite corpse mode.** Three differently-trained minds taking turns
  continuing one story. Scientifically useless. Would absolutely do it.

## Transport

Wired UART (or SPI) between neighbors is right for a desk cluster —
deterministic ~1–2 ms, and the XIAO has the pins for a ring. ESP-NOW also
works (~1–2 ms per hop, activation fits one frame) and gets the word
"mesh" into the README, which has marketing value.

## Sidequest: a diffusion model on a single S3

Feasible, but it inverts the trade the current design wins on.

The compute math is the whole story. Autoregressive generation costs L
token-passes for L tokens. A masked-diffusion LM (LLaDA/MDLM-style) costs
T × L token-passes — T denoising steps, each a full bidirectional forward
over all L positions, no KV cache. Diffusion's pitch on GPUs is that the L
positions per step run in parallel; the S3 has no parallelism to hide it,
so it is honestly T× more compute. A 256-token story at T=16 is ~5
minutes. The pragmatic config is block diffusion — 32-token blocks, 4–8
distilled steps — landing at **~2–3.5 tok/s** versus 14.

- Memory is fine, arguably better: no KV cache, same weights, attention at
  d_model=96 over 256 positions is peanuts.
- PLE survives: lookups stay per-token per-layer (masked positions fetch
  the mask row); flash traffic scales with the same T× as everything else.
- Quality-per-param suffers: at 11M params AR is a strong baseline,
  discrete diffusion typically needs more training compute to match it,
  and its "perplexity" is an ELBO bound — the clean comparison with the
  current model is lost.
- Silver lining: diffusion is embarrassingly parallel across positions, so
  the S3's second core finally earns its keep. That is a real 2x, and also
  the ceiling.

Why do it anyway:

1. **Infilling.** `MIND EDIT`: mask a span mid-story and denoise a
   replacement that fits both sides. AR fundamentally cannot. A new verb
   for the serial protocol, not a slower noun.
2. **The spectacle.** Stream every denoising step over serial and watch a
   wall of mask characters sharpen into a story. On a device whose charm
   is "the hamster is doing its best," slow-and-visible beats
   fast-and-invisible.

Caution: diffusion meshes *worse* than the transformer. Bidirectional
attention means boards must exchange K/V for all positions every step, so
the 384-bytes-per-hop free lunch from the pipeline analysis disappears.

## Verdict

- Mesh: feasible, and unusually so — PLE plus tiny d_model makes the split
  nearly free in bandwidth. The unavoidable trade is capacity-for-latency
  (~34M params at ~4–5 tok/s) unless speculative decoding is stacked on
  top, which is the genuinely interesting version.
- Diffusion: a legitimate single-board experiment — same memory story,
  ~2–3 tok/s, infilling and the denoise-on-terminal demo as the payoff. As
  a replacement it is a downgrade; as a second personality for the same
  hardware it is kind of great.

Suggested order if this ever becomes real: best-of-3 sampling (weekend),
pipeline split (the mesh), speculative decoding on top (the headline),
diffusion arm (the sideshow).
