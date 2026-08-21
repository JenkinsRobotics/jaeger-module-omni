# research/

Experiments for the staged build in
`dev/jaeger agent omni0 immediate development.md`. Not the shipped module —
this is where things get proven before they earn a `module.yaml`.

Architecture decisions here are ported from MiniCPM-o 4.5, read out of its
weights and source rather than guessed. See
`dev/minicpm-o measured architecture.md` for what was measured and how.

## Boundary

Nothing here touches `jaeger-agent`. That repo is the sequential pipeline with
text as the transfer format; this one is neural module integration with tensors
as the transfer format. Parallel developments.

Omni runs **its own** `whisper -> text -> gemma` teacher to generate training
targets. It does not import from, patch, or depend on the production agent.

## Setup

Python 3.11.9 (pinned in `.python-version`).

```sh
python -m venv .venv
.venv/bin/python -m pip install -r research/requirements.txt
```

## Run every check

```sh
cd research && ../.venv/bin/python check_all.py
```

Runs in ~2 s, fully offline, no model weights. Exits non-zero on failure.

## Modules

| Script | Stage | Proves |
|---|---|---|
| `performance_spec.py` | seam | Spec round-trips; validation rejects bad cognition output |
| `capture.py` | 1 | Turn log round-trips: manifest, wav, float32 fidelity |
| `projector.py` | 3 | `JaegerAudioProjector` shapes, token formula, ReLU, pool order |
| `distill.py` | 3/4 | Self-distillation losses: cosine floor, monotonicity, detach |
| `streaming.py` | 5 | 1 s chunking, sample fidelity, rolling 1500-frame eviction |
| `turn_end.py` | 5 | Turn-end head, hysteresis, asymmetric loss |
| `stubs.py` | — | Tiny Whisper/LM stand-ins so everything above runs offline |
| `pipeline.py` | 3 | End-to-end wiring, and that the projector actually learns |
| `inspect_gguf.py` | — | Reads GGUF headers without loading weights |
| `stage0_access.py` | 0 | Real Whisper + Gemma yield tensors — **needs weights** |

`pipeline.py` is the one that matters: it overfits a handful of samples through
the full path and asserts the loss drops below the uniform-prior floor. If the
graph is detached, the projector frozen, or a shape silently broadcasting, that
assert fails.

## Ported from MiniCPM — do not "fix" these

```text
projector       Linear -> ReLU -> Linear -> AvgPool(k=5, s=5)
                ReLU not GELU; no LayerNorm inside; pool AFTER both linears
                (audition.cpp "Step 6: audio projector")
token rate      audio_pool_step 5 on a 50 fps encoder = 10 tokens/s
streaming       1 s chunks = 50 frames, rolling 1500-frame encoder cache
                (audition.cpp:396 — 1500/50 = 30 s of context)
no resampler    no perceiver, no learned queries, no attention pooling.
                Selection is the LLM's attention's job, not the projector's.
```

## Ours, because MiniCPM's version cannot be copied

```text
turn-end head   MiniCPM predicts <|speak|>/<|listen|> as vocabulary tokens
                (omni.h:460). That needs backbone training we cannot do, so we
                replicate the behaviour with a small head over hidden state.
                Labels come free from capture.py turn boundaries.
PerformanceSpec MiniCPM has two outputs (text, audio). We have three — the
                avatar reads the same spec. An explicit intermediate is what
                makes the third output nearly free.
```

## Before `stage0_access.py`

```sh
export GEMMA_ID=...     # exact HF repo id or local snapshot for gemma-4-E4B
export WHISPER_ID=openai/whisper-medium
```

Gemma is a gated repo — accept the license on HF and `huggingface-cli login`,
or point `GEMMA_ID` at a local directory. Its printout gives the real
`hidden_size`, which is the `cognition_dim` every module here defaults to 2048
as a placeholder.

## Hardware note

This machine is an M1 Max / 32 GB. Everything in `check_all.py` runs here in
seconds. Stage 3 training does not — MPS is roughly an order of magnitude
slower than an A100 for this, and bf16 on MPS falls back silently (scripts use
fp16 on MPS for that reason). Prototype here, train on CUDA.
