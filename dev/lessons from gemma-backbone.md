# What the parallel agent's trainer does right — 2026-08-21

Their `train_audio_adapter.py` reached **8.13% held-out WER**. Mine reached
cosine +0.04 and unreadable output. The differences are not subtle, and none
of them are architectural cleverness — they are all about what is in the
training loop.

## 1. Precompute the front end. This is the big one.

```text
cached feature = AvgPool5(ReLU(linear1(APM(audio))))
```

The encoder and the first projection are FROZEN, so their output for a clip is
deterministic and identical on every epoch. Caching it means each training step
is a matmul plus a backbone forward — no whisper, no encoder, no 30 s of
padded mel per sample.

I had `training/encoder_cache.py` sitting in the repo, unwired, with a
docstring saying exactly this. I never connected it, and instead ran whisper
on every step of every run all day.

## 2. Quantise the backbone for training

```text
gemma-4-E4B-it-4bit-mlx    3.9 GB
google/gemma-4-E4B-it      15 GB fp32 / 14 GB bf16
```

They kept the backbone in the loop and it was cheap. I spent hours on OOMs,
gradient checkpointing, dtype fixes, and eventually removed the backbone
entirely — because I was carrying 14 GB when 3.9 GB was available.

## 3. Learning rate 2e-5, not 3e-4

Mine was **15x too high**. Their loss went to `train 0.538 / val CE 0.166`;
mine oscillated 16–49 and I spent several messages narrating noise as trend.

## 4. Train ONE layer, and pick the right one

```text
frozen    APM (MiniCPM's FINE-TUNED whisper) -> linear1 -> ReLU -> AvgPool5
trained   linear2: 4096 -> 2560              10.5M params
```

`linear2` is the only Qwen-specific part of MiniCPM's projector, so it is the
only part that needs replacing for Gemma. `linear1` transfers unchanged
because the input is the same whisper.

Note they use **MiniCPM's fine-tuned APM**, not vanilla whisper. It was tuned
for exactly this job. I switched to stock `whisper-medium.en` and threw that
away.

## 5. The output is a native replacement part

> "Pooling can be moved before linear2 exactly because linear2 is affine, so a
> checkpoint can be copied directly into the native GGUF's linear2 weight and
> bias."

The trained weights drop straight into MiniCPM's audio GGUF. No new runtime,
no framework bridge, no torch-beside-MLX. That is a deployment story mine did
not have.

## 6. Initialise from the token-space map

Their `init` stage writes a ridge/token-space map as the starting checkpoint.
I built `fit_alignment.py` for this, could not get Qwen's 4096-wide table
without a GGUF reader, and gave up on the init entirely rather than spending
ten minutes on `pip install gguf`.

---

# The pattern

Every one of these is about **removing work from the training loop** or
**starting from something that already partly works**. None is a modelling
insight.

I spent the day making the loop harder: full-precision 14 GB backbone,
whisper re-run every step, random init, learning rate 15x too high, and
gradients pushed through 42 layers to reach a single linear.
