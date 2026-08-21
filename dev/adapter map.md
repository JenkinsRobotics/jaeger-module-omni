# Adapter Map — every bridge between MiniCPM's frozen parts and our backbone

The whole strategy in one picture. MiniCPM-o 4.5's specialists all speak one
language internally — Qwen3-8B's 4096-dim embedding space. We freeze all of
them and train only the bridges into (and out of) our backbone's space.

```text
                    THEIRS, FROZEN                     OURS, TRAINED
                                                  ┌──────────────┐
 audio ──► whisper-medium (fine-tuned, 307M)      │              │
           └─► audio projector (21M) ── 4096 ──►  │ audio delta  │──┐
                                                  └──────────────┘  │
                                                  ┌──────────────┐  │
 image ──► SigLIP2 tower (~450M)                  │              │  ├──► backbone
 video ──► └─► perceiver resampler (109M)         │ vision delta │──┤    (Gemma;
               64 queries ───────────── 4096 ──►  │              │  │     SmolLM2 as
                                                  └──────────────┘  │     harness)
 text ───► backbone tokenizer (native, no adapter) ─────────────────┘
                                                        │
                                       ┌────────────────┼────────────────┐
                                       ▼                ▼                ▼
                                     text        PerformanceSpec   tool calls
                                    native        (non-neural       (native,
                                                   today)            outputs.py)
                                                     │
                                     ┌───────────────┴───────────┐
                                     ▼                           ▼
                              Track A TTS  (today)      speech delta (research)
                                                        backbone ─► 4096 ─►
                                                        their tts projector
                                                        ─► talker ─► token2wav
```

---

# The registry

| # | Adapter | Direction | Maps | Status | Trained on |
|---|---------|-----------|------|--------|-----------|
| 1 | **audio delta** | in | their audio 4096 → backbone | **TRAINING** | LibriSpeech + mix, self-distilled vs text teacher |
| 2 | **vision delta** | in | their resampler 4096 → backbone | front end located, port next | COCO/VQA-style captions, same recipe |
| 3 | **speech delta** | out | backbone → their tts-projector's 4096 | laid out, research | paired text through both backbones |
| 4 | text in/out | — | native tokenizer | done (nothing to train) | — |
| 5 | PerformanceSpec / avatar | out | structured, non-neural | contract shipped | Gen-2: small predictor head |
| 6 | turn-end head | control | backbone hidden → speak/listen | built | prefix-augmented corpora (labels free) |
| 7 | audio gate | control | encoder states → trigger | built | none — enrollment, not training |

All input deltas are the **same class** (`neural/delta.py DeltaAdapter`) because
the problem is identical: one linear-ish map from Qwen-space into
backbone-space, output scale pinned to the backbone's embedding norm.

## The shared-delta experiment

Audio and vision both emit into the SAME Qwen space. If the delta truly learns
space-to-space translation (rather than modality-specific correction), **one
delta should serve both modalities**. Cheap test once vision is ported: apply
the audio-trained delta to resampler output and measure. If it transfers, we
maintain one bridge, not N.

## What is measured vs assumed

```text
MEASURED  audio projector output dim 4096 (gguf), tts projector input 4096,
          resampler: query [4096,64], kv [1152,4096], pos_embed_k [4096,4900]
          = 64 output tokens per image, 70x70 pos grid, SigLIP 1152-dim patches
ASSUMED   video = interleaved image frames + audio chunks (their README's
          omni_contents pattern — sequence order carries time, no fusion module)
```

## Speech delta honesty

Their talker + token2wav are Qwen-paired and frozen-loadable, so the *interface*
is two linears — but training data for backbone→Qwen hidden-state mapping is
subtler than input deltas (hidden states, not embeddings; per-position, not
pooled). It stays research. Track A cloned TTS remains the production voice.

---

# Training recipe (shared by every input delta)

```text
freeze      their entire front end + our backbone
train       the delta alone (2-4M params)
teacher     the text path through the SAME backbone (self-distillation)
data        open, licensed, labeled — fetch_dataset.py catalog
objective   warmup CE on transcript/caption, then hidden+KL distill
guards      embed_norm pinned · chat-template spliced · held-out split ·
            prefix augmentation (audio) · free-running eval, not teacher-forced
```
