# Architecture Decision — the production voice pipeline

Decided on measurement, 2026-08-20. Supersedes the assumption that we would
train a Whisper→Gemma bridge for audio input.

---

# Decision criteria — EXTENSIBILITY, not just today's accuracy

An earlier version of this document ranked pipelines on accuracy and latency
alone. That is the wrong frame, and the user named why: **the native tower is
borrowed and frozen.**

```text
                    can we EXTEND it?     what it costs us
native Gemma tower  NO. Google's weights, fixed token budget, fixed
                    processing, no speaker branch, no prosody branch.
                    Whatever it shipped with is permanently what we get.
whisper + delta     YES. We own the bridge. Add a paralinguistic branch,
                    a speaker-identity branch, swap the encoder, retrain for
                    our domain — all without asking anyone.
```

The features we are explicitly aiming for — prosody surviving into cognition,
speaker identity, emotion, the things a transcript destroys and that MiniCPM
has — are exactly the things a frozen third-party tower cannot be taught.

So the ranking on ACCURACY (legacy 100% > native 75%) and the ranking on
CEILING both point away from native-as-primary. Native's real role is a
zero-training fallback and a cheap parallel signal, not the architecture.

A 109 ms latency edge does not buy a permanent capability ceiling.

# Decision — REVISED after measuring legacy

**Whisper's encoder bound at the TENSOR level into Gemma (the delta), our
streaming runtime around it, and an Ultravox-shaped bridge for output.**

An earlier version of this document said the native Gemma tower made the
input delta redundant. Measuring the legacy reference disproved that:

```text
legacy   12/12  100%   whisper transcript -> text -> E4B    11.8 s
native    9/12   75%   E4B native audio tower               0.53 s
hybrid   11/12   92%   both, serially                       18.9 s
```

Same backbone, same questions. **The accuracy lives in Whisper's encoder and
the speed lives in streaming** — legacy proves the information is there, and
native proves the tower loses it. The hybrid confirms it by recovering the
accuracy the moment a transcript is present, at the cost of paying full ASR
latency.

So the answer is neither: bind Whisper's ENCODER STATES directly into Gemma's
embedding space — no transcript, no text bottleneck. That is exactly the delta
we already built and trained. It should give Whisper's comprehension at
streaming speed, and keep the prosody a transcript destroys.

The native tower remains valuable as a cheap parallel signal and as the
zero-training fallback.

```text
audio  ─► E4B native tower ──┐
image  ─► E4B vision tower ──┼─► one KV cache ─► E4B ─► text + tool calls
text   ─► tokenizer ─────────┘        │
                                      ├─► turn-end head        (ours, small)
                                      └─► output delta ─► talker ─► voice
                                          (Ultravox-shaped: RMSNorm + SwiGLU)

   runtime: chunked prefill during speech · barge-in with cache rewind
```

---

# The evidence

## CORRECTION: audio fidelity measured directly, and the gap is small

Answer-correctness conflates audio quality with model quality. Measuring the
audio path alone — same backbone, same prompt, "repeat exactly what was said",
scored as WER against known text:

```text
native (Gemma tower)   12.2% pooled WER   1551 ms   9/12 clips PERFECT
legacy (whisper)        7.3% pooled WER   3770 ms
```

The native tower hears nearly as well as Whisper. The earlier "25% audio
comprehension gap" was wrong on two counts:

  1. A reasoning-channel leak ("<|channel>thought Thinking Process...") was
     appended after CORRECT transcriptions and counted as insertions —
     224% pooled WER before stripping it, 12.2% after. That measured a
     formatting quirk, not fidelity.
  2. Only TWO of the three answer failures were audio at all:

```text
q11 "spell cat backwards"   heard "drawkcab"        AUDIO
q12 "two four six eight"    heard "2 4 6"           AUDIO  (dropped a word)
q8  "how many continents"   heard PERFECTLY,
                            answered "no continents" BRAIN
```

So the input problem is smaller and better-shaped than stated: a ~5 point WER
gap and a leaky output format, not a broken front end. That reopens native as
the primary input, with the delta as the way to close the remaining 5 points
if it proves out at E4B width.

## Input: the delta, because accuracy is in Whisper and speed is in streaming

12 spoken questions, streamed as a mic delivers them, clock starting at silence:

```text
accuracy      9/12 (75%)
wait median   526 ms      p90 1345 ms
realtime      0.29x       faster than the speech consumed
time split    tail 228 ms + decode 321 ms | 334 ms hidden under speech
```

Zero training. Google's tower, our runtime. The alternative — training a
Whisper→Gemma bridge — was our plan for weeks and is now redundant for
understanding.

## But audio comprehension IS the weak link — measured

All three failures answer CORRECTLY when the same question is typed:

```text
                    via audio                  via text
q8  continents      "no continents on Earth"   "seven continents on Earth"
q11 spell backwards "D-r-a-w-k-b"              "Tac."
q12 sequence        "Seven."                   "Ten comes next."
```

The backbone knows all three. This is not quantisation and not capability —
**the audio path loses information the text path has**, and that costs 25%
accuracy. Everything downstream of the tower is fine.

That is the single most actionable number in the project: it says where the
next training effort belongs, and it is not where we spent the last week.

## Output: nothing native exists, so a bridge is required

No Gemma talker, no native voice. The only expressive cloned-voice path is
MiniCPM's talker + token2wav, which we have running (14.2 s of intelligible
cloned-voice English, verified by whisper round-trip). Reaching it needs a
trained backbone→talker bridge.

For THAT bridge, the projector A/B says use Ultravox's design:

```text
                     cross-clip   class-sep   params   tok/s
MiniCPM  (trained)      0.9052        —        21.0M   10.00
Ultravox (trained)      0.6829      0.0485     46.1M    6.25
```

Ultravox's trained projector is markedly more discriminative. Its RMSNorm
pre+mid is also a STRUCTURAL fix for the exact collapse our from-scratch run
hit (cross-clip 0.990 until we pinned output scale). Their design solves it
properly; ours patched it.

---

# What we are NOT doing, and why

```text
rely on the native tower ALONE        Loses 25% against legacy on the same
                                      backbone AND cannot be extended — it is
                                      Google's frozen weights. Every feature
                                      we are aiming for (prosody to cognition,
                                      speaker identity, emotion) requires a
                                      bridge we own. Fallback, not primary.
ship the serial hybrid                92% but 18.9 s — it pays ASR latency in
                                      full, which is the thing streaming
                                      exists to remove.
adopt MiniCPM wholesale               Its Qwen3-8B backbone is not ours, its
                                      CLI hardcodes a Chinese persona, and
                                      its English prompt is locked behind a
                                      duplex_mode the CLI never exposes. We
                                      take its TALKER, not its architecture.
adopt Ultravox wholesale              Targets models with NO native audio
                                      (Llama, Gemma 2/3). Will not load on
                                      transformers v5. We take its PROJECTOR
                                      DESIGN, not its pipeline.
12 fps video                          1035 ms/frame warmed. One frame per 5 s
                                      is 21% duty cycle and meets the bar.
```

---

# Open, in priority order

1. **Train the delta against E4B** — Whisper encoder -> delta -> Gemma 4, the
   architecture the legacy result argues for. Target: legacy's 100% at
   native's 0.53 s. The pipeline is built and proven (it reached 0.939
   teacher-forced against SmolLM2); it needs retraining against E4B's 2560
   width, with Ultravox's RMSNorm pre+mid rather than our scale pin.
2. **Streaming Whisper encoder** — the delta only beats legacy if the encoder
   runs chunked during speech. Whisper's encoder is NOT causal (it attends
   over a full 30 s window), so this needs the rolling-cache treatment in
   `streaming.py`, not naive chunking. This is the real engineering risk in
   the plan and should be prototyped before committing.
2. **Turn-end head** — the last piece of hands-free always-on. Labels come
   free from prefix augmentation, already built.
3. **Output delta** — backbone → talker, Ultravox-shaped, so the voice is ours.
4. **Fair MiniCPM baseline** — needs `llama-omni-server` with `duplex_mode`.
   Worth doing once, for an honest reference number.
