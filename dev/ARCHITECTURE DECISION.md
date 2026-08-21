# Architecture Decision — the production voice pipeline

Decided on measurement, 2026-08-20. Supersedes the assumption that we would
train a Whisper→Gemma bridge for audio input.

---

# Decision

**Native Gemma 4 E4B tower for input, our streaming runtime around it, and an
Ultravox-shaped trained bridge for output.** Not MiniCPM's architecture, not
Ultravox's, and not a pure-native design either.

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

## Input: native wins, and it is not close

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
train a Whisper->Gemma input bridge   REDUNDANT. Native tower is better and
                                      free. Our delta hit 0.939 held-out
                                      accuracy proving the technique — the
                                      technique now serves the OUTPUT side.
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

1. **Audio comprehension gap** — the 25% that text gets right and audio does
   not. Options: E4B at higher precision (we run 4-bit), a paralinguistic
   branch, or a small trained bridge ALONGSIDE the native tower. Measure
   before building.
2. **Turn-end head** — the last piece of hands-free always-on. Labels come
   free from prefix augmentation, already built.
3. **Output delta** — backbone → talker, Ultravox-shaped, so the voice is ours.
4. **Fair MiniCPM baseline** — needs `llama-omni-server` with `duplex_mode`.
   Worth doing once, for an honest reference number.
