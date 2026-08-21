# Pipeline status as of 2026-08-21

Every pipeline, what it scores, and — critically — **which benchmark the number
came from**. Not all numbers are comparable; where they are not, that is said.

Correcting a claim I made earlier: **the delta is not leading. It is last on
every measurement taken.** The argument for it is extensibility and a second
team's training evidence, not current score. Those are different claims and
were being conflated.

---

# A. OFF-THE-SHELF (someone else's, complete products)

## MiniCPM-o 4.5 — the only fully working omni system
Measured by the parallel agent (`VoiceLLM-Playground/.../BENCHMARK.md`,
2026-08-20), Q8 on the same Mac:

```text
correct answer              YES — "The capital of France is Paris."
voice out                   YES — 2.76 s waveform across 3 chunks
duplex turn-taking          12 LISTEN / 3 SPEAK decisions, model-decided
decision latency            250 ms median · 458 ms p95
audio prefill               0.88 s   ·   model load 18.35 s
peak RSS                    4,893 MB
4-gram repetition           0.00 (clean)
```

**This is the bar.** Audio in, correct answer, voice out, model-decided
turn-taking — end to end, today. Nothing of ours does all four.

Our own attempts to benchmark it failed on harness issues, not model issues
(the CLI's `--test` uses the clip as a voice-clone reference; duplex needs a
continuous decode loop). Their numbers are the trustworthy ones.

## Ultravox v0.5 — cannot run
```text
model load                  FAILS on transformers v5 (v4-era loader dies in
                            tie_weights on meta tensors), after its config
                            unconditionally fetches Meta's GATED Llama repo
projector weights           LIFTED successfully, 46.1M params, runs in our
                            ABProjector: cross-clip 0.6829 (vs MiniCPM 0.9052)
```
Value extracted: its projector DESIGN (stack-8, SwiGLU, RMSNorm pre+mid) and
the confirmation that adapter-only distribution is standard practice.

---

# B. OURS (this repo, measured on our 12-question benchmark)

All four rows below share one benchmark, so they ARE comparable to each other.

```text
pipeline   answers      audio WER   latency   notes
legacy     12/12 100%      7.3%      635 ms   whisper -> transcript -> E4B
hybrid     11/12  92%        —      ~2-3 s    audio + transcript, serial
native      9/12  75%     12.2%      526 ms   Gemma's own frozen tower
delta       4/12  33%        —       2.3 s    trained vs SmolLM2-135M
```

**The delta's 33% is not a fair comparison**: it drives a 135M backbone while
every other row drives E4B (4B). It is the only row not testing the same brain.
Its teacher-forced 0.939 means nothing — the same checkpoint free-ran at 33%.

Runtime capabilities (ours, verified live, independent of which input wins):
```text
streaming prefill    118 ms per 1 s of audio (8x realtime headroom)
barge-in             22.8 ms, conversation context preserved via cache rewind
omni input           audio + vision + text in ONE KV cache, verified
vision               1035 ms/frame, 280 tokens  (MiniCPM: 379 ms, 64 tokens)
speech out           audible cloned voice via MiniCPM talker + token2wav,
                     but wrong words without hidden-state conditioning
turn-end             NOT BUILT — the last piece of hands-free operation
```

---

# C. THE PARALLEL AGENT'S (VoiceLLM-Playground/Omni Model/gemma-backbone)

Same goal, opposite approach: replace MiniCPM's Qwen3-8B backbone with
gemma-4-E4B and train an adapter to bridge MiniCPM's audio projector into
Gemma's space.

## Before adapter training — fails
```text
correct answer      NO — repeating Qwen/omni control tokens
4-gram repetition   0.99 (degenerate)
text decode         89.94 s (vs 0.35 s for original Qwen)
duplex              3 LISTEN / 11 SPEAK, corrupted <unit> output
```

## After a 40-step pilot adapter — real learning, not yet a working answer
```text
held-out CE         5.8686 -> 1.1368     (init = ridge map W_fwd)
train CE            0.7380
data                32 utterances, 40 batch-1 steps, linear 4096->2560 only
native splice       353 chunks of ORDINARY ENGLISH (was template parroting)
held-out probes     "HE WAS IN A FEVERED STATE OF MIND" -> "WE WAS IN A
                    FEVERED STATE OF MIND"
                    "MANY LITTLE WRINKLES GATHERED BETWEEN HIS EYES" ->
                    "WE NEED TO FIND QUICKLY GATHERED BEFORE WE SEE HIS EYE'S"
```

No answer-benchmark score exists for it yet, so it cannot be placed in table B.
What it establishes is **data efficiency**: 40 steps on 32 utterances moved
held-out CE by 4.7 nats and produced recognisable transcription of unseen
audio. Ours needed 2000 steps on 6,615 turns — into a 30x smaller backbone.

---

# What this actually says

```text
1. MiniCPM is the only complete system today, and it is the bar: correct
   answers, voice out, model-decided turn-taking, 250 ms decisions.
2. Of OUR input pipelines, legacy wins outright — 100% vs native's 75%, for
   109 ms. That gap is invisible to a human; 25 points is not.
3. BOTH delta attempts are unproven as products. Ours scores 33% on a toy
   backbone; theirs has learning signal but no correct answer yet.
4. The case for the delta is (a) extensibility — native is frozen Google
   weights that can never carry prosody, speaker identity, or emotion, and
   (b) their evidence that the architecture learns fast at 4B scale.
   Neither is a current score, and should not be presented as one.
```

# The decisive experiment

Train the delta against **E4B** rather than SmolLM2-135M, and run it on the
same 12-question benchmark as everything in table B. That single run makes the
delta comparable and settles whether the architecture earns its place.

Their pilot says it should learn. Our 33% says nothing about it, because the
brain was too small to answer questions regardless.
