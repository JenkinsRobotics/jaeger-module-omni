# Review: the parallel agent's v5 input-acceptance result — 2026-08-21

Claim under review: *"Jaeger's owned audio-input pipeline now clearly exceeds
MiniCPM-o under equal conditions"* — 12/12 both, 942.45 ms vs 2153.06 ms
median complete response after speech end, 2.28x faster.

**The measurements are sound and reproduce from their artifacts.** The
methodology is better than mine was earlier today: exclusive-accelerator
audit with `competing_jobs: []` recorded per trial, WAV bytes pinned by
SHA-256, three isolated repeats per model, immutable commit ledgers. After
spending today discovering that contention silently corrupted four of my own
numbers, I take that discipline seriously.

Two things nonetheless need correcting, and one thing of theirs we should
adopt outright.

---

# 1. The 2.28x is turn DETECTION, not the audio pipeline

Their own per-trial JSON decomposes it. MiniCPM is fed paced 1-second frames
and emits a LISTEN/SPEAK decision per frame:

```text
minicpm_duplex_01.json     audio_s 1.175   frames 9   listen 7   speak 2

  frame 1  is_speak false  ''
  frame 2  is_speak false  ''
  frame 3  is_speak false  ''
  frame 4  is_speak TRUE   ' Two plus two is four.'
```

Speech ends at 1.175 s; MiniCPM commits on the **fourth one-second frame**.
Its 2196 ms is dominated by waiting for frame boundaries at 1 Hz, not by
compute. On a quiet machine I measured MiniCPM's actual audio->text work at
**0.175 s prefill + 0.350 s decode = 0.525 s**.

Jaeger's 942 ms decomposes the other way:

```text
                        turn detection      compute        total
Jaeger                  510 ms endpoint  +  ~432 ms   =    942 ms
MiniCPM (as shipped)   ~1670 ms @ 1 Hz   +  ~525 ms   =   2196 ms
```

So on raw audio->text compute the two are comparable, with MiniCPM marginally
ahead — while also producing speech-ready hidden states, which Jaeger's path
does not.

This does **not** make the result unimportant. MiniCPM's 1 Hz cadence is
intrinsic: its LISTEN/SPEAK tokens are learned per-chunk, so you cannot
shorten the frame without retraining. The latency a user feels really is
2.28x better. The correct claim is:

> Jaeger's turn-taking policy delivers responses 2.28x sooner than MiniCPM's
> learned per-frame policy. The audio-input pipelines themselves are
> comparable on compute.

"The audio-input pipeline exceeds MiniCPM" attributes to the encoder what the
endpointer earned.

---

# 2. Where their data and mine agree — the important part

Their GAP_REVIEW table and my measurements independently reach the same
conclusion about the native Gemma tower:

```text
                          theirs              mine
native Gemma tower        13.48% WER          12.2% WER
whisper (base metal)       0.98% WER           7.3% WER (medium)
```

Both of us measured the native tower at roughly an order of magnitude worse
than a real ASR front end. Notably, **their winning pipeline uses
`stt_model: base.en`, not the native tower** — so the headline result is
whisper -> Gemma, which is architecturally our `legacy`. We are not disagreeing
about architecture; we both found the same thing and their engineering around
it is further along.

Their choice of `base.en` (148 MB) is also properly evidenced rather than
assumed: 0.98% WER on 12 LibriSpeech clips, 0.49% for Turbo Q5. That is a
legitimate accuracy/latency trade, not a shortcut.

---

# 3. What we should adopt from them, unchanged

Our turn-end head is still listed as "pending". Theirs works, and the design
is in their artifacts:

```text
endpoint_speech_end_silence_ms   510      acoustic silence endpoint
semantic_confirmed_min_ms        500      semantic early-end for questions
acoustic_confirmed_guard_ms      800      guard against premature commit
end_speculation_enabled          true     speculative decode at probable-end
stt_reused_speculative_finals    1        the speculation was USED, not wasted
```

Speculating the final decode at `probably_ended` and reusing it when the
endpoint confirms is the trick that buys most of the margin — the STT result
is already in hand when the turn closes.

They also found a real bug worth inheriting: a 500 ms threshold silently
rounded **down** to 480 ms against capture-frame boundaries. Minimum
thresholds must round up. That class of error would have been invisible.

---

# 4. What the result does not cover

Their own scope note is honest about this, and it should not be lost when the
headline is quoted:

```text
speech synthesis          deliberately deferred
AEC                       deferred, needs a playback reference
semantic turn head        "injected detector; semantic head untrained"
vision                    1035 ms/frame vs MiniCPM 379 ms  (their table)
```

The comparison is **audio-in / text-out**. MiniCPM's reason for existing is
audio-in / *speech*-out, and on that axis it is uncontested because neither of
us has built it.

## The ceiling this does not raise

I measured the native Gemma tower's prosody carry directly, on embeddings,
before the LLM:

```text
same words, 4 deliveries   spread 0.0238
4 different sentences      spread 0.1243
ratio                      0.192
```

Delivery moves the representation 19% as much as content does. For a
whisper -> text -> LLM pipeline the number is not 0.192, it is **zero** —
prosody, emotion and speaker identity are discarded at the text boundary by
construction, no matter how good the endpointer is.

So their v5 result makes the sequential path faster and better engineered
than ours, and it does not move the capability ceiling at all. Voice cloning,
emotional delivery and impersonation — the YouTube-production requirements —
sit entirely on the other side of that boundary.

---

# Verdict

```text
1. Result is real, reproduces from artifacts, methodology is better than mine.
2. Headline mis-attributes: the 2.28x is turn detection, not the encoder.
   On compute the two pipelines are comparable, MiniCPM marginally ahead.
3. We independently agree the native Gemma tower is ~10x worse than a real
   ASR front end. Their winning path is whisper -> Gemma, i.e. our `legacy`.
4. Adopt their endpointer wholesale -- it is our pending turn-end head, done.
5. Nothing here addresses speech out or the prosody ceiling, which is where
   the product requirements actually live.
```
