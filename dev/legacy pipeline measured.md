# Legacy Pipeline — Measured

Read from the three working implementations in `reference/`
(`voice_assistant.py`, `voice_chat.py`, `voice_chat_streaming.py`) and from the
JFK transcripts in `training/samples/`.

These are ours and they work. This document exists so Omni does not rebuild
what already exists, and so its claims are measured against a real baseline
rather than an imagined one.

---

# The latency budget, from the tuned constants

`voice_chat_streaming.py`:

```text
FRAME_MS                30      VAD frame
PRE_ROLL_MS            240      buffer kept before speech onset
MIN_SPEECH_MS          400      shorter than this is not a turn
SILENCE_HANGOVER_MS    700      silence required to call end-of-turn
POST_PADDING_MS        250      tail kept after
BARGE_IN_MS            200      speech during TTS before we yield the floor
MAX_SPEECH_MS        30000      soft cap; waits for a pause rather than cutting
```

**`SILENCE_HANGOVER_MS = 700` plus `POST_PADDING_MS = 250` is ~950 ms of dead
time before ASR is allowed to start.** Not a tuning mistake — the comment is
explicit that end of turn is decided by silence and never by a stopwatch, which
is correct for a text-bridge design. A transcript needs a finished utterance.

This is the number the plan's "Correction 1" predicted at 500-800 ms. It is
higher, and it is confirmed from our own tuned code rather than estimated.

**That ~950 ms is what incremental prefill removes**, because a streaming
encoder does not need the utterance to be over to start consuming it. It is the
largest single latency item in the stack and it is architectural, not model
speed.

---

# What already exists — do NOT rebuild

The three implementations contain a full-duplex audio layer that is tuned and
working:

```text
Speex AEC (pyaec)          full-duplex echo cancellation, 10 ms int16 frames,
                           AEC_FILTER_LENGTH 3200
webrtcvad                  aggressiveness 2, 30 ms frames
barge-in                   200 ms of speech during TTS yields the floor
self-speech rejection      mic.pause during TTS (voice_assistant), AEC in the
                           duplex versions
noise floor calibration    auto-calibrated, plus a non-speech filter
                           (core/voice_gate.py)
wake phrase gating         so a TV cannot start a conversation
follow-up window           15 s after a reply, no wake word needed
2-pass STT                 a fast model and an accurate one, both medium.en,
                           single_segment=True, no_context=True
streaming TTS              Kokoro, streamed rather than rendered whole
```

## Scoping consequence

An earlier version of the plan said the runtime — "streaming chunking, turn-end
detection, barge-in, cancellation, the audio I/O path" — was ours to write.
Half of that is already written and tuned.

**Omni replaces what happens after the phrase queue, not the audio I/O layer.**

```text
mic -> AEC -> VAD -> phrase_queue  |  -> STT -> text -> LLM -> text -> TTS
       ^^^^^^^^^^^^^^^^^^^^^^^^^^  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
       exists, keep it             |     this is what Omni replaces
```

What Omni still owns: incremental prefill during speech, the projector, the
learned turn-end head, and cancellation across cognition rather than just
audio. Notably the turn-end head replaces `SILENCE_HANGOVER_MS` — a fixed
silence timeout cannot tell "...and then he said" from "...and that was that."

---

# STT quality baseline

18-minute JFK "we choose to go to the Moon" speech, human transcript as ground
truth, 2219 words. Normalised (lowercase, punctuation and `[APPLAUSE]` markers
stripped), Levenshtein word error rate:

```text
medium.en via pywhispercpp      2244 words    WER 3.33%
medium.en via faster-whisper    2237 words    WER 2.75%
```

Both good. faster-whisper is modestly better, presumably from decoding defaults
rather than different weights — same checkpoint.

**This is the ceiling for the Omni path.** Whisper's encoder cannot carry more
content information than its own decoder can extract. If the projector path
ends up materially worse than ~3% WER-equivalent understanding, information is
being lost in the projector, not in the encoder — and that is a measurable,
attributable failure rather than a mystery.

It also bounds the honest claim. Omni does not promise better transcription. It
promises the same content understanding *plus* what a transcript throws away,
arriving ~950 ms sooner.

---

# Corpus available today

```text
training/samples/jfk_moon_full.wav    17.8 min, 16 kHz mono
                 jfk_moon_full.txt    human transcript, 2219 words
                 jfk_moon.wav         27 s excerpt
                 + two machine transcripts for comparison
```

Segmented by `training/import_audio.py`:

```text
17.8 min -> 288 segments, 12.9 min of speech, 72% voiced
```

288 real human utterances. Small, single-speaker, and oratorical rather than
conversational — so it is a genuine Tier 1 sample and not a substitute for the
public corpora or for our own recorded sessions. Its real value is as a
**fixed evaluation set**: same audio, known ground truth, comparable across
every checkpoint we train.

---

# Settings worth inheriting

```text
single_segment=True, no_context=True   both set for latency in the live path.
                                       no_context also stops one bad
                                       transcript from biasing the next.
language pinned                        we already do this in app.py; a short
                                       clip auto-detects wrong and the
                                       transcript becomes a wrong-language
                                       teacher target rather than a noisy one.
two-pass STT                           fast model for the wake/preview path,
                                       accurate model for the real turn. The
                                       omni equivalent is a small turn-end
                                       head plus the full projector path.
16 kHz mono float32                    matches capture.py already.
```
