# Omni acceptance — the only score that counts

The product target is one continuously running audio-to-audio graph:

```text
PCM -> audio encoder -> Gemma -> speech conditioner -> Talker -> Token2Wav -> PCM
```

Text tokens may exist inside the language model, as they do in MiniCPM-o, and
text may be copied to logs in parallel.  A transcript, decoded string, text
packet, tokenizer round-trip, or TTS request may not be the transfer format at
either neural boundary.  Evaluation may transcribe the final waveform; that
transcript is a measurement, never runtime input.

## Required 80% gate

One shared spoken corpus is run through legacy, MiniCPM-o, and Jaeger.  A
Jaeger case counts only when the same invocation accepts input PCM and produces
output PCM through the graph above.  Missing output, a hidden STT hint, browser
speech synthesis, a text-to-TTS fallback, or replaying a reference waveform
scores that case as zero.

Jaeger passes the initial product milestone only when all of these are true:

1. End-to-end semantic correctness is at least 80% of the legacy pipeline on
   the same cases (10/12 when legacy is 12/12).
2. Output intelligibility, measured by an independent ASR round trip, retains
   at least 80% of the legacy pipeline's word accuracy.
3. Median time from speech end to first output audio is no more than 1.25x the
   legacy median, with wall-clock-paced input and warm models.
4. Input capture is lossless and output starts streaming before the complete
   answer is generated.
5. Barge-in stops queued/playback audio and resumes capture without duplicated
   or missing input; the next response retains conversation context.
6. Runtime provenance records the exact models, WAV hashes, process
   exclusivity, device, per-stage timings, generated code count, waveform hash,
   and architecture audit result.

MiniCPM-o remains the complete reference.  Matching legacy establishes the
80% product milestone; matching MiniCPM's learned turn-taking, voice cloning,
prosody, and duplex behavior is the next bar.

## Current evidence (2026-08-21)

- No-transcript Gemma input: 10/12, 83% of legacy, freshly reproduced.
- Talker/Token2Wav boundary: functional from exact MiniCPM condition tensors
  after fixing GGUF Q/K layout and code-head orientation.  This is a component
  proof, not an end-to-end Jaeger pass.
- Gemma-to-Talker speech conditioner: not trained or wired.
- Strict integrated Jaeger score: 0/12 because the required output PCM is
  absent.  Component successes do not override this result.
