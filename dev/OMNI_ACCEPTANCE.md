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
- That 10/12 input result uses Gemma E4B's native audio tower. It is not
  evidence for the requested Whisper adapter. The requested input graph is
  now explicitly `MiniCPM-tuned Whisper -> one MiniCPM-shaped projector whose
  final affine emits 2560-wide Gemma embeddings directly -> Gemma`; the old
  `MiniCPM projector -> separate delta -> Gemma` experiment is not the target.
- The complete strict graph now executes and writes cloned-voice PCM. The
  first learned-conditioner pilot scored only 1/4 semantic and 14.6% mean word
  accuracy.
- Replacing its weak learned lexical map with an offline-compiled, runtime
  tensor-only Gemma-ID -> Qwen-ID bridge improved the same four unseen cases
  to 3/4 semantic and 67.9% mean word accuracy. q01 failed upstream because
  Gemma answered that no image was supplied; the speech output followed it.
- A full 12-case native-tower waveform run now scores 10/12 semantic (83.3%)
  and 93.5% mean spoken-word accuracy. Talker yields codes while it is still
  decoding and Token2Wav emitted multiple live chunks on every case; median
  first audio was 2040.9 ms. This passes the core semantic/intelligibility/
  streaming sub-gate, not the complete acceptance gate.
- Complete acceptance remains false: this file also requires <=1.25x legacy
  latency under a matched warm run, wall-clock-paced input, output-side
  barge-in, and complete model/WAV/hash provenance. The current file-fed run
  did not exercise those requirements.
- A warm-resident run exposed severe unified-memory contention with 43.5 GB
  system swap. CPU Talker is more reproducible today; the Metal-F16 path runs
  but has not matched its sampled output quality or clean latency yet.
