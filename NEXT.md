# Next steps — current plan

One GPU job at a time. JaegerOmni 41M Medium SHA `a7f96bfb8898` is the current
broad-corpus winner: 3.97% free-running WER on the untouched 51,311-word
test-clean lockbox, with no regression across the five selection corpora.
The final pre-production training window is closed: the same checkpoint also
scores 3.30% normalized WER (22/667) on the source-qualified one-to-one
conversation gate and 12/12 spoken answers at 560 ms median. Do not run more
adapter training before physical production testing; every final-window
continuation, output-only refinement, seed repeat, and soup either tied the
conversation gate or regressed another gate.
The direct-file five-minute duplex gate now passes its stream-integrity checks:
300.0 seconds fed, zero drops, one real 6,010-token context roll, and
1,044/3,725 ms END-to-audio median/p95 after cadence streaming. Do not launch
more Medium training until the physical-room five-minute gate runs. The broad
43-room AEC gate has 0 false barges and 100% recall, but its 843 ms p95 still
fails the 300 ms tail gate. The old 12-case AEC lab is not a promotion signal.

The latest GemmaOmni causal model is now imported exactly, and a Jaeger 15%
causal blend is packaged separately. Keep `jaegeromni-41m-medium` as the 1:1
production candidate because it still wins dev-clean by two words. Use
`jaegeromni-41m-causal-balanced` for the next full-duplex experiment: it is
24/24 at 574.2 ms and improves four of five corpus gates, including AMI from
32.99% to 28.23%. Before any promotion, rerun the overlap/speaker/turn-end
heads on held-out real mixtures after the trunk blend and connect their
calibrated logits to policy. Structural causality and self-speech wiring pass;
head calibration does not yet.

2026-08-28 latency CPU preparation: the published 319 ms JaegerOmni 50M and
560 ms JaegerOmni 41M values came from different benchmark harnesses and must
not be treated as an architectural A/B. `research/run_latency_ab.sh` is ready
to run the 41M baseline, an opt-in mathematically equivalent packed-QKV 41M
runtime, and the 50M node with identical warmup, prompts, cases, repetitions,
and per-stage timing. It also requires identical generated replies before the
packed-QKV path can be considered. The repository GPU guard now recognizes the
parallel GemmaOmni `eval/benchmark_suite.py`, so this A/B cannot contaminate
that agent's active training.

The final GPU A/B is now complete. On 24 warmed, repeated cases the frozen 41M
scored 24/24 at 601.9 ms median, packed-QKV scored the same answers at 610.7 ms,
and 50M scored 22/24 at 619.7 ms. Keep the ordinary 41M path; do not enable
packed QKV or cite 319 ms for 50M. Reports are under
`research/results/final-gpu-20260828-final/latency/`.

The MiniCPM audit identified a larger input-latency gap: MiniCPM incrementally
encodes and prefills one-second audio chunks while the user is speaking. The
stock Jaeger path currently buffers a short utterance until turn-end and uses
12-second/one-second-overlap windows only for long continuous input. Stock
Whisper and the 41M global-attention adapter are bidirectional, so a cached
causal replacement is not a safe runtime toggle; it needs a stability/quality
gate or streaming-specific training before previously emitted audio vectors
can be committed to Gemma. Do not claim this feature until that gate exists.

`research/streaming_stability.py` now provides the missing first gate. It
compares every growing one-second prefix with a full-utterance pass at both
the stock-Whisper and final adapter boundaries, sweeping 0--4 seconds of held
right context. Its cosine/magnitude results are explicitly diagnostic, not a
WER or promotion score. The final guarded GPU queue is
`research/run_final_gpu_validation.sh`: identical latency A/B first, streaming
stability second, no training and no automatic checkpoint promotion.

The legacy PyTorch trainer's extended-run resume path is also repaired. It now
seeks a rebuilt OneCycleLR directly to the exact completed-update boundary,
including Adam momentum, rather than calling `scheduler.step()` before an
optimizer update. Its CPU self-check proves equality with an uninterrupted
new-horizon schedule and rejects an off-by-one reconstruction.

The measured stability sweep rejects cached one-second prefill on the frozen
checkpoint. With four seconds held back, the 41M adapter still had 0.977 p05
cosine and 0.206 p95 relative-L2 error against full-utterance vectors. The next
streaming experiment therefore requires prefix-consistency/causal training and
then free-running WER/answer gates; it is not a runtime flag.

## 1. Run physical production testing

Start with the frozen candidate, never an intermediate:

```text
front end: jaegeromni-41m-medium
adapter SHA: a7f96bfb889866f75df7b838f84aae368a56ab10f6469e766511d700e32fe13a
```

The physical five-minute run must retain all of the following:

- source-qualified conversation: no worse than 22/667 normalized word edits;
- cases12: 12/12;
- no lost microphone audio, no abandoned reply, and at least one deliberate
  barge-in under each policy;
- five-minute context roll plus END-to-audio median/p95;
- physical AEC attenuation and false-barge count.

The 2026-08-28 continuous-floor Elgato soak is useful but does not close this
gate: 298.0 seconds fed, zero drops, 178 ms peak lag, one successful 6,151-token
roll, and 11/11 audit-word survival, but zero endpointed turns because the room
never stayed below the measured floor for 700 ms. Repeat with a real 1:1
speaker who deliberately pauses and interrupts Jaeger's playback; retain the
continuous-floor capture as the no-yield stress case rather than weakening the
endpoint until fragments masquerade as turns.

A separate 90-second diagnostic at 300 ms silence proves the downstream
interactive mechanics: 5 turns/replies, zero drops, two continue-during-output
events, and 494/522 ms median/p95 END-to-first-audio. It also makes AEC the
remaining hard blocker rather than a paper concern. Physical attenuation was
only 0.29 dB and 25/25 decoded words of Jaeger's own repeated playback survived
into model-fed audio. Reference-lead and filter-length sweeps peaked at 2.05 dB,
so do not tune timing and claim success. The next AEC implementation must
calibrate/retain a loudspeaker-to-mic path when usable, freeze or slow adaptation
during detected double-talk, and pass both the 43-room synthetic gate and a
fresh physical raw-vs-fed audit before promotion.

Teacher forcing is permitted as a training loss. It is never an evaluation
number. Do not promote from a single clip, a teacher-forced score, cosine, or
the training-only lexical head.

## 2. How to break the 14% plateau

More passes over the same adapter/native-vector regression are closed. They
move errors between domains but have not crossed 14% general WER. The next
useful objective is a low-weight, human-target sequence loss through frozen
Gemma, while keeping the native audio-vector loss as an anchor:

```text
stock Whisper encoder (frozen)
  -> owned 50M connector
     -> native-vector regression anchor
     -> frozen Gemma + human transcript/answer CE
     -> training-only speaker and event heads
```

Only the connector and small auxiliary heads train. Gemma, Whisper, and their
decoders stay frozen. Human transcripts supervise training but are never a
runtime handoff. The other agent's 7.55% result validates this direction on a
narrow one-speaker gate, but its 54.30% room WER exposes the missing domain
robustness. The owned implementation is now `training/context_ce.py`: human
CE, same-Gemma text-equivalence KL, frozen-start output anchor, exact
source-mass sampling, holdout overlap audits, and a frame-diversity collapse
abort. It trains only the explicitly selected connector branch.

If it fails, the next structural experiment is a pretrained or distilled
monotonic sequence head at the connector bottleneck, not another randomly
initialized BiGRU CTC head. The current disposable head ended near 94% WER and
is diagnostic evidence that lexical structure is not yet usable there.

## 3. Data policy

Canonical caches live outside the repository under
`~/models/jaeger-training-data`. Current additions are:

- 12,000 previously unused LibriSpeech train-other clips;
- 4,042 new AMI turns from four hours with zero first-hour turn overlap;
- 3,249 AMI training and 750 speaker-disjoint AMI validation features;
- 14,565 deterministic RIR/MUSAN variants (26.255 h, 1,104 speakers) from
  hard LibriSpeech and AMI, retaining clean IDs and augmentation provenance;
- 29,671 previously unused train-clean-360 clips at the current <=12-second
  training limit (63.260 h): 27,911/868-speaker training and
  1,760/53-speaker validation with zero speaker overlap;
- an official-MD5-verified LibriSpeech test-clean lockbox: 2,603 usable clips
  from 40 speakers, manifest SHA
  `d8934ec73c9e1b3280f31a15946d5e7d5aa1df3319e3b912d763d251a5a1e738`;
- a measured 1,006.828-hour independent licensed source pool: LibriSpeech
  961.052 h, genuinely novel VoxPopuli 33.283 h, FLEURS 7.491 h, and AMI
  5.003 h, plus MUSAN/OpenSLR RIR augmentation assets. Raw archive hours are
  not training examples until matching stock-Whisper features are cached, and
  the current <=12-second curriculum intentionally uses a smaller subset.

Use `--balance-sources` for mixed-domain training. Keep every validation split
untouched. Before another run, add genuinely new speakers/acoustics such as
MUSAN/RIR augmentation or another licensed conversational corpus; do not cycle
the same AMI or LibriSpeech rows again merely because GPU time is available.

Secondary labels remain important but must not be conflated with actions:

- speaker identity/continuity;
- speech during playback (double-talk), distinct from the policy decision to
  stop or continue speaking;
- turn start/end and overlap;
- environmental/acoustic events.

Overlap attribution is a separate future node, not a capability of the current
turn-level A/B/C clustering.  The production order is: cancel Jaeger's known
playback reference, detect human-human overlap, separate the mixture into two
waveforms, assign each separated stream to a persistent CampPlus identity, and
feed both streams through the audio-native front end with time-aligned speaker
metadata.  Validate with overlap-aware cpWER/DER and leakage/SI-SDR on held-out
AMI/ICSI meetings.  Until those gates pass, report mixed speech as overlap and
do not claim word-level speaker attribution.

## 4. Prepared medium/large campaign

No stage starts automatically. The CPU-created large port is
`artifacts/stock-whisper-gemma-large-v3-turbo50m-port.npz` (SHA
`5e3d766034c66811c8e30d3a73e1e3e0329b88f5eeba492c9ab0a9bc22537c3a`).
It has a fresh 1280x2 input stack and byte-identical inherited downstream
blocks (52,768,768 deployed parameters versus medium's 50,146,816).
`training/test_next_window.py` passes forward, freeze-set, lineage,
and real CPU-gradient checks for both front ends.

When the GPU is assigned, run the named stages in
`training/next_5pct_window.sh`: room-feature precompute, medium CE/KL, large
native input bootstrap, large input CE/KL, then large context CE/KL. Each
stage is separately checkpointed and guarded against concurrent GPU work.
Use `training/evaluate_candidate.sh` for identical-ID free-running gates; do
not promote from loss, cosine, teacher forcing, or the large decoder control.

## 5. Live production gates

The native Gemma tower remains the desktop default until the stock connector
passes a positive-room five-minute test. Require:

- no lost audio across a 300-second capture;
- meaningful Silero-admitted speech at the selected Elgato input;
- stable A/B/C speaker telemetry without claiming word-level diarization;
- AEC/double-talk measurements and barge-in behavior;
- context roll without negative soak deltas;
- END-to-first-audio median and tail latency.

The tuned MiniCPM-derived encoder remains observer-only while its fixed corpus
WER is 13.61%, better than the stock observer. It is not an assistant default
and the Qwen language model is never loaded.

## Commit gate

```bash
./.venv/bin/python research/check_all.py
./.venv/bin/python -m jaeger_agent_omni.omni --selfcheck
./.venv/bin/python demo.py --check
git diff --check
```

Do not stage generated WAVs, local corpora, virtual environments, model
weights, or intermediate checkpoints.
