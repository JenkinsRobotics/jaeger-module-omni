# Full-duplex production test

This is the locked live comparison for the seven active Jaeger audio-model
identities. Run one command at a time against the same WAV, machine state, and
settings. Never compare a teacher-forced result or change the downstream
Gemma/Kokoro/AEC worker between ears.

The first production test target is the frozen JaegerOmni 41M Medium node,
adapter SHA `a7f96bfb8898`. It enters physical testing with 3.30% normalized
WER on the source-qualified 69-clip one-to-one conversation gate, 12/12 spoken
answers at 560 ms median, and a 21.0 ms synthetic barge-in. Run it explicitly:

```bash
python -m jaeger_agent_omni.live --front-end jaegeromni-41m-medium \
  --seconds 300 --threshold 0.20 --silence-ms 700 --cooldown 20 \
  --max-context 6000 --record /tmp/jaeger-production-1to1
```

Gemma Native remains the rollback/default until this physical phase passes.

## CPU/static readiness

These commands do not load an audio model:

```bash
./.venv/bin/python demo.py --check
./.venv/bin/python -m jaeger_agent_omni.node_registry --all --verify
./.venv/bin/python -m jaeger_agent_omni.gate_registry --verify
./.venv/bin/python research/check_all.py
./.venv/bin/python -m jaeger_agent_omni.omni --selfcheck
./.venv/bin/python -m jaeger_agent_omni.model_catalog
./.venv/bin/python -m jaeger_agent_omni.runtime.interaction
./.venv/bin/python -m jaeger_agent_omni.runtime.reporting
./.venv/bin/python -m jaeger_agent_omni.runtime.aec
./.venv/bin/python research/parity_scorecard.py --check
./.venv/bin/python research/overlap_lab.py --demo
git diff --check
```

The model catalog is shared by `omni.py`, `jaeger_agent_omni.demo`,
`jaeger_agent_omni.live`, and the full-duplex GUI. All accept the same seven
canonical identities; old `tower`, `stock`, and `cleanroom` values are aliases
only.

## Phase 1: identical five-minute file matrix

Choose one positive two-speaker room recording and keep its bytes unchanged.
The commands below are intentionally sequential because only one accelerator
job may run at a time.

```bash
COMMON="--seconds 300 --threshold 0.20 --silence-ms 700 --cooldown 20 --max-context 6000"

./.venv/bin/python demo.py --selftest-jaeger ROOM.wav --front-end gemma-native $COMMON
./.venv/bin/python demo.py --selftest-jaeger ROOM.wav --front-end gemmaomni-41m-medium $COMMON
./.venv/bin/python demo.py --selftest-jaeger ROOM.wav --front-end gemmaomni-41m-final4000 $COMMON
./.venv/bin/python demo.py --selftest-jaeger ROOM.wav --front-end jaegeromni-41m-medium $COMMON
./.venv/bin/python demo.py --selftest-jaeger ROOM.wav --front-end jaegeromni-41m-causal-balanced $COMMON
./.venv/bin/python demo.py --selftest-jaeger ROOM.wav --front-end jaegeromni-50m-medium $COMMON
./.venv/bin/python demo.py --selftest-jaeger ROOM.wav --front-end jaegeromni-50m-large $COMMON
```

The Large run is diagnostic until its transfer failure is fixed. Its presence
in this matrix does not authorize promotion.

This phase feeds the WAV directly into the worker. It measures the selected
ear, endpointing, rolling context, response behavior, queue health, and
END-to-audio latency under controlled input. It does **not** validate acoustic
echo cancellation: agent playback never travels through the test input's
physical loudspeaker-to-microphone path.

Every self-test writes a versioned JSON report under
`Full DuplexDemo App/runtime/reports/`. Use `--report PATH` to choose an exact
destination. Reports pin the canonical identity and checkpoint hash and
include configuration, fed/dropped audio, replies, errors, speakers,
interaction events, AEC state, context rolls, and latency median/p95/max.

## Phase 2: identical five-minute room matrix

For each model, select the same identity in `python demo.py`, choose the
Elgato XLR Dock, and play the same positive two-speaker recording through the
same room loudspeaker at unchanged volume and geometry. Let Jaeger answer
through its normal output device. Run for five minutes without changing the
floor, endpoint, cooldown, or context controls.

This is the AEC and real double-talk gate. Capture the final JSON report for
each identity. Include at least one deliberate near-end interruption while
Jaeger is speaking. Run the leading candidate once under each GUI policy:
`stop and listen` must cancel promptly, while `notify; keep speaking` must
record the same event without cancelling output.

## CPU AEC and overlap labs

The deterministic AEC corpus needs no model or GPU:

```bash
./.venv/bin/python research/aec_stress.py --cases 43
```

This broader real-speech/noisy/nonlinear test is the only CPU AEC promotion
gate; the old 12-case synthetic lab is diagnostic history and is not run for
selection. It currently **fails only its detection-tail gate**: 8.1 dB median
attenuation, 99% near-speech survival, 0/43 self-echo false barges, and 43/43
valid double-talk detections, but 602 ms p95 exceeds the 300 ms limit (188 ms
median; the attenuation-based double-talk override cut the tail from
843 ms). The previous rule produced 27/43 false barges and only 37% recall.

Build factual overlap mixtures from two or three known single-speaker files:

```bash
./.venv/bin/python research/overlap_lab.py \
  --speaker SPEAKER_A.flac --speaker SPEAKER_B.flac \
  --speaker SPEAKER_C.flac --out-dir /tmp/jaeger-overlap
```

The first three-speaker LibriSpeech run measured 86% turn continuity but
failed identity separation because two true speakers shared one dominant
label. That is a production blocker, not a passing diarization claim. The
tool always marks word-level overlap attribution unsupported until a separator
and overlap-aware evaluation exist.

## Optional AEC reference A/B (after the phase-2 matrix)

The experimental `--aec-reference-mode contiguous` feeds Speex one unbroken
reply-level far-end stream (clauses plus real silence, one playback
generation per reply) and ECHO-LOCKS it: a periodic mic-vs-stream
correlation servo absorbs the measured constant 614 ms output-path delay and
the ~1,250 ppm capture-clock drift that defeat any fixed offset. On the
equal-exposure fixed-waveform probe (`research/aec_path_probe.py`) it
reaches 10.5 dB echo-coupling reduction versus wallclock's run-dependent
0--7.8 dB. Wallclock remains the locked baseline until an interactive
physical run plus raw-vs-fed audit confirms; run the same 90-second
diagnostic under both modes and compare coupling, attenuation, and the
drift telemetry:

```bash
AB="--front-end jaegeromni-41m-medium --seconds 90 --threshold 0.055 \
  --silence-ms 300 --cooldown 20 --max-context 6000 --barge-policy continue"

./.venv/bin/python -m jaeger_agent_omni.live $AB \
  --record /tmp/jaeger-aec-ab-wallclock \
  --report /tmp/jaeger-aec-ab-wallclock/report.json
./.venv/bin/python -m jaeger_agent_omni.live $AB \
  --aec-reference-mode contiguous \
  --record /tmp/jaeger-aec-ab-contiguous \
  --report /tmp/jaeger-aec-ab-contiguous/report.json
./.venv/bin/python research/session_audit.py --dir /tmp/jaeger-aec-ab-wallclock
./.venv/bin/python research/session_audit.py --dir /tmp/jaeger-aec-ab-contiguous
```

A large `estimated_drift_ppm` with rising `alignment_corrections` implicates
separate ADC/DAC clocks — a contiguous stream alone does not fix that; it
measures it.

## Exclusive latency attribution

The historical 41M and 50M medians were produced by different harnesses. Run
the prepared three-way A/B only with exclusive accelerator access:

```bash
research/run_latency_ab.sh
```

It measures Whisper encoding, adapter/audio prefill, prompt closure,
first-token latency, remaining decode, and complete-reply latency separately.
The middle run packs the 41M connector's three bias-free Q/K/V projections
into one inference matrix. It is opt-in and cannot be promoted unless its
checkpoint, answers, cases, and repetitions match the baseline and the speed
gain repeats. The test does not solve MiniCPM's one-second cached audio prefill;
that remains a separate streaming architecture gate.

## Record for every model

- canonical model identity and adapter hash;
- input seconds fed and dropped;
- mic queue-overflow blocks/seconds, SEPARATELY from loop-lag dropped
  seconds (the report's `mic_queue_overflow_*` fields; overflow audio never
  reached the loop at all);
- process peak RSS plus MLX active/peak memory from the report's `memory`
  block (RSS alone cannot see Metal allocations);
- turns, completed replies, cancelled generations, and visible barge-ins;
- phase-2 raw and residual playback correlation plus AEC attenuation (mark
  these `N/A: direct file input` for phase 1);
- speaker labels observed without claiming word-level attribution;
- interaction events and their independent policy actions;
- context rolls and final cache size;
- END-to-first-audio median, p95, and maximum;
- answer text and any worker exception.

The input path must not compute or display a transcript. A diagnostic
transcript generated by Gemma may be evaluated separately, but it must never
be fed back as the input representation.

## Minimum promotion gate

A production candidate must:

1. feed all 300 seconds with zero dropped audio in both phases;
2. survive at least one context roll when the recording is long enough;
3. pass the physical room phase while listening during playback with
   measurable echo reduction;
4. detect deliberate near-end speech during playback and interrupt only when
   the configured policy says to;
5. avoid publishing an abandoned partial reply after an interruption;
6. retain its declared corpus/cases12 quality gates;
7. report stable latency tails, not only the best median;
8. pass speaker continuity and distinct-identity gates on unseen talkers;
9. pass both phases again before replacing the rollback default.

Gemma Native remains a rollback because it already has a complete five-minute
worker run. It is not presumed more accurate than the adapter models.

After adding any checkpoint-pinned corpus or duplex report, regenerate
[`PARITY_SCORECARD.md`](PARITY_SCORECARD.md):

```bash
./.venv/bin/python research/parity_scorecard.py
```
