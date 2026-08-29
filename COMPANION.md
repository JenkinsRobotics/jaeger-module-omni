# Always-on companion contract

Jaeger has two explicit interaction modes. `assistant` answers every completed
audio turn and is the production 1:1 baseline. `companion` is an always-on room
participant: it answers direct address and active follow-ups, and remains
silent for overheard conversation, media, environmental sounds, its own voice,
and incomplete thoughts.

```text
microphone
   │
   ├─ Silero VAD ─────── neural speech proposal; never means "addressed"
   │
   ├─ Whisper encoder → Jaeger adapter → Gemma
   │                                  ├─ RESPOND + response
   │                                  └─ LISTEN + reason
   │
   ├─ causal perception heads ─ shadow telemetry until calibrated
   │      speaker-change · overlap · turn-end · speaker identity
   │
   └─ AEC + double-talk event → independent stop/continue policy
```

No ASR decoder or user transcript participates in admission. Gemma receives the
native audio embeddings and emits a small control envelope. The runtime never
sends the envelope to Kokoro, and malformed output fails closed to `LISTEN`.
Follow-up windows are state supplied to the neural decision, not automatic
permission to speak. They are judged at speech-start time so a turn begun
inside the window cannot expire while the model is processing it.

## Half and full duplex

- `half` keeps capture alive but intentionally withholds microphone blocks
  while Jaeger generates or speaks. It has reliable self-rejection and cannot
  be interrupted.
- `full` continuously feeds echo-cancelled audio. Speech during playback is a
  notification; `--barge-policy stop|continue` independently selects the
  action. This is the companion target.

The packaged Silero ONNX gate now proposes speech start at 30 ms cadence. It
requires four positive frames in a 180 ms window and uses 0.30/0.125 open/hold
hysteresis. The old RMS gate remains only as explicit `--vad energy` fallback.
Silero is deliberately not described as semantic endpointing: television is
speech even when it is not meant for Jaeger. Causal
speaker/overlap/turn-end heads are now exposed in run reports under
`neural_perception_shadow`, but do not control production until checkpoint-
specific thresholds pass held-out conversational mixtures. Environmental
rejection and addressed-to-agent decisions are currently made by Gemma itself,
which is neural and audio-native; a smaller distilled admission head is an
optimization after the behavior gate is populated.

## Run

```bash
python -m jaeger_agent_omni.live \
  --front-end jaegeromni-41m-medium \
  --interaction-mode companion --duplex-mode full \
  --speakers 3 --record /tmp/jaeger-companion
```

The desktop demo exposes the same controls and a **Companion** preset. The
meeting preset also uses companion admission rather than relying on a prose
prompt plus a fixed cooldown.

## Release gate

Build a labeled JSONL manifest and run `research/companion_gate.py`. Rows are
audio paths with an expected `respond` or `listen`; rows sharing `session_id`
exercise real follow-ups with one persistent cache. Keep separate categories
for direct address, follow-up, nearby human-human conversation, television or
recorded speech, environmental noise, incomplete thought, and self-speech.

```bash
python research/make_companion_addressed.py
python research/companion_gate.py \
  --manifest /tmp/jaeger-companion-addressed.jsonl \
  --front-end jaegeromni-41m-medium \
  --out research/results/companion-gate.json
```

Promotion requires corpus-level, free-running results:

- direct-address recall at least 95%;
- natural-follow-up recall at least 90%;
- zero false responses in the initial two-minute overheard-conversation and
  environmental-noise room tests;
- no spoken control envelope or malformed-output leakage;
- full-duplex self-speech rejection plus deliberate barge-in under both action
  policies;
- p95 endpoint and END-to-audio latency reported separately;
- five-minute context roll without changing the admission behavior.

The initial gate measurement rejects promotion. On 12 unaddressed ambient
questions the frozen 41M decision listened to 11/12 (one false response). On
12 separately synthesized, explicitly named "Hey Jaeger" questions it
responded to only 4/12. A separate free-running dual-output probe answered
11/12 of those named questions and rendered the name phonetically as
"Yeeger/Yeger", so the principal failure is admission, not missing question
content. The next admission implementation needs trained positive name
variants and negative conversational speech; changing the parser to accept a
failure would only hide it.

Neural VAD is separately verified on `cases12`: 12/12 opened and 12/12
endpointed, with maximum speech probabilities 0.9970--0.9996. That does not
qualify the companion behavior gate, environmental rejection, or physical
duplex operation. A subsequent five-minute Elgato/loudspeaker observer run
found six complete turns, fed 299.5 seconds with zero drops, rolled once, and
preserved 430/430 raw-room words in model-fed audio. It validates the physical
VAD/integrity seam. It does not validate companion admission or meeting
understanding: all six running notes were the uninformative phrase "less than
10 seconds".
