# Jaeger Agent Omni — full-duplex demo

This is the repository's canonical live demo, adapted from the donated
MiniCPM-o full-duplex desktop UI. Jaeger is the default engine; MiniCPM-o 4.5
remains available as an optional local reference.

## What the Jaeger mode runs

```text
microphone → selected audio ear → rolling Gemma cache → response tokens
     ▲                                                   │
     └──────── SpeexDSP AEC ← Kokoro clause audio ◄──────┘
```

- The input path has no ASR decoder and produces no user transcript.
- Hands-free endpointing closes turns after room silence.
- Kokoro starts on complete clauses instead of waiting for the full response.
- SpeexDSP receives the exact playback waveform, allowing the microphone to
  keep feeding the model while Jaeger speaks.
- Residual near-end speech produces a timestamped interaction event. The
  default `stop` policy interrupts without discarding the conversation;
  `continue` records the same event while allowing Jaeger to finish.
- CampPlus adds inline `[speaker A/B]` metadata between audio turns.
- The cache rolls at 6000 tokens and carries recent assistant replies forward,
  so five-minute and longer sessions stay bounded.

The camera control is enabled only for the optional MiniCPM reference. Jaeger's
vision path is not connected in this demo, and the UI does not pretend it is.

The Jaeger audio selector exposes exactly five active model identities:

- **Gemma Native**;
- **GemmaOmni 41M Medium** (the frozen checkpoint currently imported here);
- **JaegerOmni 41M Medium**;
- **JaegerOmni 50M Medium**;
- **JaegerOmni 50M Large** (clearly marked experimental).

Every selection uses the same rolling Gemma, Kokoro clause output, SpeexDSP
AEC, endpointing, barge-in, and speaker-label worker. This keeps the live A/B
about the ear rather than silently changing the rest of the product.
The `omni`, spoken-demo, live-microphone, and GUI entry points all resolve
these identities through one core catalog.

## Run

```bash
./.venv/bin/python -m pip install -e '.[demo]'
brew install speexdsp                       # once on macOS
./.venv/bin/python demo.py --check
./.venv/bin/python demo.py
```

The default model location is
`~/models/gemma-4-e4b-it-mlx-full`. Override it with
`JAEGER_OMNI_MODEL=/path/to/model`.

The window defaults the room floor to `0.20`, the measured Elgato XLR Dock
setting for a continuous conversation playing in the room. Lower it toward
`0.012` for close-mic speech in a quiet room. The floor, endpoint silence,
reply cooldown, and context-roll threshold are all visible controls.
The speech-during-output control separates detection from action: choose
`stop and listen` or `notify; keep speaking` without changing AEC.

The microphone is selectable in the window and its PortAudio callback health
is visible throughout the run. Set `JAEGER_INPUT_DEVICE` to an input-device ID
or a case-insensitive name fragment (for example `Elgato`) to choose it at
launch.

`Meeting third-person` applies a selective-participant system prompt and a
20-second response cooldown. It asks Jaeger to fill gaps, answer direct
questions, and track decisions/action items without turning the room audio
into a displayed or model-input transcript.

## Exact-worker self-test

The headless test replays a WAV at wall-clock pace through the same worker the
window uses, including endpointing, clause streaming, speech output, and
metrics. It initializes AEC, but direct file input bypasses the physical echo
path; use the live Elgato/loudspeaker phase in the production protocol to
validate cancellation:

```bash
./.venv/bin/python demo.py \
  --selftest-jaeger research/cases12/q00.wav --seconds 20 --threshold 0.012
```

Every Jaeger session writes a machine-readable report beneath
`Full DuplexDemo App/runtime/reports/`; headless runs accept `--report PATH`.
The report includes model hash, complete configuration, interaction events
and actions, latency tails, AEC telemetry, speakers, replies, and errors.

The five-minute room stress recipe is:

```bash
./.venv/bin/python demo.py --selftest-jaeger meeting.wav --seconds 300 \
  --front-end gemmaomni-41m-medium \
  --threshold 0.20 --silence-ms 700 --cooldown 20 --max-context 6000
```

Use the complete sequential five-model matrix in
[`../PRODUCTION_TESTING.md`](../PRODUCTION_TESTING.md). Old `tower`, `stock`,
and `cleanroom` values remain accepted as compatibility aliases.

MiniCPM-o is optional and does not affect `--check`. The demo discovers the
existing VoiceLLM Playground build as a read-only reference, or accepts
`OMNI_SHIM`, `OMNI_REF_AUDIO`, `OMNI_QUESTION_WAV`, and `OMNI_LLM_GGUF`
overrides. If those assets are present, its historical reference test remains
available:

```bash
./.venv/bin/python demo.py --selftest-minicpm 25
```

Current read-only Playground reference check: 25 paced frames produced 22
LISTEN / 3 SPEAK decisions, answered “The capital of France is Paris,” and
played speech at 0.129 peak level. The command now fails unless frames,
SPEAK, and audible output are all observed.

Local captures, engine logs, generated WAVs, and native debug output live under
`Full DuplexDemo App/runtime/` and are intentionally ignored by git.
