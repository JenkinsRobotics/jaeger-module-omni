# Jaeger Agent Omni — half/full duplex pipeline lab

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
- Companion mode asks Gemma directly from native audio whether to `RESPOND` or
  `LISTEN`; overheard speech, environmental sounds, self-speech, and incomplete
  thoughts are explicitly silence cases. Malformed control output fails closed.
- Half/full duplex are explicit. Half duplex withholds microphone blocks while
  Jaeger is active; full duplex keeps AEC and barge-in enabled.
- The mode selector is live. Switching does not reload Gemma, replace the
  audio ear, reset conversation memory, or create a second pipeline.
- A runtime identity guard and the no-model `--check` enforce that the session,
  loop, AEC, barge detector, queues, player, companion state, and interaction
  policy remain the exact same objects across a mode change.
- The cache rolls at 6000 tokens and carries recent assistant replies forward,
  so five-minute and longer sessions stay bounded.

The camera control is enabled only for the optional MiniCPM reference. Jaeger's
vision path is not connected in this demo, and the UI does not pretend it is.

The Jaeger audio selector exposes exactly seven active model identities:

- **Gemma Native**;
- **GemmaOmni 41M Medium** (the frozen checkpoint currently imported here);
- **GemmaOmni 41M Final 4000** (latest causal meeting specialist);
- **JaegerOmni 41M Medium**;
- **JaegerOmni 41M Causal Balanced** (full-duplex development candidate);
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
setting for barge-in during a continuous conversation playing in the room.
Speech turn opening now defaults to the packaged Silero neural VAD; the GUI
can select the legacy energy gate explicitly. The VAD choice, energy/barge
floor, endpoint silence, reply cooldown, and context-roll threshold are all
visible controls.
The speech-during-output control separates detection from action: choose
`stop and listen` or `notify; keep speaking` without changing AEC.
The **Companion** preset enables audio-native admission and full duplex. The
direct-assistant mode remains available for ordinary 1:1 testing. See
[`../COMPANION.md`](../COMPANION.md) for the behavior contract and release gate.

## Compare half and full duplex in one session

Start once, then use the duplex selector while the session is live:

1. Select **Half duplex · turn based**. The input indicator changes to
   `INPUT CLOSED` from the moment Jaeger accepts the turn until cognition,
   every Kokoro clause, and the 120 ms loudspeaker tail have completed.
   Pre-suppression capture is drained before the next user turn opens. This
   follows the updated VoiceLLM sequential reference rather than merely muting
   during `sd.play()`.
2. Select **Full duplex · interruptible**. The same model session immediately
   admits microphone audio during generation and playback. The exact playback
   waveform remains the SpeexDSP far-end reference; residual near-end speech
   triggers the selected barge policy.

The right-hand mode indicator is the authoritative admission state. Reports
record the initial mode, final mode, every live transition, and the amount of
capture suppressed by half duplex, so a comparison is auditable afterward.

## Agentic execution column

The third column is driven by JaegerAgent's real callback surface. Check
**Agentic tools** before Start to run the bundled planning/tool/skill/memory
loop over the same native audio session. It displays:

- model/agent pipeline stages and liveness;
- plans derived from structured tool calls;
- sanitized tool arguments, completion state, result previews, and timing;
- loop iterations, interruptions, failures, and the currently active tools.

It does not fabricate activity when agentic mode is off, speak tool markup, or
create a transcript handoff. Executable tokens are buffered until JaegerAgent
has parsed and dispatched them; only the final user-facing answer reaches
Kokoro. The native runtime does not request Gemma's optional thinking channel.
The panel reports only observable agent-loop recovery/activity events and
structured decisions, and common credential fields in tool data are redacted.
The same sanitized events are retained in the session report.

The microphone is selectable in the window and its PortAudio callback health
is visible throughout the run. Set `JAEGER_INPUT_DEVICE` to an input-device ID
or a case-insensitive name fragment (for example `Elgato`) to choose it at
launch.

`Meeting third-person` applies companion admission plus a selective-participant
instruction. It asks Jaeger to fill gaps, answer direct questions, and track
decisions/action items without turning the room audio into a displayed or
model-input transcript. A fixed cooldown no longer substitutes for deciding
whether the room actually addressed Jaeger.

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

Use the complete sequential seven-model matrix in
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
