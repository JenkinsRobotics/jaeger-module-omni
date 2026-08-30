# Deep benchmark

`jaeger_agent_omni.benchmark` is the release gate for the combined neural and
agentic system. It composes existing factual evaluators, stores every command
log and artifact, and writes an incremental JSON/Markdown report so a failed or
interrupted long run still has evidence.

```bash
# Fast, no model load: contracts, tests, transcript ban, tools/skills, output
python -m jaeger_agent_omni.benchmark --profile cpu

# CPU plus real frozen-model audio, agent tools, duplex, Kokoro, VAD, companion
python -m jaeger_agent_omni.benchmark --profile model

# Everything above plus the 81+81 native-audio agent corpus, 43-room AEC,
# and REQUIRED physical five-minute evidence
python -m jaeger_agent_omni.benchmark --profile release \
  --physical-report research/results/live/<physical-report>.json
```

Reports land under `research/results/deep-benchmark/<timestamp>/` unless
`--out-dir` is supplied. `report.json` is the machine contract; `report.md` is
the human scorecard; `logs/` and `artifacts/` retain the evidence.

## First complete baseline (2026-08-29)

The retained release run is **NOT READY: 15 pass / 4 fail / 1 required
skip**. It measured 12/12 plain answers at 565 ms, 12/12 simple agentic
answers at 2320 ms, 17.8 ms interruption with zero drops, valid Kokoro PCM,
and neural VAD at 12/12 speech open/end with 0/5 nonspeech false opens. It
also exposed the deficits the smaller gates hid: only 28/162 full agent cases
and 20/128 required tool routes, companion admission at 1/12 ambient false
responses and 4/12 addressed recall, and an AEC barge detector with 0/43 false
barges and 43/43 recall but a 602 ms p95 tail against the 300 ms gate. The
agent corpus retained zero forbidden-tool failures and zero visible markup.
See [`report.md`](research/results/deep-benchmark/20260829-release-final/report.md).

## What is covered

| Layer | Gates |
|---|---|
| Repository | 31 module contracts, all package tests, node/checkpoint paths |
| Premise | AST-audited no-transcript input path |
| Agent framework | 19 self-checks, exact pinned source/corpus/tool parity |
| Neural ear | free-running cases12 answer accuracy and latency |
| Agent loop | native-audio answers, structured dispatch, markup rejection |
| Deep agent behavior | both 81-case corpora: routing, chaining, files, memory, skills, recovery, safety |
| Duplex | streamed input survival, endpoint, drops, barge-in cancellation latency |
| Speech output | actual resident Kokoro 24 kHz PCM synthesis |
| Perception | neural VAD open/end corpus gate |
| Companion | unaddressed false responses and named-address recall separately |
| Echo/barge policy | deterministic 43-room AEC stress gate |
| Evidence | checkpoint-pinned MiniCPM scorecard |
| Physical release | five-minute mic run with turns, zero drops, context roll, latency, and no errors |

The full agent corpus uses Kokoro to build deterministic evaluation WAVs in
`~/models/jaeger-benchmark-audio`. Its text is fixture ground truth only. The
runtime receives PCM through Whisper and the selected Jaeger adapter; no prompt
or transcript is injected into Gemma. The cache is disposable and can be
deleted without affecting checkpoints or training data.

The corpus adapts two legacy speech-tool expectations because normal Omni
speech is an automatic output branch, not a `text_to_speech` action. Read-aloud
still requires `read_file`; calculation-and-speak requires `calculate`; neither
is rewarded for calling the blocked legacy TTS tool.

Sandboxed `WRITE_LOCAL` operations are pre-approved for this explicitly
requested hermetic benchmark, matching the legacy harness. External effects,
hardware, privileged actions, and developer bypass remain denied. Web reads
are bounded by the case deadline so an unavailable service is visible as a
case error without blocking the corpus.

## Verdict semantics

- `PASS`: the measured gate met its declared threshold.
- `FAIL`: it ran and missed the threshold, errored, or timed out.
- `SKIP`: evidence was unavailable. A required skip makes the run
  **INCOMPLETE**, never green.
- Only the `release` profile can say `READY`. A green `cpu` or `model` profile
  says `PROFILE PASS`, not production readiness.

Model gates are free-running and sequential. The harness refuses to start a
model/release profile while a known training or benchmark process is active.
No teacher-forced result is accepted as a release measurement.

The physical report is intentionally a single hard integration gate. Run for
five minutes with neural VAD, full duplex, AEC, companion admission, and
`--agentic`; deliberately obtain at least one RESPOND and one LISTEN decision,
make three successful tool requests, interrupt one spoken reply, and keep the
conversation active long enough to roll context. The report must contain at
least 295 seconds of fed audio (allowing one callback of startup/teardown), no
drops or microphone overflow, actual AEC playback samples, a spoken-reply
latency, and no runtime errors. A silent observer soak cannot satisfy it.

## Direct deep-agent runs

The 162-case agent gate can be prepared or narrowed independently:

```bash
python research/benchmark_agentic_corpus.py --suite both --prepare-only \
  --out /tmp/not-used.json
python research/benchmark_agentic_corpus.py --suite a --tags routing \
  --limit 10 --out /tmp/agent-routing.json
python research/benchmark_agentic_corpus.py --suite both \
  --out research/results/agentic-full.json
```

Filtering preserves all rows of any selected multi-turn session. Every run
records raw per-case answers, tool names, loop iterations, errors, routing,
answer, safety, skill, markup, and latency judgments. Each row has a bounded
deadline (45 seconds directly, 20 seconds in the release profile), and an
incremental `*.rows.jsonl` artifact survives a tool or network stall.
