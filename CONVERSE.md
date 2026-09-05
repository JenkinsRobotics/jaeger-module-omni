# Gemma Omni V2 converse — the imported production pipeline

Imported 2026-09-03 from `VoiceLLM-Playground/Omni Model/gemma omni v2` at
its 27/27 campaign state (playground commit `9f1103b`). This is the full
pipeline — real trained weights, real node graph, real duplex ladder — not a
facade: text, vision, and audio in; text and audio out.

## Tier routing verified — 2026-09-04

Half/quasi now select the unchanged `jaegeromni-41m-medium` pairing through
its `build_converse()` entry point. Full retains `jaegeromni-41m-converse`
for previews and replies. Bare `--converse` selects quasi; explicit tiers
and `--barge stop|continue` remain available. Registry: 28 nodes / 13 profiles;
`agent-graph-v0` describes half/quasi, `agent-graph-full-v0` describes full.

Real-model campaign: **27/27** (half 12/12, quasi 6/6, barge-continue 4/4,
full 5/5), with successful process exits. The initial quasi wrapper failed
because its shell file was edited during execution; a clean-exit rerun is
retained separately. Existing barge scenarios allow retries; the clean quasi
run cut on attempt 3. This proves the campaign gate, not first-attempt reliability.

Same locked devclean64 manifest, 955 words, identical scorer:

| Ear | Transcription only | HEARD from one-generation reply |
|---|---|---|
| medium (half/quasi) | **46 edits, 4.82%** | **48 edits, 5.03%** |
| causal (full) | 48 edits, 5.03% | 52 edits, 5.45% |

HEARD tests use fresh history and no speaker context per clip. The 4.82%
transcription score must not be attributed to the conversational reply prompt.
Summary, reproducing commands, comparisons, and per-session evidence:
`research/candidate_evals/tier-matched-gpu-20260904.json`.

```sh
.venv/bin/python jaeger_agent_omni/tests/converse_sim.py --session half \
  --output research/candidate_evals/tier-half-20260904.json
# Repeat with quasi, barge-continue, full and distinct output names.
.venv/bin/python research/converse_ear_wer.py --tier half -n 64 \
  --output research/candidate_evals/tier-half-20260904-devclean64.json
.venv/bin/python research/converse_ear_wer.py --tier half --heard-reply -n 64 \
  --output research/candidate_evals/tier-half-heard-20260904-devclean64.json
# Repeat both WER protocols with --tier full for the matched causal control.
```

2026-09-05 follow-ups: speaker context now stays **off by default**. The
opt-in `OMNI_CONTEXT=1` binds identity to the current audio turn and retains
labels in history. It improves speaker-sensitive recall with ECAPA from
11/24 to 19/24 across development and held-out lexical scenarios, but raises
medium fresh-turn HEARD WER from 5.03% to 5.55%. See
`research/candidate_evals/converse-context-comparison-20260905.json`.

Image intake and shared text memory are exercised with real models: **4/4**
in `research/candidate_evals/converse-vision-verified-markers-20260905.json`.
Run with `--vision`, enter `/image /absolute/path/picture.png`, then type an
image question. The image answer can inform subsequent typed and audio turns.
The existing Gemma GGUF vision twin handles image input; audio/text use the
MLX brain with the shared text history. This does not prove simultaneous
audio+image attention or preservation of image vectors across turns.
Trimming history now preserves the shared list instead of disconnecting it.

Latest cross-pipeline review:
[`OMNI_CONSOLIDATION_REVIEW_20260905.md`](docs/OMNI_CONSOLIDATION_REVIEW_20260905.md).
The native `live.py` speech-guard improvement and physical results are separate
from this 27/27 conversation campaign; native agentic room interaction remains
unqualified.

## What runs

```
.venv/bin/python -m jaeger_agent_omni.converse --selftest       # 18 hermetic checks, no models
.venv/bin/python -m jaeger_agent_omni.converse --converse half  # turn-based, mic closed while speaking
.venv/bin/python -m jaeger_agent_omni.converse --converse quasi # open mic + Speex AEC + barge (--barge stop|continue)
.venv/bin/python -m jaeger_agent_omni.converse --converse full  # + forming-transcript preview + commit-and-trim
.venv/bin/python -m jaeger_agent_omni.converse --enroll NAME F.wav   # one-shot QbE wake enrollment
```

Semantic channel (default `--semantic nn`): stock whisper-medium encoder →
tier-selected 41M context adapter → Gemma-4 E4B, no text seam on the way
in. Reply (default `--reply nn`): the SAME Gemma answers from audio
embeddings in one `HEARD:`/`REPLY:` generation — one model, one memory.
`--reply cascade` and `--vision` load the GGUF twin lazily. Typed turns and
images ride `text-input-v0` / `semantic-vision-v0` into the same history.

## Where it lives

| upstream (playground)            | here                                             |
|----------------------------------|--------------------------------------------------|
| `run_v2.py`                      | `jaeger_agent_omni/converse.py` (docstring-only edit) |
| `core/{mlx_audio_streaming, gemmaomni_adapter, streaming_adapter, mlx_transcript_decoding}.py` | `jaeger_agent_omni/core/` (flat, byte-identical apart from one dropped sys.path insert) |
| `nodes/<13 folders>`             | `jaeger_agent_omni/nodes/` — **byte-identical**, moved under the shared manifest contract |
| `tests/converse_sim.py`          | `jaeger_agent_omni/tests/converse_sim.py`        |
| `WHITEPAPER.md`                  | `docs/GEMMA_OMNI_V2_WHITEPAPER.md`               |

The graph is selected by the `agent-graph-v0` profile in
`jaeger_agent_omni/nodes/profiles.json`; every pre-existing profile keeps its
exact selection (the new nodes arrive disabled there). Node runtimes resolve
the engine core via `../../core` by bare name, exactly as upstream — that is
why the four core modules are flat files.

Not imported: the playground's `native_runtime/` (it was vendored FROM this
repo — re-importing would be circular), `training/`, `eval/`, `research/`,
and 21 GB of `checkpoints/` (training is deferred), and the untrained
`stock-whisper-large-v3-stream-untrained` bakeoff scaffold.

## Verification in this repo (2026-09-03)

- `converse --selftest`: **17/17** hermetic checks.
- Scripted campaign (`tests/converse_sim.py`, real models, fake audio
  device): **27/27** — half 12/12, quasi 6/6, barge-continue 4/4, full 5/5;
  one-model replies ~0.9 s/turn, TTFT 0.40–0.47 s.
- Matched WER, both ears, identical locked devclean64 manifest and identical
  scorer: current default `jaegeromni-41m-medium` **4.82%** (reproduces its
  recorded number); imported `gemmaomni-v2` ear **5.65%** as wired /
  **5.76%** instructed. The current ear stays the better transcript model on
  clean audio; the imported ear is the one carrying the trained perception
  heads. Evidence in `research/candidate_evals/*-20260903-devclean64.json`;
  harness `research/converse_ear_wer.py`.

## Ear and pipeline scorecard (2026-09-03)

Every WER cell below is the SAME locked devclean64 manifest (64 clips, 955
reference words) and the SAME normalize+wer scorer; only the adapter and the
seam/decode policy vary. "Own seam" = jaeger's budget-12 free-running session
(`stock_whisper_gemma.py wer`); "converse seam" = the imported node's
production `transcribe()` (`research/converse_ear_wer.py`). AMI64 and cases12
are the recorded gates cited from `research/candidate_evals/` and node
manifests. "—" = not measured, never inferred.

| Ear (adapter) | Architecture | Seam / decode | devclean64 | AMI64 | cases12 | Latency |
|---|---|---|---|---|---|---|
| `jaegeromni-41m-medium` (native default) | context_v2, bidirectional | own | **4.82%** | 32.99% | 12/12 | 560 ms median answer |
| `jaegeromni-41m-causal-balanced` | streaming_causal_v1 | own | 5.03% | **28.23%** | 12/12 (24/24 @ 574 ms A/B) | — |
| **`jaegeromni-41m-converse` (converse default)** | streaming_causal_v1, prompt-aligned | converse, packaged node | **5.03%** | — | — | 0.61 s median transcribe |
| `jaegeromni-41m-causal-balanced` | streaming_causal_v1 | converse, naive cross-mount (no prompt) | 6.07% | — | — | 0.66 s median transcribe |
| `gemmaomni-v2-stream-bargein-93h` (retired to disabled) | streaming_causal_v1 | converse, as originally wired | 5.65% | — | — | 0.60 s median transcribe |
| `gemmaomni-v2-stream-bargein-93h` | streaming_causal_v1 | converse, instructed | 5.76% | — | — | 0.60 s median transcribe |
| `gemmaomni-41m-medium-final4000` | (imported 2026-08-28) | own | 6.81% | 22.88% | — | — |

Pipelines (behavioral, measured on their own gates — not WER-comparable
across rows):

| Pipeline | Gate | Result | Turn latency |
|---|---|---|---|
| Converse ladder (93h ear, one-model HEARD/REPLY) | converse_sim campaign | **27/27** | ~0.9 s, TTFT 0.40–0.47 s |
| Native-tower plain voice (41m-medium ear) | frozen-ear cases12 | 12/12 | 575 ms |
| Native-tower agentic (94 Omni-safe tools) | agentic cases12 | 12/12 | 2421 ms |
| MiniCPM-o comparison | [`PARITY_SCORECARD.md`](PARITY_SCORECARD.md) | see scorecard | — |

What the 2026-09-03 bakeoff and consolidation established:

1. The user's bet holds: `jaegeromni-41m-medium` remains the accuracy
   leader on clean speech at 4.82% — no imported or cross-mounted ear beats
   it. `causal-balanced` remains the hard-audio (AMI) leader.
2. The naive cross-mount penalty (5.03% → 6.07%) was the PROMPT, not the
   decode: wrapping the audio turn in the adapter's training-time
   `TRANSCRIPT_SYSTEM` recovers the own-seam number exactly. Per-row
   diagnosis showed 10 diffuse near-tie edits, no truncation or loops.
3. **Consolidation step 2 (done): `jaegeromni-41m-converse` — the
   repository's causal-balanced 41M, prompt-aligned, packaged as an
   immutable node — now IS the converse semantic channel.** Gates: node
   reproduces 5.03% through its own build path, and the full campaign
   holds 27/27 on the new ear. The imported
   `gemmaomni-v2-stream-bargein-93h` is retired to packaged-disabled. The
   checkpoint has no trained directed head, so the node suppresses that
   head and admission fails safe to the post-generation text-wake decision.

## Relation to the existing surfaces

`live.py` / the Full DuplexDemo App drive Gemma's NATIVE audio tower
(`core/runtime/`); converse drives the TRAINED whisper-encoder→adapter path
(`core/mlx_audio_streaming.py`). They are separate lineages sharing one node
registry and one Gemma family. The converse ladder's quasi tier,
commit-and-trim `stream_split()`, and ECAPA speaker/wake space exist only on
the converse path today; converging `DuplexModePolicy` (GUI/live) with the
ladder is follow-up work, deliberately not part of this import.
