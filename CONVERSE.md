# Gemma Omni V2 converse — the imported production pipeline

Imported 2026-09-03 from `VoiceLLM-Playground/Omni Model/gemma omni v2` at
its 27/27 campaign state (playground commit `9f1103b`). This is the full
pipeline — real trained weights, real node graph, real duplex ladder — not a
facade: text, vision, and audio in; text and audio out.

## What runs

```
.venv/bin/python -m jaeger_agent_omni.converse --selftest       # 17 hermetic checks, no models
.venv/bin/python -m jaeger_agent_omni.converse --converse half  # turn-based, mic closed while speaking
.venv/bin/python -m jaeger_agent_omni.converse --converse quasi # open mic + Speex AEC + barge (--barge stop|continue)
.venv/bin/python -m jaeger_agent_omni.converse --converse full  # + forming-transcript preview + commit-and-trim
.venv/bin/python -m jaeger_agent_omni.converse --enroll NAME F.wav   # one-shot QbE wake enrollment
```

Semantic channel (default `--semantic nn`): stock whisper-medium encoder →
trained 41M streaming context adapter → Gemma-4 E4B, no text seam on the way
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
  recorded number); imported `stream-bargein-93h` **5.65%** as wired /
  **5.76%** instructed. The current ear stays the better transcript model on
  clean audio; the imported ear is the one carrying the trained perception
  heads. Evidence in `research/candidate_evals/*-20260903-devclean64.json`;
  harness `research/converse_ear_wer.py`.

## Relation to the existing surfaces

`live.py` / the Full DuplexDemo App drive Gemma's NATIVE audio tower
(`core/runtime/`); converse drives the TRAINED whisper-encoder→adapter path
(`core/mlx_audio_streaming.py`). They are separate lineages sharing one node
registry and one Gemma family. The converse ladder's quasi tier,
commit-and-trim `stream_split()`, and ECAPA speaker/wake space exist only on
the converse path today; converging `DuplexModePolicy` (GUI/live) with the
ladder is follow-up work, deliberately not part of this import.
