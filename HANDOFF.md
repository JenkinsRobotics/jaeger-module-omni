# Handoff — finalize the whisper pipeline + omni pipelines

You are taking over an in-progress consolidation of Jaeger Agent Omni, a local
(MLX, Apple-silicon) omni voice agent: audio + vision + text in, text + audio
(Kokoro) out, one frozen Gemma-4 E4B brain. Your job: test the pipeline,
recommend, and further develop the **final whisper (transcription) pipeline**
and the other omni-model pipelines. Work incrementally, measure everything,
and do not regress what is already green.

Repo: `/Users/jonathanjenkins/GITHUB/jaeger-module-omni`  ·  venv: `.venv`
(Python 3.11.9). Read these first, in order: `NEXT.md` (the plan + operating
model), `CONVERSE.md` (the ear scorecard + how the pieces relate), `STATUS.md`
(measured ledger), `docs/GEMMA_OMNI_V2_WHITEPAPER.md` (architecture contract).

## The system in one screen

Two conversational lineages share one node registry and one Gemma family:
- **Native-tower path** — `jaeger_agent_omni/live.py` + the `Full DuplexDemo
  App` GUI. Drives Gemma's own MLX audio tower. Ear: `jaegeromni-41m-medium`.
- **Converse path** — `python -m jaeger_agent_omni.converse` (imported "Gemma
  Omni V2" graph). Drives a trained whisper-encoder→adapter→Gemma ear. Has the
  half/quasi/full duplex ladder, barge stop|continue, QbE wake, one-model
  HEARD/REPLY, commit-and-trim.

The **whisper pipeline** = `whisper-medium encoder → 41M adapter → Gemma`. The
encoder is identical across every ear (verified same sha); only the **adapter**
varies. Each (encoder+adapter) pairing is its own immutable node under
`jaeger_agent_omni/nodes/<id>/` (node.json + runtime.py + weights/). One node =
one pairing; a new adapter is a NEW node, never an edit (node contract in
`node_registry.py`, enforced by `validate_profiles`).

**Perception is parallel, not chained on the adapter.** Speaker ID / wake /
self-voice / turn / overlap run as independent NNs on the raw audio (ECAPA =
`para-speaker-ecapa-v1`, endpoint = `para-turn-state-v0`, VAD = Silero gate),
converge at `fusion-context-v0`. The adapter stays semantics-only. Perception
reaches Gemma two ways: **hardcoded** loop control (wake admits, turn
endpoints, self-voice drops) and a **context** text line the model reads (each
node owns a `context()`; `converse.perception_context()` assembles+injects;
`OMNI_CONTEXT=0` disables). Do NOT add a learned projection into Gemma yet —
its value is unproven; text-context first.

## Measured ear scorecard (same devclean64 manifest + scorer)

| Adapter (node) | attn | devclean64 | AMI64 | streams | heads |
|---|---|---|---|---|---|
| `jaegeromni-41m-medium` | bidirectional | **4.82%** | 32.99% | no | none |
| `jaegeromni-41m-causal-balanced` | causal | 5.03% | **28.23%** | yes | overlap/turn/spk |
| `jaegeromni-41m-converse` (converse default) | causal | 5.03% | — | yes | overlap/turn/spk |
| `gemmaomni-v2-stream-bargein-93h` (retired, disabled) | causal | 5.65% | — | yes | strong overlap 0.81F1 |
| `gemmaomni-41m-medium-final4000` | bidir | 6.81% | 22.88% (best meetings) | no | none |

Key facts: bidirectional wins clean but is worst on meetings; the causal gap
is the attention mask (proven — the cross-mount penalty was PROMPT coupling,
recovered exactly by the training-time transcript prompt). No single ear wins
all axes.

## The decided operating model (do not relitigate)

- **Default mode = quasi-duplex** (open mic + AEC + barge; agent decides).
- **Duplex tiers half/quasi/full and barge on/off stay user-selectable
  features — never remove or consolidate them.**
- **Tier-matched ear**: half + quasi are turn-based (endpoint-then-reply) →
  run `jaegeromni-41m-medium` (4.82%, best; legal because the full utterance
  exists at endpoint). Only **full** (live forming preview) needs the causal
  ear. Perception always from the parallel NNs.
- **Build order: half → quasi (default) → full.** Quasi-by-default is
  reachable now with no GPU window.

## Your work queue (ordered; measure at each step)

1. **Tier-matched ear routing (NO GPU, do first).** Make converse's turn-based
   tiers (half, quasi) transcribe with `jaegeromni-41m-medium` while `full`
   keeps the causal ear for its preview. Perception stays on the parallel NNs.
   Gate: campaign stays 27/27 AND the committed-turn WER on the turn-based
   tiers moves toward 4.82%. Start with **half-duplex on `41m-medium`**.
2. **Parallel overlap/barge NN (GPU, Phase-1).** Today quasi barge = energy
   BargeMonitor + ECAPA self-voice (passes campaign). Train a raw-audio overlap
   NN; promotion bar: overlap F1 ≥ 0.81 (match the retired 93h head) and
   event-level false-alarms/min + detection-delay gates on held-out real audio.
   This retires the last trunk-riding perception head.
3. **41m-medium streaming/causal port (GPU, Lane-1b).** Only needed to make the
   FULL-tier live preview top-accuracy. Streaming-train the medium recipe with
   the causal mask on; judge on the matched gates. Lower priority than 1–2.
4. **Speaker-sensitive eval for the perception-context value.** The campaign
   proved injection is safe (27/27) but cannot prove it helps. Build a small
   eval where speaker awareness should change the reply; only then decide to
   default `OMNI_CONTEXT` on.
5. **Other omni pipelines.** Vision (`semantic-vision-v0`) and text
   (`text-input-v0`) are ingested into the same Gemma; exercise mixed-modality
   turns in converse and confirm one shared memory. Native-tower parity items
   live in `STATUS.md`.
6. **Standing red gates (required before "locked"):** AEC 300 ms p95 tail
   (rerun `--aec-reference-mode contiguous`), agentic full corpus (28/162
   capability-reveal seam), and one physical five-minute run (tools, barge,
   context roll, zero drops). See `NEXT.md`.

## Verification commands (use these; add evidence JSON under research/candidate_evals/)

```
# hermetic engine checks (no models), must stay green
.venv/bin/python -m jaeger_agent_omni.converse --selftest        # expect 18/18
.venv/bin/python -m unittest jaeger_agent_omni.tests.test_node_registry
.venv/bin/python -m pytest jaeger_agent_omni/tests/ -q            # 46 tests

# behavioral campaign (real models + fake audio device), the primary gate
.venv/bin/python jaeger_agent_omni/tests/converse_sim.py --session half   # 12/12
#   sessions: half | quasi | barge-continue | full   -> together 27/27
#   OMNI_CONTEXT=0 disables perception-context injection (A/B)

# matched WER — ALWAYS same manifest + same scorer, never cross-population
.venv/bin/python research/converse_ear_wer.py -n 64 \
    [--node jaeger_agent_omni/nodes/<id>] [--adapter <npz>] [--system "<prompt>"] \
    --output research/candidate_evals/<name>-devclean64.json
.venv/bin/python training/stock_whisper_gemma.py wer \
    --cache ~/models/jaeger-training-data/stock-whisper-medium-eval-v1 \
    --split devclean --adapter <npz> -n 64 \
    --id-file ~/models/jaeger-training-data/stock-whisper-medium-eval-v1/devclean/manifest.jsonl \
    --output research/candidate_evals/<name>-devclean64.json
.venv/bin/python training/compare_candidate.py <baseline> <candidate>  # regression gate
```

## Guardrails (non-negotiable)

- **Measure, don't assume.** Every claim needs a reproducing command + an
  evidence JSON. Compare ears only on the SAME manifest + scorer.
- **Never commit weights** (`nodes/*/weights/` is gitignored); copy `.npz` /
  encoder by hand when adding a node. Record source sha + provenance in node.json.
- **Node contract**: new adapter → new node folder; update all profiles in
  `nodes/profiles.json` (validate_profiles requires every node explicitly
  enabled/disabled per profile) and the count in `test_node_registry.py`.
- **Keep perception parallel** on raw audio; keep the series adapter clean.
- **Keep all duplex tiers + barge selectable.** No mode removal.
- **One GPU job at a time**; respect the GPU guard. Do NOT re-run same-recipe
  41M continuations — the sanctioned training is the parallel overlap NN and
  the causal streaming port only.
- **Ponytail**: smallest change that works; one runnable check behind new
  logic; don't build the learned-projection or new abstractions until measured
  need. Commit in small verified steps with evidence in the message.

## State at handoff (latest commits)

`13da882d` operating-model plan · `7d56c2a5` perception→Gemma context route ·
`abab83ba` consolidation step 2 (converse ear = our 41M, 27/27) · `5fa28149`
step 1 (prompt-aligned 5.03%) · `be1b2653` ear rename · `b79cb682` V2 import.
Converse semantic channel currently defaults the causal `jaegeromni-41m-converse`
for every tier — step 1 above changes that to tier-matched. Working tree also
carries the user's separate in-progress GUI duplex-selector work (untracked
`core/runtime/duplex_mode.py`, `core/agentic/observer.py`, their tests) — leave
it alone unless asked.
