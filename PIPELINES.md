# Current pipeline roster

This is the canonical active-development list. Historical measurements remain
in `STATUS.md` and `research/`; the rejected Large connector is selectable for
diagnostics but is not a production candidate.

The generated, checkpoint-pinned comparison against MiniCPM-o 4.5 is
[`PARITY_SCORECARD.md`](PARITY_SCORECARD.md). Regenerate it after new fixed
corpus or duplex reports; never copy a best value from another checkpoint.

| Correct model name | What it is | Current results | Status |
| --- | --- | --- | --- |
| **Gemma Native** | Gemma's built-in audio tower | 10/12, 363 ms | Duplex-validated fallback |
| **GemmaOmni 41M Medium** | Earlier exact frozen GemmaOmni keypoint imported here | **12/12**, 4.92% clean, 34.47% AMI, 501 ms independent recheck | Retained clean reference |
| **GemmaOmni 41M Final 4000** | Latest exact imported causal checkpoint with perception heads | 10/12 default-64 (22/24 repeated-48), 6.81% clean, **22.88% AMI**, 547.6 ms warmed | Best meeting ear; QA/clean regression |
| **JaegerOmni 41M Medium** | 10% interpolation from the frozen import toward Jaeger's retained room continuation | **3.30%** source-qualified 1:1 gate; **12/12**, 4.82% clean, 32.99% AMI, 601.9 ms warmed; **3.97%** untouched test-clean | Production 1:1 candidate; physical-room gate pending |
| **JaegerOmni 41M Causal Balanced** | 85% Jaeger 41M + 15% latest causal GemmaOmni trunk; heads retained but uncalibrated | **12/12**, 5.03% clean, 28.23% AMI, **574.2 ms** warmed | Full-duplex development candidate; head revalidation pending |
| **JaegerOmni 41M Audio Perception** | Frozen causal-balanced ear plus trained scene and conversation-event heads | **12/12**, 584 ms; scene **95.31% / 0.952 macro-F1**; unseen-AMI overlap 0.742 F1 | Selectable shadow candidate; speaker-change/turn-end do not control policy |
| **JaegerOmni 50M Medium** | Jaeger's independently developed Medium connector | 11/12, 11.20% clean, 47.85% AMI, 619.7 ms same-harness | Viable historical adapter; not latency leader |
| **JaegerOmni 50M Large** | Large-v3-turbo-paired connector, exactly 52,768,768 deployed parameters | 1/12, 17.80% clean, 413 ms | Transfer failed; active experimental branch |

AMI figures throughout are the 64-clip / 673-word training-adjacent,
speaker-disjoint AMI validation slice — a ranking signal, not a deployable
meeting WER. The 3.97% untouched test-clean result was measured before the
corrected 12-token/s / 24-token-floor decoding budget and is labeled
pre-budget12 until remeasured.

MiniCPM-o 4.5, the tuned-Whisper observer, the Whisper text cascade, and the
hidden-state Talker remain references or downstream components. They are not
additional active Jaeger audio-model identities.

## What is actually selected

- The full-duplex GUI and its exact-worker self-test expose all eight canonical
  identities. `tower`, `stock`, and `cleanroom` remain compatibility aliases
  for Gemma Native, JaegerOmni 50M Medium, and GemmaOmni 41M Medium.
- JaegerOmni 41M Medium is independently packaged at
  `jaeger_agent_omni/nodes/jaegeromni-41m-medium`; it does not overwrite the frozen import. Its
  10% room blend was selected without cases12 and passed five corpus gates
  without regression before scoring 3.97% on the untouched lockbox.
- The final pre-production window kept that checkpoint frozen. On 69 English
  MINDS14 conversational clips where matched Whisper-medium is exact after
  standard normalization, it scores 22 edits / 667 words = **3.30% WER**.
  It independently rechecks at **12/12** spoken answers and 560 ms median.
  Four full/partial refinements, three independent seed repeats, and checkpoint
  soups either tied that conversation score or regressed another locked gate;
  none replaced the packaged SHA `a7f96bfb8898`.
- JaegerOmni 50M Large remains clearly marked experimental and is never
  selected by a non-diagnostic node profile.
- The stock live session now owns a persistent Gemma KV cache across turns,
  exact abandoned-reply rewind, monotonic soak counters, full-path warmup, and
  encoder-only Whisper loading. Continuous speech is prefilled in bounded
  12-second/1-second-overlap windows. The ASR decoder is neither loaded nor run.
- `meeting.py` uses the tuned observer by default; `--front-end stock` selects
  the retained general stock connector unless `--adapter` is supplied.
- The AMI checkpoint is explicit opt-in only:
  `artifacts/stock-whisper-gemma-medium50m-ami5-balanced1p5k.npz`.

## Is Jaeger equivalent to MiniCPM-o 4.5?

The clean-room stock node now matches MiniCPM's 12/12 on the shared question
suite, while the retained 50M node is 11/12 and the native tower is 10/12.
This is answer-ear parity, not full production parity. The 41M checkpoints'
answer medians (502--560 ms quick-bench, 548--602 ms on the warmed repeated
internal A/B) sit in the neighborhood of MiniCPM's separately measured
525 ms; the runs are not paired under one harness — the same checkpoint moves
~7% between this repository's own harnesses — so no faster/slower conclusion
is drawn. Jaeger's saved END-to-first-audio sessions were likewise not
run as a paired MiniCPM test and vary from 693--1,275 ms. The broader 32-room
CPU AEC stress currently fails (19 false barges, 41% recall), distinct speaker
separation fails, and the checkpoint-matched physical five-minute matrix is
unmeasured. Jaeger's owned connectors, source-balanced training path, duplex
controls, speaker telemetry, and rolling context are extensibility advantages,
not evidence of complete system parity.

The honest target is therefore two separate gates:

1. reach at least 12/12 while retaining or improving all fixed corpus WER
   gates;
2. pass repeated five-minute duplex sessions with stable latency, AEC,
   speaker telemetry, and context rolls.

Until the physical-room phase passes, MiniCPM remains the quality reference.
`jaegeromni-41m-medium` is Jaeger's default production-test ear; the native
Gemma tower remains the explicit rollback.

Canonical eight-model gate: [`PRODUCTION_TESTING.md`](PRODUCTION_TESTING.md).
Current generated comparison: [`PARITY_SCORECARD.md`](PARITY_SCORECARD.md).
One example:

```bash
python demo.py --selftest-jaeger ROOM.wav \
  --front-end gemmaomni-41m-medium --seconds 300 \
  --threshold 0.20 --silence-ms 700 --cooldown 20 --max-context 6000
```
