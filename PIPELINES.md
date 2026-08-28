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
| **GemmaOmni 41M Medium** | Exact frozen GemmaOmni checkpoint currently imported here | **12/12**, 4.92% clean, 34.47% AMI, 501 ms independent recheck | Best general accuracy; five-minute gate pending |
| **JaegerOmni 41M Medium** | 10% interpolation from the frozen import toward Jaeger's retained room continuation | **3.30%** on the source-qualified one-to-one conversation gate; **12/12**, 560 ms; **3.97%** untouched test-clean/51,311 words | Production 1:1 candidate; physical-room gate pending |
| **JaegerOmni 50M Medium** | Jaeger's independently developed Medium connector | 11/12, 11.20% clean, 47.85% AMI, **319 ms** | Fastest viable adapter |
| **JaegerOmni 50M Large** | Large-v3-turbo-paired connector, exactly 52,768,768 deployed parameters | 1/12, 17.80% clean, 413 ms | Transfer failed; active experimental branch |

MiniCPM-o 4.5, the tuned-Whisper observer, the Whisper text cascade, and the
hidden-state Talker remain references or downstream components. They are not
additional active Jaeger audio-model identities.

## What is actually selected

- The full-duplex GUI and its exact-worker self-test expose all five canonical
  identities. `tower`, `stock`, and `cleanroom` remain compatibility aliases
  for Gemma Native, JaegerOmni 50M Medium, and GemmaOmni 41M Medium.
- JaegerOmni 41M Medium is independently packaged at
  `nodes/jaegeromni-41m-medium`; it does not overwrite the frozen import. Its
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
This is answer-ear parity, not full production parity. The exact imported 41M
checkpoint independently rechecks at 501 ms versus MiniCPM's measured 525 ms
on the shared answer harness. That narrow speed crossover does not establish
speech-start superiority: Jaeger's saved END-to-first-audio sessions were not
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

Until the physical-room phase passes, MiniCPM remains the quality reference and
the native Gemma tower remains Jaeger's rollback/default. Select
`jaegeromni-41m-medium` explicitly for production testing.

Canonical five-model gate: [`PRODUCTION_TESTING.md`](PRODUCTION_TESTING.md).
Current generated comparison: [`PARITY_SCORECARD.md`](PARITY_SCORECARD.md).
One example:

```bash
python demo.py --selftest-jaeger ROOM.wav \
  --front-end gemmaomni-41m-medium --seconds 300 \
  --threshold 0.20 --silence-ms 700 --cooldown 20 --max-context 6000
```
