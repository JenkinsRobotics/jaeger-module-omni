# MiniCPM parity scorecard

Generated from checkpoint-pinned artifacts. A two-percent tolerance prevents timing noise from being called a win. WER columns are Jaeger quality gates because no same-corpus MiniCPM WER baseline is saved yet. First-audio results are comparable only when a run is explicitly marked as a paired MiniCPM harness.

## Reference

| System | cases12 | answer latency | first audio | memory | vision |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniCPM-o 4.5 | 12/12 | 525 ms | 1380 ms | 4.9 GB | 379 ms/frame |

## Audio-ear comparison

| Model | checkpoint | cases12 | vs reference | answer latency | vs reference | first audio | vs reference | clean WER | AMI WER | test-clean lockbox | physical 5 min | conclusion |
| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| Gemma Native | `built-in` | 10/12 | BEHIND | 363 ms | SURPASSED | — | UNMEASURED | — | — | — | UNMEASURED | BEHIND |
| GemmaOmni 41M Medium | `0fec52869087` | 12/12 | PARITY | 501 ms | SURPASSED | — | UNMEASURED | 4.92% | 34.47% | — | UNMEASURED | PARITY + FASTER |
| JaegerOmni 41M Medium | `a7f96bfb8898` | 12/12 | PARITY | 560 ms | BEHIND | 1044 ms | NOT COMPARABLE | 4.92% | 34.18% | 3.97% | UNMEASURED | BEHIND |
| JaegerOmni 50M Medium | `b889e5b48565` | 11/12 | BEHIND | 319 ms | SURPASSED | — | UNMEASURED | 11.20% | 47.85% | 10.68% | UNMEASURED | BEHIND |
| JaegerOmni 50M Large | `3633baa083d4` | 1/12 | BEHIND | 413 ms | SURPASSED | — | UNMEASURED | 17.80% | — | — | UNMEASURED | BEHIND |

## System-level parity

| Gate | Jaeger now | MiniCPM reference | State |
| --- | --- | --- | --- |
| AEC/double-talk | 8.1 dB room-stress attenuation, 0/43 false, 100% recall, 843 ms p95 | no comparable saved baseline | BEHIND |
| Speaker identity/overlap | 86% continuity; separation fail; word attribution unavailable | no comparable saved baseline | BEHIND |
| Speech output | decoded text → resident Kokoro | native speech generation | BEHIND |
| Vision | not connected in Jaeger GUI | 379 ms/frame | BEHIND |
| Physical five-minute matrix | no checkpoint-matched JSON report yet | complete reference system | UNMEASURED |
| Memory | no checkpoint-matched full-stack measurement | 4.9 GB | UNMEASURED |

## Current verdict

GemmaOmni 41M Medium is the current core-ear leader: 12/12 matches MiniCPM and 501 ms is faster than the 525 ms reference. JaegerOmni 41M now independently matches cases12 at 12/12, but its 560 ms answer latency is behind the reference. It is the broad lockbox leader at 3.97% free-running WER and has a zero-drop direct five-minute context-roll run. Jaeger has not surpassed MiniCPM as a complete system: physical five-minute reports are absent, speaker separation is failed, the broad AEC stress gate fails, speech still crosses decoded text into Kokoro, and Jaeger vision is not connected.

Jaeger may be called *surpassed* only after one checkpoint retains cases12 parity, is faster on the same harness, passes the locked clean/meeting gates, and passes repeated physical five-minute duplex runs. Full Omni superiority additionally requires native expressive speech and connected vision.

Regenerate with:

```bash
./.venv/bin/python research/parity_scorecard.py
```
