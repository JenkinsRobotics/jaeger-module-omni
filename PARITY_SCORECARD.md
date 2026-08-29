# MiniCPM parity scorecard

Generated from checkpoint-pinned artifacts. WER columns are Jaeger quality gates because no same-corpus MiniCPM WER baseline is saved yet. Answer-latency medians come from per-model bench runs that are NOT paired with the MiniCPM harness: the same checkpoint moves ~7% between this repository's own harnesses, so latency is reported for context only and never graded against the reference. The 41M variants sit in the neighborhood of MiniCPM's separately measured 525 ms. Latency and first-audio comparisons require a run explicitly marked as a paired MiniCPM harness. AMI WER is the 64-clip / 673-word training-adjacent, speaker-disjoint AMI validation slice — a ranking signal, not a deployable meeting-WER claim.

## Reference

| System | cases12 | answer latency | first audio | memory | vision |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniCPM-o 4.5 | 12/12 | 525 ms | 1380 ms | 4.9 GB | 379 ms/frame |

## Audio-ear comparison

| Model | checkpoint | cases12 | vs reference | answer latency | vs reference | first audio | vs reference | clean WER | AMI64 WER | test-clean lockbox | physical 5 min | conclusion |
| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| Gemma Native | `built-in` | 10/12 | BEHIND | 363 ms | NOT PAIRED | — | UNMEASURED | — | — | — | UNMEASURED | BEHIND |
| GemmaOmni 41M Medium | `0fec52869087` | 12/12 | PARITY | 501 ms | NOT PAIRED | — | UNMEASURED | 4.92%\* | 34.47%\* | — | UNMEASURED | ANSWER PARITY |
| JaegerOmni 41M Medium | `a7f96bfb8898` | 12/12 | PARITY | 560 ms | NOT PAIRED | 1044 ms | NOT COMPARABLE | 4.82% | 32.99% | 3.97%\* | UNMEASURED | ANSWER PARITY |
| GemmaOmni 41M Final 4000 | `185bd7b3f6ce` | 10/12 | BEHIND | 506 ms | NOT PAIRED | — | UNMEASURED | 6.81% | 22.88% | — | UNMEASURED | BEHIND |
| JaegerOmni 41M Causal Balanced | `e4ea88b9c99e` | 12/12 | PARITY | 502 ms | NOT PAIRED | — | UNMEASURED | 5.03% | 28.23% | — | UNMEASURED | ANSWER PARITY |
| JaegerOmni 50M Medium | `b889e5b48565` | 11/12 | BEHIND | 319 ms | NOT PAIRED | — | UNMEASURED | 11.20%\* | 47.85%\* | 10.68%\* | UNMEASURED | BEHIND |
| JaegerOmni 50M Large | `3633baa083d4` | 1/12 | BEHIND | 413 ms | NOT PAIRED | — | UNMEASURED | 17.80%\* | — | — | UNMEASURED | BEHIND |

\* measured before the corrected 12-token/s / 24-token-floor decoding budget; remeasure at the authoritative budget before any cross-checkpoint comparison.

## System-level parity

| Gate | Jaeger now | MiniCPM reference | State |
| --- | --- | --- | --- |
| AEC/double-talk | 8.1 dB synthetic room-stress attenuation, 0/43 false, 100% recall, 843 ms p95 | no comparable saved baseline | BEHIND |
| Speaker identity/overlap | 86% continuity; separation fail; word attribution unavailable | no comparable saved baseline | BEHIND |
| Speech output | decoded text → resident Kokoro | native speech generation | BEHIND |
| Vision | not connected in Jaeger GUI | 379 ms/frame | BEHIND |
| Physical five-minute matrix | no checkpoint-matched JSON report yet | complete reference system | UNMEASURED |
| Memory | no checkpoint-matched full-stack measurement | 4.9 GB | UNMEASURED |

## Current verdict

GemmaOmni 41M Medium matches MiniCPM at 12/12 on the shared question suite. Every 41M variant's answer median (502–560 ms here) sits in the neighborhood of the 525 ms MiniCPM reference, but the runs are not paired under one harness, so no faster/slower conclusion is drawn. JaegerOmni 41M independently matches cases12 at 12/12. The new causal-balanced Jaeger checkpoint retains 12/12 and reduces the 673-word AMI64 validation slice to 28.23%, but its perception heads still require post-blend calibration. The frozen Jaeger 41M remains the broad lockbox leader at 3.97% free-running WER (measured before the corrected 12-token/s budget) and has a zero-drop direct five-minute context-roll run. Jaeger has not surpassed MiniCPM as a complete system: physical five-minute reports are absent, speaker separation is failed, the broad AEC stress gate fails, speech still crosses decoded text into Kokoro, and Jaeger vision is not connected.

Jaeger may be called *surpassed* only after one checkpoint retains cases12 parity, is faster on the same harness, passes the locked clean/meeting gates, and passes repeated physical five-minute duplex runs. Full Omni superiority additionally requires native expressive speech and connected vision.

Regenerate with:

```bash
./.venv/bin/python research/parity_scorecard.py
```
