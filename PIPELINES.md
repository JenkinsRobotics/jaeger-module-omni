# Current pipeline roster

This is the canonical list. Historical measurements remain in `STATUS.md` and
`research/`, but they are not selectable production pipelines.

| Pipeline | Role | Measured quality | Speech out | Status |
| --- | --- | --- | --- | --- |
| Gemma native audio tower | live assistant/demo | 10/12 cases12; 363 ms text median | Kokoro clause streaming; 693--1,275 ms measured END-to-audio medians | **Live default** |
| Stock Whisper-medium -> owned 50M connector -> Gemma | Qwen-free assistant development | 10/12; 322 ms; 14.03% dev-clean64; 19.10% MINDS/Vox591 | compatible with the same output layer, but not live-soak promoted | **Active development** |
| Stock Whisper-medium -> AMI-balanced 50M connector -> Gemma | meeting-domain research | 49.93% AMI64 versus 53.79% general baseline; 10/12; one clean error worse | not production gated | **Specialist only** |
| MiniCPM-tuned Whisper -> sparse connector -> Gemma | transcript/meeting observer | 13.61% on its fixed 977-word observer gate | minutes/output path, not assistant speech | **Retained observer** |
| MiniCPM-o 4.5 native | external reference | 12/12; 525 ms to text; 1,380 ms to speech | native | **Reference, not Jaeger runtime** |
| Whisper/`pywhispercpp` -> text -> Gemma -> Kokoro | shipped sequential reference | 12/12; approximately 0.5--1% reported STT WER | Kokoro | **Reference; violates omni text-free premise** |

## What is actually selected

- `demo.py` and `jaeger_agent_omni.live` use the Gemma native tower.
- `omni.py --front-end stock` uses
  `artifacts/stock-whisper-gemma-medium50m-expanded-blend25.npz`.
- `meeting.py` uses the tuned observer by default; `--front-end stock` selects
  the retained general stock connector unless `--adapter` is supplied.
- The AMI checkpoint is explicit opt-in only:
  `artifacts/stock-whisper-gemma-medium50m-ami5-balanced1p5k.npz`.

## Is Jaeger equivalent to MiniCPM-o 4.5?

Not yet in model quality. MiniCPM is 12/12 on the shared question suite while
both Jaeger assistant paths are 10/12. Jaeger has demonstrated a faster best
END-to-first-audio session than MiniCPM's measured 1,380 ms speech start, but
Jaeger's 693--1,275 ms session variance means latency superiority is not yet a
stable claim. Jaeger does own its 50M stock-Whisper connector, source-balanced
training path, AEC/duplex controls, speaker telemetry, and rolling Gemma
context; those are product and extensibility advantages, not evidence of equal
perception accuracy.

The honest target is therefore two separate gates:

1. reach at least 12/12 while retaining or improving all fixed corpus WER
   gates;
2. pass repeated five-minute duplex sessions with stable latency, AEC,
   speaker telemetry, and context rolls.

Until both pass, MiniCPM remains the quality reference and the native Gemma
tower remains Jaeger's live default.
