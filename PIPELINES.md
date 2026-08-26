# Current pipeline roster

This is the canonical list. Historical measurements remain in `STATUS.md` and
`research/`, but they are not selectable production pipelines.

| Pipeline | Role | Measured quality | Speech out | Status |
| --- | --- | --- | --- | --- |
| Gemma native audio tower | live assistant/demo | 10/12 cases12; 363 ms text median | Kokoro clause streaming; 693--1,275 ms measured END-to-audio medians | **Live default** |
| Stock Whisper-medium -> clean-room 41M context adapter -> Gemma | Qwen-free clean-speech leader | **12/12**, 572 ms; **0.00%** Gemma-generated transcript WER/67 words; 3.234% instructed WER on matched 2,041 words | same Kokoro output; selectable in live/GUI | **Accuracy candidate; room gate pending** |
| Stock Whisper-medium -> owned 50M connector -> Gemma | Qwen-free production candidate | 11/12; 319 ms; 11.20% dev-clean64; 17.86% MINDS/Vox591; 13.51% clean-360val128; 12.56% novel-Vox128 | same Kokoro clause-streamed output; selectable in live/GUI | **Candidate; positive-room gate pending** |
| Stock Whisper-medium -> AMI-balanced 50M connector -> Gemma | meeting-domain research | 49.93% AMI64 versus 53.79% general baseline; 10/12; one clean error worse | not production gated | **Specialist only** |
| MiniCPM-tuned Whisper -> sparse connector -> Gemma | transcript/meeting observer | 13.61% on its fixed 977-word observer gate | minutes/output path, not assistant speech | **Retained observer** |
| MiniCPM-o 4.5 native | external reference | 12/12; 525 ms to text; 1,380 ms to speech | native | **Reference, not Jaeger runtime** |
| Whisper/`pywhispercpp` -> text -> Gemma -> Kokoro | shipped sequential reference | 12/12; approximately 0.5--1% reported STT WER | Kokoro | **Reference; violates omni text-free premise** |

## What is actually selected

- `demo.py` and `jaeger_agent_omni.live` retain the Gemma native tower as the
  rollback default and accept `--front-end stock`.
- `omni.py --front-end stock`, `jaeger_agent_omni.demo --front-end stock`, and
  the full-duplex GUI's Whisper selector use the paired, pinned node at
  `nodes/stock-whisper-medium-cekl-blend50`.
- The full-duplex GUI also exposes the independent clean-room node at
  `nodes/stock-whisper-medium-cleanroom-ctx93h-step2000`; CLI callers select it
  with `--front-end stock --adapter NODE_DIR`.
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
This is answer-quality parity, not full production parity: the clean-room node
has not passed the independent lockboxes or live-room gate and its 572 ms
median is slower than MiniCPM's measured 525 ms to text. Jaeger has demonstrated a faster best
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

Canonical candidate gate:

```bash
python demo.py --selftest-jaeger ROOM.wav --front-end stock --seconds 300 \
  --threshold 0.20 --silence-ms 700 --cooldown 20 --max-context 6000
```
