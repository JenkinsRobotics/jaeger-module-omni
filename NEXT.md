# Next steps — current plan

One GPU job at a time. The other agent's clean-room CE/KL run owns the current
training window; do not launch evaluation or another trainer until it exits.

## 1. Gate the checkpoint now training

Compare it against the two retained stock checkpoints, never an intermediate:

```text
general: artifacts/stock-whisper-gemma-medium50m-expanded-blend25.npz
meeting: artifacts/stock-whisper-gemma-medium50m-ami5-balanced1p5k.npz
```

Promotion requires all of the following free-running measurements:

- cases12: at least 10/12;
- dev-clean64: no worse than 134/955 errors (14.03% archived gate);
- MINDS/Vox591: no worse than 1,860/9,736 errors (19.10%);
- speaker-disjoint train-other64: no worse than 224/1,143 (19.60%);
- AMI64: report the exact 673-word result, but do not trade a general gate for
  a meeting-only gain;
- median latency and parameter count reported alongside quality.

Teacher forcing is permitted as a training loss. It is never an evaluation
number. Do not promote from a single clip, a teacher-forced score, cosine, or
the training-only lexical head.

## 2. How to break the 14% plateau

More passes over the same adapter/native-vector regression are closed. They
move errors between domains but have not crossed 14% general WER. The next
useful objective is a low-weight, human-target sequence loss through frozen
Gemma, while keeping the native audio-vector loss as an anchor:

```text
stock Whisper encoder (frozen)
  -> owned 50M connector
     -> native-vector regression anchor
     -> frozen Gemma + human transcript/answer CE
     -> training-only speaker and event heads
```

Only the connector and small auxiliary heads train. Gemma, Whisper, and their
decoders stay frozen. Human transcripts supervise training but are never a
runtime handoff. The other agent's CE/KL result is the first test of this
direction; evaluate it before implementing a duplicate.

If it fails, the next structural experiment is a pretrained or distilled
monotonic sequence head at the connector bottleneck, not another randomly
initialized BiGRU CTC head. The current disposable head ended near 94% WER and
is diagnostic evidence that lexical structure is not yet usable there.

## 3. Data policy

Canonical caches live outside the repository under
`~/models/jaeger-training-data`. Current additions are:

- 12,000 previously unused LibriSpeech train-other clips;
- 4,042 new AMI turns from four hours with zero first-hour turn overlap;
- 3,249 AMI training and 750 speaker-disjoint AMI validation features.

Use `--balance-sources` for mixed-domain training. Keep every validation split
untouched. Before another run, add genuinely new speakers/acoustics such as
MUSAN/RIR augmentation or another licensed conversational corpus; do not cycle
the same AMI or LibriSpeech rows again merely because GPU time is available.

Secondary labels remain important but must not be conflated with actions:

- speaker identity/continuity;
- speech during playback (double-talk), distinct from the policy decision to
  stop or continue speaking;
- turn start/end and overlap;
- environmental/acoustic events.

## 4. Live production gates

The native Gemma tower remains the desktop default until the stock connector
passes a positive-room five-minute test. Require:

- no lost audio across a 300-second capture;
- meaningful Silero-admitted speech at the selected Elgato input;
- stable A/B/C speaker telemetry without claiming word-level diarization;
- AEC/double-talk measurements and barge-in behavior;
- context roll without negative soak deltas;
- END-to-first-audio median and tail latency.

The tuned MiniCPM-derived encoder remains observer-only while its fixed corpus
WER is 13.61%, better than the stock observer. It is not an assistant default
and the Qwen language model is never loaded.

## Commit gate

```bash
./.venv/bin/python research/check_all.py
./.venv/bin/python -m jaeger_agent_omni.omni --selfcheck
./.venv/bin/python demo.py --check
git diff --check
```

Do not stage generated WAVs, local corpora, virtual environments, model
weights, or intermediate checkpoints.
