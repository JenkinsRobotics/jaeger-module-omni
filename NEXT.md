# Next steps — measured order

Run one GPU command at a time. `training/run.sh` now detects this repo's
PyTorch/MLX trainers and the VoiceLLM clean-room/adapter jobs; evaluation and
generation still require the same manual discipline.

## 1. Keep the two production objectives separate

Assistant/demo mode remains the native Gemma tower because it scores 10/12;
the tuned-Whisper answer adapter scores 6/12 and is not promoted.

```bash
./.venv/bin/python -m jaeger_agent_omni.omni \
    --bench research/cases12
```

Transcript/observer mode uses the Qwen-free tuned-Whisper path and promoted
sparse adapter. On the same 64-clip validation slice it improves 14.74% to
13.61% free-running corpus WER.

```bash
./.venv/bin/python research/wer_corpus.py \
    --adapter ~/models/gemma-audio-adapter/adapter_sparse_vox1375_k022_b000.npz \
    -n 64
```

Do not swap the sparse transcript adapter into the assistant demo merely
because its WER is better. It correctly emits the question rather than its
answer; those are different trained objectives.

## 2. Repeat the five-minute positive-room gate

The 2026-08-22 five-minute Elgato capture retained all 300 seconds but the
supposed conversation was not present at that input: every 9-second window had
only 3–19% Silero-positive frames. Speech admission now prevents those windows
from hallucinating text. Repeat while the recording is visibly reaching the
Elgato and require a meaningful admitted-window fraction before judging WER or
diarization.

```bash
./.venv/bin/python -m jaeger_agent_omni.meeting \
    --seconds 300 --device 1 --k 3 \
    --record /tmp/meeting-positive.wav \
    --out research/meeting-positive.md
```

The retained 90-second positive replay is the current functional proof: 11
admitted neural windows, coherent cloud-storage/meal-kit transcript and
minutes, clean exit. Remaining observer defects are overlap de-duplication and
word-level speaker attribution. Do not convert online A/B/C turn labels into a
claim of word-level diarization; the offline k=3 centroids separate at
0.369/0.401/0.592, but most long windows contain mixed voices.

## 3. Improve answer alignment at the proven boundary

Stock OpenAI Whisper plus a fresh adapter is rejected at this data scale:
99.27% WER after CTC warm-up and 96.88% after 100 Gemma CE updates. Do not spend
another long run on stock features without a new hypothesis and a same-corpus
ablation.

Continue training only the proven 4096→2560 boundary over cached
`AvgPool5(ReLU(linear1(tuned Whisper(audio))))` features. Build response targets
for the exact 12-case failures and broader spoken QA, then require both:

- free-running 12-case accuracy at least 10/12 (native-tower parity);
- no regression from 13.61% on the fixed transcript corpus when the intended
  checkpoint is transcript-capable.

The stock experiment remains reproducible as a control:

```bash
./.venv/bin/python training/stock_whisper_gemma.py --help
```

## 4. Live duplex gaps

The canonical desktop app and `jaeger_agent_omni.live` remain on the native
tower. Speex AEC passes the controlled echo corpus at 24.1 dB but measured only
about 0.1 dB aggregate during continuous room double-talk. Long clauses still
produce occasional ~1 second END→AUDIO outliers despite comma/semicolon/em-dash
streaming. Repeat positive five-minute duplex captures before changing either
threshold.

## 5. Speech out

```bash
./.venv/bin/python research/verify_talker_text_following.py
```

This needs Qwen3-8B hidden states as an offline teacher (~16 GB), with no other
GPU job. Runtime input remains Qwen-free. Train SpeechOutDelta only after the
teacher-following gate is clean; zero-conditioning was measured to produce
fluent but incorrect speech.

## Commit gate

```bash
./.venv/bin/python demo.py --check
./.venv/bin/python -m jaeger_agent_omni.omni --selfcheck
./.venv/bin/python research/check_all.py
git diff --check
```

The `check_all.py` command must still report exactly 31/31 green. Never stage
regenerated `demo_out/*.wav` files unless the user explicitly asks to replace
the checked-in audio fixtures.
