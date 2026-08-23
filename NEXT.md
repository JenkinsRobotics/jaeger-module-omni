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

The **active development** checkpoint is now the fully owned Gemma-native
teacher path:

```text
artifacts/stock-whisper-gemma-native-stack-swiglu-refine5k.npz
27.75% WER — exact historical dev-clean64 / 955 words
27.38% WER — held-out MINDS/Vox591 / 9,736 words
8/12 answers — canonical dual-output prompt, 532 ms median
20.90% generated transcript WER — cases12 diagnostic (was 43.28%)
```

It remains a challenger, not the live or observer default. The older paired
stock adapter remains the accuracy gate at 16.34%/9 of 12; the native tower
remains live at 10/12; the tuned-Whisper observer remains 13.61% WER.

Frame stacking plus RMSNorm/SwiGLU is now measured, not merely proposed. It
improves the prior native-teacher adapter by 3.66 WER points on dev-clean and
4.80 points on MINDS/Vox, but does not improve the 8/12 answer gate. Do not
continue the selected recipe at a lower learning rate: a complete 5e-6 sweep
regressed from 29.82% to at least 31.99% on the ranking evaluator. The next
controlled experiment must change the learning signal or temporal connector
while keeping encoder, corpora, prompts, and free-running gates fixed.
Sampled CTC against human transcripts and frozen Gemma input-token rows is
also closed for this checkpoint: even 250 steps at 1e-6 regressed the exact
dev-clean gate from 27.75% to 28.59%, reaching 33.61% by step 1,000. It used no
Gemma output, but still pulled the vectors away from the native audio manifold.
Do not add a fixed lag or first-difference loss: 64/64 temporal-audit clips
prefer zero lag, and a 0.25 delta-loss sweep regressed every checkpoint from
29.82% to 33.95–40.87% on the external 969-word ranking corpus.

Stock OpenAI Whisper plus a fresh adapter is now viable through paired
pre-Gemma representation distillation. The promoted local checkpoint is
`artifacts/stock-whisper-gemma-medium-multidomain.step005500.npz`: 16.34% WER
on 64 LibriSpeech clips/955 words and 25.52% on 64 FLEURS clips/1,403 words.
The original direct-answer prompt made it repeat questions and scored 0/12,
but that was not its answer ceiling. A single-pass diagnostic that asks Gemma
to generate `TRANSCRIPT:` then `ANSWER:` scores **9/12** answers and 28.36%
generated-transcript WER over the suite's 67 words. The three misses all have
incorrect generated transcripts. Keep the native 10/12 tower as the desktop
default, but use `omni.py --dual-output` when separating perception failures
from response failures.

The next stock experiment should add paired MINDS-14/VoxPopuli/far-field audio
only when raw waveforms are locally recoverable. Continue from step 5,500 at a
low learning rate and require both fixed gates: no regression above 16.34% on
LibriSpeech and an improvement below 25.52% on FLEURS. Step 7,000 is already
rejected (18.32%/25.59%). Do not select by nearest vocabulary-row recall; the
step it preferred produced 106.28% free-running WER.

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
