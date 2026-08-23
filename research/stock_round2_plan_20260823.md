# Stock-Whisper round 2 launch sheet — 2026-08-23

Do not launch until the operator says the GPU window is open. Every command in
this sheet is sequential. `stock_whisper_gemma.py` now refuses known jobs,
including `cleanroom_distill.py`.

## Admitted source archive

Common removable root:

```text
~/models/jaeger-training-audio/stock-round-20260823/
```

Measured admission result:

```text
5,795 unique PCM files / 10.7665 hours / 2.1 GiB
5,204 train / 591 validation
1,795 MINDS-14 en-US/en-GB/en-AU (CC BY 4.0)
4,000 VoxPopuli English (CC0)
0 missing files
0 train/validation ID overlap
0 exact transcript overlap with research/cases12
duration range 0.680–28.757 seconds
```

The archive manifest joins every waveform to an existing tuned-Whisper teacher
feature ID. Transcripts are human dataset labels, never model generations.

## Window 1: stock feature cache

```bash
RAW=~/models/jaeger-training-audio/stock-round-20260823/manifest.jsonl
TEACH=~/models/gemma-audio-adapter/cache
NEW=~/models/stock-whisper-gemma/cache-medium-minds-vox-paired-v1

./.venv/bin/python -u training/stock_whisper_gemma.py precompute-paired \
  --cache "$NEW" --whisper openai/whisper-medium --max-seconds 29 \
  --teacher-cache "$TEACH" \
  --teacher-split train_minds14 --teacher-split train_voxpopuli \
  --raw-manifest "$RAW" --output-split train

./.venv/bin/python -u training/stock_whisper_gemma.py precompute-paired \
  --cache "$NEW" --whisper openai/whisper-medium --max-seconds 29 \
  --teacher-cache "$TEACH" \
  --teacher-split validation_minds14 --teacher-split validation_voxpopuli \
  --raw-manifest "$RAW" --output-split validation
```

Before training, require 5,204/591 stock rows, exact ID intersections with the
four teacher splits, and student/teacher temporal-length agreement.

## Baseline new-domain gates

Measure the promoted step-5,500 checkpoint before changing it. These are
free-running corpus metrics, not teacher-forced losses.

```bash
BASE=artifacts/stock-whisper-gemma-medium-multidomain.step005500.npz

./.venv/bin/python training/stock_whisper_gemma.py wer \
  --cache "$NEW" --split validation --source-prefix minds14: \
  --adapter "$BASE" -n 64 \
  --output research/stock_round2_base_minds64_wer.json

./.venv/bin/python training/stock_whisper_gemma.py wer \
  --cache "$NEW" --split validation --source-prefix voxpopuli: \
  --adapter "$BASE" -n 64 \
  --output research/stock_round2_base_vox64_wer.json
```

## Branch A: tuned-Whisper observer continuation

Continue from the promoted checkpoint, not the rejected step 7,000. Use a
lower `1e-5` rate, save every 250 steps, and stop at 6,500 for a short sweep.
The training target remains pre-Gemma tuned-adapter vectors.

```bash
./.venv/bin/python -u training/stock_whisper_gemma.py distill \
  --cache "../VoiceLLM-Playground/Omni Model/gemma-backbone/cleanroom_cache50" \
  --train-split train100 --validation-split devclean \
  --teacher-cache "$TEACH" --teacher-train-split train_100 \
  --teacher-validation-split validation \
  --teacher-adapter \
    ~/models/gemma-audio-adapter/adapter_sparse_vox1375_k022_b000.npz \
  --extra-pair \
    ~/models/stock-whisper-gemma/cache-medium-fleurs-paired-v1 train validation \
    "$TEACH" train_fleurs validation_fleurs \
  --extra-pair "$NEW" train validation \
    "$TEACH" train_minds14 validation_minds14 \
  --extra-pair "$NEW" train validation \
    "$TEACH" train_voxpopuli validation_voxpopuli \
  --resume "$BASE" --steps 6500 --learning-rate 1e-5 \
  --save-every 250 --eval-examples 64 \
  --output artifacts/stock-whisper-gemma-medium-round2.npz
```

Evaluate 5,750/6,000/6,250/6,500 sequentially. Reject any checkpoint above
16.34% on fixed dev-clean64 or above 25.52% on fixed FLEURS64. Only survivors
advance to separate MINDS64 and Vox64 free-running gates.

## Branch B: Gemma-native teacher

The VoiceLLM clean-room run is a separate architecture:

```text
stock Whisper 50 Hz -> LayerNorm -> Linear/GELU -> pool 2 -> Linear 2560
teacher = Gemma audio_tower -> embed_audio at ~25 Hz
loss = smooth-L1 + cosine
```

Do not blend its weights with Branch A. Sweep its 2,500-step checkpoints with:

1. cosine to native teacher plus cross-clip collapse/diversity;
2. free-running generated-transcript corpus WER;
3. `TRANSCRIPT:` plus `ANSWER:` cases12 accuracy.

The current run's inline decoded sample probe was broken by an MLX
`generate_step` call signature and is not a quality result. Use the corrected
runtime path after training. If a checkpoint beats the promoted stock adapter,
port its LayerNorm/GELU checkpoint format and runtime as a named native branch;
do not silently reinterpret it as the MiniCPM-shaped adapter.

## Commit gates

```bash
./.venv/bin/python demo.py --check
./.venv/bin/python -m jaeger_agent_omni.omni --selfcheck
./.venv/bin/python research/check_all.py
git diff --check
```

Report only free-running, corpus-level figures. Leave `demo_out/*.wav`
unstaged.
