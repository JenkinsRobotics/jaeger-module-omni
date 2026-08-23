# 50M contextual stock-Whisper adapter — joint promotion plan

The selected control remains:

```text
artifacts/stock-whisper-gemma-native-temporal256x2-k31-6k.npz
28,858,368 parameters
15.60% dev-clean64 WER
19.86% MINDS/Vox591 WER
10/12 cases12 answers
```

The verified lexical source control is the production `pywhispercpp`
large-v3-turbo decoder: 1 edit / 205 words = **0.4878% pooled WER** on 12
held-out LibriSpeech `dev-clean` clips from four speakers. It ran at 2.63x
realtime with 558 ms median finalization. The raw report is
`VoiceLLM-Playground/Voice LLm production/tests/results/
librispeech_2026-08-20_213921/results.json`.

That figure proves the Whisper encoder/decoder family can preserve nearly all
lexical information on that material. It is not yet a directly comparable
ceiling for this adapter: the selected adapter uses Whisper-medium and its
authoritative gate contains 955 different words, whereas the 0.4878% control
uses large-v3-turbo and 205 words. Before claiming progress toward the source
ceiling, run the pywhispercpp decoder, native Gemma tower, and adapter on the
same waveforms with the same normalizer and pooled word counts.

The challenger adds one exact-zero residual branch after the selected local
and long-temporal branches:

```text
RMSNorm(2560)
  -> SwiGLU(1536)
  -> 12-head bidirectional self-attention
  -> depthwise Conv1d(k=31)
  -> RMSNorm
  -> zero-init Linear(1536, 2560)
  -> residual add
```

The branch adds 21,288,448 parameters for 50,146,816 total. Its final
projection is zero at initialization, so step zero produces exactly the
selected control output. All earlier branches remain frozen.

## Training objectives

Primary supervision remains paired, pre-language-model representation
distillation from Gemma's own `audio_tower -> embed_audio` vectors. No Gemma
output, generated transcript, teacher forcing, Whisper decoder, MiniCPM
projector, or Qwen model supplies a target.

The training mix uses LibriSpeech, FLEURS, MINDS-14, and VoxPopuli. A small
same/different-speaker contrastive loss is applied only to rows carrying real
speaker labels. It reads the new contextual bottleneck and never emits a
speaker label into Gemma's content stream.

Native-vector distillation is a bootstrap and manifold anchor, not the final
lexical objective. Once a student approaches the native teacher, pure
distillation cannot systematically target the lower pywhispercpp WER. The next
stage must use human transcripts through a training-only lexical head attached
to the contextual bottleneck, while retaining a smaller native-vector anchor.
Do not force the final Gemma-space audio vectors toward individual vocabulary
rows; that sampled-CTC objective has already regressed free-running WER.

Environmental robustness is represented by held-out far-field/domain gates,
not by inventing environment labels. Turn-end and barge-in remain separately
trained/tested runtime capabilities; an adapter WER score cannot establish
either behavior.

## Joint promotion gates

| Capability | Control / requirement | Measurement |
| --- | --- | --- |
| lexical perception | 15.60% | historical dev-clean64, 955 words |
| varied-room perception | 19.86% | held-out MINDS/Vox591, 9,736 words |
| answering | 10/12 | canonical cases12, free-running |
| speaker preservation | no regression | held-out same/different-speaker retrieval plus room k=3 |
| environment/far-field | no regression | per-domain MINDS/Vox and positive-room meeting capture |
| turn taking | no regression | hands-free turn completion and false-handover count |
| barge-in | no regression | `research/duplex_check.py --barge-in` and AEC double-talk capture |
| live speed | no material regression | adapter latency plus five-minute END->AUDIO p50/p95 |
| meeting continuity | thread survives | five-minute capture, context roll, speaker tags, minutes completeness |

A checkpoint is not promoted merely because its native-vector training loss or
one WER slice improves. Corpus results are free-running, never teacher-forced.
The checked-in control remains active until the complete matrix is measured.

## Owned run

The trainer, context architecture, checkpoint writer, contention guard, and
probes now live in this repository. The cache and Gemma directory below are
read-only data/model inputs; no Python module is imported from either one:

```bash
./.venv/bin/python -u training/context_distill.py \
  --cache "$HOME/GITHUB/VoiceLLM-Playground/Omni Model/gemma-backbone/cleanroom_cache50" \
  --split train100 \
  --extra-split mindsvox_train \
  --extra-split fleurs_train \
  --model ~/models/gemma-4-e4b-it-mlx-full \
  --resume artifacts/stock-whisper-gemma-native-temporal256x2-k31-6k.npz \
  --output artifacts/stock-whisper-gemma-native-context1536-8k.npz \
  --context-hidden 1536 --context-heads 12 --context-kernel 31 \
  --speaker-weight 0.01 --speaker-margin 0.30 \
  --steps 8000 --learning-rate 5e-5 --warmup 200 \
  --max-seconds 12 --short-weight 2 --short-seconds 3 \
  --log-every 250 --probe-every 1000 --checkpoint-every 1000
```

## Sequential large-v3-turbo A/B

Do not train the large-v3-turbo branch until the 50.15M Whisper-medium run is
fully swept. If no medium checkpoint clears the joint gates, retain the
28.86M selected control as the architectural baseline. If the 50.15M branch
wins, promote it first and use that complete downstream structure as the large
branch baseline.

The large-v3-turbo branch is a separate pipeline and artifact. Its encoder is
not checkpoint-compatible with Whisper-medium: large-v3-turbo emits 1280-wide
frames while the current medium encoder emits 1024-wide frames. At pool/stack
2 the first normalization/projection therefore changes from 2048 input values
to 2560. Do not reshape or pad the medium input weights and call that a port.

The reusable downstream 2560-wide residual, local-temporal, long-temporal and
context blocks may be copied from the winning medium architecture. Train a new
large input stack against paired native Gemma vectors before unfreezing any
shared downstream block. Then apply the same truth-supervised lexical stage
and secondary objectives used for medium.

The A/B must use identical waveform IDs and pooled scoring:

1. pywhispercpp large-v3-turbo decoder source control;
2. 50M Whisper-medium adapter;
3. separately trained large-v3-turbo adapter;
4. native Gemma tower and legacy cascade references.

Report dev-clean64/955, MINDS/Vox591/9,736, cases12, speaker retrieval,
far-field/room capture, adapter latency, memory, and five-minute duplex
behavior. Keep both `--front-end stock-medium` and `stock-large-v3-turbo`
selectors until the large branch wins the complete matrix.
