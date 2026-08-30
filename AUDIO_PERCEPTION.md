# Neural audio perception

Jaeger's production speech gate remains the packaged Silero VAD. The selectable
`jaegeromni-41m-audio-perception` node is the first richer neural observation
path: the same stock-Whisper-medium → causal 41M → Gemma embeddings plus heads
for scene, overlap, speaker state/change, and turn completion.

This node is **after Whisper**. It is not the intended unified pre-Whisper
acoustic controller and cannot cancel echo or save Whisper work on rejected
audio. The production target is a separate small causal network that consumes
microphone PCM plus Jaeger's exact playback reference and emits both enhanced
PCM and typed control tensors:

```text
mic PCM + playback reference
  -> neural acoustic controller (continuous, low-rate causal state)
       -> enhanced external-speech PCM -> Whisper -> JaegerOmni 41M -> Gemma
       -> speech/self/external/music/noise/overlap/turn-state tensors ----^
```

That controller may eventually absorb Silero, Speex, fixed energy thresholds,
and endpoint heuristics. It must first prove echo removal, near-end speech
survival, false-open rate, endpoint delay, and compute latency. Classification
heads alone cannot replace AEC because AEC must produce a cleaned waveform, not
merely label the echo.

## Ternary go/no-go

Ternary weights are deferred for the first controller. A synchronized,
batch-one Metal benchmark measured the exact dispatch pattern of four causal
layers invoked once per 10 ms audio hop. Across three 2,000-frame trials on the
M1 Max, the representative 256-wide/262K-parameter controller took **0.330 ms**
in FP16 and **0.337 ms** with a custom packed {-1,0,+1} W2A16 kernel: **0.98x**
the FP16 speed, below the predeclared 1.5x development gate. At width 512 it
was 0.97x and at 1024 it fell to 0.77x. Merely storing ternary values and using
FP16 matmul also provides no ternary compute benefit.

Packing does reduce these dense weights by 87%, but the representative model
shrinks only from 512 KiB to about 66 KiB. That is not worth a custom QAT and
kernel path while endpoint and echo quality remain open. Build and validate the
controller in FP16 first; use supported quantization only after profiling the
trained topology. This benchmark is a kernel result, not an acoustic-quality
claim, and can be rerun if the topology or MLX kernels materially change:

```bash
./.venv/bin/python research/ternary_controller_bench.py \
  --widths 128 256 512 1024 --layers 4 --iterations 2000 --repeats 3
```

The saved measurements are in
`research/results/ternary-controller-m1max-20260829.json`.

That tiny-controller latency result must not be generalized into "ternary is
useless." A second test converted the real JaegerOmni 41M adapter with
per-output-row least-squares post-training ternarization. Converting all eight
matrices damaged the free-running answer gate from 12/12 to 8/12. Sensitivity
was highly structured: ternarizing only the two 16.78M-parameter feed-forward
matrices retained 12/12 at 506 ms versus the unchanged checkpoint's 12/12 at
505 ms. Across three locked corpora it changed normalized pooled WER from
210/4,482 (**4.69%**) to 204/4,482 (**4.55%**): one corpus lost one edit and
two gained seven combined. Treat that as quality retention, not a claimed
quality improvement.

A real mixed FP16/ternary implementation would reduce the adapter's matrix
weights from about 78 MiB to 50 MiB, a **36% whole-adapter-matrix reduction**.
The evaluation checkpoint reconstructs ternary weights as FP32 so it validates
quality only; it does not validate packed 41M latency or process memory. The
candidate is therefore research evidence, not a selectable node. Its roughly
28 MiB advantage over simply storing this adapter in FP16 is immaterial beside
Whisper, Gemma, activations, and KV cache on the current 32 GB host, but it is a
credible future embedded-device path. Full reports:
`research/results/ternary41m-20260829/summary.json`.

This is deliberately one neural node because all outputs consume the same
causal adapter states. It does not contain an STT decoder and never hands text
from Whisper to Gemma.

## What is validated

| Signal | Held-out result | Runtime authority |
| --- | ---: | --- |
| Speech / music / noise / silence | 95.31% accuracy, 0.952 macro-F1 on 128 source-file-disjoint clips | Appended to Gemma as non-transcript turn metadata; shadow only |
| Overlap | 0.723 F1 on unseen synthetic speakers; 0.742 on unseen AMI meetings | Telemetry only |
| Speaker change | 0.202 synthetic; 0.088 AMI F1 | Telemetry only |
| Turn end | 0.221 synthetic; 0.254 AMI F1 | Telemetry only |
| Spoken answers | 12/12, 584 ms median | Same frozen causal-balanced embeddings |

The scene corpus uses disjoint MUSAN source files across train, validation, and
test. Event thresholds are selected on speaker-disjoint synthetic validation
plus meeting-disjoint AMI validation, then reported once on the held-out folds.
All non-head parameters are bit-identical to the causal-balanced source node.

## Why this is not yet the live VAD

The adapter mask is causal, but stock Whisper is bidirectional within each
bounded encoder window. Consequently these are honest per-turn/bounded-window
observations, not a 30 ms streaming replacement for Silero. The event heads
also produce too many meeting false positives to control endpointing or
barge-in. Live policy therefore remains:

```text
PCM -> Silero speech admission -> transcript-free neural ear -> Gemma
                              \-> audio-scene metadata (v2 node only)
playback reference -> AEC/double-talk policy -------------------^
```

The next production-grade version needs a prefix-consistent streaming encoder,
labeled conversational endpoint data, and event-level gates reporting false
events/minute plus detection delay. A rejected causal temporal-difference head
improved one synthetic endpoint score but regressed unseen AMI speaker change
and endpoint F1; it was not added to runtime.

## Reproduce

```bash
./.venv/bin/python research/perception_gate.py \
  --adapter jaeger_agent_omni/nodes/jaegeromni-41m-audio-perception/weights/adapter.npz
./.venv/bin/python research/audio_scene_gate.py \
  --adapter jaeger_agent_omni/nodes/jaegeromni-41m-audio-perception/weights/adapter.npz
./.venv/bin/python -m jaeger_agent_omni.omni \
  --front-end jaegeromni-41m-audio-perception --bench research/cases12
```
