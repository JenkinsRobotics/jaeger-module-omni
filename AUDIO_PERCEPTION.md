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
