# JaegerOmni 41M Audio Perception

Selectable shadow candidate built on the exact JaegerOmni 41M causal-balanced
audio/Gemma trunk. Frozen-head training adds four-way audio-scene telemetry and
recalibrates speaker-change, overlap, turn-end, and speaker-state signals.

Every non-head parameter is bit-identical to the source checkpoint, so this
stage cannot change its audio embeddings, WER, or generated answers. The scene
head is source-file-disjoint validated and its metadata may be appended between
audio and response generation. Event heads remain diagnostic only: their
meeting false-positive rates are not safe enough to control live endpoints or
barge-in policy.

This is not yet a neural replacement for the production Silero VAD. Stock
Whisper's encoder is still bidirectional and evaluated in bounded windows, so
the new heads are causal at the adapter but not yet a low-latency 30 ms gate.
