# GemmaOmni 41M Medium Final 4000

Exact immutable import of the other GemmaOmni effort's 2026-08-28 final-window
checkpoint. It pairs stock Whisper-medium with a 40.9M-parameter causal
adapter, four frame-level perception heads, and a learned self-speech input.

This node is deliberately separate from `jaegeromni-41m-medium`. Its training
shifted the clean/meeting frontier: the source repo reports 23.03% AMI WER but
6.49% dev-clean WER and 8/12 cases12. Those numbers are provenance, not
promotion evidence; Jaeger's own locked harness must reproduce them.

The adapter file is hard-linked from the local VoiceLLM-Playground artifact so
the imported bytes remain exact. Its SHA-256 is pinned in `node.json`. The
Whisper encoder is the same pinned stock snapshot used by the other medium
nodes. No Whisper decoder or transcript handoff exists on this path.
