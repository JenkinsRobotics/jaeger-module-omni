# neural-acoustic-controller

Separate causal pre-Whisper controller for the eventual neural replacement of
the independent AEC, VAD, sound-category, self-speech, overlap, and acoustic
endpoint heuristics.

```text
microphone PCM + exact Jaeger playback reference
  -> causal shared encoder
       -> estimated echo -> echo-reduced PCM -> Whisper -> JaegerOmni 41M
       -> voice / scene / self / external / overlap tensors -----------^
       -> forming / acoustically-ended / wake tensors -----------------^
```

The node is intentionally disabled and has no packaged checkpoint yet. Its
untrained echo head is initialized to exact passthrough, but untrained control
scores have no authority. Speex and Silero remain production fallbacks until a
trained checkpoint passes every gate in `node.json`.

The five companion stages map as follows:

1. `hear` consumes the controller's neural voice probabilities.
2. `categorise` consumes the same shared acoustic state; it never consumes a
   decoded transcript.
3. `state_of` combines causal acoustic completion with a later semantic
   completion probability from JaegerOmni/Gemma.
4. `addressed_to_me` consumes later neural semantic/wake/conversation scores.
5. `decide` emits `answer` or `log_only`; it does not infer perception itself.

This boundary preserves the no-transcript input path. Raw 10 ms acoustics can
detect energy, identity, overlap, and pauses; whether a sentence is meaningful
and directed at Jaeger requires the later language-capable neural state.

CPU contract test:

```bash
./.venv/bin/python -m unittest \
  jaeger_agent_omni.tests.test_acoustic_controller
```

Training is not launched automatically. `train.py` accepts a JSONL manifest of
paired microphone/reference/echo-free targets plus frame labels and writes an
explicit checkpoint and report.
