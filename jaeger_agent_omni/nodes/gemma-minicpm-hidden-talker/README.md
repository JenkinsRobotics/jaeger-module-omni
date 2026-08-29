# gemma-minicpm-hidden-talker

Experimental embedded speech-output node:

```text
Gemma token IDs + matching 2560-wide hidden states
  -> offline-compiled token-ID lookup + trained rank-128 hidden bridge
  -> MiniCPM 20-layer Talker (speech codes at 25 Hz)
  -> StepAudio flow decoder + HiFi-GAN
  -> cloned 24 kHz voice
```

No decoded string is passed to TTS. The 12-case free-running result is 10/12
semantic answers, 93.52% mean spoken-word accuracy, and 2040.9 ms median first
audio. It remains experimental because that latency misses the live target,
the run was not wall-clock paced or barge-in tested, and its inherited Talker
still needs an integer Gemma-vocabulary to Qwen-vocabulary lookup.

This is richer than Kokoro's current text boundary: Gemma hidden state reaches
the Talker and a reference waveform provides voice identity. It is not the
final native design. That design trains a Talker directly on Gemma token
embeddings and hidden states, eliminating the vocabulary lookup while keeping
the speech-code decoder and voice conditioning.
