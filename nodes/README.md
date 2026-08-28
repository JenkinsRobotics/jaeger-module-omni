# Neural nodes

`nodes/` contains learned neural components in the Omni matrix. Signal
conditioning and admission policy are deliberately separate under `gates/`.
Every node owns a manifest, local runtime entry point, dependency declaration,
and locally materialized immutable weights. Model files may be hard-linked to
the machine's model cache so the node is self-contained without duplicating
gigabytes on disk.

Each manifest declares typed `consumes` and `produces` ports. Adapter nodes also
verify encoder identity, encoder boundary width, pooling, architecture, and
checkpoint hash before loading. Medium and large Whisper adapters are separate
nodes because changing the encoder changes the trained transfer contract.

Status is explicit. A rejected experimental node remains reproducible but is
never selected as a live default.

The registry stays flat; capability is typed in each manifest:

```text
nodes/
  stock-whisper-medium-cekl-blend50/
  stock-whisper-medium-cleanroom-ctx93h-step2000/
  jaegeromni-41m-medium/
  stock-whisper-large-v3-turbo50m/
  whisper-medium-stt/
  whisper-large-v3-turbo-stt/
  gemma-4-e4b-cognition/
  kokoro-tts/
  gemma-minicpm-hidden-talker/
  campplus-speaker/
  profiles.json
```

Medium and large Whisper perception nodes are different nodes because their
encoder boundaries and trained adapters differ. Medium and large Whisper STT
decoders are also different nodes, but they are not components of the
transcript-free perception path. Typed `consumes`/`produces` ports prevent an
STT transcript node from being silently substituted for an embedding node.

All ten currently registered nodes have a local `runtime.py`, dependencies,
and installed immutable assets under their own ignored `weights/` directory.
`profiles.json` explicitly enables or disables every node. Rejected nodes may
only appear in a profile carrying `allow_rejected: true`.

Kokoro is a genuine neural TTS node, but its declared `text` input is the live
production bottleneck: Gemma's language tokens are decoded to text before
Kokoro. `gemma-minicpm-hidden-talker` is the measured experimental alternative.
It consumes Gemma token IDs and `cognition_hidden_states_2560` without decoded
text, emits speech codes, clones a reference voice, and decodes those codes to
waveform. Its 12-case result is 10/12 semantic answers, 93.52% mean spoken-word
accuracy, and 2040.9 ms median first audio, so it is not the latency default.

That experimental node still uses an offline integer lookup between Gemma's
vocabulary and the inherited MiniCPM/Qwen-paired Talker vocabulary. Continuous
Gemma hidden state is preserved, but the final native design will train the
Talker directly against Gemma token embeddings and hidden states and remove
the lookup. The earlier direct-conditioner pilot scored only 1/4, so it is not
registered as a working node.

CampPlus remains here because learned speaker embeddings and labels carry
conversation identity into cognition. Silero VAD and Speex AEC are in
`../gates/`: Silero happens to use a neural classifier but only admits PCM;
Speex is adaptive DSP operating on microphone and playback-reference PCM.

`KOKORO_HIDDEN_CONDITIONING.md` records the exact trainable seam for a future
low-latency Gemma-conditioned Kokoro variant: monotonic phoneme planning plus
duration, pause, F0, energy/noise, and style conditioning. It is deliberately
not registered as a node until trained weights and free-running evidence exist.
