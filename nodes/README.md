# Audio model nodes

Each directory is an inseparable encoder/adapter/backbone contract. Passing a
node directory anywhere an adapter path is accepted loads `node.json`, verifies
the adapter SHA-256, then checks encoder identity, encoder boundary width,
pooling, and adapter architecture against the checkpoint before model loading.

Large immutable model and adapter weights remain in the repository's existing
artifact/model stores and are referenced by pinned identity and hash. Copying
hundreds of megabytes into every folder would not make the pairing safer.
Likewise, shared neural code remains versioned once under
`jaeger_agent_omni/runtime`; duplicating it per node would create divergent,
unreviewable implementations. A node pins the exact implementation symbol and
architecture name it requires.

Status is explicit. A rejected experimental node remains reproducible but is
never selected as a live default.

The registry stays flat; capability is typed in each manifest:

```text
nodes/
  stock-whisper-medium-cekl-blend50/
  stock-whisper-large-v3-turbo50m/
  whisper-medium-stt/                 (when installed)
  whisper-large-v3-turbo-stt/         (when installed)
  kokoro-tts/                          (when installed)
  campplus-speaker/                    (when installed)
  silero-vad/                          (when installed)
  speex-aec/                           (when installed)
  profiles/
```

Medium and large Whisper perception nodes are different nodes because their
encoder boundaries and trained adapters differ. Medium and large Whisper STT
decoders are also different nodes, but they are not components of the
transcript-free perception path. Typed `consumes`/`produces` ports prevent an
STT transcript node from being silently substituted for an embedding node.
