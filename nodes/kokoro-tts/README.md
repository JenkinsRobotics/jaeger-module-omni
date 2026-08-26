# kokoro-tts

Current production speech-output node. The CLI, live worker, and full-duplex
demo load this exact node and keep its model resident for clause streaming.
Its packaged snapshot includes every voice exposed by
`jaeger_agent_omni.voice`.

It consumes text and is therefore the simple, low-latency production fallback,
not a claim of hidden-state speech generation. Rich hidden conditioning and
voice cloning remain separate future work.
