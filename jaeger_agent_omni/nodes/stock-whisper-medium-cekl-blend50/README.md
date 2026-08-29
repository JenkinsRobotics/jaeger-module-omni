# stock-whisper-medium-cekl-blend50

Complete installed node layout:

```text
node.json
runtime.py
requirements.txt
weights/adapter.npz
weights/encoder/{config, tokenizer, processor, model.safetensors}
```

This node consumes 16 kHz mono PCM and emits Gemma-width audio embeddings. It
does not run Whisper's decoder or hand text to Gemma. The adapter and encoder
are a pinned pair; `node.json` hashes and runtime validation reject substitution.

Offline candidate: 11/12 cases, 11.20% dev-clean64 WER. It is not the live
default until the positive-room duplex gate passes.
