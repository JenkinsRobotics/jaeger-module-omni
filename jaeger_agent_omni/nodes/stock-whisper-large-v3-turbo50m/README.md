# stock-whisper-large-v3-turbo50m

Complete installed node layout:

```text
node.json
runtime.py
requirements.txt
weights/adapter.npz
weights/encoder/{config, tokenizer, processor, model.safetensors}
```

This node is retained as a rejected, reproducible experiment. On the same 12
waveforms its stock decoder made 0/67 lexical errors, but adapter-to-Gemma
transcription made 63/67 errors and answering scored 1/12. It cannot be enabled
in a production profile. Its failure belongs to this paired connector, not to
the large-v3-turbo decoder control.
