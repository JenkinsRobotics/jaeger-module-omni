# Experimental single-brain multimodal demo

This entrypoint accepts local images, raw audio and text in one persistent MLX
Gemma-4 E4B session. It uses the frozen medium ear and frozen vision projector.
Kokoro can save each spoken reply. No caption or transcript bridges the inputs.
The production profiles and GUI are unchanged.

Create `turns.json` next to your image and audio files:

```json
[
  {"image": "picture.png", "text": "Describe what you see."},
  {"audio": "follow-up.wav"},
  {"text": "Please correct the label to cedar."},
  {"image": "another-picture.png", "audio": "question.wav"}
]
```

Run from the repository root, with no other GPU job active:

```sh
training/run_gpu.sh research/omni_multimodal_demo.py \
  --turns turns.json --speak --output artifacts/my-multimodal-demo.json
```

Paths in the JSON are relative to that JSON file. Each turn can combine image,
audio and text. All turns share the same brain and cache. Start a new process
for a fresh conversation. The report records input hashes, answers, vision
encoding time, complete-text request time and MLX peak memory. Spoken WAVs are
saved under `artifacts/my-multimodal-demo.audio/`; saved speech does not prove
physical playback delivery.

The initial research probe passes 20/20 synthetic shape, color and label turns.
This does not establish natural-image accuracy, tool completion, MiniCPM
superiority, or production readiness. Image attention currently uses the causal
language-model path. Long-context rolling and live camera capture are not
implemented in this experimental entrypoint.

Evidence: [shared-cache probe](../research/candidate_evals/joint-vision-probe-20260908.json).
