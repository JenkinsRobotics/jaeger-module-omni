# Deprecated artifacts — do not use, do not cite

## audio-delta__minicpm-o-4.5__smollm2-135m

Trained against `HuggingFaceTB/SmolLM2-135M-Instruct`, not Gemma E4B. A
projector is valid **only** for the backbone it was trained against, so this
adapter cannot be loaded by the pipeline and its scores describe nothing we
ship.

It is kept for provenance because its numbers were quoted, and the record
should show what they actually were:

```text
reported     "90+ accuracy"   -> 0.939 TEACHER-FORCED token agreement
free-running  4/12 = 33%      -> the same checkpoint, answering on its own
checkpoint    agreement 0.0 at steps 1600 / 2000 / 2400
```

Teacher-forcing hands the model the correct previous token at every step, so
it measures "predict token N+1 given a perfect prefix". A model can score 94%
there and collapse the instant its own output feeds back — which is the only
mode that runs in production. Report free-running numbers.

The 135M stand-in existed because two E4B config-shape bugs made the real
backbone look untrainable, not because of hardware:

```text
per_layer_inputs   E4B feeds 256 dims per token per layer; passing embeds
                   without ids sends transformers into a 29 GiB dead end
hidden_size        Gemma4Config is multimodal -- it lives on get_text_config()
```

Both are fixed. E4B trains here at 0.58 s/step in 16.6 GB.
