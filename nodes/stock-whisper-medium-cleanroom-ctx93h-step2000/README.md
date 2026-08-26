# stock-whisper-medium-cleanroom-ctx93h-step2000

Controlled Qwen-free candidate imported byte-for-byte from the clean-room
campaign. This is an independent neural node, not an overwrite of the retained
50M medium connector.

```text
PCM 16 kHz
  -> pinned stock OpenAI Whisper-medium encoder (decoder not loaded)
  -> LayerNorm 1024 -> Linear/GELU 2048 -> mean-pool/2
  -> 8-head bidirectional self-attention + residual LayerNorm
  -> 4096-wide GELU FFN + residual LayerNorm
  -> Linear 2560
  -> frozen Gemma-4 E4B
```

The adapter has 40,915,456 parameters and was trained on 93 hours / 42,498
clips / 744 speakers, including 37% RIR/MUSAN room degradation. On the old
bridge's exact 128 clips / 2,041 words it measured 3.234% WER with the
instruction used by the application and 5.047% with an empty prompt. The same
material has only two speakers. Cases12, independent lockboxes, room/meeting,
and five-minute duplex validation are deliberately still null in `node.json`.

After import, the canonical spoken-question gate scored **12/12 at 572 ms
median**. The independent diagnostic generation produced all 67 spoken words
with **0.00% generated-transcript WER** and again answered 12/12. That
transcript is Gemma's diagnostic output, never an input handoff.

Select without changing the default:

```bash
python -m jaeger_agent_omni.omni --bench research/cases12 \
  --front-end stock \
  --adapter nodes/stock-whisper-medium-cleanroom-ctx93h-step2000
```
