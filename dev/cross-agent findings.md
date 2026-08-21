# Findings from the parallel Gemma-backbone experiment

A second agent worked the same problem from the opposite direction in
`VoiceLLM-Playground/Omni Model/gemma-backbone/`: instead of driving Gemma's
NATIVE audio tower (our path), it replaced MiniCPM-o's Qwen3-8B backbone with
gemma-4-E4B and trained an adapter to bridge MiniCPM's audio projector into
Gemma's embedding space.

Reviewed 2026-08-21. Verified their two load-bearing technical claims against
our own model before adopting anything.

---

# Verified against our model — both true, both matter

```text
embed_scale = 50.596 (= sqrt(2560)) is applied to TEXT embeddings.
    Raw external audio embeddings must NOT be multiplied by it. Our streaming
    path is correct by construction because we scatter through
    get_input_embeddings rather than splicing embeddings ourselves — but this
    is precisely the class of bug that produced our "fluent garbage" phase.

per_layer_inputs at audio positions == per_layer_inputs of pad token 0.
    Confirmed exactly (max diff < 1e-5). We discovered per_layer_inputs
    empirically after fluent-garbage output; they documented WHY it is shaped
    that way. Anyone hand-building E4B input needs both facts.
```

# Their result, and why it is significant

Stage-1 LLaVA-style alignment: MiniCPM APM features -> trained linear ->
frozen 4-bit Gemma, next-token CE on LibriSpeech transcripts.

```text
init (W_fwd ridge map)   held-out CE 5.8686
after 40 steps / 32 utts held-out CE 1.1368,  train CE 0.7380
```

Held-out greedy probes show genuine acoustic learning, not memorisation:

```text
expected  "HE WAS IN A FEVERED STATE OF MIND ..."
generated "WE WAS IN A FEVERED STATE OF MIND ..."
expected  "MANY LITTLE WRINKLES GATHERED BETWEEN HIS EYES ..."
generated "WE NEED TO FIND QUICKLY GATHERED BEFORE WE SEE HIS EYE'S ..."
```

**40 steps on 32 utterances.** Our delta ran 2000 steps on 6615 turns to reach
a teacher-forced 0.939 whose free-running score was 33%. Theirs is a far
better data-efficiency signal, and the difference is instructive: they train
into a 4B backbone, we trained into a 135M one.

# What we should adopt

```text
1. TRAIN THE DELTA AGAINST E4B, NOT SmolLM2.
   Their pilot is the evidence that the architecture works at 4B scale with
   trivial data. Our 33% free-running was very likely the 135M backbone, not
   the bridge. This was already our next step; their result raises confidence
   from "worth trying" to "expected to work".

2. PRECOMPUTE APM FEATURES TO DISK.
   They cache AvgPool5(ReLU(linear1(Whisper(audio)))) as [T,4096] at 10 Hz and
   note that moving the affine linear2 AFTER pooling is exact. We already
   built encoder_cache.py for the same reason; their boundary choice (cache
   pre-linear2) is cleaner than ours (cache post-tower) because it lets the
   trained adapter be spliced straight back into linear2.

3. THE 660 MB AUDIO GGUF IS SUFFICIENT.
   They confirmed the F16 audio GGUF holds the exact APM + projector weights,
   so the 20 GB HF checkpoint is unnecessary. We reached the same conclusion
   via inspect_gguf.py; independent confirmation.

4. SPLICE-BACK VERIFICATION.
   surgery.py writes the trained adapter into the GGUF and reads it back
   bit-for-bit (max error 0). Our artifact export should assert the same
   round-trip property — currently we verify state_dict equality in Python but
   never a written-then-reread comparison.
```

# What we should NOT adopt

```text
Replacing MiniCPM's backbone wholesale.
    Their TTS path is blocked on three hardcoded 4096 shapes in the engine
    (omni.cpp ~1472, ~1790-1900, 1937). Ours sidesteps this entirely by using
    Gemma's native towers for input and treating MiniCPM's talker as a
    separate output stage.

The ridge-regression alignment map as a starting point.
    W_fwd holdout cosine 0.68 / W_rev 0.59 — and their own verdict is that a
    linear remap alone is "too lossy"; the model parroted the Qwen chat
    template until the adapter was TRAINED. Confirms our finding that
    embedding spaces do not transfer without learning.
```

# Gotchas they paid for that we should not repeat

```text
ctypes without argtypes truncates 64-bit pointers -> fake "engine segfaults"
n_ctx = 0 makes the engine silently generate nothing
duplex init forces LISTEN for the first 3 decodes ("force_listen 1/3") —
    we hit exactly this from the other side; use duplex=False for simplex tests
duplex REQUIRES wall-clock pacing; burst-feeding never produces SPEAK
    (our DuplexLoop already paces, and our own burst test produced the same
    silent failure before we fed audio at true realtime)
the engine defaults to Chinese prompts; set language before any prefill, and
    the reference audio's language leaks into replies
```

# Cross-check: where the two experiments agree

Both independently found that MiniCPM/Whisper front ends are FINE-TUNED, not
stock; that a linear map between embedding spaces is insufficient without
training; that duplex forces listen for 3 decodes and needs real-time pacing;
and that the audio GGUF alone is enough to work with. Two different attack
angles, same conclusions — which is the strongest evidence either of us has.
