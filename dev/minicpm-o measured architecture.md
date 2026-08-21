# MiniCPM-o 4.5 — Measured Architecture

Answers to the questions in `dev/reference architecture minicpm-o 4.md`, read
out of the artifact at `~/models/MiniCPM-o-4_5-gguf` rather than from papers or
recollection.

Method: `research/inspect_gguf.py` reads GGUF headers (tensor names, shapes,
metadata) without loading weights; the shipped 85 KB README for API shape; and
`~/GITHUB/llama.cpp-omni/tools/omni/audition.cpp` for the actual forward pass.

Four confidence levels are marked throughout:

```text
MEASURED   read directly from tensor headers or model metadata
SOURCE     read from the llama.cpp-omni implementation
API        visible in the shipped README's documented calls
UNKNOWN    not answerable from what we have
```

---

# Component inventory

```text
MiniCPM-o-4_5-Q8_0.gguf              8.7 GB   Qwen3 backbone
audio/MiniCPM-o-4_5-audio-F16.gguf   660 MB   Whisper encoder + audio projector
vision/MiniCPM-o-4_5-vision-F16.gguf 1.1 GB   CLIP/SigLIP2 vision tower
tts/MiniCPM-o-4_5-tts-F16.gguf       1.2 GB   the talker (llama-arch, 20 blocks)
tts/...-projector-F16.gguf            15 MB   LLM hidden -> talker
token2wav-gguf/encoder.gguf          151 MB   upsample_encoder_v2
token2wav-gguf/flow_matching.gguf    458 MB   flow-matching decoder
token2wav-gguf/flow_extra.gguf        14 MB
token2wav-gguf/hifigan2.gguf          83 MB   vocoder
token2wav-gguf/prompt_cache.gguf     212 MB   voice identity + decoder state
```

Five separate neural systems, shipped as separate files, assembled at runtime.
Note this is itself an architectural statement: **the "end-to-end model" is a
set of independently loadable components**, which is the same posture as our
slot design.

---

# The numbers

## Cognitive backbone — MEASURED

```text
general.architecture                 qwen3
general.size_label                   8.2B
block_count                          36
embedding_length                     4096          <- the projector's target
feed_forward_length                  12288
attention.head_count                 32
attention.head_count_kv              8             GQA
context_length                       40960
rope.freq_base                       1000000.0
tokenizer                            gpt2 / qwen2 pre
chat_template                        stock Qwen tool-calling
                                     (<tools>, <tool_call> XML)
```

The chat template is **unmodified Qwen tool-calling**. No multimodal tokens in
the template — modality splicing happens below the template layer.

## Audio encoder — MEASURED

```text
general.architecture                 whisper
encoder_attention_heads              16
d_model                              1024
encoder.conv1.weight                 [3, 80, 1024]   80-bin mel frontend
encoder.conv2.weight                 [3, 1024, 1024] stride-2 downsample
371 tensors                          = whisper-medium encoder, decoder absent
```

The Whisper **decoder is not shipped**. Confirms the ASR path is not used at
all — only the encoder's hidden states.

## Audio projector — MEASURED

```text
audio_pool_step                      5              50 fps -> 10 tokens/s
audio_projector.linear1.weight       [1024, 4096]
audio_projector.linear1.bias         [4096]
audio_projector.linear2.weight       [4096, 4096]
audio_projector.linear2.bias         [4096]
                                     21.0M params
```

Forward pass, from `llama.cpp-omni/tools/omni/audition.cpp` ("Step 6: audio
projector") — SOURCE, no longer inferred:

```text
encoder -> ln_post -> linear1 -> ReLU -> linear2 -> AvgPool(k=5, s=5)
```

```cpp
cur = ggml_mul_mat(ctx0, model.whisper_proj_1_w, cur);
cur = ggml_add(ctx0, cur, model.whisper_proj_1_b);
cur = ggml_relu(ctx0, cur);                              // ReLU, not GELU
cur = ggml_mul_mat(ctx0, model.whisper_proj_2_w, cur);
cur = ggml_add(ctx0, cur, model.whisper_proj_2_b);
cur = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, 5, 5, 0);  // AFTER the linears
```

Three details that are easy to get wrong:

```text
activation      ReLU, not GELU
normalisation   NONE inside the projector. The norm ahead of it is Whisper's
                own final ln_post, part of the encoder.
pool position   AFTER both linears, not before. The linears run at the full
                50 fps and averaging happens in projected space — 5x the
                projector FLOPs, and what the trained weights expect.
token count     (in - k)/s + 1, floor    (audition.cpp:1385, :2059)
```

**Two linears with a ReLU, then average pooling. That is the entire audio
interface.**

No perceiver resampler. No learned latent queries. No attention pooling. No
cross-attention. The projector cannot select or reject anything — it averages
groups of five frames and applies two matrices. Whatever filtering of
irrelevant audio happens, happens in Qwen's own attention downstream.

This is the single most useful finding for us: the component we were most
tempted to over-build is, in the reference implementation, the simplest thing
that could work.

## Vision tower — MEASURED

```text
general.architecture                 clip
clip.vision.embedding_length         1152           SigLIP2-so400m shape
clip.vision.block_count              27
clip.vision.attention.head_count     16
```

## Talker — MEASURED

```text
general.architecture                 llama
general.name                         Tts
block_count                          20
embedding_length                     768
attention.head_count                 12
```

The talker is **a small LLM in its own right** — 20 blocks, 768 wide. It is not
a vocoder; it is an autoregressive model that consumes projected cognitive
state and emits speech tokens.

## Talker projector — MEASURED

```text
linear1.weight                       [4096, 768]    Qwen hidden -> talker
linear2.weight                       [768, 768]
                                     3.7M params
```

Same two-linear pattern as the audio side, mirrored. **The hidden-state ->
speech interface is 3.7M parameters.** This substantially lowers our estimate
of what native speech output would cost architecturally — the difficulty is in
training it, not in its size or shape.

## token2wav — MEASURED

```text
upsample_encoder_v2   ->  flow_matching  ->  hifigan2
                              458 MB          83 MB
```

CosyVoice2's decoder stack: speech tokens through a flow-matching model to mel,
then a HiFi-GAN vocoder to waveform.

## Voice identity — MEASURED

```text
prompt_cache.spk_cb                  [192]                  speaker embedding
prompt_cache.conformer_cnn_cache     [512, 6]
prompt_cache.conformer_att_cache     [128, 302, 8, 10]
prompt_cache.estimator_cnn_cache     [1024, 2, 80, 2]
prompt_cache.estimator_att_cache     [128, 302, 640, 2]
                                     52.9M floats, 212 MB
mtmd.prompt_cache.chunk_main         25
mtmd.prompt_cache.chunk_total        28
mtmd.prompt_cache.up_rate            2
```

**A voice is a 192-float speaker vector plus precomputed decoder caches.**

Built once from reference audio, then reused for every generation. This
confirms at tensor level what the API implies: voice identity is a *load-time
property*, not a per-utterance cost. It also explains why cross-turn timbre
consistency should hold — every turn conditions on the identical cached vector.

---

# Streaming and duplex — API

From the shipped README:

```python
sys_msg = model.get_sys_prompt(ref_audio=ref_audio, mode="omni", language="en")
model.streaming_prefill(session_id=session_id, msgs=[sys_msg])

# "For each 1s audio chunk, perform streaming_prefill once
#  to reduce first-token latency"
for chunk in chunks:
    model.streaming_prefill(session_id=session_id,
                            msgs=[{"role": "user", "content": [chunk]}],
                            omni_mode=False,
                            is_last_chunk=is_last)

iter_gen = model.streaming_generate(session_id=session_id, generate_audio=True)
```

Facts this establishes:

```text
chunk size            1 second
prefill               incremental, per chunk, during user speech
                      -> this is where the latency win comes from
state                 session_id keyed
turn boundary         explicit is_last_chunk flag
voice                 set in the system prompt at session init
```

## The streaming encoder — SOURCE

`audition.cpp:396-397` settles how the 30-second window is handled:

```cpp
const int n_audio_ctx = hparams.n_ctx;      // 1500
const int n_iter = n_audio_ctx / n_tokens;  // 1500 / 50 = 30
```

The Whisper encoder runs with a **rolling KV cache of 1500 frames**, consuming
50 new frames (1 second) per step. Thirty steps fill the 30-second context.

So Whisper's fixed window is not worked around by padding, and not by a
modified encoder — it becomes the *depth of a streaming cache*. Each second of
audio costs one second of encoder compute, while the model still attends over
the previous 30 seconds. This is the design Stage 5 should copy directly, and
it is considerably better than the chunked-with-restart approach we had
assumed.

Multimodal time alignment is **interleaving, not fusion**:

```python
for i in range(len(video_frames)):
    omni_contents.append(video_frames[i])
    omni_contents.append(audio_segments[i])
```

Modalities are appended to one time-ordered content list and prefilled in
order. There is no cross-modal fusion module — sequence position carries the
temporal relationship.

---

# Turn-taking is LEARNED — SOURCE

`tools/omni/omni.h:460-469`. This is the most important finding in the file and
it overturns an earlier assumption in our plan.

```c
llama_token special_token_speak;        // <|speak|>          model starts speaking
llama_token special_token_listen;       // <|listen|>         model starts listening (duplex)
llama_token special_token_chunk_eos;    // <|chunk_eos|>      semantic chunk end
llama_token special_token_chunk_tts_eos;// <|chunk_tts_eos|>  TTS chunk end
llama_token special_token_turn_eos;     // <|turn_eos|>       turn end
llama_token special_token_unit_end;     // </unit>            duplex chunk boundary
llama_token special_token_tts_pad;      // <|tts_pad|>        sampling forbidden in duplex
```

**Whether to speak or keep listening is a token prediction.** At chunk
boundaries the model emits `<|speak|>` or `<|listen|>`, and turn-taking is
therefore a learned behaviour trained into the backbone's vocabulary — not a
VAD threshold, not a caller decision, not application-level cancellation.

`<|tts_pad|>` ("sampling forbidden in duplex mode") means the model emits
something on every step even while silent. That is closer to frame-synchronous
operation than a request/response loop.

Output side, for symmetry: `omni.h:76` — 25 speech tokens per chunk, matching
`mtmd.prompt_cache.chunk_main = 25` in the prompt cache metadata.

## Corrections this forces on our plan

Two earlier claims in `dev/jaeger agent omni0 immediate development.md` were
wrong and have been fixed there:

```text
WAS   "MiniCPM's full duplex is chunked-with-cancellation; frame-synchronous
       is a different family."
IS    Learned turn-taking via <|speak|>/<|listen|>, generating every step.
       Meaningfully closer to frame-synchronous.

WAS   "is_last_chunk is a caller argument, so turn-end is probably the
       application's problem in every architecture."
IS    Wrong. is_last_chunk drives prefill bookkeeping; the turn decision is
       the model's, made in token space.
```

## What this means for Jaeger — the one thing we cannot copy

Replicating `<|speak|>`/`<|listen|>` requires **new vocabulary trained into the
backbone on full-duplex conversational data**. Gemma has no such tokens and no
such training, and adding them is not a projector-scale job.

So we replicate the *function*, not the *mechanism*: a small **turn-end head**
on top of E4B's hidden state (or the projector output) predicting
"user still speaking" vs "my turn now." Trainable from our own captured
sessions, because a logged turn boundary is exactly that label — `capture.py`
is already recording it.

This upgrades the Stage 5 turn-end head from a nice-to-have to **the component
that determines whether the system feels conversational**. It is the single
piece of MiniCPM's behaviour that no off-the-shelf part supplies, and the
reference implementation confirms it is worth having rather than approximating
with silence timeouts.

---

# What Jaeger adopts

```text
ADOPT AS-IS
  pool(stride 5) + 2 linears           audio projector — already implemented
  10 audio tokens/second               the compression ratio
  whisper encoder only, no decoder     ASR path unused
  1-second streaming chunks            solves the 30 s padding problem
  incremental prefill during speech    the actual latency mechanism
  session-keyed state + last-chunk flag
  voice as a load-time cached vector   maps onto our Soul binding
  interleaving for time alignment      not a fusion module
  <audio_bos> embeds <audio_eos>       marker tokens bracket spliced
                                       embeddings (mtmd.cpp:526); some models
                                       in the same file use none at all

REPLICATE THE FUNCTION, NOT THE MECHANISM
  <|speak|> / <|listen|> tokens   ->   small turn-end head over E4B hidden
                                       state. New backbone vocabulary is out
                                       of reach; the behaviour is not.

ADAPT
  backbone Qwen3-8B (4096)   ->  gemma-4-E4B (hidden TBC from Stage 0)
                                 projector output dim changes to match;
                                 architecture unchanged
  stock tool-calling template ->  Gemma's template, already working in
                                 jaeger-agent

DO NOT ADOPT
  talker + token2wav stack     ->  Track A cloning TTS instead.
                                   Their speech output is the weakest part
                                   for character work, and we need Mochi's
                                   voice, not a generic one.
  two outputs (text, audio)    ->  we add a third: avatar, off PerformanceSpec
```

---

# Still UNKNOWN — needs the Python source

The artifact cannot answer these. If they matter, read the modeling code.

```text
training stages          which components frozen when; not recoverable from
                         a released checkpoint. This is the gap that matters
                         most and the artifact cannot close it.
self-hearing             how generated audio is prevented from re-entering as
                         user input
duplex data              what conversational corpus taught <|speak|>/<|listen|>
```

Everything else in the original question set is now answered above.
