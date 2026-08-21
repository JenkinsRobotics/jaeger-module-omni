# STATUS

> Truthful, or it's worthless. Any commit that changes behavior updates
> this file in the same commit — see `CONVENTIONS.md` → "STATUS stays
> truthful."

## Current state — 2026-08-19

- **Repo:** `jaeger_agent_omni` — native multimodal neural agent for JaegerOS
- **Version:** 0.1.0
- **JaegerOS pin:** not set. Nothing is discoverable yet; `module.yaml`
  declares the intended contract, not a working one.
- **Python:** 3.11.9 (`.python-version`), torch 2.13, transformers 5.15

### It runs

Voice in, cognition out, end to end on real models — verified 2026-08-19 on
CPU with `openai/whisper-tiny` + `HuggingFaceTB/SmolLM2-135M-Instruct` and a
1.3 s speech clip:

```text
TEXT PATH  (works today; also the Stage 2 teacher)
  input       "What is two plus two?"   (real speech)
  transcript  "What is 2 plus 2?"
  response    "2 + 2 is a classic arithmetic operation that can be
               understood by people with a basic understanding of numbers..."
  timing      transcribe 390ms + generate 990ms = 1380ms
  captured    1 turn written to the corpus

OMNI PATH  (plumbing proven; weights untrained)
  transcript  none — never computed
  response    coherent-shaped nonsense, as expected
  timing      encode 181ms + project 3ms + generate 800ms
```

The measurement that matters: **Whisper's 508ms autoregressive decode becomes
181ms of encode-only.** Generation times are not comparable between paths (they
produced different token counts), but eliminating the ASR decode pass is real
and measured. The remaining latency is dominated by generation, exactly as
predicted — which is why Stage 5's incremental prefill, not the projector, is
where the conversational win lands.

### Three-way audio comparison — 2026-08-20 night

Same clip ("What is two plus two?", synthesised speech) into three systems:

```text
gemma-4-E4B NATIVE audio    "What is two plus two"      perfect, zero training
  (ggml-org Q4 GGUF + mmproj, llama-mtmd-cli --jinja; the E4B audio tower is
   streaming-causal: attention_context_right=0)

our delta @ step 1000       "The answer is 23."         understood it was
  (MiniCPM front end -> 2.4M delta -> SmolLM2-135M)     arithmetic; answered
                                                        (old assistant prompt),
                                                        wrong number — 135M
                                                        backbone, mid-training

MiniCPM-o 4.5 (C++ defaults) Chinese persona self-intro — the test harness
  hardcodes a zh voice-clone system prompt and uses the input clip as the
  VOICE SAMPLE. Not a fair content read; redo via llama-omni-server with an
  EN prompt.

ultravox-v0_6-gemma-3-27b   queued (GGUF exists; too big beside training)
```

**Consequence, and it is the big one: the production input side is already
solved by Google.** E4B's native tower transcribed perfectly from a Q4 GGUF
with no adapter and no training, and it is built causal/streaming. The local
MLX 4-bit copy is text-only (963 missing tower params) — the ggml-org GGUF +
mmproj is the runnable native path today.

Where our work redirects, with the input side native:

```text
speech OUT        nothing native exists. Talker + output delta remains the
                  only path to their expressive cloned voice — and the whole
                  delta pipeline just proved itself on the input side.
paralinguistics   test whether native E4B audio hears tone/emotion or is
                  ASR-flavoured; the MiniCPM front end stays as the
                  paralinguistic branch if so.
duplex runtime    turn-end, barge-in, PerformanceSpec, avatar — unchanged.
harness lessons   embed_norm, prompt provenance, prefix augmentation, WER
                  evals — all transfer to the output delta.
```

Lane B's own result is quietly respectable: audio -> "this is an arithmetic
question, answer it" through a 135M backbone and a 2.4M single-linear delta at
step 1000. The architecture works; the backbone is just tiny.

### Zero-conditioning disproven; hidden states are load-bearing — 2026-08-20 late

Second synthesis attempt with the FULL measured cadence — audio_bos (151687),
text_eos (151692), code EOS (6561), their exact sampling (t 0.8 / top_p 0.85 /
top_k 25 / rep 1.05 / min 100 tokens) — still produces fluent babble:
whisper hears "I'm not going to be a good one" for "We choose to go to the
moon." Same failure as attempt one, now with the cadence ruled out.

Conclusion, measured twice: **the talker does not follow text without the
L2-normalised hidden-state term.** It was trained with that term at every
position; text embeddings alone are out-of-distribution and it improvises
fluently. The output side therefore has a hard dependency order:

```text
1. real Qwen3-8B hidden states  -> verify the talker follows text (one-time
                                   download, also the output-delta teacher)
2. output delta                 -> backbone hidden -> Qwen tts space, trained
                                   on paired text through both models
```

Silver lining: babble in a cloned voice through the whole ported stack is the
strongest possible verification of the code->audio half.

### First audible speech out — 2026-08-20 evening

`text -> talker (our port) -> speech codes -> token2wav (their service) -> wav`
produced 14.2 s of **intelligible cloned-voice English**, round-tripped through
whisper. It said the wrong words — fluent speech, not the requested text — and
that split is the diagnosis:

```text
token2wav      VERIFIED  codes -> fluent English in the reference voice.
                         Driven via OpenBMB's own pyt2w JSON service from an
                         isolated venv (.venv-t2w); no flow-matching port.
talker port    ALIVE, conditioning incomplete. Zero-conditioning free-runs in
                         speech space instead of reading the text. The missing
                         piece is their input cadence: 25-token text/code
                         interleaving with <|tts_bos|>/<|chunk_eos|>, plus
                         proj_spk speaker conditioning — all readable in
                         omni.cpp.
```

Shims that made it run on a Mac (research/t2w_bootstrap.py): .cuda() no-ops +
device='cuda' factory remaps (stepaudio2 hardcodes both), ruamel.yaml pinned
to 0.17 for HyperPyYAML. Flow constraints measured: reference audio and each
generation window cap at 1000 mel frames (~10 s), hence 25-token streaming
chunks with the service cache carrying state — the same cadence as their C++.

### Copy-their-baseline architecture + demo GUI — 2026-08-20

Direction change, at Jonathan's call, and the measurements back it:

**Their whisper encoder is NOT stock** — 17-31% relative weight changes from
OpenBMB's alignment training. So "copy their baseline" means both halves:
`research/load_minicpm.py` lifts their fine-tuned encoder (307M, all 371
tensors remapped into a stock HF WhisperEncoder) and their projector (21M)
out of the GGUF. Their full pairing in our code: cross-clip cosine **0.8607**,
more discriminative than stock-encoder mixing (0.9052).

New architecture (`--arch minicpm_delta`, now the default):

```text
audio -> THEIR encoder (frozen) -> THEIR projector (frozen) -> Qwen space
                                                                   |
                                                          DeltaAdapter  TRAINED
                                                                   |
                                                            our backbone
```

The delta (`neural/delta.py`) maps LLM-space to LLM-space — a far smaller ask
than audio-to-language from scratch, and front-end collapse is structurally
impossible because the front end is not trainable. embed_norm pinning kept.

**Version check** (asked and answered): MiniCPM-o 4.5 (Feb 2026) is the newest
omni model. The `OpenBMB/MiniCPM` repo is their text-only line — MiniCPM5-1B
has no audio. Our source is current.

**Demo GUI** — `demo.py`, one file, stdlib server, verified headless:

```sh
python demo.py --backbone <hf-id>        # http://localhost:7860
```

Mic stays open (browser AEC), RMS VAD with the legacy 700 ms hangover, barge-in
cancels speech, text box, conversation history, replies spoken via browser
speechSynthesis (stand-in for Track A TTS). Camera is preview-only and labeled
so — vision lands at Stage 7, nothing fake happens to frames. Curl-tested:
text turn, real-speech audio turn, history follow-up all work.

### Architecture verified against MiniCPM's own weights — 2026-08-20

The decisive experiment, and it should have been run first.

MiniCPM's trained audio projector was lifted out of its GGUF and loaded into
**our** `JaegerAudioProjector` with no adaptation — the shapes matched the port
exactly (21.0M params, 1024 -> 4096 -> 4096).

```text
cross-clip cosine (1.0 = every clip identical, mean-pooled over 8 clips)

  whisper-medium encoder output   0.9097
  MiniCPM projector output        0.9052   preserves discriminability
  our trained projector           0.9903   destroys it
```

**The port is faithful.** Their weights come out slightly *more* discriminative
than the encoder states they were given, running in our code. ReLU, no
LayerNorm, pool-after — all correct.

**Our training is what collapses the representation.** The projector learned to
emit a near-constant and let the backbone produce generic text, because at our
scale that minimises loss better than a weak audio signal does.

A note on the metric: speech shares enough structure that ~0.91 is the natural
floor for mean-pooled encoder states. The scale is compressed near 1.0, so the
meaningful comparison is a projector against **its own input**, never against an
absolute threshold. An earlier version of this tool used a fixed 0.9 and would
have reported MiniCPM's own weights as collapsed.

What this reprioritises: several rounds of "needs more data" were wrong.
More data cannot help a projector that has learned to ignore its input. The
levers are capacity (554K vs their 21M), encoder strength (whisper-tiny's states
start at 0.866 diversity, medium's at 0.910), and an objective that rewards
discrimination.

### End-to-end measured — 2026-08-20

The pipeline runs. The model does not yet work. Both halves are true and the
numbers say which is which.

Text vs omni on 20 held-out JFK turns, 120 reference words, projector at step
156 (whisper-tiny + SmolLM2-135M):

```text
text path   WER   6.67%
omni path   WER 591.67%
```

The omni output is topically right and specifically wrong — "the first American
space program to be launched" where the reference is "Senator Wiley and
Congressman Miller." At this scale the projector has learned *this audio is the
moon speech* and the backbone confabulates from there.

WER above 100% is legitimate: insertions exceed the reference length. That
pointed at a real bug — **labels carried no EOS**, so the model was never
taught to stop and ran to `max_new_tokens` every time. Fixed in `dataset.py`;
targets now end with EOS, teacher inputs do not.

The other gap is exposure bias, and it is the number to watch going forward:

```text
target_acc 0.33   teacher-forced — given the correct prefix, the next token is
                  right a third of the time
WER 591%          free-running — the first error compounds into the next, with
                  no correct prefix to recover from
```

A teacher-forced accuracy that would look encouraging on a loss curve is
nowhere near enough to survive free generation. Any future claim that the
projector "works" has to be a free-running number.

Scale is the honest explanation: ~11 minutes of single-speaker audio, 224
training turns, 156 steps, a 554K projector on a 135M backbone. Published
speech-LLM work uses thousands of hours.

### Real speech, held-out — 2026-08-20

Stage 3 warmup on the JFK corpus (288 imported segments, 260 above the 0.8
confidence gate, 224 train / 36 val), whisper-tiny + SmolLM2-135M, MPS:

```text
loss         6.63 -> 2.43
target_acc   0.18 -> 0.35     HELD OUT — not seen during training
```

Against 0.91 on the 60-turn synthetic corpus, which was memorisation. The
held-out split was added precisely to expose that gap, and it did on the first
run that used it.

35% of transcript tokens recovered from audio alone is real signal and far
from useful. Scale explains it: ~11 minutes of single-speaker oratory, a 554K
projector, a 135M backbone, whisper-tiny. Published speech-LLM work uses
thousands of hours. The pipeline is correct; the corpus is three orders of
magnitude too small.

Two bugs this run caught, both in the unattended machinery:

```text
English-only whisper   medium.en rejects language/task kwargs. The legacy
                       stack runs medium.en, so this would have hit the first
                       production import. Fixed with transcribe_kwargs().
no checkpoint at all   save_every=2000 never fired in a 300-step run, so a
                       SIGTERM left only a truncated .tmp. Atomic writes
                       protect the previous checkpoint; they cannot conjure
                       one that was never taken. save_every is now clamped so
                       every run checkpoints at least 4 times, and an orphaned
                       .tmp is reported rather than silently ignored.
```

The signal handling itself worked: SIGTERM was caught, the step finished, the
save began. It just had nothing to fall back on.

### First real training run — 2026-08-19

Stage 3 warmup: 150 steps, 60 synthetic turns, whisper-tiny + SmolLM2-135M, MPS.

```text
                      loss             target_acc      grad norm
no output scaling     5.39 -> 3.17     0.76 -> 0.38    47 -> 648    diverged
embed_norm pinned     6.07 -> 0.268    0.66 -> 0.91    24 -> 5.6    converged
```

`target_acc` = transcript tokens predicted from audio alone, teacher-forced,
no text input. On 60 synthetic turns most of it is memorisation; the
**stability** difference is the finding.

Cause: with a frozen backbone nothing constrains the projector's output scale.
It drifted to 4.4x the backbone's mean embedding norm while its weight norms
barely moved — it rotated out of the distribution the backbone was trained on.
MiniCPM does not meet this because it trains the backbone jointly. Recorded in
`neural/audio.py` and the plan doc.

### What works

Sixteen self-checking modules, offline, ~8 s via `research/check_all.py`:

```text
RUNTIME  jaeger_agent_omni/
  performance_spec.py     voice+face intent, validated
  outputs.py              text / speech / tool-call routing
  capture.py              turn log + transcript-confidence gate
  streaming.py            1 s chunking + rolling 1500-frame cache
  neural/base.py          ModalityProjector contract + registry
  neural/audio.py         JaegerAudioProjector (MiniCPM port)
  neural/heads.py         turn-end head + hysteresis

HARNESS  research/
  stubs.py                tiny Whisper/LM stand-ins
  pipeline.py             end-to-end wiring; asserts the projector learns
  inspect_gguf.py         GGUF header reader
  stage0_access.py        real-weights access check (manual, needs download)

TRAINING  training/
  data.py                 tiered corpus mixing policy
  config.py               run config, validated presets
  distill.py              self-distillation losses
  train.py                the loop — runs end to end on stubs
```

Plus the application itself:

```text
jaeger_agent_omni/app.py        OmniApp — both paths, per-stage timing,
                                capture-on-text-path
                                StreamingOmni — prefill during speech
jaeger_agent_omni/__main__.py   python -m jaeger_agent_omni --audio ...
training/import_audio.py        long audio -> corpus, with confidence
training/evaluate.py            text vs omni WER on held-out turns
training/nightly.sh             unattended overnight runs
```

### Streaming prefill

`StreamingOmni` consumes 1-second chunks as they arrive, so encoder and
projector work happens *while the user is still speaking*. `Timing` reports it
as `overlapped_ms` and **excludes it from the total** — that exclusion is the
latency thesis made measurable, and it is what turns the legacy stack's ~950 ms
of endpoint dead time into zero.

One honest limitation: HF Whisper exposes no encoder KV cache, so each chunk
re-encodes the accumulated audio rather than extending a cache. That is
O(n^2) over a turn — fine at conversational lengths, wrong for long-form. The
rolling 1500-frame cache in `streaming.py` is the shape to grow into, and it is
what MiniCPM does (`audition.cpp:396`).

Two load-bearing checks:

- `research/pipeline.py` runs a training loop on stubs and asserts the loss
  falls below the uniform-prior floor `ln(vocab)`. A detached graph or a frozen
  projector fails here rather than in a run.
- `training/train.py --selfcheck` exercises the real loop — optimiser,
  OneCycle schedule, gradient clipping, checkpointing — and asserts the
  backbone stayed frozen while the projector did not.

### What is stubbed

- `jaeger_agent_omni/node.py` — untouched template stub. No bus wiring.
- `jaeger_agent_omni/config.py` — untouched template stub.
- No *real* training has been run and no model weights downloaded.
  `train.py --stub` runs the full loop on tiny random models; `--backbone`
  deliberately refuses rather than guessing, because a projector is only valid
  for the encoder/backbone pairing it was trained against.

### Known gaps / blockers

```text
GEMMA_ID unknown        the app runs against any HF causal LM; it has not
                        been run against gemma-4-E4B because the exact repo
                        id is unconfirmed. cognition_dim is read from the
                        backbone config at runtime, so nothing is hardcoded —
                        this is a one-flag change, not a code change.
training hardware       M1 Max / 32 GB is ~10x too slow for Stage 3.
                        Decide the box before starting.
no corpus yet           research/capture.py is ready; no recorder is
                        feeding it. This is the item that gets worse
                        the longer it waits.
Track D not run         MiniCPM-o gguf is on disk at
                        ~/models/MiniCPM-o-4_5-gguf but has not been
                        measured against the legacy stack.
```

### Verified findings

MiniCPM-o 4.5's architecture was read from its weights and llama.cpp source
rather than inferred — see `dev/minicpm-o measured architecture.md`. Four
things we had guessed wrong are now corrected in the code: ReLU (not GELU), no
LayerNorm in the projector, pooling after the linears, and learned turn-taking
via `<|speak|>`/`<|listen|>` tokens.
