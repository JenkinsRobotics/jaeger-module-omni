# Jaeger Agent Omni — Near-Term Plan

**Supersedes** the previous version of this file (the Omni-0…Omni-7 ladder).
That ladder still exists, unchanged in substance, as **Track B** below — it was
never wrong, it was just sequenced as though it were the only project.

`dev/jaeger long term neural architecture.md` is **unchanged and still governs**.
Nothing here contradicts it; this document is how we get to Generation 1 of that
ladder while producing content.

---

# Repository boundary — Jaeger Agent vs Jaeger Agent Omni

These are **parallel developments**, not a migration. Nothing here modifies
`jaeger-agent`.

```text
JAEGER AGENT                       JAEGER AGENT OMNI
sequential pipeline                neural module integration
text is the transfer format        tensors are the transfer format
stt -> text -> llm -> text -> tts  audio -> latents -> cognition -> spec
ships today, keeps shipping        research, proves its own path
```

## What this means in practice

```text
DO NOT
  patch jaeger-agent to emit training data
  import from jaeger-agent
  share runtime state or model instances
  make production depend on anything in this repo

DO
  run our own teacher here: whisper -> text -> gemma, from transformers,
  purely to generate training targets. It does not need to BE jaeger-agent;
  it only needs to be the same sequential arrangement.
  record our own sessions into research/capture.py
  compare against jaeger-agent's behaviour as a benchmark, never as a
  dependency
```

Owning the teacher is better than borrowing it: reproducible, version-pinned,
controlled prompts, and no risk to a system that is working.

The two converge later at the **slot** level — Omni eventually ships a module
that a JaegerOS project can select instead of the sequential stack. Until then
the only thing they share is the contract.

---

# Why this document changed

Three requirements arrived that the old plan absorbed into one track:

1. **Low-latency voice-to-voice conversation** — the current sequential stack is
   noticeably laggy and we have hit its architectural ceiling.
2. **Highly dynamic voice output** — cloning, emotion, singing, impersonation.
   Production-critical for YouTube.
3. **Real-time character rendering** — driven, not generated.

These are three problems with three different cost profiles. Fusing them means
the cheap ones wait on the expensive one — but *separating* them wrongly means
assuming one engine must serve both conversation and production. It must not.

## Conversation and production have opposite requirements

```text
live conversation    latency-critical, expressivity-tolerant
produced content     expressivity-critical, latency-tolerant
```

Nobody watching a video cares that a line took four seconds to render. Nobody
talking to Jaeger cares that it cannot sing. These are two callers of the same
cognition, and the slot architecture already exists so they need not share an
engine. **PerformanceSpec is what lets one cognition drive either.**

## The correction that drove the split

**Off-the-shelf components did not make MiniCPM-o cheap.** The parts (Whisper,
SigLIP2, Qwen3, CosyVoice2) are free. The multi-stage alignment training that
taught them to speak to each other in latent space was not — it is the single
most expensive thing OpenBMB did, and it is the part we cannot replicate this
year. Assembling pretrained models at the *runtime* level is days of work.
Assembling them at the *neural* level is a training program.

Both are on the roadmap. Only the first one ships content.

Second correction: **end-to-end omni reduces expressive control.** When speech
is generated from cognitive hidden states, the control surface becomes "hope the
model felt angry." For conversation that trade is fine. For produced content —
where takes are repeated and the third one must differ deliberately from the
second — explicit conditioning is the feature, not the legacy. MiniCPM-o's
speech head is its weakest component; copying it faithfully would give us *less*
voice control than we can buy off the shelf today.

---

# The three tracks

```text
TRACK D — Reference baseline        run MiniCPM-o. Cheapest, highest-information.
                                    Days of work. Redirects every other track.
TRACK A — Expressive voice          production-critical, off-the-shelf
TRACK C — Character rendering       production-critical, off-the-shelf
TRACK B — Omni neural research      strategic, gated on what D measures
```

D is measurement. A and C are assembly. B is research.

---

# The shared seam: PerformanceSpec

The thing to design before writing any of the three tracks.

Cognition emits **one structured object per utterance**. Three consumers read it.
Nothing else crosses between the tracks.

```text
                    mind (cognition)
                          │
                   PerformanceSpec
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
      text            tts slot        animation slot
    (chat/log)      voice + style      face + gesture
                          │                 │
                       audio ──────────► lipsync
```

## Schema (v1)

```text
PerformanceSpec
  generation_id   str        # cancellation key, see Track B
  text            str        # what is said
  soul            str        # which character — resolves the bindings below
  delivery        enum       # normal | whisper | shout | sung
  emotion         enum       # neutral | happy | sad | angry | surprised | afraid | disgusted
  intensity       float      # 0.0 .. 1.0
  pace            float      # 0.5 .. 2.0, multiplier on baseline
  energy          float      # 0.0 .. 1.0
  emphasis        [(start, end, strength)]   # char spans into `text`
  pauses          [(after_char, ms)]
  gaze            enum|vec3  # camera | away | <target>
```

Deliberately small. Add fields when a track is blocked without one, not before.
`gesture`, `blink_rate`, `head_pose` are the likely v2 additions.

## Why the emotion set is Ekman-6 + neutral

Not because it is scientifically complete, but because it is the set that both
TTS instruct modes and ARKit blendshape presets already speak. Continuous
emotion space is a Track B question.

## Character-neutral intent, character-specific binding

The spec carries **intent**, never rig or voice specifics. The Soul carries the
binding:

```text
Soul: mochi
├── voice
│   ├── reference_audio → speaker embedding, extracted and cached at load
│   ├── (or finetuned voice model, for the dedicated-TTS path)
│   └── baseline pace, energy, pitch range
├── rig
│   ├── asset (VRM / 2D)
│   └── emotion → blendshape target map
└── style
    └── instruct-prompt phrasing for this character's TTS engine
```

This is the same shape as the modality-adapter pattern in the long-term doc:
a shared representation plus a per-endpoint adapter. `emotion: angry` means the
same thing everywhere; what an angry *mochi* sounds and looks like lives in the
Soul. Adding a character is authoring a binding, not touching cognition.

---

# TRACK D — Reference baseline: run MiniCPM-o

**Goal:** not to build an omni model. To *run* one, instrument it, and find out
what we are actually chasing.

Someone already shipped approximately our end state. Measuring their artifact
beats reasoning about architecture, and it costs days. This is the same thing we
did with the current stt/llm/tts stack: build it, use it, learn its ceiling.
That ceiling is why this document exists.

## Why this is first

Every other track is currently making assumptions that Track D converts into
numbers:

```text
Track B assumes    native audio understanding is noticeably better in
                   conversation                              → D shows whether
Track A assumes    omni speech output is not expressive enough for content
                                                             → D shows how bad
Track C assumes    nothing                                   → unaffected
this doc assumes   the latency win is architectural and large → D measures it
```

If MiniCPM-o's speech output turns out good enough for some content, Track A
shrinks. If its latency floor is as good as claimed, that becomes the number
Track B must justify itself against. If full duplex feels materially different
from chunked-and-cancel, that reopens a decision recorded below as closed.

## What to instrument

Run both stacks on the same material, same hardware, same session.

```text
LATENCY, broken out — not a single end-to-end number
  end of user speech  →  turn-end detected
  turn-end            →  first cognition token
  first token         →  first audio sample out       (TTFA)
  total perceived gap
  legacy: report the VAD endpoint timeout separately — it is dead time and we
  need to know how much of the gap it is

BEHAVIOUR
  interruption: end of user's interrupting speech → old audio stops
                                                  → new audio starts
  does it hear itself
  turn-taking on hesitation, mid-sentence pauses, thinking noises
  exposed knobs: turn-end sensitivity, interruption threshold, chunk size,
                 barge-in policy — what can we tune, what is welded shut

UNDERSTANDING — the paralinguistic question
  same sentence, four deliveries (flat / angry / sad / sarcastic).
  Does the response differ? This is the whole justification for Track B.
  Legacy stack scores zero here by construction.

OUTPUT QUALITY — the voice-identity question
  cloning fidelity        against the reference clip
  cross-turn consistency  turn 1 vs turn 40 of one session   ← decides RVC
  expressivity under cloning  can cloned-Mochi still do angry
  cloning in streaming mode   or offline only
  singing (expect failure — record how it fails)

COST
  VRAM, whether it can run concurrent with a rig render, tokens/s
```

## The actual benefit: integrated full duplex

The reason to run MiniCPM-o rather than keep optimizing the legacy stack. One
model, one process, one streaming loop — no IPC boundaries, no endpoint barrier,
no text bridge. Perception and generation coexist because the model was designed
that way, not because three services were persuaded to overlap. This is not
reachable by tuning the current pipeline.

Two properties of that simplicity to plan around:

**It ends at the model boundary.** Inside the checkpoint, duplex is free.
Alongside Gemma it is ours again: MiniCPM speaking, Gemma mid-tool-call, user
interrupts — cancelling coherently across two models with different notions of
state is the hard part, and the integrated design contributes nothing to it.
That is where the runtime engineering goes. Expect it rather than discover it.

**Integration costs tunability.** In the legacy stack every stage is ours — VAD
timeout, endpoint sensitivity, interruption threshold. Integrated, we get
whatever knobs OpenBMB exposed. If turn-end detection is too eager or too slow
for an actor's delivery, we may not be able to fix it.

Track D question: enumerate what is actually exposed — turn-end sensitivity,
interruption threshold, chunk size, barge-in policy — before building a
production workflow on top of it.

## Target architecture if D succeeds: two brains

MiniCPM-o owns the conversational surface — audio in, audio out, the thing it
was trained to do. Gemma stays the agentic runtime. They are not in series.

```text
                    audio in ──┬──────────────────────────┐
                               ▼                          │
                    ┌──────────────────────┐              │
                    │  MiniCPM-o           │              │
                    │  fast conversational │              │
                    │  native audio in/out │              │
                    └──────┬───────────────┘              │
                           │                              │
              answers directly          needs tools / memory / reasoning
                           │                              │
                           │                    ┌─────────▼──────────┐
                           │                    │  Gemma             │
                           │                    │  tools, RAG, plan  │
                           │                    └─────────┬──────────┘
                           │                              │
                           └──────────► PerformanceSpec ◄─┘
                                              │
                                       audio out + rig
```

**MiniCPM-o owns the floor. Gemma owns the answer.** The common path never
touches Gemma and pays nothing for its existence.

### Hard constraint on routing

**Never ask Gemma whether Gemma is needed.** Routing the turn to Gemma for
classification pays Gemma's latency on every turn, including "how are you," and
destroys the entire point. The decision comes from MiniCPM-o itself — it either
answers, or it emits a tool-request marker mid-generation — or from a classifier
cheap enough to be free. This is the difference between a fast dual stack and a
slow one, and it is easy to get wrong.

### Floor-holding

Perceived latency is time-to-first-**sound**, not time-to-answer. When a turn
does need Gemma, MiniCPM-o speaks immediately — an acknowledgement, a hesitation
noise, "let me check" — natively and in character. Gemma's work happens
underneath audio that is already playing. The conversation never has a hole.

A single omni model must choose between fast and smart. This one is both,
provided it is honest about which mode it is in.

### Production note: tools are rare in acted conversation

An actor talking to a character on camera almost never triggers a tool call.
The YouTube path is effectively pure MiniCPM-o. Gemma exists for the assistant
use case where someone asks Jaeger to actually do something.

## The gap that actually threatens production: voice identity

Not latency. If MiniCPM-o owns audio output, Mochi must sound like Mochi. If
every Soul sounds the same, the thing the channel is built on disappears.

MiniCPM-o's speech side is CosyVoice2-derived and therefore **inherits zero-shot
voice cloning from a reference clip**. So this is likely already solved, and not
as a stage in the path — as a **load-time property**.

The assistant voice is set through the session's system prompt with reference
audio at init. The speaker embedding is extracted once and cached; every
generation after that is conditioned on it. **Load the Soul, you loaded the
voice.** Zero per-turn cost, nothing in the runtime path.

This also removes one drift source: fixed cached conditioning means every turn
receives identical reference, rather than re-extracting per utterance.

### The four things that decide it

Separate properties, and passing one says nothing about the others:

```text
fidelity        does the clone sound like the reference          usually fine
consistency     turn 40 vs turn 1, same session                  expected to
                                                                 pass given
                                                                 cached
                                                                 conditioning;
                                                                 the talker is
                                                                 still
                                                                 generative, so
                                                                 verify
expressivity    can cloned-Mochi still do angry, or does
                conditioning on a reference flatten delivery
streaming       does cloning work in the streaming path, or
                only offline — cloning we cannot use at low
                latency is cloning we do not have
voice switching CAN THE VOICE CHANGE MID-SESSION, or does a new
                character require tearing down and rebuilding the
                session? Shapes how Mochi handles scenes — see below
```

Test: clone one voice, run 40 turns in one session, compare turn 1 to turn 40.
Then say one line four ways.

### Multi-character scenes

If the voice lives in the session's system prompt, switching characters may
require a session reset. Fine for one character per stream; awkward for two
characters conversing in one scene.

If a reset is required, the pattern is **one session per Soul**, with
arbitration deciding who currently holds the floor. Establish this early — it
shapes Mochi's scene model, and discovering it late means rewriting the
conversation manager.

### Fallback, only if the tests fail

Realtime voice conversion as a post-stage:

```text
mic → MiniCPM-o → audio (expressive, wrong timbre) → RVC → audio (Mochi)
                                                     ~20-50 ms
```

MiniCPM-o supplies prosody, emotion, and timing; RVC supplies identity. RVC runs
realtime — it is what live voice-changers use — so the cost is tens of
milliseconds. If this proves necessary, Track A's Layer 2 moves from an offline
production tool into the live path. Additional risk to check in that case:
conversion models are trained mostly on human speech and synthetic input
sometimes converts with artifacts.

**Default assumption: no conversion stage in the live path.** Add it only when a
measurement says built-in cloning is insufficient. Singing and impersonation
stay with offline conversion regardless — no TTS or omni model does those.

## What "feels like a person" means numerically

So the target is testable rather than a vibe:

```text
human conversation gap   ~200 ms median
under 500 ms             natural
500 ms - 1 s             slightly slow, acceptable
over 2 s                 broken
legacy stack today       1.5 - 2.5 s        ← the ceiling that produced this doc
target                   sub-second TTFA, RVC included
```

If D lands under a second end-to-end with conversion in the path, production is
unblocked and Track B becomes purely strategic.

## Explicitly not the goal

Not forking MiniCPM-o. Not training it. **Not swapping Gemma into it** — see
below. Track D produces a measurement report plus, if the numbers hold, a
working conversational engine.

## Why we cannot simply put Gemma inside MiniCPM-o

Recorded because it is the obvious idea and it does not work.

Every trained weight in MiniCPM-o is keyed to Qwen3-8B specifically. The audio
projector was trained to write into Qwen's embedding space; the vision projector
likewise; the talker was trained to read Qwen's hidden states. Substitute the
backbone and all three emit noise — not degraded output, noise. "Same structure,
different LLM" means rebuilding the model from raw components and redoing
OpenBMB's full alignment program.

This is not a Gemma problem. It is true of any backbone substitution in any omni
model. **The backbone is not a part; it is the coordinate system every other
part was fitted to.** That is also precisely what Track B would be signing up
for, at small scale, which is why B is gated on D.

Two related facts:

* MiniCPM's Qwen is no longer stock Qwen — multimodal alignment reliably costs
  some text and agentic ability. Its tool-calling is likely *worse* than base
  Qwen3-8B. Keeping Gemma as the tool brain is correct, and the gap is wider
  than the model card suggests.
* Our Gemma-vs-Qwen comparison is not apples-to-apples: a 26B MoE against an 8B
  backbone. The lesson is "omni backbones are currently small," not "Qwen is
  bad." It does not change the decision.

## Where this lives

Not merged into `jaeger_agent`. **This repo** — it is already the right home. It
ships as a module in a slot and `jaeger_agent` consumes it like any other
engine. Merging a 9B research checkpoint into the main agent repo couples their
release cycles for no benefit; the slot system exists to prevent exactly that.

## On owning the pipeline

Almost none of the lag lives in the weights. It lives in the **runtime**.

Revised after reading `reference/` — half that runtime already exists and is
tuned: Speex AEC, webrtcvad, barge-in, self-speech rejection, noise-floor
calibration, wake gating, streamed TTS. See `dev/legacy pipeline measured.md`.

**Omni replaces what happens after the phrase queue, not the audio I/O layer.**

```text
mic -> AEC -> VAD -> phrase_queue  |  -> STT -> text -> LLM -> text -> TTS
       exists, keep it             |     this is what Omni replaces
```

What Omni still owns: incremental prefill during speech, the projector, the
learned turn-end head — which replaces the 700 ms silence timeout, because a
stopwatch cannot tell "...and then he said" from "...and that was that." — and
cancellation across cognition rather than only across audio.

Ownership order: **runtime now → routing now → interface now → weights later**,
if Track B earns it. The runtime built in step one is the same runtime a
Jaeger-native model plugs into later. That part is not throwaway.

## On Gen 1 not needing to be modular

Accepted. Gen 1 exists to produce understanding and content, and a later rewrite
is expected and fine — that is exactly how the current stack was built. The one
thing worth defining early is **PerformanceSpec**, and not for modularity's
sake: three consumers exist on day one, and Mochi cannot have characters without
it. Everything else can be hardcoded and thrown away.

# TRACK A — Expressive voice

**Goal:** replace Kokoro in the creative path with a cloning, style-controllable
engine. Kokoro stays registered in the `tts` slot as the low-latency fallback.

This is a **slot implementation**, not an architecture change. `kokoro_tts` is
the canonical node shape; the new engine mirrors it.

## Two layers

**Layer 1 — cloning TTS with style control.** Handles conversational speech and
most narration. Candidates for bake-off (verify current state of each before
committing — this field moves monthly):

```text
CosyVoice 2/3     zero-shot cloning, instruct-mode natural-language style,
                  streaming, low TTFA, Apache-2.0. Same lineage MiniCPM-o uses.
                  → default starting point
GPT-SoVITS        finetune on ~1 min, very high character fidelity,
                  strong track record in fandub/character work
IndexTTS2         separates emotion reference from timbre reference
Fish / OpenAudio  emotion + tone markers
Chatterbox        single exaggeration dial, MIT
Orpheus           LLM-backboned, emotion tags, streaming
```

**Layer 2 — voice conversion.** Handles what TTS structurally cannot: singing,
impersonation, and any acted delivery a text prompt cannot specify.
RVC / so-vits-svc / seed-vc family. Perform-then-convert: the performance is
recorded (sung badly is fine), conversion carries pitch and timing into the
character's timbre. Offline is acceptable — this is not in the realtime path.

## Bake-off scoring

Score on our own material, not published benchmarks:

```text
cloning fidelity        vs a held-out clip of the target voice
consistency             same voice across 20 separate generations
emotion range           can it actually do angry, or only "slightly tense"
whisper / shout         delivery modes, not just volume
pace control            without pitch artifacts
emphasis                per-word, controllable
TTFA                    time to first audio, streaming
VRAM                    concurrent with backbone + rig render
license                 commercial use on a monetized channel
```

## Zero-shot vs finetune

For recurring characters, **finetune**. 10–30 min of a character's voice into
GPT-SoVITS or an RVC model beats any zero-shot clone on cross-episode
consistency, and consistency is what an audience notices. Zero-shot is for
one-off voices and prototyping.

## Deliverable

`tts` slot gains a second engine. PerformanceSpec is its input contract.
Kokoro still boots. Nothing else in JaegerAI changes.

---

# TRACK C — Character rendering

**Goal:** real-time rendered character driven by PerformanceSpec.

## Why this is cheap

We are not generating pixels. We are rasterizing an authored asset with a small
parameter vector changed per frame — what game engines have done for twenty
years. Consequence: 1080p/4K at 60fps on hardware we own, with perfect
frame-to-frame identity consistency, because it is literally the same asset
every frame. The "1000×1000 realtime" target is not a constraint; it is
exceeded by roughly two orders of magnitude.

Temporal consistency — the hardest failure mode in generated video — does not
exist as a problem here.

## Rig choice: 3D blendshapes

Recommend **VRM (3D)** over Live2D (2D), for one reason beyond the render:
the long-term doc has `JaegerExpression` driving a physical robot face. A
blendshape vector maps onto servos and displays. A 2D mesh deformation does not.
Same control signal drives the YouTube avatar and, later, the robot.

If the flat 2D look is required for a specific Soul, Inochi2D is the open
Live2D alternative and avoids Cubism's commercial licensing tiers. The
PerformanceSpec does not care — that is the point of the Soul binding.

Control surface: **ARKit 52 blendshapes**. De-facto standard, every tool speaks
it, and it is 52 floats.

## Lipsync — the lazy path

We generate the audio and we know the exact text, so we do not need audio→viseme
inference. Forced-align the generated audio against the known transcript
(torchaudio MMS_FA, or Rhubarb for CPU-only), map phonemes to visemes.
Deterministic, accurate, effectively free.

Audio-driven models (Audio2Face lineage) are a later refinement for a subtler
mouth, not a v1 requirement.

## Emotion → face

`emotion` + `intensity` from PerformanceSpec index the Soul's blendshape map,
smoothed over ~100ms. A lookup table, not a network. Upper face from emotion,
lower face from visemes, blended — which is exactly why lipsync and emotion stay
coherent here and are an emergent prayer in generated video.

## Upgrade path to photoreal — do not discard the rig

Render the rig, then a low-strength img2img pass with ControlNet conditioned on
the rig output. The rig supplies pose, timing, identity, and temporal stability;
diffusion supplies style and detail. Realtime-feasible because it is 1–4 steps
with strong conditioning, not 30 unconditioned. Gaussian-splat avatars are the
alternative route for photoreal-of-a-real-person.

## Deliverable

`animation` slot consumes PerformanceSpec + generated audio, drives a VRM rig,
renders. Mochi is the first Soul to carry a full binding.

---

# TRACK B — Omni neural research

Unchanged in substance from the previous plan. Reordered, and with three
corrections found during review.

## What this track is actually for

Not latency, and not speech quality — Track A owns quality. This track exists so
Jaeger can **understand** voice: sarcasm, hesitation, emotional state, who is
speaking. Nothing off the shelf does that well, and no amount of TTS work
produces it.

## Correction 1 — the latency win is real, and it is architectural

An earlier draft of this plan claimed the latency case for omni was weak on the
grounds that omni does not remove the LLM. That reasoning costed the *models*
and ignored the *barriers*, and it was wrong.

The lag in the current stack is not the sum of its inference times. It is the
serialization:

```text
VAD endpoint timeout    MEASURED at 700 ms + 250 ms post-padding = ~950 ms,
                        from our own tuned voice_chat_streaming.py. Dead time.
                        Structural — a text bridge needs a complete utterance
                        before transcription means anything.
ASR full pass           could not have started earlier
LLM prefill             cold start; nothing was precomputed during the three
                        seconds the user was talking
TTS first clause        cannot emit until enough text exists
IPC                     process boundaries between separate services
```

An omni model with a streaming audio encoder prefills **while the user is still
speaking**. At end-of-speech most of the input is already ingested, and turn-end
can be predicted semantically by the model rather than by an energy timeout.
That is the gap between ~1.5-2.5 s and sub-second time-to-first-audio, and it
has nothing to do with whether Whisper or Kokoro is fast.

The counter-cost is real but second-order: audio tokens are denser than text —
Whisper's encoder runs at 50 Hz, so even at 4x pooling a 5 s utterance is ~60
tokens against ~15 for its transcript, so prefill grows. It grows against a
window that was previously dead time. Track D measures the actual numbers.

## Correction 2 — Whisper discards what we said we want to keep

Whisper's encoder is trained for ASR; invariance to speaker and delivery is the
objective, not a side effect. A Whisper-only path is a fancier ASR. If prosody,
emotion, and speaker state must survive to cognition, a second branch is
required — WavLM/HuBERT (WavLM is the standard backbone for speaker
verification for exactly this reason), or a small paralinguistic encoder
contributing extra tokens alongside the Whisper tokens. MiniCPM-o has this same
hole; copying it faithfully would copy the limitation.

## Correction 3 — "swappable" is true in code, false in training

A projector is trained against a *specific pair*: this encoder's representation,
that backbone's embedding space. Swap the backbone and the projector is garbage.
Swap the encoder, same. The interfaces buy clean code, not free swaps.

Consequence: **do the learning experiments on a 1B-class backbone.** The
projector gets retrained for the production backbone regardless, and training a
projector requires backprop *through* the frozen LLM — frozen is not cheap, it
still needs forward, backward, and stored activations. A frozen 26B sits near
full-finetune VRAM minus optimizer state.

## First experiments — both are decision-making, not building

**B1 — ceiling probe.** Freeze the Whisper encoder, dump hidden states for a few
hundred clips, train three linear probes: content, speaker ID, emotion label.
Answers in a day how much of the requirement the representation can carry, and
whether the second branch from Correction 2 is mandatory. Do this before writing
a projector.

**B2 — legacy latency budget.** Instrument the current stack: ASR / prefill /
decode / TTFA. Establishes what "faster" would even mean.

## Then, if the probes justify it

```text
Whisper encoder    FROZEN
JaegerAudioProjector   TRAIN       Linear → norm → GELU → Linear, nothing fancier
small backbone     FROZEN          1B class, not the production model
        ↓
    text answer
```

Success condition unchanged and non-negotiable: the backbone must never receive
the transcript as text. Otherwise nothing has been demonstrated.

Progressive unfreezing (backbone LoRA, then selected layers) only if measurement
demands it.

## Verify before committing to a production backbone

`Gemma 4 26B-A4B` needs checkpoint/runtime verification, and an MoE backbone
makes hidden-state-conditioned heads more awkward — which experts fire varies
per token. Keep `CognitiveBackbone` as an abstraction; do not commit early.

## Cancellation architecture — carried forward unchanged

`generation_id` on every cognitive and speech generation. On interrupt:
invalidate old generation, cancel speech, discard stale audio, revise state,
start new. `generation_id` is already a PerformanceSpec field so Tracks A and C
inherit this for free.

## Full duplex is a fork, not a phase

Two different things get that name:

```text
frame-synchronous     model ticks at fixed rate, models both streams at once
                      (Moshi lineage) — requires a backbone trained that way,
                      cannot be bolted onto Gemma

chunked + cancel      VAD + generation_id invalidation, works with any LLM,
                      delivers most of the perceived benefit
```

**Decision: chunked + cancel, plus a learned turn-end head.**

Revised after reading MiniCPM's source (`dev/minicpm-o measured
architecture.md`). The earlier framing here — that MiniCPM was itself doing
chunked-with-cancellation — was wrong. It emits `<|speak|>` / `<|listen|>`
tokens and decides turn-taking as a prediction, generating on every step even
while silent.

We cannot copy that: it requires new vocabulary trained into the backbone on
full-duplex conversational data. So we copy the behaviour instead — a small
turn-end head over E4B's hidden state, trained on our own captured turn
boundaries.

Frame-synchronous generation stays out of scope. The turn-end head does not.

---

# Training strategy for Track B: self-distillation

The legacy stack is the teacher. Text-Gemma and audio-Gemma are the same
weights — only the input path differs — so we are not teaching Gemma anything.
We are teaching the projector to hit a target we can compute exactly.

## Loss targets internal state, not tokens

```text
text path    "what time is it" → tokenizer → embeddings → Gemma → H_text
audio path   waveform → Whisper → projector → Gemma        → H_audio

loss = distance(H_audio, H_text)   at several layers
     + KL(logits_audio ‖ logits_text)
     + cross-entropy on the response
```

Hidden-state distillation gives a dense signal — every layer, every position,
every sample — rather than one sparse next-token signal. This is materially more
data-efficient than aligning against an ASR corpus, and it is why the data
requirement here is tens to low hundreds of hours rather than thousands.

## Data

Three sources, and they do different jobs. Verify licences before use — several
of the well-known corpora are research-only or need an access agreement.

### Tier 1 — labeled public corpora (ground truth)

**Worth more than volume alone.** A human transcript removes ASR error from the
teacher path: with Whisper pseudo-labels the teacher reads whatever Whisper
misheard, and the projector learns to reproduce the mistake faithfully. Ground
truth breaks that loop, and needs no confidence gate.

```text
LibriSpeech         ~960 h   clean read speech. The standard warmup set.
People's Speech     ~30k h   volume, wide acoustic variety.
Common Voice        large    accent + speaker diversity.
VoxPopuli           large    multilingual, parliamentary.
GigaSpeech          ~10k h   varied domains; check the access agreement.
```

Use these for Stage 3 warmup, where the job is landing the projector in roughly
the right region and read speech is perfectly adequate.

### Tier 2 — far-field / real rooms

```text
AMI meeting corpus  real rooms, mic arrays, distance, overlap
CHiME               noisy multi-party
```

Closer to a robot's actual inference condition than anything read into a
close mic. This is the tier that stops the projector overfitting to
studio-clean input.

### Tier 3 — emotion / paralinguistic labels

This tier is what makes **Stage 6** possible, and it is the hardest thing to
synthesise, so labeled data matters most here.

```text
CREMA-D, RAVDESS    acted emotion, clean labels, permissive
MELD                conversational emotion from TV dialogue
IEMOCAP             widely used; access-restricted, check terms
```

An emotion label is exactly the enrichment the teacher needs:

```text
teacher sees   "what time is it"  [tone: irritated]      <- from the label
student sees   audio only
```

With Tier 3 the annotation is ground truth rather than a classifier's guess,
which makes it the right data to *validate* the classifier-derived annotations
we use elsewhere.

### Adapters, not special cases

Every corpus stores audio and text differently. Rather than branch per corpus,
each one **declares how its schema maps onto ours** and the pipeline is shared:

```python
Corpus(key="tedlium", hf_id="LIUM/tedlium", text_field="text",
       text_transform=strip_ted_markers)
```

Adding a corpus is a few lines of mapping, not new code. Same pattern as the
modality projectors — an adapter at the boundary, one shared path behind it.
`training/fetch_dataset.py --list` shows the catalog and which schemas are
verified against the live hub.

### Text normalisation: lowercase, everywhere

**Casing carries no loudness.** Delivery is a `PerformanceSpec` field
(`normal | whisper | shout`) precisely so text does not have to encode it, and
`energy` covers the continuous part. A model that learns SHOUTING from capital
letters has learned the wrong signal from the wrong place.

The forcing issue is mixing: LibriSpeech is UPPERCASE with no punctuation,
TED-LIUM and AMI are natural. Raw, a mixed corpus teaches the model those
conventions rather than the speech. So normalisation applies to **all** sources,
not just the offender:

```text
lowercase          uniform across corpora
drop markers       [laughter] <unk> {breath} are noise, not words
collapse space
```

What this gives up: punctuation and capitalisation in the output. That is a
separate task with its own training data (inverse text normalisation), and a
separate head can restore it later. Not this stage's job.

### Tier 4 — our own sessions

Our room, our actors, our mic distance, our character personas. Lowest volume,
highest relevance, and the only source of real turn boundaries for the Stage 5
turn-end head. Accumulates from day one; see `research/capture.py`.

Set `confidence=1.0` on human-labeled turns so a single gate expression works
across all four tiers.

### How they combine

```text
Stage 3 warmup      Tier 1, plus Tier 2 for robustness
Stage 4 distill     Tier 1 + 2 + 4. Our own data weighted up — it is the
                    distribution we actually serve.
Stage 5 turn-end    Tier 4 only. Nobody else's turn boundaries are ours.
Stage 6 paralinguistic  Tier 3 for ground truth, Tier 4 for realism.
```

Free and unbounded on top of all this: any audio plus our own teacher pipeline
produces its own targets, in our domain, with our prompts and our characters.

**Distribution is the trap.** Training on TTS-generated audio alone teaches the
projector the narrow slice of Whisper's output space that synthetic speech
occupies. At inference the input is an actor in a room at distance through a
real mic — somewhere else entirely. Whisper is robust to this; a small projector
trained only on clean synthetic audio is not.

Two mitigations, both cheap:

```text
augmentation      room impulse responses, noise, distance/gain variation,
                  mic frequency response. Standard tooling, no recording.

far-field corpora  AMI / CHiME — recorded at real distance in real rooms
                  rather than simulated. See Tier 2 above.

real session logs  our own recordings. Perfectly paired, our exact acoustics.
                  → START CAPTURING NOW. See sequencing.
```

## Finding: a frozen backbone constrains the projector's output scale

Measured, not predicted. First real Stage 3 run — 150 steps, 60 synthetic
turns, whisper-tiny + SmolLM2-135M:

```text
                      loss             target_acc      grad norm
no output scaling     5.39 -> 3.17     0.76 -> 0.38    47 -> 648    diverged
embed_norm pinned     6.07 -> 0.268    0.66 -> 0.91    24 -> 5.6    converged
```

Cause, from the checkpoint:

```text
                  out-norm/token   ratio to text emb   w1 norm   w2 norm
fresh                      3.97          1.24x          13.85     13.86
after 150 steps           14.23          4.44x          14.08     14.14
```

The weights barely grew. The projector **rotated** into a configuration
emitting embeddings 4.4x larger than any token the frozen backbone had ever
seen — it drove its own input out of distribution. Loss climbed while
gradients exploded.

**MiniCPM never meets this because it trains the backbone jointly**, so the
backbone adapts to whatever the projector emits. Training against a frozen
backbone is a different regime, and porting their architecture faithfully is
necessary but not sufficient.

Fix: re-scale each output token to the backbone's own mean embedding norm.
Direction stays free, magnitude is pinned. `JaegerAudioProjector.for_backbone()`
reads both `cognition_dim` and the target norm from the model, so neither is
ever a constant.

This is the first thing we have learned that the reference implementation could
not have told us, and it will apply to every future modality projector trained
against a frozen backbone.

## Known open questions

```text
temporal compression   Whisper emits ~50 frames/s — far too many tokens.
                       Pool to ~6-12 tokens/s. How much trades context and
                       latency against information loss. Main design decision,
                       no obvious answer, tune empirically.

projector-only ceiling Dropping novel vectors into an embedding space Gemma
                       never trained on often plateaus. Expect backbone LoRA
                       as stage two — the original freeze schedule anticipated
                       this.

turn-end detection     Once audio is native, something must decide the actor
                       finished. Becomes a small trained head. No longer free
                       from VAD.

streaming window       Whisper pads to 30 s. Encoding a 1 s chunk costs the
                       same as 30 s. Needs a chunked/masked variant.
```

## Why the same trick is NOT the answer for speech output

Output can be distilled too: Gemma generates text, TTS renders it, the codec
encodes to speech tokens, train a head from hidden states to those tokens.
Defined targets, real recipe.

But **the teacher is our own TTS, so the student inherits our TTS's prosody.**
The gain is latency — no waiting for text before speech begins — not
expression. Expressivity beyond the TTS requires training on real human speech
with real delivery, which is the expensive path.

Therefore: self-distill the input side, where the win is large and the teacher
is exact. Keep Track A's TTS on the output side, where it already beats
anything we would distill. Revisit native speech output only when latency alone
justifies it.

---

# Build status

```text
jaeger_agent_omni/   contracts that will ship
research/            experiments — promoted into the package once proven
training/            corpus policy, run configuration
```

**Promotion rule:** code stays in `research/` until the stage validating it
passes, then moves into the package. `performance_spec.py` and `outputs.py` are
contracts rather than experiments, so they start there. Models do not.

Every module ships one runnable check; `research/check_all.py` runs all of them
offline in ~2 s.

```text
DONE, offline-verified  (16/16 checks green)
  package/performance_spec  the three-consumer seam, with validation
  package/outputs           text / speech / tool-call routing off one pass
  research/capture          Stage 1 turn log + confidence gate — START RECORDING
  research/projector        Stage 3 network, ported from MiniCPM's forward pass
  research/distill          Stage 3/4 losses, warmup + distill presets
  research/streaming        Stage 5 chunking + rolling 1500-frame cache
  research/turn_end         Stage 5 turn-end head (our <|speak|>/<|listen|>)
  research/stubs            tiny Whisper/LM so all of the above runs weight-free
  research/pipeline         end-to-end wiring; asserts the projector learns
  research/inspect_gguf     GGUF header reader (how the above was grounded)
  training/data             tiered corpus mixing policy
  training/dataset          corpus -> batches, audio prefix + teacher forcing
  training/make_corpus      synthetic smoke corpus via system TTS
  training/config           run configuration, validated presets
  training/train            the loop: resume, time budget, signal-safe
  package/app               OmniApp — both paths, per-stage timing
  package/__main__          python -m jaeger_agent_omni --audio ...

VERIFIED ON REAL MODELS
  voice -> cognition    whisper-tiny + SmolLM2-135M, real speech, both paths.
                        text: transcribe 390ms + generate 990ms, correct answer.
                        omni: encode 181ms + project 3ms, no transcript at all.
  Stage 3 warmup        150 steps, 60 synthetic turns, MPS. 91% of transcript
                        tokens predicted from audio alone. See the embed_norm
                        finding above.

BLOCKED ON WEIGHTS
  gemma-4-E4B run       one flag (--backbone). cognition_dim and embed_norm are
                        both read from the model, so nothing is hardcoded — the
                        blocker is the repo id, not the code.

BLOCKED ON HARDWARE
  full Stage 3          M1 Max managed 150 steps in 5 min at tiny scale. A real
                        corpus at medium/E4B scale needs CUDA. nightly.sh makes
                        unattended runs survivable in the meantime.

NOT STARTED
  Stage 2 teacher harness — Omni's own whisper->E4B pipeline (needs weights)
  session recorder feeding capture.py — subscribes /sense/mic/pcm alongside
    JaegerWhisperSTT; audio_io owns the device, so no contention
  training/train.py — the loop itself (blocked on Stage 0 + hardware)
  Track A bake-off, Track C rig, Track D measurement

AGENTIC — reserved, not built
  outputs.py carries a `tool_calls` route so the path exists. No tool library
  here: jaeger-agent already has one and those tools are text-in/text-out,
  modality-agnostic. Duplicating them would be bloat. MiniCPM inherits stock
  Qwen tool-calling (measured: <tools>/<tool_call> template) — how well it
  survives multimodal alignment, and how calls coexist with <|speak|>, are
  Track D measurements.
```

---

# Tools: two paths, and the rule that decides between them

## Symbolic tools work today, unchanged

Omni's backbone **is** `gemma-4-E4B`. Same weights, same chat template, same
dialect quirks `jaeger-agent`'s parsers already handle. The backbone cannot
tell whether its input arrived as tokens or as projected audio, so the entire
tool-calling stack transfers with zero changes.

This is a concrete payoff from keeping Gemma. MiniCPM's Qwen backbone would
have meant redoing every dialect parser, every tool-presentation path, and
every regression test that exists for Gemma's specific failure modes.

`outputs.py` carries the `tool_calls` route. We do not reimplement the tools
themselves — they are text-in/text-out and modality-agnostic, so duplicating
them here would be bloat.

## Neural tool modules — a neural tool is just another projector

The long-term direction, and it reuses exactly what Stage 3 is building:

```text
audio    → whisper encoder → JaegerAudioProjector  ─┐
vision   → SigLIP           → JaegerVisionProjector ─┼→ shared cognition
memory   → retrieval net    → JaegerMemoryProjector ─┘
body     → proprioception   → JaegerBodyProjector   ─┘
```

Same pattern, same shape, same training recipe as the audio path. The encoder
work is not a one-off — it is the **template every specialist follows**. Once
`JaegerAudioProjector` is proven, adding a modality is a known job rather than
a research question.

## The rule: symbolic stays text, perceptual becomes latent

Not everything should become a neural module. The deciding question is whether
text is a *lossy* representation of the answer.

```text
STAYS TEXT — the answer genuinely is a symbol
  get_time         "16:32" loses nothing
  file_read        the file is text
  web_search       results are documents
  calculate        the number is the answer
  run_python       code in, output out
  remember/recall  when the fact is propositional

BECOMES NEURAL — text throws away the answer
  look_at          "a man in a red shirt" discards the scene
  listen           prosody, speaker, emotion do not survive a transcript
  recall           when the memory is episodic/perceptual, not propositional
  motion           trajectories are not sentences
  expression       a face is not an adjective
```

This is the project's own thesis applied to tools. The long-term architecture
doc already says it: text remains excellent for code, logs, files, APIs, RAG,
symbols. The mistake is making it *mandatory* for things it cannot carry.

A useful consequence: `look_at()` returning a caption is not a tool we should
improve — it is a modality we have not connected yet. Vision-as-tool becomes
vision-as-input at Stage 7, and the caption-returning version quietly stops
being needed.

## Open question — tool calls in a duplex stream

Unresolved, and it needs answering before Stage 5 ships.

When the model generates continuously (speech tokens, or `<|tts_pad|>` while
silent), *when* does a tool call get emitted? A call mid-utterance means either
pausing speech or interleaving a non-speech token into an audio stream.

MiniCPM's vocabulary shows `<|speak|>` / `<|listen|>` / `<|chunk_eos|>` but the
artifact does not show how tool calls coexist with them. **Track D measurement:
ask MiniCPM something requiring a tool during a live duplex session and observe
what happens to the audio.**

Our own answer is likely floor-holding: emit the call at a chunk boundary, keep
speaking through the latency. That is the mechanism already recorded under the
two-brain design — it applies whether the tool brain is Gemma-elsewhere or
Gemma-itself.

---

# Staged build — Jaeger Agent Omni

Decided. Each stage is small, leaves **one runnable check** behind, and produces
something usable. Any stage can be the last one without stranding the product:
Tracks A and C ship the channel independently, and the legacy stack keeps
running throughout.

## Fixed decisions

```text
audio encoder     openai/whisper-medium        multilingual — matches MiniCPM's
                                               encoder, so a failing projector
                                               cannot be blamed on it
cognitive model   gemma-4-E4B                  the realtime model, already the
                                               conversational surface
                  bf16, NOT the Q4 GGUF        train and deploy at the same
                                               precision
deep think        gemma-4-26B-A4B unchanged    text in, no audio, stays quantized
speech output     Track A cloning TTS          not native audio-out
avatar            derived from PerformanceSpec not a trained head, yet
stack             torch + transformers         + peft at Stage 4, datasets
```

The 26B never sees audio. It receives a task from E4B and answers in text,
exactly as today. **This is why there is no memory conflict** — the realtime
path is small by construction.

---

## Stage 0 — Raw access

Get both models under `transformers` and pull tensors out of them.

```text
build   whisper-medium encoder → last_hidden_state
        gemma-4-E4B bf16 → input embeddings + hidden states
```

```text
CHECK   assert whisper_out.shape == (1, 1500, 1024)
        assert gemma_embed.shape[-1] == gemma.config.hidden_size
        print peak VRAM for both resident
```

Gives: proof the interfaces exist, and the real VRAM number before anything is
built on top. Half a day. Expect the GGUF→safetensors swap to be the only
friction.

---

## Stage 1 — Capture

Raw mic audio + transcript + response, logged per turn. **Start this before
Stage 0** — it costs nothing and compounds.

Recorded by Omni's own recorder into `research/capture.py`'s format. Not by
instrumenting `jaeger-agent` — see the repository boundary above.

Two sources, both ours:

```text
recorded sessions   our room, our actors, our mic distance. Highest value.
public corpora      AMI / CHiME for far-field, People's Speech for volume.
                    Fills the gap while our own corpus accumulates.
```

```text
CHECK   write a turn, read it back, assert wav + manifest fidelity
        (capture.py demo — already passing)
```

Gives: the training corpus, and the turn boundaries the Stage 5 turn-end head
trains on.

---

## Stage 2 — Teacher harness

**Omni's own sequential pipeline**, built here from `transformers`, existing
only to generate training targets:

```text
audio -> whisper (transcript) -> E4B -> hidden states + logits, cached
```

This mirrors what `jaeger-agent` does architecturally without touching it. Same
arrangement, our own instance, pinned versions, controlled prompts.

```text
CHECK   assert same input twice → identical hidden states  (determinism)
        assert the teacher's text response is stable across runs
```

Gives: exact training targets for every captured turn. No labeling.

---

## Stage 3 — Projector, transcript warmup

`JaegerAudioProjector`: Linear → ReLU → Linear → AvgPool(5). Train on
audio→text. Whisper frozen, E4B frozen.

Architecture ported from MiniCPM's actual forward pass — see
`dev/minicpm-o measured architecture.md`. ReLU not GELU, no LayerNorm inside
the projector, and pooling comes **after** both linears.

### Architecture confirmed from MiniCPM's own weights

Read out of `~/models/MiniCPM-o-4_5-gguf` with `research/inspect_gguf.py`,
not inferred:

```text
audio_pool_step                  5              50 fps -> 10 tokens/s
audio_projector.linear1.weight   [1024, 4096]   whisper d_model -> Qwen hidden
audio_projector.linear2.weight   [4096, 4096]
                                 21.0M params

tts/projector.linear1.weight     [4096, 768]    Qwen hidden -> CosyVoice
tts/projector.linear2.weight     [768, 768]
                                 3.7M params
```

**There is no resampler.** No perceiver, no learned queries, no attention
pooling. Mean-pool with stride 5, then two linears. Selecting what matters is
left to the LLM's own attention downstream — the projector is deliberately
dumb.

This settles the "reject irrelevant input" question: MiniCPM does not solve it
in the projector, and neither should we until a measurement says the projector
is the bottleneck. `research/projector.py` implements this design as the
default, with a `stack` mode (concatenate adjacent frames instead of averaging)
as the A/B candidate — stacking preserves frame order that pooling destroys,
at the cost of a wider first matrix.

The TTS-side projector being the same shape is worth remembering: the
hidden-state → speech interface is also just two linears, which lowers the
estimated difficulty of native speech output if we revisit it.

**This is the milestone.** From the original plan, unchanged:

```text
CHECK   speak: "what is two plus two"
        assert "four" in response.lower()
        assert transcript NOT in prompt          ← the whole point
```

Gives: E4B understands speech with no ASR text in the loop. If this works, the
architecture is proven and everything after it is refinement.

---

## Stage 4 — Hidden-state distillation

Swap the loss from next-token onto internal state:

```text
loss = distance(H_audio, H_text) at several layers
     + KL(logits_audio ‖ logits_text)
     + cross-entropy on the response
```

```text
CHECK   held-out set, same question via both paths
        assert agreement_rate > baseline_from_stage_3
```

Gives: understanding rather than transcription. Add LoRA on E4B only if
agreement plateaus — measure first.

---

## Stage 5 — Streaming

Copy MiniCPM's design directly (`dev/minicpm-o measured architecture.md`):

```text
1 s chunks = 50 encoder frames = 10 audio tokens
rolling 1500-frame encoder KV cache   (audition.cpp: 1500/50 = 30 steps)
incremental prefill per chunk during user speech
```

Whisper's 30 s window is not padded around and not patched — it becomes the
depth of a streaming cache. Each second of audio costs one second of encoder
compute while still attending over the previous 30 seconds.

Plus the **turn-end head** — our stand-in for `<|speak|>`/`<|listen|>`, which
we cannot replicate without backbone vocabulary training. This is the component
that decides whether the system feels conversational, and `capture.py` is
already logging the labels it needs.

### Built: `StreamingOmni` (jaeger_agent_omni/app.py)

Consumes 1 s chunks as they arrive; encoder and projector run during speech.
`Timing.overlapped_ms` records that work and is **excluded from the total**,
because it costs the listener nothing. That exclusion is the whole thesis,
measurable.

Known gap: HF Whisper exposes no encoder KV cache, so each chunk re-encodes the
accumulated audio instead of extending a cache — O(n^2) per turn. Acceptable at
conversational lengths, wrong for long-form, and the reason `streaming.py`'s
rolling 1500-frame `EncoderCache` exists as the target shape.

```text
CHECK   assert TTFA < legacy_TTFA_from_track_D
        assert turn_end_false_trigger_rate < threshold on real recordings
```

Gives: **the latency win lands here.** Everything before this is offline
correctness; this is where an actor stops waiting.

---

## Stage 6 — Paralinguistics

Enrich the teacher's input with classifier-derived annotations so the student
has gradient pressure to preserve delivery:

```text
teacher sees   "what time is it"  [tone: irritated, pace: fast]
student sees   audio only
```

```text
CHECK   same sentence, four deliveries (flat / angry / sad / sarcastic)
        assert responses differ meaningfully
        legacy stack scores zero here by construction
```

Gives: the capability nothing off the shelf has. Skip this stage and we have
built a fast ASR-equivalent.

---

## Stage 7 — Output wiring

E4B emits PerformanceSpec → Track A TTS → audio; avatar reads the same spec.

```text
CHECK   assert spec.emotion drives both voice and face from one source
        assert visemes align with generated audio (forced alignment)
```

Gives: the third output modality MiniCPM does not have, at near-zero cost
because the intermediate is explicit.

---

## What "done" looks like

```text
audio ──► whisper-medium ──► JaegerAudioProjector ──► gemma-4-E4B
                                                          │
                                                  PerformanceSpec
                                    ┌─────────────────────┼──────────────┐
                                    ▼                     ▼              ▼
                                  text                 TTS(clone)     avatar
                                    │                     │              │
                              tools / 26B              audio           rig
```

Our version of MiniCPM: same structure, our models, one output more, and the
one component we chose not to replicate is the one we can buy better.

---

# Sequencing

```text
TODAY      Turn on raw mic capture in the legacy stack. Log waveform +
           pipeline text output per turn. Costs nothing, compounds daily,
           and is the training corpus for Track B — our room, our actors,
           our mic distance. The only item here that gets more valuable
           the earlier it starts. Do this before anything else.

NOW        Track D: MiniCPM-o running + instrumented, legacy stack instrumented
           identically, same hardware, same session. One report, side by side.
           PerformanceSpec schema pinned         blocks A and C, hours of work
           Track C rig + forced-align lipsync spike   independent of D

NEXT       two-brain routing: MiniCPM-o emits tool-request, Gemma handles it,
           floor-holding while Gemma works
           realtime RVC stage ONLY if D's cloning tests fail
           Track A bake-off, scoped by what D showed about omni speech quality
           animation slot: mochi Soul binding complete
           offline voice-conversion stage for singing / impersonation

THEN       Track B re-planned against D's numbers. B1 ceiling probe only if D
           shows paralinguistic understanding is worth chasing and MiniCPM-o
           does not already deliver it.

LATER      style-spec prediction from cognition (small NN — the first custom
           network that is actually justified)
           photoreal img2img pass over rig output
```

No dates. Track D determines the shape of everything after it — that is the
point of running it first.

---

# What we are explicitly NOT doing

Deletions from the previous plan, and things not to reintroduce:

* **Realtime video diffusion.** Deleted entirely. The rig replaces it and beats
  it on control, consistency, resolution, and cost.
* **Training a custom TTS or vocoder.** Voice generation quality is a thing to
  buy. The gap we feared does not exist.
* **Building an omni stack before running one.** Track B is gated on Track D.
  Reasoning about an architecture somebody has already shipped, instead of
  measuring theirs, is the expensive way to learn this.
* **Omni-0 as a milestone.** It is ~20 lines of `encoder(...).last_hidden_state`.
  Folded into B1.
* **The five-ABC interface layer up front** (`ModalityEncoder`,
  `ModalityProjector`, …). One implementation each is a speculative
  abstraction. Write the concrete path; extract the interface when a second
  implementation exists. PerformanceSpec is the one contract worth defining
  early, because three tracks consume it on day one.
* **`NeuralPacket`.** Premature. Revisit at Generation 2.
* **Removing or rewriting the legacy pipeline.** It stays, it boots, it is the
  benchmark.
* **Connecting neural output to actuators.** Deterministic authority stays
  deterministic — see the long-term doc.

---

# Relationship to the long-term architecture

Everything above is **Generation 1** on the long-term ladder: software modules
plus pretrained neural modules, communicating through a defined contract.

PerformanceSpec is the seam that makes Generation 2 an upgrade rather than a
rewrite. Today cognition authors it, or a lookup table fills it. Later a small
trained network *predicts* it from audio input and cognitive state. **The
interface does not change** — the consumers never learn that their input started
being predicted instead of authored.

That was the architectural bet in the long-term doc. This plan places it where
it can be collected.
