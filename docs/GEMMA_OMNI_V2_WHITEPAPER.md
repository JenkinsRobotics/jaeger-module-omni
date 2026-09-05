# GemmaOmni V2 — Parallel Paralinguistics for a Cascaded Voice Agent
### White paper and execution plan · 2026-09-01

## Abstract

Every voice assistant in this repository shares one defect by construction:
the language model experiences conversation through text, and text is a
lossy compression of the waveform. Speaker identity, pace, prosody,
hesitation, "I'm not finished," and "that phrase was addressed to you" all
exist in the audio, are destroyed by transcription, and are then partially
reconstructed downstream by hand-tuned logic. V2 restructures the pipeline
so the paralinguistic information never passes through the text bottleneck:
a PARALLEL channel of small, independently trained neural heads reads the
waveform directly, at frame rate, alongside the SERIES semantic channel
(Whisper → adapter → Gemma), and the two converge at a fusion point with
its own projection into the language model. The long-term end state is a
single end-to-end omni model; V2 is the intermediate architecture that (a)
works today with hand-coded baselines, (b) generates its own training
data, and (c) lets each learned component be trained, measured, and
promoted independently against a yardstick that already runs.

## 1. What the baseline taught us (measured, 2026-08-29 → 09-01)

The turn-based cascade is production-grade at what text can carry:
large-v3-turbo whole-turn decoding is ceiling-exact on clean speech
(2.78% / 1.18% WER — the model floor). Everything beyond the words needed
rigid recovery, and each recovery mechanism exists because a live failure
demanded it:

| recovered signal | mechanism | the failure that forced it |
|---|---|---|
| who is speaking | 5-feature fingerprint + centroids | v1 cosine scored 0.990–0.998 between DIFFERENT live voices; nobody ever split |
| still speaking | looks_complete + HOLDING (8 s) | agent cut the user off mid-thought, live |
| turn boundaries in latched dialogue | voice-change sentry | 16 s+ "listening" while two voices merged into one S1 block |
| decode stability on long buffers | DriftAnchor (≥6 s + drift) | the "architectural" 576-class failure; anchor fixed it: 33.6→19.6% @120 s, no losing class in 4 streams |
| not answering itself | self-voice fingerprint rejection | AEC leakage: its own reply transcribed as user turns at Kokoro's pitch |
| not answering silence | stock-phrase + wpm/energy guard | Whisper hallucinated "Thank you." from residue; a reply feedback loop ran for a full screen |
| being interruptible | windowed BargeMonitor (10-of-12) | consecutive-streak reset on inter-word dips; interruptions transcribed but never cut playback |

Two lessons generalize. First: every one of these is a fixed threshold
found by watching a failure — rigid logic does work, but it does not
transfer; each new content class costs a new constant. Second, and more
important: the wake path is text-matched, and text matching inherits every
STT failure — the misheard-name rescue table (yeager→jaeger), the wake
vocabulary drift bug ("hey google" silently unmatched for a day), and
hallucinated phrases that look identical to real ones once they are text.

## 2. Architecture

```
                       ┌─────────────────────────────────────────┐
                       │                SERIES                   │
  audio ──┬──────────▶ │ whisper ──▶ trained adapter ──▶ Gemma   │
  16 kHz  │            │ (utterance rate · semantics only)       │
          │            └─────────────────────────────────────────┘
          │                                              ▲
          │            ┌───────────────────────────┐     │ separate
          └──────────▶ │         PARALLEL          │     │ projection
             frames    │ speaker-embed   (0.6 s)   │     │ (context
             30 ms     │ turn-state      (30 ms)   ├──▶ fusion tokens)
                       │ wake-spotter    (0.3 s)   │
                       └───────────────────────────┘
```

Principles:

1. **The series channel stays pure.** Whisper/adapter carry words. No
   paralinguistic recovery lives behind the transcription layer — that was
   V1's mistake: context computed in the adapter, downstream of the layer
   that discards it.
2. **The parallel channel never waits.** Frame-rate heads read raw audio;
   speaker, turn state, and wake are available in milliseconds, before —
   and independent of — any decode. A hallucinated transcript cannot
   produce a wake event; a mis-transcribed name cannot lose one.
3. **Convergence, not premixing.** Fusion assembles one record per
   utterance and (target state) feeds Gemma through its own small
   projection as dedicated context tokens. Channels remain independently
   trainable and independently attendable.
4. **Wake is waveform-matched.** Query-by-example: enrollment stores an
   embedding template from ONE example; spotting is distance in embedding
   space. One-shot new phrases by construction. No STT in the loop.

## 3. The node system

Same contract as V1 (`node.json` manifest with typed consumes/produces,
`runtime.py:build`, weights owned by the folder), plus `stats/` — every
session appends what the node measured about itself. A node is one
independently trainable, independently promotable capability:

    nodes/para-speaker-embed-v0/   fingerprint + centroids (weights/: centroids.npz)
    nodes/para-turn-state-v0/      envelope + end-of-turn pressure
    nodes/para-wake-spotter-v0/    QbE template match (weights/: templates.npz)
    nodes/semantic-whisper-v0/     off-the-shelf turbo (series stand-in)
    nodes/fusion-context-v0/       fused record + the Gemma projection seam

**The yardstick rule.** Every v0 runtime is the hand-coded baseline
vendored from the measured voice family. A trained replacement swaps in at
exactly one function seam (`fingerprint()`, `embed()`, the EOT rule) and
ships only if it beats the v0 on the node's own metric. This repo has
reverted five plausible streaming "improvements" that measured worse; the
process is the point.

## 4. Training plan

**Supervision already exists.** The structured voice apps log, per turn:
speaker id + confidence, voiceprint vector, complete/holding, addressed,
prosody, pause_before_ms, boundary — and crucially the OUTCOME (a pause
followed by the same speaker continuing is a ground-truth "hold was
right"). These JSONL sidecars are labeled training data generated from
real conversations by the rigid logic: the teacher network, in effect.

**Phase 1 — shared contrastive embedding (speaker + wake).**
One small encoder (log-mel → ~256-d), trained contrastively: same-speaker
segments attract, different-speaker repel (LibriSpeech clean360 +
`ami-annotations-v1` + our logged voiceprint sessions); same-phrase
positives from augmented copies (speed/pitch/noise) for the wake use.
Speaker head = centroid matching in the learned space (replaces
`_fingerprint`); wake head = template matching in the same space (replaces
`embed`). Gates: speaker — clusters ≥3-voice live sessions with 0
merges of different-gender pairs and beats the fingerprint on same-gender
pairs (its stated ceiling); wake — ≥95% detection at ≤1 false wake/hour on
enrolled phrases, ONE enrollment example, tested against the v1 node's
measured bar (barge-in head: 81.0% F1) as the floor.

**Phase 2 — end-of-turn prediction.**
Replace silence-ramp `eot_pressure` with a learned P(turn ends | audio so
far): supervision from logged endpoints + continuations (the holding
outcomes). Gate: earlier median commit than the 500 ms hangover at equal
or fewer false endpoints on the sim + live replay suites.

**Phase 3 — fusion projection into Gemma.**
Freeze both channels; train the small projection that turns fusion records
into context tokens, against response-quality preferences (the 02 `--plain`
A/B finally answers "does context improve replies" — run it first; it is
the cheapest unrun experiment in the repo). V1\'s `training/` stack is the
machinery; checkpoints land in each node\'s `weights/`.

**Phase 4 — collapse toward omni.**
With heads proven, the cascade is a curriculum: distil the whole graph
(series + parallel + fusion decisions) into the V1 encoder+adapter trunk —
the adapter learns to carry paralinguistics natively, and the parallel
heads become training scaffolding. Native full duplex (emit while
receiving) remains the horizon beyond that, per DUPLEX.md.

## 5. Evaluation

Per-node, before promotion: speaker (cluster purity on 3-speaker live
replays), wake (detection/false-alarm curves per enrolled phrase,
one-example enrollment only), EOT (latency vs false-endpoint tradeoff),
fusion (the --plain A/B). System: the scripted end-to-end campaign built
for the voice family (sim harnesses drive real models through wake /
barge / hold / dismissal conversations) adapted to run_v2, plus the WER
matrix (clean210/576, AMI×2) to prove the parallel channel never degrades
the series one. Every claim traces to a run; anything unmeasured is
labeled unmeasured.

## 6. Risks, stated

- v0 wake embedding is speaker-biased (voice+prosody > phrase content) —
  acceptable for a personal device, wrong for multi-user; Phase 1 fixes.
- Same-gender speaker separation is the fingerprint ceiling — stated since
  v2 of the fingerprint; Phase 1's gate targets it directly.
- Fusion tokens are unproven value: the [context:] line's effect on reply
  quality has never been measured. Phase 3 begins with that A/B, and if
  context shows no lift, the parallel channel still pays for itself in
  wake robustness and turn-taking latency.
- The frame bus is single-process today; real parallelism (threads per
  node) is an engineering step deferred until a head is heavy enough to
  need it. (`ponytail:` sequential bus, thread it when a node exceeds ~5 ms/frame.)

## 7. Current status

Scaffold complete and green (7/7 node-contract checks, no models): two
synthetic voices separate in the parallel channel; utterances endpoint
exactly once; one-shot enrollment persists and spots with no STT; a
different voice does not false-wake; fusion converges the channels.
`run_v2.py --wav/--mic` runs the full pipeline; `--no-stt` demonstrates
the thesis — the parallel channel functioning with the text layer absent.


## Addendum 2026-09-02 — full multimodal surface + the beat-the-baseline gates

The lessons of the cascade demo apps are now applied to V2 wholesale. V1's
flaw was bolting context into the adapter behind Whisper; V2 runs the
parallel channel beside the series one — and as of this addendum the graph
carries EVERY modality and the verified interaction layer:

    series:    audio → whisper → Gemma        (semantic-whisper-v0)
               image → mmproj  → Gemma        (semantic-vision-v0)
               typed text      → Gemma        (text-input-v0 — typing IS
                                               addressing; full duplex
                                               covers typed interruptions)
    parallel:  speaker-embed · turn-state · wake-spotter (QbE, one-shot)
    fusion:    context record (+wpm, +energy → the guards run on GRAPH
               evidence, not bolt-on logic)
    output:    llm-gemma-v0 (rolling log = the only memory) → tts-kokoro-v0
    device:    audio-duplex-io-v0 (48 kHz AEC, windowed barge, floor policy)

`run_v2.py --converse half|quasi|full` is the functioning agent on this graph:
endpointing comes from the para channel's eot_pressure, addressing prefers
the WAVEFORM wake (QbE templates; text-wake only as unenrolled fallback),
HOLDING-lite, farewell-then-close, garble guard — the cascade's behaviors,
re-homed onto the channels that will be trained.

### Gates — "beats everything we have done" means these numbers

Baseline = the measured cascade (gemma_multimodal + the voice family).
The trained V2 ships only when it clears ALL of:

| gate | baseline to beat | how measured |
|---|---|---|
| time-to-first-audio | < 2.0 s (cascade ≈2.0–2.6; MiniCPM ref 1.6) | converse probe, wake→first TTS sample |
| turn-taking | fewer premature replies than 500 ms + HOLDING at equal latency | sim campaign ported to converse |
| wake | ≥95% detect, ≤1 false/hr, ONE enrollment example | QbE eval vs family text-wake |
| speaker | beats fingerprint on same-gender pairs (its stated ceiling) | 3-voice live replays |
| WER | not worse than turbo turn path (2.78/1.18 clean) | the WER matrix |
| probes | text memory, vision, voice memory — all pass | mm_probe suite vs converse |
| duplex | barge cut ≤400 ms; full duplex on all inputs incl. typed | sim + live |

Until a trained head clears its row, the v0 (hand-coded, cascade-verified)
holds the slot. That is the whole method: the scaffold converses today,
and every seam has a yardstick bolted to it.


## Addendum 2026-09-02 (later) — the pipeline collapses to ONE model

Probed live (nn_reply_probe, 5/5): the frozen Gemma that receives the
adapter's audio embeddings FOLLOWS INSTRUCTIONS about them. With a
system prompt + rolling history around the audio marker, one generation
emits `HEARD: <verbatim transcript>` + `REPLY: <answer>` — transcript for
the guards, answer for the mouth, no second model, no input text seam.

Measured, same machine, warm turns:

| path | models | per-turn | TTFT |
|---|---|---|---|
| cascade (transcribe → GGUF Gemma → reply) | 2 Gemmas | ~5.5 s | — |
| **direct reply (`--reply nn`, default)** | **1 Gemma** | **1.4–1.6 s** | **0.44 s** |

Memory works through the audio path (name recall PASS), arithmetic PASS,
format compliance 3/3. `--converse` now defaults to semantic=nn +
reply=nn: the second Gemma loads lazily only for typed/image turns.
Gating is paralinguistic-first (QbE wake / follow-up window); text guards
run on HEARD post-generation and may discard a reply — one wasted
generation per guard trip, the honest cost of understanding without text.

Remaining text seam in the entire app: reply text → Kokoro. That is the
output decoder the training stack trains next — and with TTFT at 0.44 s, the
first-audio gate (<2.0 s; MiniCPM 1.6 s) is now expected to fall to the
TTS handoff, not the understanding path.


## Addendum 2026-09-02 (final) — trained networks mounted for the para channel

Smoke-verified on this machine (ecapa_smoke, all PASS): the parallel
functions now run on trained networks where trained networks exist:

- **speaker ID**: ECAPA-TDNN via the training stack's SpeakerEmbedder/Registry —
  two synthesized voices separate with perfect purity (S1/S2/S1,
  conf 1.0) and re-identify across turns.
- **wake by waveform**: one-shot templates in the LEARNED embedding
  space — enrolled "hey computer" matches at cosine 0.993; unrelated
  speech in another voice rejects outright. `--enroll` writes both
  spaces (hand features remain only as the frame-rate early latch).
- **turn detection**: the last hand-coded para function. The trained
  perception heads (81% F1 self-voice-aware barge) already live in the
  adapter checkpoint; mounting their 25 Hz events into converse gating
  is the next work order — wiring, not training.

Hand-coded logic survives only as policy (floor policy, dismissal) or as
the yardstick v0s, exactly per the method.


## Addendum 2026-09-02 (heads mount) — trained perception heads live, calibration pending

`para-perception-heads-v1` mounts the checkpoint's five per-frame heads on
the shared trunk (one 0.25–0.29 s pass, no second model). Smoke, synthetic
audio: **speaker_change correctly localized a two-voice boundary**
(frames 55–63 at the clip midpoint); every head ranked in the right
direction (directed: addressed 0.602 > overheard 0.549; overlap highest
on the mixed clip at 0.975) — but margins are thin because Kokoro speech
is out-of-distribution for heads trained on real room audio. Therefore:
the converse `directed` gate runs CONSERVATIVE (< 0.35 blocks; cannot
false-block either measured case), and full-strength head gating
(turn_end replacing the envelope, overlap driving barge) is gated on the
per-head eval against real recorded data — the training stack's eval stack, real
sessions, per-head ROC vs the cascade yardsticks.


## Gate scorecard — measured 2026-09-02

| gate | target | measured | verdict |
|---|---|---|---|
| time-to-first-audio | < 2.0 s (MiniCPM 1.6 stretch) | **1.38 / 1.47 / 1.64 s** incl. the 500 ms endpoint | **PASS — beats MiniCPM on 2 of 3 turns** |
| WER clean210 (same harness) | ≤ 2.78 (turbo) | 9.09% | FAIL |
| WER clean576 | ≤ 1.18 | 5.34% | FAIL |
| WER AMI | ≤ 26.52 | 69.06% raw / 64.64% gain-normalized | FAIL — and the gain diagnostic proves it is a DISTRIBUTION gap, not level sensitivity |
| memory/probes | all pass | voice 5/5, text unified (one model, one history) | PASS |
| wake / speaker | one-shot, learned | ECAPA 0.993 / conf 1.0 | PASS |

**The demo story as of today**: the omni concept is PROVEN on latency and
architecture — one model, embeddings-native input, trained para heads,
1.4–1.6 s to first audio — and honestly SHORT on transcription accuracy
off the training distribution. The training work order that closes it is
already in-tree: the `stock-whisper-large-v3-stream-untrained` scaffold
(the stronger encoder turbo uses) + the 600 h campaign with domain mix
extended toward conversational audio (AMI-like). Numbers to beat:
2.78 / 1.18 / 26.52. The output speech decoder remains the second
training campaign. Everything else on the gap list is closed.


## Addendum 2026-09-03 — the three-tier duplex ladder, verified vs the baseline

`--converse` now runs the SAME three-tier duplex ladder as the multimodal
baseline (`gemma_multimodal.py --audio …`), tier for tier, so the omni graph
and the cascade baseline compare like for like:

| tier | flag | what it adds |
|---|---|---|
| half | `--converse half` | mic closed while speaking; wake + follow-up window |
| quasi | `--converse quasi` | open mic + 48 kHz AEC duplex device + continuous barge watcher + `--barge stop\|continue` |
| full | `--converse full` | + streaming semantics: a live **forming** preview while the user is still talking, and commit-and-trim so an unbounded monologue never grows without limit |

### Scripted converse campaign (real models, fake audio device) — 27/27

`scratchpad omni_sim.py` drives `run_v2.converse` with a fake `sounddevice`
(scripted mic, acoustic-echo coupling, playback capture) and real models
(trained adapter + Gemma + Kokoro + ECAPA), one scenario per behavior:

| session | scenarios | result |
|---|---|---|
| half | wake→reply, follow-up at turn-rate, hold→merge, hold→expiry, one-memory across voice+text, farewell-then-close | **12/12** |
| quasi | wake, **barge cuts playback**, farewell-then-close | **6/6** |
| barge-continue | overtalk heard, **playback NOT cut** (floor policy = continue) | **4/4** |
| full | live forming preview while listening, long monologue answered | **5/5** |

Per-turn NN latency in-campaign: **0.67–0.91 s** (TTFT 0.25–0.36 s), holding
the half-duplex latency contract on the one-model reply path.

### Engine defects the campaign found and fixed (all live-relevant)

1. **Format collapse with history** — with a rolling history in context the
   one-model reply path silently dropped the `HEARD:/REPLY:` format, so every
   follow-up parsed as an empty transcript and the agent held→expired→said
   nothing. Fixed by prefilling the assistant turn with `HEARD:` in
   `core/mlx_audio_streaming.respond()` (zero added latency).
2. **Punctuation-only completeness** — the trained adapter transcribes
   ALL-CAPS with no punctuation, so "ends with . ! ?" judged EVERY spoken turn
   incomplete and every follow-up paid the 8 s hold. Ported the family's
   word-level `looks_complete` into `fusion-context-v0`.
3. **QbE wake suppressed the text fallback** — the pre-gate hard-returned on
   `wake.templates`, so enrolling a (crude v0) QbE template that never fires
   made the assistant deaf to the reliable text-wake path. QbE is now
   ADDITIVE: a miss falls through to directed-head + text-wake.
4. **Main loop stole barge frames** — during playback the main loop dequeued
   `io.q` frames *before* the `if speaking: continue` check and discarded
   them, racing the barge watcher for the same queue. Fixed so the watcher
   owns the queue during playback.
5. **Stale queue starved barge** — `speak()` now `io.drain()`s at playback
   start, so the watcher evaluates audio DURING playback, not backlog buffered
   before the agent spoke.
6. **Hold continuation dropped / expiry leaked to 2nd Gemma** — held thoughts
   now merge with the follow-up and regenerate on the whole thought on the ONE
   model; empty transcripts are dropped, never held.

### Self-echo and commit-and-trim — proven off the flaky E2E path

- **Self-voice rejection**: measured offline — the agent's own voice through
  the playback-capture path matches its ECAPA self-vector at **cosine 0.91**
  (dropped); a different-voice user turn falls below the 0.45 gate (kept).
  (The E2E round-trip is not used: the fake device's crude resampling mangles
  ECAPA features, and true 48 kHz resampling per frame starves the threads.)
- **Commit-and-trim**: the 12 s long-turn safety valve is extracted to a pure
  `stream_split()` and covered by `run_v2 --selftest` (17/17). It rarely fires
  live: real and synthetic speech both carry sub-500 ms pauses, so turns
  almost always endpoint before 12 s.

### Parity vs the multimodal baseline

| capability | `gemma_multimodal` (cascade baseline) | `gemma omni v2 --converse` |
|---|---|---|
| half / quasi / full duplex | ✓ (`--audio plain·structured·quasi·full`) | ✓ (`half·quasi·full`), campaign 27/27 |
| barge `stop`\|`continue` floor policy | ✓ | ✓ |
| speaker ID | pitch fingerprint | **ECAPA (learned)** |
| wake | text + waveform | QbE waveform + text fallback |
| HOLDING incomplete-turn | ✓ | ✓ (+ continuation merge) |
| self-voice echo reject | fingerprint | **ECAPA, cosine 0.91** |
| stock / garble guards | ✓ | ✓ |
| vision (mmproj) + typed text | ✓ | ✓ |
| memory | rolling log | one model, ONE history across voice+text |
| **input → LLM** | **TEXT transcript seam** | **NN audio embeddings, no text seam** |
| **reply** | **second Gemma** | **same Gemma, one generation** |
| per-turn latency | ~5.5 s cascade | **0.67–0.91 s** |
| recognition accuracy on hard audio | turbo-class | **behind** — the training gap above |

Reading: as a running agent the omni graph now MATCHES the baseline across the
whole duplex ladder and BEATS it on the two things the architecture promised —
no text seam on the way in, one model on the way out — at a fraction of the
latency. It stays behind on exactly one axis, recognition accuracy off the
training distribution, which is the training work order, not a wiring gap.


## Addendum 2026-09-05 — measured consolidation

Tier-matched routing is implemented: medium for half/quasi, current causal for
full, quasi default, 27/27 conversation checks. Speaker context remains opt-in
because the measured recall gain carries a transcription tradeoff. Real
image→typed→voice shared-memory turns pass 4/4 through the existing vision twin.

Two raw-audio overlap TCN trials and three variants of a heads-free medium
causal port failed their qualification gates and were not promoted. An existing
Silero NN now independently guards generation-time barge in native neural-VAD
mode; it is speech detection, not an overlap-head replacement. The physical
retest passes 12/13 mechanical checks but fails agent tool use. AEC tail and
native agentic action fulfillment remain red. Results, counterexamples, hashes
and reproducing commands are collected in the
[consolidation review](OMNI_CONSOLIDATION_REVIEW_20260905.md).
