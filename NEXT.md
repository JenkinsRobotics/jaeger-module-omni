# Next steps — current plan

2026-09-13 two action-dispatch defects fixed and measured: bare `call:` after
prose is now dispatched, and `append_file` line-separates. Text actual file
outcomes **10/12 → 12/12**; speech unchanged at **1/12**, so the frozen demo
suite stays **35/48** and demo acceptance stays FAILED. The speech action gap
is now isolated to the ear and the frozen spoken fixtures (wrong filenames,
wrong content, turns that stop after the first tool) — no further parser or
tool-layer work will move it. Barge-continue overtalk is reproduced at 3/4
uninstrumented; the utterance is lost before `on_utterance`, not rejected by a
gate, and remains open. Evidence:
[`action-dispatch-fixes-20260913.json`](research/candidate_evals/action-dispatch-fixes-20260913.json).

Unchanged blockers to shippable: 95% demo completion (now 35/48), three
consecutive 30-minute physical sessions (zero attempted), physical
uninterrupted speech (0 of 3 replies survived; all cut with `played_clauses: 0`
under 0.47 dB measured AEC attenuation), and the 300 ms AEC p95 gate (601.9 ms).

2026-09-08 implementation and measured decisions:
[`docs/OMNI_DEMO_PROGRESS_20260907.md`](docs/OMNI_DEMO_PROGRESS_20260907.md).
Keep quasi default in converse, medium for half/quasi and the existing causal ear
for full. No trial is promoted. The frozen demo suite is **35/48 (72.9%)**;
112 CPU tests, 14 subtests and 18/18 selftest pass. The default barge campaign
first fails 3/4, then passes 4/4; other tiers pass 23/23. A neural endpoint passes
27/27 but triggers an extra reply to self-echo and is rejected.

The five-minute physical replay records 304.9 seconds, zero drops/overflows and
clean shutdown, but **zero successful actions and zero uninterrupted replies**.
The corrected physical gate is 14/16, failed. Do not start the three 30-minute
acceptance sessions until these behavioral failures are fixed.

Next: fix self-echo admission and AEC tail together with an independent overlap
model that passes event reliability, not just frame F1; then prove actual speech
multi-step outcomes using acoustically unambiguous frozen requests. WavLM offline
and bounded-window trials are rejected. Do not continue their recipes unchanged.
The distinct causal audio-KL trial is also rejected: steps 50/150 score
5.97/5.76% clean and 32.10/31.95% meeting versus the full control 5.03/28.23%.
Keep both immutable candidates disabled; no continuation.

A runnable [single-brain multimodal demo](docs/MULTIMODAL_DEMO.md) now accepts
image/audio/text turns and saves spoken replies. It passes a three-turn smoke
and the joint probe passes 20/20 synthetic checks. Expand to natural images,
long sessions and matched MiniCPM visual/action/duplex tests. Current MiniCPM
comparison covers ASR and 12 questions only: MiniCPM wins both ASR sets; Jaeger
has lower complete-text request latency, excluding startup, endpointing and TTS.

The dated checkpoints below remain historical evidence, not new release passes.

2026-09-07 user-directed side research: develop biologically inspired local
learning and internal prediction models. The CPU-only
[`predictive coding lab`](research/predictive_coding/README.md) now demonstrates
sensory inference and actual local-error learning against matched backprop
controls. Three toy-world seeds reach near-identical quality; no production
benefit or biological equivalence is established. Next research step is a
controlled causal-audio experiment on the disjoint overlap corpus. This research
does not promote a node or change the release gates below.

2026-09-05 tested selection: **quasi default**, medium on half/quasi, current
causal ear on full. Final conversation campaign **27/27**, selftest **18/18**,
package tests **65**, module checks **31/31**, vision/text/voice memory **4/4**.
The selected adapters and all shared Whisper-medium encoder bytes are unchanged.
No weights were committed; 31 immutable nodes / 13 profiles are registered.

The ordered work queue has now been exercised, including two parallel-overlap
training recipes, the heads-free causal port, speaker-context controls, mixed
vision turns, three full native agentic corpus runs, and two physical five-minute
runs. **The release gates are not all green.** Keep the failed training variants
disabled and context opt-in/off by default. Native tool results now return to
Gemma, and a separate raw-audio Silero instance prevents background RMS from
continually cancelling generation. The physical retest improves **9/13→12/13**
checks, but still makes zero tool calls and suffers playback interruptions.

Next work must address the measured deficits: a stronger independent overlap
model that qualifies on F1 AND event reliability, the AEC tail (601.9 ms p95),
and native action fulfillment (31/162 original scorer; 35/162 after a separate
paired spoken-integer diagnostic). The discovery prompt and an initial-schema
ordering pilot did not resolve action fulfillment. No same-recipe 41M or scratch
TCN continuation is supported by these results. Preserve the current full ear
until a distinct causal method beats both matched WER gates.

Full results, limits and evidence pointers:
[`docs/OMNI_CONSOLIDATION_REVIEW_20260905.md`](docs/OMNI_CONSOLIDATION_REVIEW_20260905.md).

## Pipeline lockdown — what we have and what we refine (2026-09-03)

Everything below this section remains the standing plan except where a lane
explicitly supersedes it. Import record: [`STATUS.md`](STATUS.md),
[`CONVERSE.md`](CONVERSE.md).

### What we have (measured)

Two Gemma-4 E4B conversational lineages in one repository:

1. **Native-tower path** (`live.py`, the GUI, `omni.py`): plain voice 12/12
   at 575 ms; agentic 12/12 at 2421 ms over 94 Omni-safe tools; Silero VAD
   green; 17.8 ms barge cancellation; physical five-minute capture integrity
   green. Red: companion admission (4/12 named recall), AEC p95 tail
   (602/843 ms vs the 300 ms gate), agentic full corpus (28/162).
2. **Trained-adapter converse path** (imported Gemma Omni V2,
   `python -m jaeger_agent_omni.converse`): campaign 27/27 across the
   half/quasi/full duplex ladder with barge stop|continue, wake/hold,
   one-model HEARD/REPLY at ~0.9 s per turn (TTFT 0.40–0.47 s),
   commit-and-trim; text+vision+audio in, text+audio out.

Ears at the series seam, matched devclean64, identical scorer:
`jaegeromni-41m-medium` **4.82%** (cases12 12/12 at 560 ms, 3.97% on the
51,311-word test-clean lockbox, AMI 32.99%) versus the imported
`gemmaomni-v2-stream-bargein-93h` **5.65%** as wired / 5.76%
instructed (streaming causal, 0.60 s median transcribe, carries the only
trained directed/overlap heads — overlap F1 0.81).
`jaegeromni-41m-causal-balanced` stays packaged as the duplex experiment
variant (24/24 at 574 ms, AMI 28.23%).

Perception, two generations side by side:
- **Parallel on raw audio** (the architecture we are locking): v0
  turn-state / speaker-embed / wake-spotter on 30 ms frames, trained ECAPA
  (wake similarity 0.993, speaker separation 0.51/0.17, self-voice cosine
  0.91), CampPlus live labels, Silero gate.
- **Chained on the adapter trunk** (transitional, to be retired): the V2
  directed/overlap heads riding the 93h adapter; the 41M audio-perception
  shadow node (scene 0.952 F1, events 0.088/0.254 — policy-banned).

Output: Kokoro is the production mouth on both lineages; the hidden-state
talker stays experimental. Harnesses: converse selftest 18/18, the
converse_sim campaign, the tiered release benchmark, the locked WER gates
plus the new twin `research/converse_ear_wer.py`, and
`training/compare_candidate.py`.

### Operating model (user, 2026-09-04) — quasi-duplex default, tier-matched ear

The payoff of parallelizing perception: the transcription ear no longer
carries speaker/overlap/turn, so it is free to be the BEST ear per tier
instead of the one with heads.

- **Half + quasi (turn-based, endpoint-then-reply)** run on
  `jaegeromni-41m-medium` (4.82%, bidirectional) — legal because the whole
  utterance exists at endpoint. Perception comes from the parallel NNs:
  ECAPA (speaker/wake/self-voice, live), para-turn-state (endpoint, live).
- **Full (live forming preview)** is the ONLY tier that needs the causal ear
  (`jaegeromni-41m-converse`, 5.03%), and only for the pre-endpoint preview.

Default operating mode is **quasi-duplex** (open mic + AEC + barge, agent
decides) — the point of an end-to-end omni agent. Build order for simplicity:
**half → quasi (default) → full**. Quasi-by-default is reachable NOW with the
best ear + existing parallel NNs; no GPU window needed to ship it.

The one perception gap for robust quasi barge is a **parallel overlap/barge
NN** — today barge runs on the energy BargeMonitor + ECAPA self-voice (passed
the campaign); a trained overlap NN (Lane-1 Phase-1) upgrades it. speaker ID
is already ECAPA; turn is already para-turn-state-v0.

Tier matching is implemented and verified: `41m-medium` (4.82%) serves
the turn-based tiers and the current causal ear serves full's preview. The causal
streaming-port GPU job (Lane-1b) therefore matters only to unlock/upgrade
full-duplex, not to ship the quasi default. The causal-port trials failed their matched gates; the current full ear remains selected.

### Lane 1 — update the 41M omni to the parallel idea, then the ear bakeoff

User-directed 2026-09-03; this supersedes the "no more adapter training"
rule below for THIS architectural change only (that rule was about squeezing
the same recipe, not about the perception split).

1. Train the Phase-1 parallel perception encoder (shared contrastive
   speaker+wake on raw log-mel, whitepaper §Phase 1). Promotion gates:
   event-level false-alarms/minute, detection delay, wake recall,
   same-gender separation — and overlap/directed parity with the trunk
   heads (F1 ≥ 0.81) before any trunk head is retired.
2. Bring the proven 41M quality into the clean series seam: port or retrain
   `jaegeromni-41m-medium` into the streaming-causal, heads-free form
   factor at the converse adapter boundary (one function seam, per the
   yardstick rule).
3. Judge with the matched framework only: devclean64 + cases12 + ami64 (+
   lockbox at the end) through `compare_candidate.py`, using the committed
   twin harnesses. Expected outcome (stated bet): the whisper-adapter 41M
   still wins accuracy — and with perception parallel, it drops into
   converse with zero perception cost. If the 93h ear wins any gate, keep
   the loser packaged, dead, per registry policy.

Bakeoff outcome (2026-09-03): the naive cross-mount penalty (6.07% vs 5.03%
own-seam) was PROMPT coupling — aligning the adapter's training-time
transcript prompt recovers 5.03% exactly. Consolidation step 2 is DONE:
`jaegeromni-41m-converse` (the repository's causal-balanced 41M,
prompt-aligned, immutably packaged, untrained directed head suppressed
fail-safe) is now the converse semantic channel — node reproduces 5.03%
through its own build, campaign 27/27 on the new ear, imported 93h retired
to packaged-disabled. The bet direction held: nothing beat 41m-medium's
4.82%. Remaining Lane-1 work is unchanged in kind: (a) Phase-1 parallel
perception encoder, (b) the 41m-medium streaming port to reclaim the last
0.21 points at the seam — both GPU-window jobs judged by the same matched
gates. Scorecard table: [`CONVERSE.md`](CONVERSE.md).

### Lane 2 — the conversational front (duplex tiers stay selectable)

Ruling (user, 2026-09-03): the duplex tiers are a FEATURE, not duplication
to remove. Half, quasi, and full stay user-selectable modes, with barge-in
enable/disable as an independent choice — like a person choosing how to
hold a conversation. Do not consolidate the three existing duplex
implementations (live.py, the GUI's DuplexModePolicy, the converse ladder)
now; each is verified on its own surface, and a tidiness refactor risks
measured behavior for zero user-visible gain. Revisit only if the
duplicated policies ever produce an actual divergence bug. The possible
future evolution — one always-full-duplex engine where the agent itself
throttles capture/barge to IMITATE the lower tiers — is noted and
deliberately deferred.

What this lane still does: expose the quasi tier as a selectable option on
the GUI surface once the in-flight duplex-selector/observer work lands
(adding a mode choice, not removing one); decide the final conversational
spine — current recommendation: the converse ladder becomes the
conversation engine, the agentic runtime mounts on it through the same
session-adapter pattern `live --agentic` uses, and the native-tower path
remains the vision-interleave and AEC bench. Companion admission's red gate
(4/12 recall) gets attacked with the imported addressing stack (additive
QbE wake + text-wake + directed head + follow-up window) against the
COMPANION.md categories — not by weakening the parser.

### Lane 3 — standing red gates (unchanged, still required for "locked")

AEC 300 ms tail (contiguous mode rerun next), agentic full-corpus
capability-reveal seam (28/162), and the one physical five-minute release
run with tools, barge, roll, and zero drops. Closed and staying closed:
ternary packaging, packed QKV, hidden-talker promotion, and same-recipe
41M continuations.

---

The new release-depth benchmark supersedes the earlier inference that source
parity plus cases12 meant broad agent parity. Source/code parity is exact and
simple native-audio questions remain 12/12, but the full 81+81 corpus is only
28/162 with 20/128 required tool routes. Fix the capability-reveal seam first:
the compact eight-tool voice prefix keeps simple latency reasonable, but Gemma
usually declines file, memory, scheduling, skill, code, and workflow requests
instead of calling `load_tools` and continuing. Develop against category-level
held-out improvements, not individual prompt exceptions. Keep the bounded web
case deadlines; network unavailability must score as an error without erasing
the other 161 rows. Preserve zero safety failures and zero markup leakage.

Companion admission is the other immediate production blocker. Its current
operating point is simultaneously one false response among 12 unaddressed
questions and only 4/12 recall for explicit “Hey Jaeger” questions. Do not
loosen the parser globally. Train/calibrate an audio-native addressee signal
with phonetic name variants, conversation state, speaker continuity, and
hard-negative nearby questions, then require zero ambient false responses and
at least 90% addressed recall.

AEC passes false-barge and recall gates but misses tail latency: 0/43 false,
43/43 detected, 188 ms median, 602 ms p95 versus the 300 ms gate. Work the
outlier path rather than optimizing the median. Release evidence must finally
come from one five-minute physical run with neural VAD, full duplex, active
AEC, companion RESPOND and LISTEN decisions, at least three agent tool calls,
a deliberate stopped barge-in, a context roll, spoken END-to-audio timing,
zero drops/overflow, and no runtime errors. The run report now records agentic
tool names/counts so this gate is machine-verifiable.

The copied Jaeger Agent control plane is integrated CPU-side and the package
layout is now `nodes/`, `tools/`, `skills/`, `memory/`, plus internal `core/`.
Do not add a transcript bridge: `live --agentic` must continue from the native
audio-prefilled Gemma cache, and the copied `listen`/`text_to_speech` tools
remain denied. Before making agentic mode a default, run a physical five-minute
gate covering ordinary conversation, at least three tool calls, skill loading,
memory persistence across turns, deliberate barge-in during both plain and
tool-using answers, forbidden-tool rejection, and context roll. Report
END-to-first-audio separately because safe tool-markup buffering currently
removes sentence streaming. The next output-side architecture experiment is a
gated hidden-state talker; do not describe decoded-text Kokoro as tensor-native
in the meantime.

The post-alignment automated gates are green: frozen-ear audio remains 12/12
at 575 ms, native-audio agent mode is 12/12 at 2421 ms with three valid math
dispatches and zero unexpected calls, embedded/source corpora and tool names
are exact, and the same 101 loop/parser/safety tests pass in both namespaces.
Do not spend GPU time on this seam. The remaining agentic deficit is runtime
latency: the eight-tool compact voice prefix is much better than the original
18-tool prefix (2421 ms versus 6766 ms median) but still materially slower than
plain Omni. Optimize schema routing/prefix attention only against corpus-level
tool-routing accuracy; never obtain a latency win by silently removing access
to registered capabilities.

The richer neural observation node is now packaged as
`jaegeromni-41m-audio-perception`. Its four-way scene head passes a
source-file-disjoint gate at 0.952 macro-F1 and its frozen ear retains 12/12;
use it for shadow-mode live collection. Do not replace Silero or drive
barge-in/endpoint policy from its event heads: unseen-meeting speaker-change
and turn-end are only 0.088/0.254 F1. The next useful training data is labeled
physical 1:1 conversation with true speech start, semantic completion,
interruption intent, playback overlap, music, and household noise. Promotion
requires event-level false alarms/minute and detection-delay gates, then a
prefix-consistent streaming encoder; more framewise linear-head epochs are
closed.

The separate pre-Whisper neural acoustic controller now has its FP16 causal
baseline, typed runtime contract, factual JSONL/NPZ trainer, source-disjoint
split guard, CPU gradient check, and shadow profile. The 262,060-parameter node
consumes microphone PCM plus the exact playback reference and emits enhanced
PCM plus typed speech/self/external/music/noise/overlap/turn state. The next
GPU window is data curation and training, not more architecture churn. Do not
spend it on ternary QAT: the packed Metal go/no-go benchmark delivered 0.98x
FP16 speed at the representative shape despite 87% smaller weights. Keep
Silero and Speex authoritative until the node independently passes waveform
AEC, near-end survival, false-opens/minute, endpoint-delay, and 10 ms compute
gates.

Keep the separate JaegerOmni 41M feed-forward-only ternary result as an
embedded-deployment lead, not a production option. One-shot full-adapter
ternarization fails (8/12), while selective FF ternarization retains 12/12 and
4.55% pooled normalized WER versus 4.69% unchanged across 4,482 words. Do not
package it until a truly packed mixed FP16/W2 runtime demonstrates total-process
memory and end-to-end latency; the theoretical saving is only about 28 MiB
beyond an FP16 adapter on the current 32 GB host.

The always-on companion control path is now implemented behind
`--interaction-mode companion`: Gemma emits a fail-closed RESPOND/LISTEN
decision from native audio, follow-up state is judged at speech start, and
half/full duplex are explicit. The desktop demo exposes the same modes. Causal
speaker-change/overlap/turn-end/speaker-ID probabilities are recorded as
shadow telemetry and do not control policy because the blended checkpoint's
heads remain uncalibrated. Populate and run the corpus-level gate described in
`COMPANION.md` before calling this production-qualified; specifically require
direct address, follow-up, overheard human conversation, recorded media,
environmental noise, incomplete thought, and self-speech categories. The
first named-address run is red at 4/12 recall even though the audio-answer
probe is 11/12; train/calibrate a dedicated addressed-to-agent head with name
variants and negative human-human speech rather than weakening the parser.

The repository-owned Silero neural VAD is now the default acoustic gate in
live and the GUI; `--vad energy` is an explicit diagnostic fallback. It opened
and endpointed 12/12 cases12 clips, then found six complete turns in a
five-minute continuous-floor physical run while feeding 299.5 seconds with
zero drops and preserving 430/430 audited words. This verifies speech
admission and fixes the old zero-turn RMS soak, not environmental rejection or
semantic turn completion. Next collect the labeled physical noise/media/
quiet-speech corpus and report false opens, missed speech, open delay, and
endpoint delay independently of companion admission. Also investigate the
separate audio-understanding failure: every silent-observer running note in
that run was only "less than 10 seconds" despite full input survival.

One GPU job at a time. JaegerOmni 41M Medium SHA `a7f96bfb8898` is the current
broad-corpus winner: 3.97% free-running WER on the untouched 51,311-word
test-clean lockbox (pre-budget12; remeasure at the corrected 12-token/s
budget before cross-checkpoint comparison), with no regression across the
five selection corpora.
The final pre-production training window is closed: the same checkpoint also
scores 3.30% normalized WER (22/667) on the source-qualified one-to-one
conversation gate and 12/12 spoken answers at 560 ms median. Do not run more
adapter training before physical production testing; every final-window
continuation, output-only refinement, seed repeat, and soup either tied the
conversation gate or regressed another gate.
The direct-file five-minute duplex gate now passes its stream-integrity checks:
300.0 seconds fed, zero drops, one real 6,010-token context roll, and
1,044/3,725 ms END-to-audio median/p95 after cadence streaming. Do not launch
more Medium training until the physical-room five-minute gate runs. The broad
43-room AEC gate has 0 false barges and 100% recall, but its 843 ms p95 still
fails the 300 ms tail gate. The old 12-case AEC lab is not a promotion signal.

The 2026-08-29 physical five-minute observer gate now passes capture integrity
for both 41M variants: 299.9/300 seconds fed, zero drops/queue overflow, one
6,151-token roll, 100% raw-vs-fed word survival, and grounded post-roll
minutes. Do not call full-meeting retention passed: a silent observer has no
assistant replies to reseed, so the roll discarded the first four minutes of
user audio. Preserve recent native audio embeddings or produce incremental
audio-native notes before the next observer promotion gate.

k=3 offline diarization separated the two dominant alternating recorded
talkers in both captures (maximum centroid cosine 0.477/0.467); validate the
small 12-second third cluster against labeled source timing, then feed stable
speaker changes between turns. Do not claim overlapping word attribution.

The AEC root cause is now measured, not hypothesized
(`research/aec_path_probe.py`): a constant 614 ms output-path delay beyond
the 500 ms filter span, plus ~1,250 ppm capture-clock drift that no fixed
offset can survive (offline replay shows cancellation dies when the
reference runs >20 ms late). Echo-locked contiguous mode self-aligns onto
the measured echo and reaches 10.5 dB coupling reduction on the
equal-exposure fixed-waveform probe — matching the 10.3 dB offline ideal —
while wallclock varies 0--7.8 dB run-to-run. Drift resampling is demoted:
drift is real but the servo already absorbs it. Next: rerun the 90-second
interactive diagnostic with `--aec-reference-mode contiguous`, then a fresh
physical raw-vs-fed audit; promotion still requires the 43-room gate
(now 0/43 false, 43/43 recall, 188/602 ms median/p95 — the 300 ms tail
gate is still red) and the audit showing Jaeger's own words no longer
survive into model-fed audio.

The latest GemmaOmni causal model is now imported exactly, and a Jaeger 15%
causal blend is packaged separately. Keep `jaegeromni-41m-medium` as the 1:1
production candidate because it still wins dev-clean by two words. Use
`jaegeromni-41m-causal-balanced` for the next full-duplex experiment: it is
24/24 at 574.2 ms and improves four of five corpus gates, including the
64-clip/673-word training-adjacent AMI validation slice from 32.99% to
28.23%. Before any promotion, rerun the overlap/speaker/turn-end
heads on held-out real mixtures after the trunk blend and connect their
calibrated logits to policy. Structural causality and self-speech wiring pass;
head calibration does not yet.

2026-08-28 latency CPU preparation: the published 319 ms JaegerOmni 50M and
560 ms JaegerOmni 41M values came from different benchmark harnesses and must
not be treated as an architectural A/B. `research/run_latency_ab.sh` is ready
to run the 41M baseline, an opt-in mathematically equivalent packed-QKV 41M
runtime, and the 50M node with identical warmup, prompts, cases, repetitions,
and per-stage timing. It also requires identical generated replies before the
packed-QKV path can be considered. The repository GPU guard now recognizes the
parallel GemmaOmni `eval/benchmark_suite.py`, so this A/B cannot contaminate
that agent's active training.

The final GPU A/B is now complete. On 24 warmed, repeated cases the frozen 41M
scored 24/24 at 601.9 ms median, packed-QKV scored the same answers at 610.7 ms,
and 50M scored 22/24 at 619.7 ms. Keep the ordinary 41M path; do not enable
packed QKV or cite 319 ms for 50M. Reports are under
`research/results/final-gpu-20260828-final/latency/`.

The MiniCPM audit identified a larger input-latency gap: MiniCPM incrementally
encodes and prefills one-second audio chunks while the user is speaking. The
stock Jaeger path currently buffers a short utterance until turn-end and uses
12-second/one-second-overlap windows only for long continuous input. Stock
Whisper and the 41M global-attention adapter are bidirectional, so a cached
causal replacement is not a safe runtime toggle; it needs a stability/quality
gate or streaming-specific training before previously emitted audio vectors
can be committed to Gemma. Do not claim this feature until that gate exists.

`research/streaming_stability.py` now provides the missing first gate. It
compares every growing one-second prefix with a full-utterance pass at both
the stock-Whisper and final adapter boundaries, sweeping 0--4 seconds of held
right context. Its cosine/magnitude results are explicitly diagnostic, not a
WER or promotion score. The final guarded GPU queue is
`research/run_final_gpu_validation.sh`: identical latency A/B first, streaming
stability second, no training and no automatic checkpoint promotion.

The legacy PyTorch trainer's extended-run resume path is also repaired. It now
seeks a rebuilt OneCycleLR directly to the exact completed-update boundary,
including Adam momentum, rather than calling `scheduler.step()` before an
optimizer update. Its CPU self-check proves equality with an uninterrupted
new-horizon schedule and rejects an off-by-one reconstruction.

The measured stability sweep rejects cached one-second prefill on the frozen
checkpoint. With four seconds held back, the 41M adapter still had 0.977 p05
cosine and 0.206 p95 relative-L2 error against full-utterance vectors. The next
streaming experiment therefore requires prefix-consistency/causal training and
then free-running WER/answer gates; it is not a runtime flag.

## 1. Run physical production testing

Start with the frozen candidate, never an intermediate:

```text
front end: jaegeromni-41m-medium
adapter SHA: a7f96bfb889866f75df7b838f84aae368a56ab10f6469e766511d700e32fe13a
```

The physical five-minute run must retain all of the following:

- source-qualified conversation: no worse than 22/667 normalized word edits;
- cases12: 12/12;
- no lost microphone audio, no abandoned reply, and at least one deliberate
  barge-in under each policy;
- five-minute context roll plus END-to-audio median/p95;
- physical AEC attenuation and false-barge count.

The 2026-08-28 continuous-floor Elgato soak is useful but does not close this
gate: 298.0 seconds fed, zero drops, 178 ms peak lag, one successful 6,151-token
roll, and 11/11 audit-word survival, but zero endpointed turns because the room
never stayed below the measured floor for 700 ms. Repeat with a real 1:1
speaker who deliberately pauses and interrupts Jaeger's playback; retain the
continuous-floor capture as the no-yield stress case rather than weakening the
endpoint until fragments masquerade as turns.

A separate 90-second diagnostic at 300 ms silence proves the downstream
interactive mechanics: 5 turns/replies, zero drops, two continue-during-output
events, and 494/522 ms median/p95 END-to-first-audio. It also makes AEC the
remaining hard blocker rather than a paper concern. Physical attenuation was
only 0.29 dB and 25/25 decoded words of Jaeger's own repeated playback survived
into model-fed audio. Reference-lead and filter-length sweeps peaked at 2.05 dB,
so do not tune timing and claim success. The next AEC implementation must
calibrate/retain a loudspeaker-to-mic path when usable, freeze or slow adaptation
during detected double-talk, and pass both the 43-room synthetic gate and a
fresh physical raw-vs-fed audit before promotion.

Teacher forcing is permitted as a training loss. It is never an evaluation
number. Do not promote from a single clip, a teacher-forced score, cosine, or
the training-only lexical head.

## 2. How to break the 14% plateau

More passes over the same adapter/native-vector regression are closed. They
move errors between domains but have not crossed 14% general WER. The next
useful objective is a low-weight, human-target sequence loss through frozen
Gemma, while keeping the native audio-vector loss as an anchor:

```text
stock Whisper encoder (frozen)
  -> owned 50M connector
     -> native-vector regression anchor
     -> frozen Gemma + human transcript/answer CE
     -> training-only speaker and event heads
```

Only the connector and small auxiliary heads train. Gemma, Whisper, and their
decoders stay frozen. Human transcripts supervise training but are never a
runtime handoff. The other agent's 7.55% result validates this direction on a
narrow one-speaker gate, but its 54.30% room WER exposes the missing domain
robustness. The owned implementation is now `training/context_ce.py`: human
CE, same-Gemma text-equivalence KL, frozen-start output anchor, exact
source-mass sampling, holdout overlap audits, and a frame-diversity collapse
abort. It trains only the explicitly selected connector branch.

If it fails, the next structural experiment is a pretrained or distilled
monotonic sequence head at the connector bottleneck, not another randomly
initialized BiGRU CTC head. The current disposable head ended near 94% WER and
is diagnostic evidence that lexical structure is not yet usable there.

## 3. Data policy

Canonical caches live outside the repository under
`~/models/jaeger-training-data`. Current additions are:

- 12,000 previously unused LibriSpeech train-other clips;
- 4,042 new AMI turns from four hours with zero first-hour turn overlap;
- 3,249 AMI training and 750 speaker-disjoint AMI validation features;
- 14,565 deterministic RIR/MUSAN variants (26.255 h, 1,104 speakers) from
  hard LibriSpeech and AMI, retaining clean IDs and augmentation provenance;
- 29,671 previously unused train-clean-360 clips at the current <=12-second
  training limit (63.260 h): 27,911/868-speaker training and
  1,760/53-speaker validation with zero speaker overlap;
- an official-MD5-verified LibriSpeech test-clean lockbox: 2,603 usable clips
  from 40 speakers, manifest SHA
  `d8934ec73c9e1b3280f31a15946d5e7d5aa1df3319e3b912d763d251a5a1e738`;
- a measured 1,006.828-hour independent licensed source pool: LibriSpeech
  961.052 h, genuinely novel VoxPopuli 33.283 h, FLEURS 7.491 h, and AMI
  5.003 h, plus MUSAN/OpenSLR RIR augmentation assets. Raw archive hours are
  not training examples until matching stock-Whisper features are cached, and
  the current <=12-second curriculum intentionally uses a smaller subset.

Use `--balance-sources` for mixed-domain training. Keep every validation split
untouched. Before another run, add genuinely new speakers/acoustics such as
MUSAN/RIR augmentation or another licensed conversational corpus; do not cycle
the same AMI or LibriSpeech rows again merely because GPU time is available.

Secondary labels remain important but must not be conflated with actions:

- speaker identity/continuity;
- speech during playback (double-talk), distinct from the policy decision to
  stop or continue speaking;
- turn start/end and overlap;
- environmental/acoustic events.

Overlap attribution is a separate future node, not a capability of the current
turn-level A/B/C clustering.  The production order is: cancel Jaeger's known
playback reference, detect human-human overlap, separate the mixture into two
waveforms, assign each separated stream to a persistent CampPlus identity, and
feed both streams through the audio-native front end with time-aligned speaker
metadata.  Validate with overlap-aware cpWER/DER and leakage/SI-SDR on held-out
AMI/ICSI meetings.  Until those gates pass, report mixed speech as overlap and
do not claim word-level speaker attribution.

## 4. Prepared medium/large campaign

No stage starts automatically. The CPU-created large port is
`artifacts/stock-whisper-gemma-large-v3-turbo50m-port.npz` (SHA
`5e3d766034c66811c8e30d3a73e1e3e0329b88f5eeba492c9ab0a9bc22537c3a`).
It has a fresh 1280x2 input stack and byte-identical inherited downstream
blocks (52,768,768 deployed parameters versus medium's 50,146,816).
`training/test_next_window.py` passes forward, freeze-set, lineage,
and real CPU-gradient checks for both front ends.

When the GPU is assigned, run the named stages in
`training/next_5pct_window.sh`: room-feature precompute, medium CE/KL, large
native input bootstrap, large input CE/KL, then large context CE/KL. Each
stage is separately checkpointed and guarded against concurrent GPU work.
Use `training/evaluate_candidate.sh` for identical-ID free-running gates; do
not promote from loss, cosine, teacher forcing, or the large decoder control.

## 5. Live production gates

The native Gemma tower remains the desktop default until the stock connector
passes a positive-room five-minute test. Require:

- no lost audio across a 300-second capture;
- meaningful Silero-admitted speech at the selected Elgato input;
- stable A/B/C speaker telemetry without claiming word-level diarization;
- AEC/double-talk measurements and barge-in behavior;
- context roll without negative soak deltas;
- END-to-first-audio median and tail latency.

The tuned MiniCPM-derived encoder remains observer-only while its fixed corpus
WER is 13.61%, better than the stock observer. It is not an assistant default
and the Qwen language model is never loaded.

## Commit gate

```bash
./.venv/bin/python research/check_all.py
./.venv/bin/python -m jaeger_agent_omni.omni --selfcheck
./.venv/bin/python demo.py --check
git diff --check
```

Do not stage generated WAVs, local corpora, virtual environments, model
weights, or intermediate checkpoints.
