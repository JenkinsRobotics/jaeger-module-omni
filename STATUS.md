# Jaeger Agent Omni — what works, 2026-08-22

Measured ledger. Every claim here has a command that reproduces it.

> Current selection and comparison: [`PIPELINES.md`](PIPELINES.md). Sections
> below the current 50M results are retained experiment history, not additional
> production choices.

---

# The pipeline

```text
mic ─► audio encoder ─► Gemma E4B ─► text
                            │
                            └─► hidden states ─► talker ─► token2wav ─► speaker
```

**No transcription step anywhere in that chain.** Audio becomes embeddings and
stays embeddings until the backbone emits language. That is the point of this
repository: the sequential stack sends TEXT between modules and loses
everything text cannot carry.

The invariant is enforced, not documented. `omni.py --selfcheck` reads its own
source and fails the build if `transcribe(` or `feed_text(` ever appear:

```bash
python -m jaeger_agent_omni.omni --selfcheck
```

---

# Works today

```bash
python -m jaeger_agent_omni.omni --bench research/cases12
```

```text
omni [tower]   10/12 = 83%   median 363 ms
legacy         12/12 = 100%  (text between modules, 9.8 GB, two model stacks)
MiniCPM-o 4.5  12/12 = 100%  525 ms to text, 1380 ms to speech
```

83% of legacy, integrated, in one process. The two misses are the frozen
tower's: one read audio as an image, one never named Shakespeare.

| capability | state | command |
| --- | --- | --- |
| audio -> text, no transcript | **10/12, 363 ms** | `omni.py --bench` |
| hands-free turn detection | **works** | `python -m jaeger_agent_omni.runtime.endpoint` |
| always-on duplex | **works** | `python -m jaeger_agent_omni.runtime.duplex` |
| barge-in | **20.7 ms**, context kept | same |
| omni input (a/v/text, one cache) | **works** | `research/check_all.py` |
| speech codes -> audible cloned voice | **works** | `research/t2w_client.py --say` |
| cognition -> talker conditioning | **built, untrained** | `neural/delta.py` |
| tuned Whisper -> sparse adapter -> E4B | **observer: 13.61% WER/64** | `meeting.py` |
| stock Whisper -> owned 50M adapter -> E4B | **candidate: 11.20% WER/64, 11/12** | `omni.py --front-end stock --dual-output` |

31/31 module self-checks green: `python research/check_all.py`

## Stock-Whisper live integration — 2026-08-25

The strongest pinned medium node is now selectable, without copying a loose
checkpoint path, from `omni.py`, `jaeger_agent_omni.demo`,
`jaeger_agent_omni.live`, and the full-duplex desktop app:

```text
nodes/stock-whisper-medium-cekl-blend50
50,146,816 deployed adapter parameters
11.20% WER — dev-clean64
17.86% WER — MINDS/Vox591
13.51% WER — clean-360 validation128
12.56% WER — novel Vox128
11/12 answers — canonical cases12, 319 ms median
```

This integration removes three prior live blockers: the stock session now
keeps one Gemma KV cache across microphone turns, interrupted generations
rewind exactly without discarding the preceding conversation, and explicit
context rolls retain monotonic lifetime prefill counters. Startup warms the
real Whisper/adapter/Gemma path. Uninterrupted speech is encoded and prefilled
in bounded 12-second windows with one-second overlap (28 windows / 15,000
non-duplicated encoder frames for the five-minute contract), matching the
adapter's training-length curriculum rather than deferring one unbounded
operation until the room falls silent. For packaged nodes, only Whisper's encoder
weights are materialized; the ASR decoder is neither loaded nor run.

The native tower remains the selected rollback default until the stock node
passes the positive-room, five-minute full-duplex gate. Static/runtime gates
are green; a fresh physical measurement is intentionally not claimed while
the one allowed accelerator job belongs to the active trainer.

## Stock-Whisper medium 50M development winner — 2026-08-23

The current Qwen-free development winner is the 25% interpolation between
the selected native/contextual connector and its new-data continuation:

```text
artifacts/stock-whisper-gemma-medium50m-expanded-blend25.npz
50,146,816 deployed adapter parameters
14.03% WER — dev-clean64, 134 edits / 955 words
19.10% WER — MINDS/Vox591, 1,860 edits / 9,736 words
19.60% WER — speaker-disjoint train-other64, 224 edits / 1,143 words
10/12 answers — canonical cases12, 322 ms median
```

The training-only lexical probe is a bidirectional GRU/character-CTC head on
the 1,536-wide contextual bottleneck. Human transcripts supervise this head;
they are never fed into Gemma or used as a runtime handoff. Gemma-native audio
vectors remain the deployed-output anchor, and the lexical head is ignored by
the runtime. Direct CTC against Gemma vocabulary rows remains rejected.

The expanded pool adds 12,000 previously unused LibriSpeech
`train-other-500` clips: 11,316 training clips from 1,095 speakers and 684
validation clips from 60 disjoint speakers. It has zero ID overlap with the
earlier train-clean/dev-clean/MINDS/Vox/FLEURS cache. The raw new-data
continuation improved hard speech to 18.72% but regressed clean speech to
14.45%, so it was not promoted. A 25% weight interpolation improved the prior
selected checkpoint by one clean edit and one hard-speech edit while tying
the 9,736-word conversational result and retaining 10/12 answers. The 50%
and 75% blends both regressed dev-clean to 14.55% and were rejected.

Five hours of AMI meeting speech were then added without replaying source
turns. The first hour contains 1,006 turns; the next four contain 4,042 turns
from 12 speakers, seven not present in hour one, with zero turn-ID overlap.
Their stock-Whisper cache has 3,249 training and 750 speaker-disjoint
validation examples. A source-balanced half-pass sampled the new AMI and
train-other domains with exactly 0.50 probability each, about 750 updates per
domain, rather than repeatedly cycling either corpus.

The raw balanced checkpoint is a useful AMI specialist, not a general winner:

```text
artifacts/stock-whisper-gemma-medium50m-ami5-balanced1p5k.npz
49.93% AMI64 — 336 / 673 errors, versus 53.79% (362 / 673)
14.35% rebuilt dev-clean64 — 137 / 955, one worse than paired baseline
10/12 cases12 — 321 ms median
```

A 50% interpolation preserved rebuilt dev-clean exactly at 136/955 and gained
nine AMI errors, but regressed MINDS/Vox591 from 1,860 to 1,875 errors. Smaller
10% and 5% interpolations still lost one and two MINDS/Vox errors respectively.
All are therefore rejected as the general default. This establishes a real
domain tradeoff rather than a route below 10% WER; the direct adapter's best
verified general result remains 14.03%.

This checkpoint is the development baseline, not the live desktop default.
The native Gemma tower remains live until five-minute duplex, speaker, AEC,
and context-roll gates are repeated.

## Canonical live demo

```bash
python demo.py --check
python demo.py
```

The donated full-duplex desktop app now runs the real Jaeger live path by
default: Gemma audio embeddings in, Kokoro clause streaming out, SpeexDSP AEC,
hands-free endpoints, residual barge-in, inline speaker metadata, and bounded
rolling context. It does not display or compute an input transcript. MiniCPM-o
remains an optional reference selector.

The reciprocal VoiceLLM Playground review retained the useful product/runtime
parts: its existing MiniCPM shim/model can be discovered read-only, the GUI now
selects a physical input device and exposes callback failures, and a selective
third-person meeting preset is available. Its owned-seam transcript ledger and
Gemma transcription decoder were deliberately not imported: they are useful
ASR research, but violate this repository's no-transcript input invariant.
The optional native reference was then verified through this app: 25 paced
frames yielded 22 LISTEN / 3 SPEAK, the answer “The capital of France is
Paris,” and real playback at 0.129 peak level.

Exact GUI-worker replay of the five-minute room recording, free-running at
`--threshold 0.20 --silence-ms 700 --cooldown 20 --max-context 6000`:

```text
audio fed        300.0 / 300.0 s
audio dropped      0.0 s
turns              28
interrupted gens    22 (no abandoned partial reply published)
visible barge-ins  27
completed replies   6
speaker labels      A and B
context rolls       1 (cache ended at 3054 tokens, thread continued)
peak loop lag      347 ms
END->AUDIO        1091 ms median across completed starts
```

The short exact-worker proof with diarization enabled produced one correct
reply, speaker `A`, 20.0/20.0 seconds fed, zero loss, and 458 ms END->AUDIO.

## Tuned-Whisper meeting observer

The meeting observer loads the audio-tuned
Whisper encoder and `audio_projector.linear1` from MiniCPM's audio GGUF, then
uses Jaeger's trained 4096→2560 adapter to enter Gemma 4 E4B. It never loads
the Qwen language model, the original Qwen-facing `linear2`, or Whisper's
decoder. The audio GGUF is a weight container, not a Qwen runtime dependency.

The reciprocal stock-Whisper hypothesis is now a viable Qwen-free alternative,
but not the default. Sampled CTC plus 250 Gemma-CE updates still collapsed at
**117.17% free-running WER** on a fixed 64-clip/955-word `dev-clean` slice.
The successful recipe instead distils paired *pre-Gemma audio vectors* from
the proven tuned path into the same MiniCPM-shaped 1024→4096→2560 projector.
No Gemma layer, logit, generated token, Whisper decoder, or Qwen model supplies
the distillation target. There are 4,786/136 paired LibriSpeech train/held-out
utterances plus 2,466/388 paired FLEURS utterances.

The selected multidomain checkpoint at step 5,500 scores **16.34% WER** on
64 held-out LibriSpeech clips/955 words and **25.52% WER** on 64 held-out
FLEURS clips/1,403 words. The clean-only step 4,500 baseline was 16.34% and
27.37% respectively, so the varied continuation gained 1.85 FLEURS points
with no measured clean regression. A longer step-7,000 continuation reached
25.59% FLEURS but regressed clean speech to 18.32%, and was rejected. Direct
nearest-token recall was retained only as a collapse diagnostic: its preferred
step 1,500 scored 106.28% free-running WER, proving that soft audio prompts do
not need to sit on vocabulary rows.

The stock adapter's first assistant score of 0/12 was a prompt/probe error,
not its answer ceiling. With one Gemma generation explicitly formatted as
`TRANSCRIPT:` followed by `ANSWER:`, it scores **9/12 answers** at 660 ms
median and 28.36% generated-transcript WER over 67 words. All three answer
misses correspond to bad generated transcripts. That transcript is Gemma's
post-transformer interpretation—not raw tokens entering Gemma; the actual
input remains continuous 2560-wide adapter vectors. `--dual-output` exposes
both fields without feeding the generated transcript back into the session.

### Archived native-teacher lineage — superseded, 2026-08-23

This historical development track was fully owned and competitor-free in
runtime and training lineage. Its selected connector used the independently
proven Ultravox projector structure at Gemma's measured 25 Hz boundary:

```text
stock OpenAI Whisper-medium encoder (50 Hz, decoder unused)
  -> stack 2 frames (2048) -> RMSNorm -> Linear(5120) -> SwiGLU(2560)
  -> RMSNorm(2560) -> Linear(2560), no affine biases
  -> zero-init residual: RMSNorm -> Linear(2048) -> SwiGLU(1024)
     -> RMSNorm(1024) -> Linear(2560) + skip
  -> zero-init temporal residual: RMSNorm -> SwiGLU(256)
     -> depthwise Conv1d(k=3) -> RMSNorm -> Linear(2560) + skip
  -> zero-init long temporal residual: RMSNorm -> SwiGLU(256)
     -> depthwise Conv1d(k=31) -> RMSNorm -> Linear(2560) + skip
  -> Gemma 4 E4B
teacher: Gemma's own audio_tower -> embed_audio vectors at 25 Hz
```

The initial 2048-wide clean-room run scored 52.32% WER on its 64-clip
development slice. Widening to 4096, then using a short-utterance curriculum
and balanced LibriSpeech/MINDS-14/VoxPopuli replay produced the selected local
mean-pool checkpoint
`artifacts/stock-whisper-gemma-native-h4096-replay-step1000.npz`. Replacing
mean pooling with frame stacking plus RMSNorm/SwiGLU, then sweeping a 15k
fresh run, 10k continuation, and 5k low-rate refinement produced
`artifacts/stock-whisper-gemma-native-stack-swiglu-refine5k.npz`.

Freezing that entire connector and training only a zero-initialized 7.86M
parameter residual block preserves its output exactly at step zero while
adding error-correction capacity. A 15k residual-only run, an 8k selected
half-rate continuation, and a 4k selected low-rate refinement produce
`artifacts/stock-whisper-gemma-native-stack-residual1024-refine4k.npz`.

Before adding temporal context, the main-repo evaluator measured this residual
checkpoint at **20.52% free-running WER on the exact same 64 clips/955 words**
as the 16.34% paired-adapter reference, down from 27.75%. It also measured
**22.63% WER on 591 held-out MINDS/Vox clips/9,736 words**, down from 27.38%,
and **8/12 answers at 511 ms median** with the dual transcript/answer prompt.
Generated transcript WER on cases12 was **20.90%**; answer accuracy did not
rise because the remaining four suite misses were perception errors.

This was a large clean-room improvement, but at that point it was not promoted over the
paired stock adapter (16.34%/9 of 12), tuned-Whisper observer (13.61%), or
native-tower live default (10 of 12). The residual path is strongly
non-monotonic: half-rate step 8,000 beats its final checkpoint, and low-rate
step 4,000 beats step 5,000. Further same-branch training is closed at that
selected point. A separate human-transcript CTC A/B also
failed the primary corpus gate: at a conservative 1e-6, checkpoints 250/500/
750/1,000 regressed to 28.59/29.21/31.52/33.61% WER on the same 955 words.
This confirms token-row alignment destroys useful native-tower geometry; no
Gemma-generated output was used in that test. A 64-clip temporal audit also
rules out a frame-offset bug: all 64 clips prefer lag zero (cosine 0.7155,
versus 0.6483/0.6168 at -1/+1). First-difference cosine is only 0.3229, but
adding a 0.25 temporal-difference loss regresses every 1k-step checkpoint to
33.95–40.87% on the 969-word ranking corpus. That objective is closed too.

Follow-up sweeps confirmed that per-token residual connector was locally
saturated. A second
zero-init residual regresses every ranking checkpoint. Training only <=3 s
clips reaches 9/12 but regresses dev-clean to 21.88%; 0.875 weight
interpolation still scores 21.47%. Full replay with 4x short weighting improves
dev-clean to 20.00% but regresses MINDS/Vox to 22.78%; 2x weighting scores
20.31%/22.83%. A zero-init 5,120-value output affine never beats 20.52%.
Raising cosine loss from 0.1 to 1.0 tied dev-clean and improved the cases12
transcript diagnostic to 19.40%, but loses the 9,736-word mixed gate by exactly
one edit (2,204 versus 2,203); 0.5 loses by eight. The retained checkpoint
therefore remained unchanged through those per-token sweeps; the temporal
result below supersedes it. The complete free-running matrix is in
`research/stock_whisper_native_followup_sweeps_20260823.json`.

The available new-data lever was tested as well. A training-only FLEURS
manifest added 1,937 CC-BY-4.0 English-US clips (1--12 s) without touching
either validation partition. Step 1,000 improves dev-clean from 20.52% to
20.00%, but regresses MINDS/Vox from 22.63% to 22.82% (2,222 versus 2,203
edits) and remains 8/12. This is another domain trade, not a promotion.

Changing the temporal connector succeeds where the per-token sweeps do not.
A 1.97M-parameter Conformer-style branch adds local context across three 25 Hz
tokens while preserving the selected residual connector exactly at step zero.
Training only that branch for 8,000 steps produces a non-monotonic exact-corpus
winner at step 7,000:

```text
artifacts/stock-whisper-gemma-native-temporal256-7k.npz
18.01% WER — dev-clean64, 172 edits / 955 words
20.95% WER — MINDS/Vox591, 2,040 edits / 9,736 words
8/12 answers — dual-output cases12, 525 ms median
20.90% generated transcript WER — 67-word cases12 diagnostic
```

This improves the prior native-teacher checkpoint by 2.51 WER points on
dev-clean and 1.68 points on MINDS/Vox without using a transcript, answer, or
Gemma output as supervision. Step 8,000 wins the 969-word ranking proxy but is
worse on both authoritative corpora (18.12%/21.29%), so step 7,000 is selected.
At that stage the native-teacher path was not promoted over the
16.34%/9-of-12 paired adapter or the 10-of-12 native live tower because answer
accuracy remained 8/12.

A lower-rate continuation improved both corpora again to 17.07%/20.72%, but
remained 8/12. Upweighting <=3 s open-corpus clips 4x also left every 500-step
answer checkpoint at 8/12, so data frequency was not the missing lever. The
successful change was a second exact-zero Conformer-style residual with a
31-token depthwise kernel. At 25 Hz its 1.24-second receptive field covers most
of the short question suite while preserving the complete selected connector
bit-for-bit at step zero. Training only its 1.98M parameters for 8,000 steps
produced another non-monotonic winner at step 6,000:

```text
artifacts/stock-whisper-gemma-native-temporal256x2-k31-6k.npz
15.60% WER — dev-clean64, 149 edits / 955 words
19.86% WER — MINDS/Vox591, 1,934 edits / 9,736 words
10/12 answers — dual-output cases12, 530 ms median
8.96% generated transcript WER — 67-word cases12 diagnostic
```

This checkpoint cleared all three predeclared promotion gates at that stage.
It beats the paired adapter on the same dev-clean
gate (15.60% versus 16.34%) and reaches native-tower answer parity (10/12),
without transcript, answer, Gemma-generated output, MiniCPM, or Qwen
supervision. It has since been superseded by the retained 50M general
checkpoint at 14.03%/19.10% and 10/12. Step 8,000 improved dev-clean further
to 15.18% but fell to 9/12. The native tower remains the live desktop default
until the bidirectional stock encoder passes live-room latency and soak gates.

The reported 10/12 is after fixing a grader false positive: expected digit
`7` had incorrectly matched inside `0.137`. Numeric answers now match complete
normalized tokens, and the selected step 6,000 still scores 10/12.

On the exact same existing 64-clip/977-word validation slice used by
`research/wer_corpus.py`, the promoted sparse tuned adapter improves the prior
adapter from **14.74% to 13.61% WER** (113 fewer basis points, 7.7%
relative). The answer-distilled checkpoint reaches only **6/12** on the spoken
question suite at **283 ms median**, versus the native tower's 10/12, so the
canonical assistant and desktop demo correctly remain on `tower`. Transcript
and answer adapters are not interchangeable objectives.

```bash
python -m jaeger_agent_omni.meeting --seconds 300 --device 1 --k 3 \
    --record /tmp/meeting.wav --out meeting.md
python -m jaeger_agent_omni.meeting --wav room.wav --front-end stock
```

The meeting observer uses 9.0-second neural windows with 0.5-second overlap;
the 44.9-second one-shot decode was measured to collapse to one clause, so a
five-minute tensor-shape check is not presented as long-window accuracy.
Silero must admit at least 25% of a window's 30 ms frames before Gemma runs.
This fixed a concrete failure: the new five-minute Elgato capture retained all
300.0 seconds but contained broadband room/cable noise (3–19% Silero-positive
frames). Before admission it hallucinated refusal/loop text; replay after the
fix completed 36/36 time ranges in 2.9 seconds, exit 0, and marked every range
`[no speech admitted]`, contributing nothing to minutes.

The retained positive 90.0-second room conversation admitted all 11 windows,
covered the cloud-storage and meal-kit discussion, and produced coherent
minutes. It also exposed the remaining defect: one overlap contains a spurious
phrase and turn-level online labels are not word-level diarization. With
`--k 3`, online labels A/B/C all fire; the independent offline audit finds
three centroids at cosine 0.369/0.401/0.592. Those are useful clusters, not a
claim that the two humans are perfectly separated at every word.

The new stock observer also completed an exact 300.0-second Elgato capture,
but this was an input-routing failure rather than an accuracy sample. Only
2/36 windows crossed the 25% speech gate; the Elgato signal was loud
(0.100 RMS, peak 1.0) but predominantly non-speech. Twelve-second probes found
0% Silero speech on the C920 and digital silence on both Wave Link routes.
The WAV and minutes are retained under `/tmp/jaeger-stock-meeting-5m.*`; none
of its two decoded windows is counted as a corpus measurement.

---

# The reference, measured stage by stage

MiniCPM-o 4.5 on the same 12 wavs, quiet machine:

```text
STAGE A  pre-LLM      0.175 s prefill    0.11x realtime
STAGE B  LLM          0.350 s decode     12/12 = 100%
STAGE C  speech out   1380 ms to first audio, vocoder RTF 0.675
```

It is the only complete system today and it is genuinely ahead: same accuracy
as legacy, faster, half the memory (4.9 GB doing audio+vision+LLM+speech,
against legacy's 9.8 GB doing audio+LLM alone), and it speaks.

---

# Archived rationale for the original tuned-encoder bridge

Measured on embeddings, before the LLM sees them, so it separates "the encoder
discarded it" from "the model ignored it":

```text
                              delivery carried
text transcript                    0.000     by construction
Gemma native tower                 0.192     frozen, cannot be extended
whisper encoder (pooled)           0.240
whisper encoder (temporal)         0.323     what a delta actually reads
```

Four takes of one sentence — different rate, emotion, and speaker — returned
the same description from the native tower every time. Voice cloning, emotion
perception and impersonation are structurally unavailable through it.

The delta's case is not only prosody. It is all three:

```text
MORE INFORMATION   0.323 vs a transcript's 0.000
FASTER             skips whisper's autoregressive decoder entirely
INTEGRATED         one graph, one footprint, not two model stacks
```

---

# Measurement rules, learned the hard way

Six wrong numbers today, and every one had one of two causes.

```text
11.8 s legacy      whisper on CPU, not MPS
48.5 s legacy      a stale job had the GPU
RTF 1.23           same stale job
17.1 s legacy      the parallel agent's benchmark, 12 GB and 73% CPU
9/12 MiniCPM       "twenty-five" normalised to "twentyfive"
11/12 legacy       "24" not matched against "twenty four"
```

1. **One job at a time.** `research/sweep.sh` enforces it. Concurrent GPU work
   made a 635 ms pipeline read 48514 ms.
2. **One grader.** `research/grading.py`, with all three historical failures
   pinned in its self-check. No single normalisation works: `T-A-C.` needs
   punctuation deleted, `twenty-five` needs it spaced.
3. **Free-running only.** A checkpoint scoring 0.939 teacher-forced scored
   **33%** answering on its own. Teacher-forcing hands the model the correct
   previous token at every step; it measures nothing that runs in production.
4. **RSS cannot see MPS.** A 15 GB model reads as 1.2 GB. Use
   `torch.mps.driver_allocated_memory()`.
5. **First step is Metal compile.** 12 s against 0.58 s steady state.

---

# E4B was always trainable here

The SmolLM2-135M stand-in existed because of three config-shape bugs, none of
them hardware:

```text
per_layer_inputs   E4B feeds 256 dims per token per layer. Passing embeds
                   without ids makes transformers try to recover the ids by
                   comparing against all 262144 vocab rows -- 29 GiB, and it
                   fails at any memory size because projector-made embeddings
                   are not in the vocab.
hidden_size        Gemma4Config is multimodal: it lives on get_text_config().
                   Four call sites, now one shared helper.
```

Fixed. E4B trains at **0.58 s/step in 16.6 GB** — 2000 steps in 19 minutes.

---

# Historical next steps (superseded by `NEXT.md`)

```text
1. delta front end swaps in at ONE seam in omni.py; nothing else moves.
2. train SpeechOutDelta -> audible speech that follows the words.
3. vision is 1035 ms/frame against MiniCPM's 379 ms. Not yet touched.
```

---

# The E4B bug ledger

Seven blockers, all in code calibrated against a 135M stand-in. None were
hardware. Each presented as "E4B does not fit on this machine", which is what
justified the stand-in — so the substitution was self-perpetuating: it made
the real backbone look impossible while silently mis-calibrating every knob.

```text
per_layer_inputs   E4B feeds 256 dims per token per layer. Passing embeds
                   without ids makes transformers recover the ids by
                   comparing against all 262144 vocab rows: 29 GiB, and it
                   fails at any size because projector embeddings are not in
                   the vocab.
hidden_size        Gemma4Config is multimodal — get_text_config(). 4 sites.
batch_size 8       fine at 135M, 42.4 GiB at 4B. --batch 1 --grad-accum 32
                   keeps the effective batch and the optimisation identical.
dtype fp32         0.5 GB at 135M, 27.9 GB at 4B. Size-aware now.
trimming           `len(set(durations)) == 1` is always true at batch 1 and
                   never above it, so trimming never ran: every clip cost the
                   full 30 s / 300-token budget. 46% waste on a real batch.
embed_scale        `.weight.norm()` reads 1.20; the model receives 61.75
                   (x sqrt(2560)). The audio prefix was 51x quieter than the
                   text beside it. Would have produced a useless adapter even
                   without the NaN.
no NaN guard       a run reached step 120, went nan, and kept "training" for
                   the rest of its budget producing a dead checkpoint.
```

## The NaN hunt, including the wrong turns

Asserted a root cause three times, wrong twice. Recording that because the
pattern matters more than the answer.

```text
RULED OUT BY MEASUREMENT
  bf16 encoder      fp32 vs bf16 agree to 3 sig figs across 12 batches
  bf16 CE forward   finite at logit scales 12 / 30 / 60
  bf16 CE backward  finite, |g|max identical to fp32
  embed_scale       a real bug, fixed — NaN persisted, so not the cause
  learning rate     changed twice on theory, no effect either time
  embed_norm pin    prefix norms exactly 61.75, no drift

STILL UNDER TEST
  gradient checkpointing — recomputes 42 layers of activations in bf16
  during backward. The only link never tested alone, and the only thing
  ADDED on the day the NaNs began.
```

Every fix that worked came from reading an error or measuring the specific
thing. Every fix that failed came from reasoning at a distance. That is the
same lesson as `sweep.sh` (one GPU job at a time) and `grading.py` (one
grader, all failures pinned), learned a third time.

## Numbers that were wrong today, and why

```text
11.8 s legacy      whisper on CPU, not MPS
48.5 s legacy      a stale job had the GPU
RTF 1.23           same stale job
17.1 s legacy      the parallel agent's benchmark, 12 GB and 73% CPU
9/12 MiniCPM       "twenty-five" normalised to "twentyfive"
11/12 legacy       "24" not matched against "twenty four"
"fits at 16.6 GB"  measured at seq 64, batch 1 — a case training never runs
"90+ accuracy"     teacher-forced; free-running was 33%
```

Two causes, over and over: **something else was on the machine**, or **the
measurement was of a different thing than the claim**.
