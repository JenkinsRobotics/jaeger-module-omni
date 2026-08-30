# Jaeger Agent Omni — what works, 2026-08-22

Measured ledger. Every claim here has a command that reproduces it.

> Current selection and comparison: [`PIPELINES.md`](PIPELINES.md). Sections
> below the current 50M results are retained experiment history, not additional
> production choices.
>
> Current MiniCPM comparison: [`PARITY_SCORECARD.md`](PARITY_SCORECARD.md).
> It is generated from exact checkpoint SHAs and supersedes stale latency or
> promotion language in the historical sections below.
>
> 2026-08-29 release-depth benchmark: the repository now owns one tiered,
> durable gate (`python -m jaeger_agent_omni.benchmark`) spanning CPU
> contracts, the transcript ban, the complete package tests, source parity,
> the frozen neural ear, the native agent loop, duplex/barge-in, Kokoro PCM,
> neural VAD, companion admission, both 81-case agent corpora, the 43-room AEC
> matrix, and required physical five-minute evidence. The first release run
> is honestly **NOT READY: 15 pass / 4 fail / 1 required skip**. Stable green
> measurements are 12/12 plain answers at 565 ms, 12/12 simple agentic answers
> at 2320 ms, 17.8 ms barge cancellation with zero drops, valid 24 kHz Kokoro,
> and neural VAD at 12/12 speech open/end plus 0/5 synthetic nonspeech false
> opens. Red gates are companion admission (1/12 false responses and only
> 4/12 named-address responses), the full native-audio agent corpus (28/162
> full cases, 20/128 required routes, 14 bounded web/network errors), and AEC's
> 602 ms p95 detection tail despite 0 false barges and 100% recall. Safety and
> output hygiene held across the deep corpus: zero forbidden-tool failures and
> zero visible tool markup. A real physical integration report remains
> required. Full row evidence is in the retained `20260829-release-final`
> deep-benchmark directory; harness contract: [`BENCHMARK.md`](BENCHMARK.md).
>
> 2026-08-29 agentic integration (CPU-complete, live gate open): the Jaeger
> Agent framework is now copied into this repository at pinned upstream commit
> `a3696e9e`, with its required JaegerOS substrate pinned at `e05dc24e`. The
> package presents `nodes/`, `tools/`, `skills/`, and `memory/` as sibling
> extension surfaces and keeps implementation plumbing under `core/`. The
> native adapter starts from audio already embedded in Gemma's session, blocks
> the copied `listen` and `text_to_speech` tools, dispatches structured calls,
> and appends trusted results to the same cache without an input transcript.
> The copied catalog contains 97 registered tools and 95 Omni-safe tools. The
> voice bootstrap exposes 8 common tools; every other safe tool remains
> registered and can be exposed by `load_tools`. Seven focused integration
> tests pass, as do 101 selected copied-core tests after namespace relocation.
> The same 101 source tests also pass against the read-only legacy package;
> both 81+81 case corpora have identical SHA-256s and both catalogs contain the
> same 97 names. After fixing stripped Gemma tool-boundary tokens, the
> free-running native-audio agent gate scores **12/12**, makes three valid
> `calculate` calls, makes zero unexpected calls, and has **2421 ms** median
> turn latency (versus an initial broken **9/12 at 6766 ms**). The ordinary
> non-agentic frozen-ear gate remains **12/12 at 575 ms**. This mode is opt-in
> behind `live --agentic` pending a physical five-minute tool/memory/barge-in
> gate. Tool-capable replies are currently buffered to guarantee markup is
> never spoken; agentic latency is therefore functional, not yet production
> parity with the plain voice path.
> Speech output remains decoded response text into resident Kokoro; the hidden-
> state talker is experimental.
> Loading the installed skill set adds ten runtime tools, so the recorded live
> agent surface is 107 registered / 105 Omni-safe; 97/95 is the exact base
> catalog comparison before skill registration.
>
> 2026-08-29 neural audio-perception window: a separately selectable
> `jaegeromni-41m-audio-perception` node now keeps the exact causal-balanced
> audio/Gemma trunk frozen and adds trained scene and conversation-event heads.
> It retains **12/12** free-running answers at **584 ms** median. On a
> source-file-disjoint 128-clip MUSAN/silence test it scores **95.31%** scene
> accuracy and **0.952 macro-F1**. On held-out speakers / unseen AMI meetings,
> overlap reaches **0.723 / 0.742 F1**, but speaker-change is only 0.202/0.088
> and turn-end 0.221/0.254. Scene metadata can reach Gemma between turns;
> conversation-event heads remain shadow-only and Silero remains the live VAD.
> A 262K causal temporal-head experiment was rejected because AMI
> speaker-change/turn-end regressed to 0.040/0.234. Full evidence and limits:
> [`AUDIO_PERCEPTION.md`](AUDIO_PERCEPTION.md).
>
> 2026-08-29 ternary controller gate: a custom packed {-1,0,+1} Metal kernel
> was benchmarked as four synchronized batch-one causal layers at the live
> 10 ms-hop invocation pattern. At the representative 256-wide/262K-parameter
> size, three 2,000-frame trials measured **0.330 ms FP16 vs 0.337 ms ternary**
> (**0.98x**, not a speedup); width 512/1024 were 0.97x/0.77x. Although packing
> removes 87% of weight storage, it saves less than 0.5 MiB at the target size.
> Ternary QAT/kernel work is therefore deferred. This is a compute result, not
> an acoustic-quality result; the separate pre-Whisper controller should be
> developed and gated in FP16 first.
>
> A separate real-model counter-test prevents overgeneralizing that decision.
> Post-training ternarization of every JaegerOmni 41M matrix regressed cases12
> from 12/12 to 8/12, but ternarizing only its two feed-forward matrices
> (16.78M/40.89M matrix parameters) retained **12/12, 506 ms** versus **12/12,
> 505 ms** unchanged. Three locked corpora pooled at **4.55% normalized WER
> (204/4,482)** versus **4.69% (210/4,482)** unchanged; this establishes quality
> retention, not improvement. A packed mixed runtime would reduce adapter
> matrix storage about **36% versus FP16 (78 to 50 MiB)**, but is not built, so
> actual packed latency/RAM remain unclaimed and no production node was added.
>
> 2026-08-29 CPU acoustic-controller scaffold: a separate flat neural node,
> `neural-acoustic-controller`, now implements the pre-Whisper boundary as a
> **262,060-parameter** causal network. It consumes 10 ms microphone frames and
> Jaeger's exact playback reference, returns waveform-safe echo-residual PCM,
> a 128-dimensional acoustic state, and typed scene/self/external/overlap/turn/
> wake probabilities. Its factual source-disjoint training contract and loss
> path pass a one-step CPU gradient check (loss 1.9138); 6 focused tests and
> the 31/31 repository suite pass. No checkpoint is packaged: the node is
> shadow-disabled and Speex/Silero remain authoritative until physical AEC,
> VAD, endpoint, prefix-causality, and latency promotion gates pass.
>
> Runtime identity correction: CLI, live, spoken demo, desktop GUI, and the
> node profile now default to `jaegeromni-41m-medium`; `gemma-native` is the
> explicit rollback. The post-Whisper perception node remains opt-in. When it
> is selected, both CLI and GUI append its validated scene result before Gemma
> responds. It is not represented as neural AEC: the planned pre-Whisper
> controller must consume microphone plus playback reference and emit enhanced
> PCM as well as typed control tensors.
>
> 2026-08-27 current result: JaegerOmni 41M Medium SHA `a7f96bfb8898`
> passes five locked corpus gates without regression and scores **3.97%**
> free-running WER on the untouched 2,603-clip/51,311-word test-clean lockbox
> (pre-budget12: measured before the corrected 12-token/s/24-token-floor
> budget; remeasure before cross-checkpoint comparison). AMI results
> throughout are the 64-clip/673-word training-adjacent, speaker-disjoint
> validation slice.
> Its direct AMI five-minute duplex run fed 300.0 seconds with zero drops,
> produced 14 replies, recorded 13 listen-while-speaking events, and rolled
> context once at 6,010 logical tokens. A 12-word streaming cadence reduced
> END-to-first-audio from 3,899/8,953 ms median/p95 to **1,044/3,725 ms** on
> the same input. The physical-room gate remains open. The 43-room synthetic
> AEC stress gate improved from 27 false/37% recall to **0 false/100% recall**,
> but remains red because its 843 ms p95 exceeds the 300 ms tail gate.
> Historical 41M numbers below describe superseded checkpoints.
>
> 2026-08-28 final pre-production window: the packaged JaegerOmni 41M Medium
> remains frozen at SHA `a7f96bfb8898`. It scores **3.30% normalized WER**
> (22/667) on a locked 69-clip English MINDS14 one-to-one conversation gate
> where matched Whisper-medium is exact, and **12/12** spoken answers at
> 560 ms median. The no-transcript selfcheck, 31/31 module checks, and duplex
> barge-in gate pass (21.0 ms, zero dropped audio). Qualified Clean-360 remains
> narrowly above target at 5.09% normalized WER. Final-window full-adapter,
> output-only, three-seed, and soup candidates were retained as experiments but
> rejected because none improved every production gate. Physical-room five-
> minute testing is the remaining promotion gate.
>
> 2026-08-28 final GPU validation used one identical warmed harness for all
> adapter timings. JaegerOmni 41M Medium answered **24/24** repeated cases at
> **601.9 ms** median end-to-complete-reply; JaegerOmni 50M Medium answered
> **22/24** at **619.7 ms**. The historical 319 ms 50M result is therefore not
> an architectural advantage. Mathematically equivalent packed-QKV inference
> retained 24/24 identical replies but regressed 41M latency to 610.7 ms, so it
> remains disabled. Prefix-vs-final representation testing also rejects an
> immediate cached-Whisper prefill: even with four seconds of held right
> context, adapter cosine p05/relative-L2 p95 were 0.977/0.206 versus the strict
> 0.999/0.05 gate.
>
> A physical Elgato continuous-floor run then fed **298.0 seconds with zero
> dropped audio**, rolled context once at 6,151 tokens, retained positive soak
> accounting after the roll, and held peak loop lag to 178 ms. Raw-vs-fed audit
> survival was 11/11 words. It is a runtime soak, not the final interactive
> promotion: the source never provided 700 ms silence, so no turn closed and
> AEC/playback, reply latency, or barge-in could be measured. k=3 produced two
> nearly identical clusters (cosine 0.961), so shared-loudspeaker talkers remain
> unresolved.
>
> A separate 90-second interactive diagnostic shortened endpoint silence to
> 300 ms solely to exercise playback. It completed 5 turns with zero dropped
> audio, two continue-during-output events, and **494/522 ms** median/p95
> END-to-first-audio. Physical AEC is still a production blocker: attenuation
> was only **0.29 dB** over 9.5 playback seconds, and raw-vs-fed audit preserved
> 25/25 words of Jaeger's repeated “sound of a gong” reply. An offline sweep of
> 0--900 ms reference lead and 500/1200 ms filter length peaked at only 2.05 dB;
> timing/filter length alone does not solve continuous double-talk adaptation.
>
> 2026-08-28 GemmaOmni refresh: the exact latest causal checkpoint SHA
> `185bd7b3f6ce` is packaged as `gemmaomni-41m-medium-final4000`. Independent
> free-running reproduction at the corrected 12-token/s transcript budget is
> **6.81% dev-clean, 13.19% MINDS/Vox, 6.35% clean-360, 7.57% novel Vox, and
> 22.88% AMI**, with 22/24 repeated cases and 547.6 ms warmed median. The old
> evaluator's 4.48-token/s cap truncated fast speech; the authoritative harness
> now records its 12-token/s/24-token-floor budget in every artifact.
>
> A predeclared interpolation sweep produced
> `jaegeromni-41m-causal-balanced` SHA `e4ea88b9c99e`. Versus the frozen
> JaegerOmni 41M Medium it changes the five gates from 4.82/15.35/5.07/9.48/
> 32.99% to **5.03/14.60/4.89/9.27/28.23%**, retains 24/24 repeated answers,
> and improves warmed median latency from 601.9 to **574.2 ms**. It is a
> development candidate, not a replacement: dev-clean loses two words and its
> imported perception heads require recalibration after the trunk blend. Both
> causal nodes pass prefix stability at <4.8e-6 max drift; head shapes are
> finite and self-speech conditioning is connected.
>
> 2026-08-29 physical Elgato test: both JaegerOmni 41M Medium and Causal
> Balanced fed **299.9/300 seconds**, dropped zero audio, had zero mic-queue
> overflow, rolled once at 6,151 tokens, and preserved **100.0%** of the
> offline-audited raw-room words (657/657 and 630/630). Peak loop lag was
> 160.2/172.0 ms and MLX peak memory was 5,867.3/5,867.4 MB. Their native
> post-roll meeting minutes were grounded in the final 50 seconds at 120/120
> and 108/108 word survival. This is runtime parity, not a summary-quality A/B:
> the continuous recording advanced between windows, and the context roll
> discarded all pre-roll user audio because silent-observer mode had no prior
> replies to reseed.
>
> Offline k=3 diarization now separated two dominant alternating recorded
> talkers in both five-minute captures (pairwise centroid cosine maxima
> 0.477/0.467); a 12-second third cluster remains unlabeled. k=4 over-split a
> talker at cosine 0.912, so k=3 is the useful setting. This does not yet mean
> live word attribution. Physical AEC remains blocked: forced-playback runs
> measured only **0.17 dB wallclock** and **0.51 dB contiguous** attenuation;
> the runs had unequal 19/61-second playback exposure, and contiguous word
> survival regressed from 96.8% to 82.7%. Keep contiguous experimental.
>
> 2026-08-29 afternoon AEC root cause, measured with
> `research/aec_path_probe.py`: the loudspeaker-to-mic path delay is a
> **constant 614.1 ms** (three chirps, 0.0 ms spread, 0 ppm drift, coupling
> 0.56--0.62) — beyond Speex's 500 ms filter span — and the capture
> wall-clock-vs-sample mapping drifts **~1,250 ppm**, so no fixed offset can
> hold the ±20 ms alignment window offline replay proves is required.
> Offline replay of the real room recording with perfect alignment reaches
> **10.3 dB echo-coupling reduction**, proving the echo is linearly
> cancellable. The new echo-locked contiguous mode (mic-vs-stream
> correlation servo, deadband, early-side target) reaches **10.5 dB
> physical coupling reduction** on the equal-exposure fixed-waveform probe;
> a paired wallclock run measured 7.8 dB but wallclock varies 0--7.8 dB
> across runs with device buffering. Aggregate attenuation stays ~1 dB only
> because continuous room speech (which must survive) dominates total
> energy. Separately, the barge detector gained an attenuation-based
> double-talk override (loud + filter-failed-to-suppress + 6 dB
> filter-proven guard + 200 ms leaky sustain): the 43-room gate improved
> from 188/843 ms to **188/602 ms** median/p95 at 0/43 false and 43/43
> recall. The 300 ms tail gate remains red. The live mouth is now a
> two-stage synth-ahead pipeline, removing one full Kokoro synthesis of
> dead air between clauses.
>
> 2026-08-29 companion control implementation: the CLI and desktop demo now
> expose explicit `assistant|companion` behavior and `half|full` duplex.
> Companion admission is a fail-closed Gemma RESPOND/LISTEN decision made
> directly from native audio embeddings; control text is withheld from Kokoro,
> malformed output stays silent, and contextual follow-up permission is judged
> at speech-start time with speaker-continuity metadata. Causal speaker-change,
> overlap, turn-end, and speaker-ID probabilities are recorded as shadow
> telemetry only; their blended-checkpoint calibration is still invalid. A
> corpus-level free-running gate now measures direct-address recall, false
> responses, follow-ups, categories, and latency. Initial results reject
> promotion: 11/12 unaddressed questions were ignored, but only 4/12 explicit
> "Hey Jaeger" questions received a response. The same named audio produced
> 11/12 correct answers in the dual-output probe, so content transfer is not
> the main blocker; address admission is. The frozen model commonly renders
> the name as "Yeeger/Yeger", motivating a trained address head with phonetic
> positives rather than a looser output parser.
>
> 2026-08-29 neural VAD integration: the packaged, hash-verified Silero ONNX
> gate is now the default live and GUI speech detector. Four positive 30 ms
> frames within 180 ms open a turn at probability 0.30; a 0.125 hold threshold
> preserves quiet syllables. Exact-frame residual buffering prevents the 30 ms
> gate from losing 10 ms from each 100 ms capture block. The old RMS detector
> is retained only as explicit `--vad energy` fallback. On cases12, Silero
> opened and endpointed 12/12 clips (maximum probabilities 0.9970--0.9996).
> A five-minute Elgato/loudspeaker run then produced **6 complete neural-VAD
> turns**, fed 299.5/302.3 seconds with zero drops/queue overflow, held peak
> loop lag to 158.4 ms, rolled context once, and preserved 430/430 audited
> raw-room words in model-fed audio. Across 9,977 frames, mean/max speech
> probability was 0.0572/0.9191. The prior RMS continuous-floor run produced
> zero turns. This promotes neural VAD as the acoustic default, not the whole
> companion: its six generated observer notes were uninformative ("less than
> 10 seconds"), and environmental/addressee rejection remain separate gates.

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
| hands-free turn detection | **works** | `python -m jaeger_agent_omni.core.runtime.endpoint` |
| always-on duplex | **works** | `python -m jaeger_agent_omni.core.runtime.duplex` |
| barge-in | **20.7 ms**, context kept | same |
| omni input (a/v/text, one cache) | **works** | `research/check_all.py` |
| speech codes -> audible cloned voice | **works** | `research/t2w_client.py --say` |
| cognition -> talker conditioning | **built, untrained** | `neural/delta.py` |
| tuned Whisper -> sparse adapter -> E4B | **observer: 13.61% WER/64** | `meeting.py` |
| stock Whisper -> owned 50M adapter -> E4B | **candidate: 11.20% WER/64, 11/12** | `omni.py --front-end stock --dual-output` |

31/31 module self-checks green: `python research/check_all.py`

## Clean-room 41M crossover node — 2026-08-25

The independent 93-hour clean-room checkpoint is now packaged as its own
immutable, selectable node rather than overwriting the retained 50M adapter:

```text
jaeger_agent_omni/nodes/stock-whisper-medium-cleanroom-ctx93h-step2000
40,915,456 adapter parameters
12/12 canonical spoken answers, 572 ms median
0.00% Gemma-generated transcript WER, 67/67 words, 12/12 answers
3.234% instructed WER on the bridge's matched 128 clips / 2,041 words
5.047% empty-prompt WER on that same material
```

The q11 failure is closed: stock Whisper already decoded “Who wrote Romeo and
Juliet?” perfectly; the retained 50M adapter distorted it into “it worked
ramillo.” The clean-room node sends Gemma enough information to generate the
exact question and answer “William Shakespeare.” It therefore reaches native
MiniCPM's 12/12 answer accuracy, but not yet full production parity: its
572 ms median trails MiniCPM's 525 ms-to-text measurement, and independent
lockbox, live-room/meeting, and five-minute duplex gates remain open.

## Stock-Whisper live integration — 2026-08-25

The strongest pinned medium node is now selectable, without copying a loose
checkpoint path, from `omni.py`, `jaeger_agent_omni.demo`,
`jaeger_agent_omni.live`, and the full-duplex desktop app:

```text
jaeger_agent_omni/nodes/stock-whisper-medium-cekl-blend50
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
