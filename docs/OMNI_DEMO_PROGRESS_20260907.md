# Omni demo development · 2026-09-07–08

This execution follows [the frozen plan](DEMO_READINESS_PLAN_20260907.md).
Production selection remains quasi by default in converse, medium for half/quasi,
and the current causal ear for full. Speaker context remains opt-in. No new
adapter or overlap model has been promoted. The separate native `live.py` and
user-owned GUI currently expose half/full; the GUI work has not been edited.

## Matched reference benchmark

On the same M1 Max, identical 140 PCM fixtures and existing scorers:

| Measure | Jaeger medium | MiniCPM-o 4.5 |
|---|---:|---:|
| Clean ASR, 64 clips / 955 words | 46 edits · 4.82% | 21 edits · 2.20% |
| Meeting ASR, 64 clips / 673 words | 224 edits · 33.28% | 93 edits · 13.82% |
| Spoken factual questions | 12/12 | 12/12 |
| Clean complete-text request median | 591 ms | 1074 ms |
| Meeting complete-text request median | 430 ms | 852 ms |
| Question complete-text request median | 608 ms | 1341 ms |

MiniCPM required its native audio generation template even when synthesis was
disabled. The initial incorrect-template diagnostics are retained and excluded.
The qualified text-only template matches the two diagnostic outputs from the
native TTS-enabled template. These are practical deployed configurations:
Jaeger uses 4-bit MLX Gemma; MiniCPM uses Q8 llama.cpp. Latency excludes loading,
warmup, endpointing and speech synthesis. These development sets are not new
lockboxes; AMI is training-adjacent. No overall superiority claim is supported.

Evidence and reproducible commands:
[comparison](../research/candidate_evals/paired-omni-comparison-20260907.json),
[protocol audit](../research/candidate_evals/paired-protocol-audit-20260907.json).
The comparator rejects differing audio, scorers, prompts, decode budgets or hosts.

## Listening during playback

Frozen WavLM-base-plus plus a separately trained GRU head achieves test frame F1
**.873 on barge data and .837 on AMI** after selecting epoch 6 using validation.
It fails promotion: the encoder sees future audio, false events are
**4.50/min and 14.45/min**, and barge p95 delay is **400 ms** before buffering
and compute. The completed Python job saved its evidence; the subsequently
edited shell wrapper exited 2. The incident is recorded, not counted as a clean
launcher run. Launchers and active experiment code must remain unchanged during jobs.

The next trial uses 1280 ms past-only windows, published every 160 ms, with
availability timestamps attached to all four 40 ms predictions. A CPU probe
measured about 50 ms p95 encoder compute per update; this is a small compute
probe, not physical end-to-end latency. All 7,146 bounded-context feature rows completed successfully. The eight-epoch
head selected epoch 4 on validation and is rejected: test F1 **.819/.798**,
false events **3.30/23.84 per minute**, buffered p95 **720/480 ms** before
compute, and barge event recall **.886**. The AMI frame target is missed.
[Bounded decision](../research/candidate_evals/wavlm-bounded-decision-20260908.json).
A separate validation-selected alarm policy requires onset inside the truth
interval. Test event recall is **.835/.869**, false events **1.88/12.92 per
minute**, and p95 **780/640 ms**. Early alarms no longer receive later-overlap
credit. This stricter diagnostic also rejects promotion; it preserves the
original frame scorer and excludes encoder compute. [Alarm-policy evidence](../research/candidate_evals/wavlm-bounded-alarm-policy-20260908.json).

Evidence: [offline decision](../research/candidate_evals/wavlm-overlap-decision-20260907.json),
[bounded CPU probe](../research/candidate_evals/wavlm-bounded-cpu-probe-v1-20260907.json).

A separate AEC experiment rejected 20 ms callbacks: 29/43 detections, one false
interruption and 1736 ms p95. The 100 ms control exactly reproduces the previous
43/43, zero false interruptions and 601.9 ms p95. Default DSP/policy is unchanged.
[AEC decision](../research/candidate_evals/aec-cadence-decision-20260907.json).

## Actual actions

The file-schema pilot adds `write_file`, `read_file`, `append_file` and
`delete_file` to the initial tool prefix. It is currently a research flag.

| Twelve matched file tasks | Text control | Native speech |
|---|---:|---:|
| Existing bootstrap, verified file outcomes | 0/12 | 0/12 |
| File schemas exposed, original scorer | 10/12 | 8/12 |
| File schemas exposed, verified file outcomes | 10/12 | 1/12 |

Tool routing is not completion. Speech examples include `level_test.txt` instead
of `level3_test.txt`, a space instead of an underscore, changed capitalization,
and omitted later steps. Exact text/capitalization checks also expose information
that ordinary speech may leave ambiguous. Those frozen failures remain visible;
the scorer has not been relaxed. The identical-waveform ASR diagnostic scores
45/148 word edits (30.41%), including spoken-number and punctuation differences.
Both calculation requests retain the instruction to save, and both chained
file requests retain their later steps. This supports an action-following
problem beyond transcription; the separate ASR prompt is not a hidden-state audit.

A reproducible phonemizer audit finds that `level3_test.txt` loses both the
digit 3 and underscore before synthesis in two requests. These frozen cases
remain failed, but are not evidence that the ear dropped audible digits. Future
spoken-file fixtures need explicit, acoustically recoverable spelling; do not
change these inputs retroactively. See [phoneme audit](../research/candidate_evals/agentic-phoneme-audit-20260907.json)
and [ASR diagnostic](../research/candidate_evals/agentic-file-asr-diagnostic-20260907.json).

[Matched input comparison](../research/candidate_evals/agentic-files-input-comparison-20260907.json).
The full 162-case speech run with file schemas scores **45/162**, compared with
historical **31/162**, and routing **35/128** versus **21/128**. Actual file
outcomes remain **1/12**. There are 18 old-scorer gains and four losses, including
a forbidden read attempt that returned no contents. One clarification loss is
an old substring-grader limitation. Production promotion is rejected. This is
not a same-day control; external facts/network conditions vary, and CPU work
was shared with AEC/testing. [Full comparison](../research/candidate_evals/agentic-files-full-comparison-20260907.json).

A generic completion instruction improves speech actual files to **2/12**;
text stays **10/12**. One response leaks an unexecuted `call:write_file` line.
[Completion decision](../research/candidate_evals/agentic-completion-hint-decision-20260908.json).
A separately frozen v2 fixture arm explicitly pronounces digits, underscores
and extensions and reaches **4/12** actual files (old scorer 10/12). It repairs
the `chain_note.txt` sequence, but produces `level_3_underscore_test.txt` for
the other filename. This changes the inputs, not the ear; it cannot replace
the v1 suite score. Two list tasks execute all tools but join item strings
without separators. The exact line-format contract is stronger than the
spoken request’s explicit formatting requirements and remains reported.
[V2 speech report](../research/candidate_evals/agentic-spelled-files-speech-v2-20260908.json).

A predeclared two-case code-budget control compares 96 versus 256 tokens on
identical audio, prompts, adapter, runtime and 20-second deadline. Both arms
complete **0/2** requests. At 256, one previously truncated response creates a
valid `skills/fib10.py`; independent execution confirms the expected numbers,
but the agent chooses `computer_do` instead of `run_python`, and the benchmark
policy denies that desktop effect. The other case repeatedly omits `path`.
No parser was relaxed, no independent execution was counted as an agent pass,
and the default budget is unchanged.
[Budget decision](../research/candidate_evals/agentic-code-budget-decision-20260908.json).

## Runtime and acceptance

Device callback errors and missing callbacks now surface to the owner. Stream
start/stop failures close resources. Speech clauses carry reply identities so a
cancelled synthesis result cannot restart playback. Reports distinguish generated
answers from fully delivered, uninterrupted speech. Context rolls run between
model operations on the owner thread. Worker shutdown failures are recorded.

Startup validates the selected audio formats before loading Gemma. Teardown
closes the microphone before freeing AEC, records failures, and exits nonzero
on a stuck/failed device close. Source identities are captured at run start.

Current-workspace CPU regression validation passes **112 tests and 14 subtests**, plus converse
**18/18**. The `live --demo` check verifies wiring only.
[CPU evidence](../research/candidate_evals/demo-hardening-cpu-final-20260908.json).

Current tier runs: half 12/12, quasi 6/6 and full 5/5. Barge-continue first
failed 3/4, then passed 4/4 under frame diagnostics. All 27 checks have passed
across the latest tier runs, but this is not a clean consecutive campaign;
the first failure remains a stability risk. Echo-cleaned speech repeatedly
falls below the energy endpoint threshold.
[Repeatability record](../research/candidate_evals/converse-repeatability-20260908.json).

The physical acoustic replay captured **304.9 seconds with zero drops or
overflows**. Teardown completed in 134 ms with no errors or source changes.
Behavior failed: **zero completed uninterrupted replies**, three interrupted
replies, and **zero successful action results** from three tool calls. The old
physical scorer incorrectly passed 13/13 by counting attempts. The hardened
scorer requires successful action results, completed speech and clean shutdown;
regrading gives **14/16, failed**. Original reports remain unchanged and are
superseded by this explicit decision. There is no 30-minute acceptance run.
[Physical decision](../research/candidate_evals/physical-demo-hardening-decision-20260908.json).

The frozen development suite currently scores **35/48 (72.9%)**, below 95%: 12 answers,
one exact file outcome, 15 visual checks and seven duplex checks pass. This
uses the first failed barge run, rather than substituting its diagnostic retry.
This aggregates the declared research QA, file-schema, legacy vision and duplex
reports; it is not one physical session or proof of a single deployed configuration.
Input defects and implicit formatting requirements limit what individual
failures establish; neither changes the frozen score. Physical acceptance stays
red. [Acceptance report](../research/candidate_evals/demo-acceptance-current-20260908.json).

The frozen [48-case suite](../research/demo_suite_v1.json) requires at least 95%
completion. The acceptance checker also requires evidence-bound false-success
reviews for scripted actions and all three physical sessions. A physical tool-call
counter alone cannot certify completed actions. Each physical review must name
the native report SHA, verify at least three actual action outcomes and explain
why the observed success confirmations are supported. Three matching 30-minute
reports must have no crashes, capture drops or speech failures. Acoustic replay
is labeled separately from human conversation.

## Joint vision experiment

The existing converse vision fallback uses a separate GGUF Gemma session with
shared text memory. It does not prove image/audio coexistence in one brain cache.
New disabled node `gemma4-vision-mlx-v1` contains only frozen vision/projector
weights. The research probe feeds projected image features, raw audio and text
into the existing MLX Gemma session. Four real CPU encodes produced 256×2560
features. The GPU probe passes **20/20** semantic turns in the same unchanged
brain instance/cache, including simultaneous image + raw audio. This is narrow
synthetic coverage with cached vision features; encoding latency is excluded.
[Joint probe](../research/candidate_evals/joint-vision-probe-20260908.json). There are now
35 immutable nodes and 13 profiles. No profile enables the new node. The previously measured speaker-sensitive
control remains 19/24 recall with ECAPA labels versus 11/24 without, accompanied
by worse HEARD WER (5.55% vs 5.03%) and one annotation leak; context stays opt-in.
See [the prior controlled evaluation](OMNI_CONSOLIDATION_REVIEW_20260905.md).

The legacy shared-text vision fallback scores **19/20** on expanded coverage:
it identifies the green triangle correctly, then incorrectly recalls blue in
the spoken follow-up. The new 20/20 shared-cache probe uses a different prompt
protocol, so this is not a controlled claim that one change caused the gain.

An additional disabled `para-turn-state-neural-v1` node reuses frozen Silero
weights with independent recurrent state. In a CPU gain/noise diagnostic it
detects all four speech gains and rejects all four noise controls; the old
energy gate gets 4/8. One synthetic phrase is not room qualification. The neural endpoint passes a full sequential 27/27 campaign; its barge-continue
run passes the original 4/4 checks but admits the agent’s own “It’s four” as a
new user turn and replies “That’s correct.” That extra response blocks
promotion regardless of the original scripted score. A separate one-question
self-echo audit confirms old energy **2/2** versus neural **1/2**: the neural
candidate produces two replies for one request. It stays disabled in all profiles.
[Decision](../research/candidate_evals/neural-turn-decision-20260908.json).

## Causal ear experiment

The distinct audio-conditioned logit-KL trial completed 150 steps in 233 seconds,
with 11.2 GB peak Metal allocation. It starts from unchanged medium, uses human
transcript CE plus frozen-medium audio-logit distillation, and has no future-frame
output anchoring. Step 50 and step 150 are separate disabled immutable nodes;
both are rejected on matched clean/meeting WER. No continuation is planned.

| Causal ear | Clean, 955 words | Meeting, 673 words |
|---|---:|---:|
| Current full control | 48 edits · 5.03% | 190 edits · 28.23% |
| Audio-KL step 50 | 57 edits · 5.97% | 216 edits · 32.10% |
| Audio-KL step 150 | 55 edits · 5.76% | 215 edits · 31.95% |

The same ordered references, manifest hashes and scorer hashes validate both
comparisons. Loss reduction did not translate into improved recognition.
[Quality decision](../research/candidate_evals/causal-audiokl-decision-20260908.json).
[Training record](../research/candidate_evals/causal-audio-kl-training-20260908.json).

On the same M1 Max/32 GB and one frozen 12-second waveform, six previews at
2/4/6/8/10/12 seconds take **4.940 s total for the current full ear and 4.949 s
for step 150**. Final-prefix requests take 1.314/1.325 s. Both adapter prefix
checks have zero measured drift, but the encoder processes 42 seconds of audio
for 12 seconds of unique input: **3.5× re-encoding**. Peak MLX allocation is
about 5.25 GB in each run, excluding memory owned by other runtimes. This is
one warm ASR trace, not endpoint latency or a general speed comparison.
[Prefix-cost comparison](../research/candidate_evals/prefix-cost-comparison-20260908.json).

## Next measured changes

Playback self-echo and action completion remain the first demo blockers. A
replacement overlap/admission model must distinguish the speaker from the agent,
pass false-event and publication-delay gates, and survive the new one-question
self-echo audit. Improving speech-presence sensitivity alone failed that audit.
AEC still needs an independent tail reduction below 300 ms.

For actions, freeze acoustically unambiguous requests with explicit formatting
contracts before another matched text/speech experiment. Keep v1 scores visible.
Measure safe, structured multi-step execution and actual artifacts, including
error reporting; do not execute tool-looking prose or claim success from attempts.
The current generic instruction and file-schema pilots do not qualify for default.

The runnable [multimodal demo](MULTIMODAL_DEMO.md) passes three mixed turns with
saved, non-silent speech. Broaden it to natural images and long sessions, then
extend the qualified MiniCPM comparison to vision, actions and interruptions.
A five-minute physical behavioral pass must precede the three 30-minute sessions.
The AEC tail, action completion and physical acceptance remain red. Training
and test failures are evidence for the next change, not release passes.
