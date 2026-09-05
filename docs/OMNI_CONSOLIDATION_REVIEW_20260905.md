# Omni consolidation review — 2026-09-05

Keep quasi-duplex as the default. Half/quasi use `jaegeromni-41m-medium`;
full retains `jaegeromni-41m-converse`. Keep speaker context opt-in. The
measured training candidates do not qualify to replace these ears or the
existing overlap policy. This is not a release-ready declaration.

## Selected pipeline and verified changes

| Area | Measured outcome | Decision |
|---|---|---|
| Tier routing | Final default campaign 27/27; selftest 18/18 | Keep medium on half/quasi, causal on full |
| Transcription | Medium 46/955 edits, 4.82%; current causal 48/955, 5.03% | Medium remains the turn-based ear |
| One-generation HEARD/REPLY | Fresh-turn medium 48/955, 5.03%; causal 52/955, 5.45% | Keep this protocol separate from transcription-only WER |
| Parallel overlap | Best tested F1 0.633 synthetic / 0.728 AMI; event false alarms 20.93 / 45.77 per minute | Reject both scratch TCN trials; do not retire existing policy |
| Heads-free causal port | Mask, step 250 and step 500 all lose both matched WER gates to current full ear | Three immutable nodes packaged, disabled in every profile |
| Speaker context | ECAPA recall 19/24 versus 11/24 without labels; medium HEARD WER 5.55% versus 5.03% | Keep off by default; opt-in also has one observed spoken annotation leak |
| Vision/text/audio | Real-model image→typed→voice memory test 4/4 | `/image PATH` works; shared history now survives rolling |
| AEC tail | 43/43 events, zero false barges, p95 601.9 ms; no global policy trial meets all gates | Keep existing AEC policy; 300 ms gate remains red |
| Full agentic corpus | Current 31/162 original scorer, 35/162 with the separate integer-equivalence audit; routing 21/128 | Native continuation is fixed; action fulfillment remains below the gates |
| Physical five-minute run | Neural generation guard: 299.8 s, no drops/overflows, 3 spoken replies; checks 9/13→12/13 | Still fails tool use; playback interruptions and ungrounded clock replies remain |

The default simulator campaign now starts interruptions only after actual
nonzero playback. Quasi's stop test succeeded on its second allowed attempt;
27/27 is not a first-attempt barge-recall claim. Assertion output cannot
satisfy simulator waits for real runtime replies. The first vision report is
retained but explicitly invalidated because its waits could match assertion
output; the corrected report is the 4/4 result used here.

Vision currently uses the existing GGUF Gemma vision twin and the MLX
audio/text model with shared **text history**. The test establishes image
intake and cross-modality recall; it does not establish simultaneous
image/audio attention or persistent image vectors in one MLX cache.

Native deterministic-tool results now return to Gemma, because the inherited
text multi-step detector cannot inspect an opaque audio turn. This preserves
12/12 native answers with no unexpected tools (2.26 s median), and allows
Gemma to emit the companion envelope after a tool. It does not make Gemma
fulfill every requested action. The full corpus remains 31/162 under its
unchanged scorer; four correct spelled-out integers lose their former numeric
substring matches. A separate, identical integer-equivalence audit on all
three runs yields baseline/discovery/fixed counts of 35/42/35, with identical
baseline/fixed pass sets. The release gate has not been replaced or relaxed.
The rejected discovery prompt scores 42/162 but increases median latency from
2.88 to 4.46 s and adds unwanted tool use. The initial-schema-order pilot
yields no gain after applying the same number-equivalence diagnostic.

The first physical run exposed generation cancellation on ordinary background
energy. Neural mode now uses an independent instance of the existing Silero
NN on raw waveform frames for this decision. On the recorded pre-actor
background, RMS triggers in 50/50 blocks and the neural speech guard in 0/50.
The same-device/actor retest reduces generation-stage barges from 34 to 2;
AEC runs for 10.3 seconds and both companion decisions occur. Live NN
processing p95 is 2.40 ms per 100 ms block. Playback AEC policy is unchanged,
and this speech detector is not a trained overlap detector. All 65 package
tests pass (14 subtests); no new weights or adapter node were required.
Selftest 18/18, module checks 31/31, native selfcheck and demo check also pass.

## Training and comparison discipline

All five compared ear packages contain the same Whisper-medium encoder bytes.
The two selected adapter hashes are unchanged. The causal trials contain
40,915,456 trainable adapter parameters; Whisper and Gemma remained frozen.
No weights were committed. The registry contains 31 nodes and 13 profiles.

The causal trial uses 45,725 human-audio rows, speaker-disjoint holdouts,
500 steps, and a causal attention mask with text KL and bidirectional output
anchoring. This is one completed port trial, with separate mask-only,
250-step and 500-step nodes, not a same-recipe continuation. Adapter prefix
causality is checked; the frozen Whisper encoder still re-encodes prefixes.

| Ear | devclean64 WER | AMI64 WER |
|---|---:|---:|
| Current medium | 4.82% | 33.28% |
| Current full causal | 5.03% | 28.23% |
| Medium causal mask only | 5.24% | 33.58% |
| Medium causal port, step 250 | 7.43% | 29.12% |
| Medium causal port, step 500 | 7.64% | 28.38% |

These are matched converse decoding results, with identical ordered examples,
references and scorer. Historical native-route AMI measurements are a
separate protocol. No new test-clean lockbox evaluation was justified because
the new causal candidates failed the development gates. All five ears passed
the coarse cases12 answer grader; that does not override their WER failures.

## Evidence and reproduction

The following JSON artifacts include commands, identities and raw outcomes;
paths below are relative to `research/candidate_evals/`:

- `final-default-campaign-20260905.json` and its four session reports.
- `tier-matched-gpu-20260904.json` for the paired WER protocols.
- `parallel-overlap-data-20260904.json`,
  `parallel-overlap-trial-20260904.json`, and
  `parallel-overlap-trial2-20260904.json`.
- `medium-causal-port-training-20260904.json` and
  `medium-causal-quality-comparison-20260904.json`.
- `converse-cases12-summary-20260904.json`.
- `converse-context-comparison-20260905.json` and
  `final-context-optin-half-20260905.json`.
- `converse-vision-verified-markers-20260905.json`.
- `aec-contiguous-20260904.json` and `aec-policy-trials-20260904.json`.
- `agentic-discovery-comparison-20260905.json` and its two full reports.
- `agentic-continuation-comparison-20260905.json` and
  `agentic-numeric-audit-20260905.json`.
- `agentic-bootstrap-comparison-20260905.json`: initial-schema-order pilot,
  no action-fulfillment gain (same diagnostic 5/11 and routing 4/10).
- `physical-generation-barge-comparison-20260905.json` and its two physical reports;
  `generation-barge-replay-20260905.json` isolates the pre-output noise trigger.
- `final-cpu-checks-20260905.json` and `final-ear-identity-20260905.json`.

## Next experiments warranted by the evidence

1. Before another overlap training run, change the representation or learning
   objective and pre-register the same disjoint F1 and event gates. Two scratch
   TCN recipes do not support another continuation of either recipe.
2. Preserve the current full ear until a distinct causal training method beats
   both matched corpus gates. A mathematically causal adapter alone is not an
   accuracy improvement or a cached streaming encoder.
3. Keep speaker context experimental until its recall gain survives broader
   speakers and room recordings without transcription or spoken-label
   regressions. Do not change the global default on this two-voice pilot.
4. Discovery was rejected for default use. Native continuation is fixed, but
   Gemma still often answers without completing the remaining actions. Further
   agentic work must target action fulfillment, using the full corpus and the
   unchanged transcription/answer controls.
5. The five-minute physical replay has run twice. Keep native agentic room
   interaction experimental: the retest still made no tool calls, clock
   answers were ungrounded, and all three spoken replies were interrupted
   during playback. The pre-output speech guard is useful but does not
   establish reliable room interaction. AEC tail remains a separate red gate.
