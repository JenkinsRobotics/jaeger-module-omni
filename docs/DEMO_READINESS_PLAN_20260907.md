# Demo readiness execution contract · 2026-09-07

User authorized the six-stage development plan, including GPU trials and
physical tests. This document fixes the interpretation before new measurements.
The predictive-coding playground is separate research; it does not replace this
production work queue.

1. **Paired reference:** MiniCPM-o 4.5 and unchanged Jaeger medium, identical
   PCM and scoring, fresh state, disclosed prompts/decode budgets, warmup outside
   reported samples. Start with devclean64, AMI64 and cases12; extend to visual
   turns, actions and interruption scenarios. These development populations are
   not fresh lockboxes. Record latency scope and memory coverage rather than
   combining incomparable engine counters into a claimed total.
2. **Playback listening:** evaluate a frozen pretrained WavLM representation
   with a separate learned overlap detector. Preserve the speaker/AMI-group
   disjoint labels. Any bounded-context implementation must measure availability
   delay and compute, not report offline bidirectional predictions as streaming.
   Gates remain F1 >= .81 on each corpus, event recall >= .90, <= 1 false event
   per minute, and p95 detection <= 300 ms including buffering and inference.
   AEC independently retains its <= 300 ms p95 gate.
3. **Ear quality:** categorize current errors before choosing a distinct
   prefix-consistency experiment. Retain existing ears until matched clean and
   meeting quality plus latency improve; keep the lockbox closed until then.
4. **Action fulfillment:** compare identical requests as text and native audio.
   Audit actual file/memory/result state and multi-step completion. Preserve
   original corpus scores alongside any additional outcome checks. Ground-truth
   request text belongs only in the explicitly labeled text control.
5. **Multimodal coverage:** broaden unseen visual questions, follow-ups,
   corrections and retained memory; speaker context remains opt-in. Distinguish
   shared text history from simultaneous image/audio attention.
6. **Demo hardening:** preserve quasi default and every selectable tier; measure
   recovery, cancellation, startup failure and physical capture. Acceptance is
   >= 95% completion on a frozen representative demo suite, zero fabricated
   action-success confirmations, and three consecutive 30-minute physical runs
   with no crashes/capture drops. Replay and human interaction must be labeled
   separately. Existing red gates are not waived by a narrow demo pass.

One GPU job at a time, no weight commits, no same-recipe failed continuations,
and no changes to the user's separate GUI/selector work. Each stage retains
commands, hashes, raw outcomes and a pass/reject decision before promotion.

Implementation checkpoint · 2026-09-08:

- `research/paired_omni_benchmark.py` freezes identical PCM and uses the existing
  WER/answer scorers. Jaeger reproduces 46/955 clean errors and answers 12/12.
  The MiniCPM native audio template is now qualified. Matched clean WER is
  Jaeger **4.82%** vs MiniCPM **2.20%**; AMI **33.28%** vs **13.82%**;
  both answer **12/12**. Jaeger has lower measured complete-text request latency.
  These are deployed configurations (Jaeger 4-bit MLX, MiniCPM Q8 llama.cpp),
  not equal-precision training comparisons. See `paired-omni-comparison-20260907.json`.
  No full-system superiority or fresh-lockbox claim is supported.
- `research/ear_error_audit.py` reconciles every alignment with the fixed WER
  scorer. The meeting sample contains 123 substitutions, 93 deletions and
  8 insertions; clean speech 39/4/3. Neither sample hit the decode cap.
  These slices do not establish noise/overlap causation without audio labels.
- Commit `f43bbaf9` adds the explicit text control and real state checks for
  12 file tasks. With four concrete file schemas exposed, text yields **10/12**
  actual file outcomes and speech **1/12** (the old scorer says **8/12**).
  Without the extra schemas both arms yielded **0/12** actual outcomes.
  Wrong file names, content case and incomplete multi-step actions remain
  visible failures. Full corpus yields 45/162 old-scorer passes, versus historical
  31/162, but actual files remain 1/12 and four old passes are lost. File-schema
  promotion is rejected. The phonemizer drops the digit/underscore in two frozen
  requests; these fixture limitations are recorded without changing scores.
- Frozen WavLM-base-plus offline head training completed: test F1 **.873/.837**
  meets the frame target, but **4.50/14.45 false events per minute** and future
  audio disqualify it. The 1280 ms past-window/160 ms hop trial with
  explicit feature availability timestamps completed all 7,146 cached rows and
  eight head epochs. It is rejected: F1 .819/.798, false events 3.30/23.84 per
  minute, buffered p95 720/480 ms. No live replacement is qualified.
- A 20 ms AEC callback trial was rejected: p95 **1736 ms**, recall **29/43**,
  one false interruption. The unchanged 100 ms control reproduces **601.9 ms**,
  **43/43** and zero false interruptions. Physical/compute delay is additional.
- Device callback failures now surface to the owner. Speech clauses carry
  reply IDs, preventing a synthesis result from reviving cancelled speech.
  The report counts fully delivered speech separately from generated replies.
  CPU checks pass: 112 tests, 14 subtests and 18/18 selftest. The current default
  barge campaign first failed 3/4 then passed 4/4; the other tiers pass 23/23.
  A more sensitive neural endpoint passes 27/27 but fails a separate self-echo
  audit; it stays disabled. Physical replay captures 304.9 seconds with zero
  drops/overflows and clean shutdown, but zero successful actions and zero
  uninterrupted replies. The hardened physical gate correctly fails 14/16.
- `research/demo_suite_v1.json` freezes 48 tasks before the new action/vision
  results. `research/demo_acceptance.py` requires matching audio, outcome
  evidence, an explicit false-success review and three distinct 30-minute
  physical reports with the same runtime/configuration. The current suite is
  **35/48 (72.9%)** and physical acceptance is red. Three 30-minute runs have
  not been attempted while the five-minute behavioral gate fails. The existing
  release gates remain separately required.
- A frozen vision/projector-only node and `research/omni_multimodal_demo.py`
  now feed image/audio/text into one persistent MLX Gemma brain. The synthetic
  joint probe passes 20/20; the runnable demo passes three mixed turns and saves
  three valid speech WAVs. Natural-image and MiniCPM multimodal coverage remain.
- The distinct audio-logit-KL causal trial completed 150 steps. Two disabled
  checkpoint nodes are rejected: clean WER 5.97/5.76%, meeting 32.10/31.95%,
  both worse than the full control 5.03/28.23%. No failed recipe is continued. Detailed evidence and current decisions are in
  [the execution report](OMNI_DEMO_PROGRESS_20260907.md).

Production selection remains medium for half/quasi, causal for full, and
speaker context opt-in. No new candidate has been promoted.
