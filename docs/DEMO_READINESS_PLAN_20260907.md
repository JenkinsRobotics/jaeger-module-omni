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

Implementation checkpoint (in progress):

- `research/paired_omni_benchmark.py` freezes identical PCM and uses the existing
  WER/answer scorers. Jaeger reproduces 46/955 clean errors and answers 12/12.
  The initial MiniCPM ASR template is **not qualified**: unexpected task output
  requires native prompt validation. `paired-protocol-audit-20260907.json`
  explicitly prevents those diagnostic runs supporting a superiority claim.
- `research/ear_error_audit.py` reconciles every alignment with the fixed WER
  scorer. The meeting sample contains 123 substitutions, 93 deletions and
  8 insertions; clean speech 39/4/3. Neither sample hit the decode cap.
  These slices do not establish noise/overlap causation without audio labels.
- Commit `f43bbaf9` adds the explicit text control and real state checks for
  12 file tasks. GPU text/speech comparison is pending.
- Frozen WavLM-base-plus features are being cached outside git. This is an
  offline representation trial; it cannot pass the live promotion gate.
- Device callback failures now surface to the owner. Speech clauses carry
  reply IDs, preventing a synthesis result from reviving cancelled speech.
  The report counts fully delivered speech separately from generated replies.
  CPU fault checks pass; the GPU campaign and physical validation are pending.
- `research/demo_suite_v1.json` freezes 48 tasks before the new action/vision
  results. `research/demo_acceptance.py` requires matching audio, outcome
  evidence, an explicit false-success review and three distinct 30-minute
  physical reports with the same runtime/configuration. The initial report is
  correctly red. The existing release gates remain separately required.

Production selection remains medium for half/quasi, causal for full, and
speaker context opt-in. No new candidate has been promoted.
