# Independent production-readiness review prompt

Use this same prompt with two agents independently. Do not let one agent's
answer influence the other before both reports are complete.

---

Review the Jaeger Agent Omni repository as an independent senior audio/ML and
real-time systems reviewer.

Repository: `~/GITHUB/jaeger_agent_omni`

This is a **CPU-only, read-only review**. Do not start training, model
inference, live benchmarks, downloads, or any GPU job. Do not edit files. A
separate GemmaOmni agent currently owns the GPU. Do not inspect, pull, or
import that agent's newer in-progress GemmaOmni work; assess only the frozen
GemmaOmni checkpoint already present in this repository.

Read these first, completely:

1. `STATUS.md`
2. `NEXT.md`
3. `dev/lessons from gemma-backbone.md`
4. `PIPELINES.md`
5. `PRODUCTION_TESTING.md`
6. `Full DuplexDemo App/README.md`

Then inspect the implementation and saved evaluation artifacts needed to
verify the claims. The canonical active model roster is exactly:

| Correct model name | What it is | Current measured results | Current status |
| --- | --- | --- | --- |
| **Gemma Native** | Gemma's built-in audio tower | 10/12, 363 ms | Duplex-validated fallback |
| **GemmaOmni 41M Medium** | Exact frozen GemmaOmni checkpoint imported here | **12/12**, 4.92% clean, 34.47% AMI, ~501 ms | Best general accuracy; five-minute gate pending |
| **JaegerOmni 41M Medium** | 10% blend toward Jaeger's retained room continuation | **3.97%** untouched test-clean/51,311 words; 4.92% clean64; 34.18% AMI; 15.22% MINDS/Vox591 | Broad-corpus winner; five-minute gate pending |
| **JaegerOmni 50M Medium** | Jaeger's independently developed Medium connector | 11/12, 11.20% clean, 47.85% AMI, **319 ms** | Fastest viable adapter |
| **JaegerOmni 50M Large** | Large-v3-turbo-paired connector, exactly 52,768,768 deployed parameters | 1/12, 17.80% clean, 413 ms | Transfer failed; experimental |

The product target is a production-ready full-duplex system comparable to or
better than MiniCPM-o 4.5, ultimately with one excellent Medium and one
excellent Large ear. The runtime must preserve the no-transcript input-path
premise: no ASR decoder, `transcribe()`, or transcript-to-Gemma handoff. Inline
speaker metadata between turns is allowed. Current production output is
Gemma-decoded text streamed by clause into Kokoro; the native hidden-state
Talker is experimental and must not be described as production.

Evaluate, with file-and-line evidence where possible:

- whether the GUI actually selects all five distinct checkpoints while holding
  downstream Gemma, Kokoro, AEC, endpointing, context, and policy constant;
- checkpoint identity, architecture compatibility, registry/profile safety,
  and whether any alias or fallback can silently select the wrong ear;
- free-running answer accuracy and corpus WER evidence, including sample size,
  speaker/domain diversity, prompt conditions, checkpoint-selection leakage,
  and whether every comparison is genuinely apples-to-apples;
- causal/streaming behavior, bounded audio windows, dropped-input risk,
  rolling-context correctness, abandoned-generation rewind, and five-minute
  stability;
- END-to-first-audio median, p95, and maximum, including clause boundaries and
  long-first-clause risk;
- full-duplex AEC using the exact generated playback reference, residual
  double-talk behavior, and the distinction between a direct-WAV test and a
  physical loudspeaker-to-Elgato test;
- barge-in as a real-time semantic notification/policy input, not merely a
  hard-coded stop action;
- speaker-change, identity, overlap, and per-speaker context capability, while
  distinguishing segment labels from true word-level overlapping-speaker
  attribution;
- why the 41M and 50M Medium results differ and why the Large transfer failed;
- the decoded-text Kokoro seam: its present production value, latency cost,
  and a safe future route toward richer prosody without misrepresenting it as
  hidden-state end-to-end speech today;
- whether `PRODUCTION_TESTING.md` is sufficient to promote a model without
  benchmark gaming or a narrow matched-domain win.

Do not recommend tuning to one fixed evaluation set. Prefer universal changes
that should improve clean speech, room speech, meetings, and live duplex use.
Clearly separate verified facts, reasonable inferences, and hypotheses that
require a GPU experiment.

Return:

1. a concise verdict on current production readiness;
2. findings ordered by severity, with file/line or artifact evidence;
3. a corrected five-model comparison table if any listed claim is unsupported;
4. CPU-only fixes or analyses that can be completed before the next GPU window;
5. the smallest high-information next GPU experiment for Medium, followed by
   the corresponding Large plan only after Medium is understood;
6. explicit pass/fail promotion gates for accuracy, latency tails, streaming,
   AEC, barge-in notification, speaker notification, overlap handling, and
   five-minute stability;
7. the top three changes most likely to produce a universal improvement rather
   than a benchmark-specific score.

Do not call a model production-ready solely because it reaches 12/12 or a low
WER on one corpus. Do not combine historical best numbers from different
checkpoints into a fictional pipeline. Cite uncertainty directly.
