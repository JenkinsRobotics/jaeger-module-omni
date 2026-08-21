# Jaeger Agent Omni — what works, 2026-08-21

One page. Every claim here has a command that reproduces it.

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
omni [tower]   10/12 = 83%   median 482 ms
legacy         12/12 = 100%  (text between modules, 9.8 GB, two model stacks)
MiniCPM-o 4.5  12/12 = 100%  525 ms to text, 1380 ms to speech
```

83% of legacy, integrated, in one process. The two misses are the frozen
tower's: one read audio as an image, one never named Shakespeare.

| capability | state | command |
| --- | --- | --- |
| audio -> text, no transcript | **10/12, 482 ms** | `omni.py --bench` |
| hands-free turn detection | **works** | `python -m jaeger_agent_omni.runtime.endpoint` |
| always-on duplex | **works** | `python -m jaeger_agent_omni.runtime.duplex` |
| barge-in | **22.8 ms**, context kept | same |
| omni input (a/v/text, one cache) | **works** | `research/check_all.py` |
| speech codes -> audible cloned voice | **works** | `research/t2w_client.py --say` |
| cognition -> talker conditioning | **built, untrained** | `neural/delta.py` |
| whisper -> delta -> E4B | **training** | `training/train.py` |

30/30 module self-checks green: `python research/check_all.py`

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

# Why the native tower is a dead end, and the delta is not

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

# Next

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
