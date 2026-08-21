# Voice LLM Pipeline Taxonomy

Three independent axes. The reference implementations in `reference/` are named
for the first two; the third is what Jaeger Agent Omni changes.

```text
INTERCHANGE   what crosses between stages     text  |  latent
TEMPORAL      do stages overlap in time       sequential  |  concurrent
DUPLEX        can it hear while speaking      half  |  full
```

---

# The axes are independent

```text
                    text interchange          latent interchange

sequential   voice_sequential_halfduplex.py   (possible, pointless — latents
half-duplex  A0                                buy nothing without overlap)

sequential   voice_sequential_fullduplex.py   —
full-duplex  A0's brain, harness audio

concurrent   voice_concurrent_fullduplex.py   JAEGER AGENT OMNI
full-duplex  A1  ← as far as text goes        MiniCPM-o
```

Sequential + full-duplex is not a contradiction: the audio layer runs in its own
threads with AEC while cognition proceeds one stage at a time. That is exactly
what `voice_sequential_fullduplex.py` is.

---

# The ceiling on the text column

From `voice_concurrent_fullduplex.py`'s own notes — the most useful sentence in
the reference set:

> What deliberately does not [overlap]: ASR still completes before the LLM
> starts on the same turn. The installed pywhispercpp has no partial-transcript
> API (verified, not assumed), and feeding the model text that may still be
> revised has no correction path.

That is not a library gap. It is structural. **A text bridge cannot overlap
ASR with cognition**, because a transcript is only meaningful once the utterance
is over, and a revisable transcript has no correction path once tokens are
committed.

Everything else in that file overlaps beautifully — Kokoro renders phrase N
while the LLM writes N+1, the speaker plays N while Kokoro renders N+1, the mic
never closes. A1 is text-interchange concurrency taken as far as it goes.

**The one stage that cannot be made concurrent in a text pipeline is exactly
the one Omni deletes.** Latents have no such barrier: a partial audio encoding
is a valid partial audio encoding, and prefill can consume it immediately.

---

# Default: concurrent, full duplex

That is the operating mode Jaeger targets, so it is the mode the model is
**trained** for. Everything else is the organiser choosing a policy on top:

```text
"speak and ignore input while I do"   half duplex   — a policy
"wait until I have heard it all"      sequential    — a policy
"listen and think while you talk"     the default   — what training assumes
```

The weights do not change between these. The scheduler does.

# One model, four modes

Sequential/concurrent and half/full duplex are **orchestration**, not weights.
The four reference implementations share three identical models and differ only
in scheduling. Omni inherits that: one projector, one backbone, and the runtime
decides how to drive them.

Two exceptions, both worth knowing before training rather than after:

## Turn-taking is partly in the weights

```text
VAD timeout          orchestration. A stopwatch. Cannot tell
                     "...and then he said" from "...and that was that."
<|speak|>/<|listen|> MiniCPM: model property, trained into the vocabulary.
                     Needs backbone training we cannot do.
turn-end head        ours: a small separate model the organizer consults.
                     Swappable, trainable from our own turn boundaries.
```

## Mid-utterance robustness IS a training concern

Concurrent operation hands the projector audio that is **not finished** — 1.2
seconds of a sentence still being spoken. A projector trained only on complete
utterances has never seen that shape, and the mismatch degrades streaming
rather than raising an error. This is the one place where "the organiser
decides" is not sufficient.

**Prefix augmentation** (`prefix_fraction`, default 0.3) truncates a share of
training clips at a random point. Two things fall out:

```text
1. mid-utterance audio is in-distribution for the projector
2. turn-end labels come FREE — a truncated clip is "still speaking" (0),
   a complete one is "turn ended" (1) — from ANY corpus, with no
   conversational data required
```

That second point is worth more than the first. The Stage 5 turn-end head was
the piece we could not source labels for without our own recorded sessions;
prefix augmentation supplies them from LibriSpeech.

Two guards on it:

```text
prefixes are training-only   validation stays whole-utterance, so the number
                             means the same thing across runs and stays
                             comparable to the text path
prefixes are NOT supervised  a truncated clip's transcript no longer matches
                             its audio, so its labels are masked out entirely.
                             It supervises turn-end and nothing else.
```

Cuts are deterministic per index, so a resumed run does not silently change its
data distribution.

---

# Gates are policy, not mode

A wake word does not switch the system into a different mode. Everything
upstream keeps running — mic open, encoder running, cognition available. The
gate only decides whether what was heard should **act**.

And a gate needs no trained detector, because we already run an audio encoder:

```text
enroll   reference audio -> encoder -> normalised embedding, stored
listen   incoming chunk  -> encoder -> cosine similarity -> threshold
```

Hear it once, keep the vector, trigger when it comes round again.
`jaeger_agent_omni/neural/gate.py`. **One encode, two consumers** — the gate
reads the same encoder output the projector reads, so adding a trigger costs a
dot product, not a model.

Honest about what it is: good at matching a specific phrase said in a similar
way — a catchphrase, a recorded wake word, a sound cue. Enroll several takes
and it tolerates more. It is **not** speaker-independent; for
anyone-can-say-it triggers, enroll several speakers or gate on the transcript.

## What is policy and what is perception

The line worth holding, because it decides what needs training:

```text
POLICY — a prompt, no training
  "wait until I have heard it all"
  "ignore input while I speak"
  "only act after the wake phrase"
  "answer everything you hear"

PERCEPTION — a judgment about the signal
  "has this person finished their sentence?"
  "was that my name or a similar-sounding word?"
```

A human needs no training to *decide* to wait. Knowing *when* the other person
is done is learned perception — prosody, syntax, breath. The turn-end head does
the second, not the first.

That head is interim, though. Once the backbone genuinely understands the
audio it is given, "have they finished?" becomes a prompt like the rest. It is
scaffolding for a model that is not good enough yet, not a permanent component.

# Where the modes matter for us

```text
production, YouTube   concurrent + full duplex. An actor cannot wait, and
                      cannot be cut off mid-sentence.
batch content         sequential is fine. Nobody watches the render.
benchmarking          keep A0 (sequential half-duplex) as the control. It is
                      the honest baseline every latency claim is measured
                      against.
```
