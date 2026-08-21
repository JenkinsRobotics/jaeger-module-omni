# Stage-wise baseline — 2026-08-21

End-to-end duplex benchmarking of MiniCPM stayed blocked on harness bugs, so
it is measured at every boundary that IS reachable instead. All numbers below
are on the SAME 12 wavs (`research/bench_audio/`), through one shared grader
(`research/grading.py`).

---

# MiniCPM-o 4.5 — the reference, measured on a quiet machine

```text
STAGE A  pre-LLM     wav -> whisper enc -> projector -> LLM-space
                     median 0.175 s prefill   ·   0.11x realtime
STAGE B  LLM         LLM-space -> text
                     median 0.350 s decode    ·   12/12 = 100%
STAGE C  post-LLM    hidden -> talker -> token2wav -> waveform
                     first audio 1380 ms (n=11)
                     vocoder 578 ms/chunk     ·   RTF 0.675
```

`RTF 0.675` means it renders 1.0 s of speech in 0.58 s — roughly 1.5x
headroom, so it sustains continuous speech rather than falling behind.

An earlier reading of `RTF 1.23` was taken while another job held the GPU and
is withdrawn. Same for a `48514 ms` legacy median. See CONTENTION below.

---

# The native Gemma tower is transcription-shaped

`research/prosody_carry.py` measures the tower's embeddings directly, before
the LLM sees them, so it separates "the tower discarded it" from "the LLM
ignored it".

```text
same words, 4 deliveries   spread 0.0238     (rate, emotion, and speaker)
4 different sentences      spread 0.1243     known-good control
ratio                      0.192
```

Changing HOW something is said moves the representation **19% as much** as
changing WHAT is said. The narrative form of the probe agrees — four takes at
different rates, emotions and speakers all returned "The speaker sounds calm
and even. Their tone is neutral."

Consequences, and these are structural rather than training problems:

```text
voice-cloning conditioning   unavailable — speaker identity not encoded
emotion perception           unavailable — delivery not encoded
"say that angrily"           unavailable — no prosody to condition on
impersonation                unavailable — timbre not encoded
```

The tower is Google's frozen weights. We cannot add a speaker branch or a
prosody branch to it. This is the measured form of "it boxes us in".

It also reframes native's two misses (`The color of the image is not
provided`, and never naming Shakespeare): with a transcription-shaped channel
there is no richer signal underneath to fall back on when the words go wrong.

---

# Accuracy — trustworthy, since contention does not change correctness

```text
MiniCPM  12/12  100%
legacy   12/12  100%
hybrid   12/12  100%
native   10/12   83%
```

Three separate grader defects were found and fixed getting here, each of
which had made a working system look broken:

```text
"T-A-C."       needs punctuation DELETED   -> tac
"twenty-five"  needs punctuation as SPACE  -> twenty five
"24 hours"     needs digit/word equivalence -> twenty four
```

No single normalisation satisfies all three, so `grading.py` tries each form
and accepts a hit under any. All three cases are pinned in its self-check.
Before the fix MiniCPM read 9/12 and legacy read 11/12; both are 12/12.

---

# Latency — only two rows are currently trustworthy

```text
native    461 ms median   CLEAN   (single model stack)
MiniCPM   525 ms to text  CLEAN   (single integrated model)
          1380 ms to speech
legacy    ~640-880 ms     DISPUTED — best observations only
hybrid    ~62 s           DISPUTED — but the mechanism is real
```

## CONTENTION — why several numbers today were wrong

Every bad measurement in this project has had the same cause: something else
was using the machine.

```text
11.8 s legacy    Whisper running on CPU instead of MPS
48.5 s legacy    a stale job resumed and took the GPU
RTF 1.23         same stale job
17.1 s legacy    the parallel agent's benchmark_spoken_qa, 4.5 GB, 13.7% CPU
```

The machine has 32 GB and was at **17.6 GB of 18.4 GB swap** during the legacy
run. Legacy's per-question times were not a warmup ramp but erratic thrash:

```text
q00 944ms · q04 32244ms · q08 40971ms · q11 877ms
```

Fast at both ends, 35x worse in the middle. The ends (877-944 ms) agree with
the earlier quiet-machine figure of 635 ms; the middle is swap.

**Two agents share this Mac.** Latency work has to be scheduled against that,
which is why `research/sweep.sh` runs strictly one job at a time.

## The memory finding, which is NOT an artifact

```text
                       resident      does
MiniCPM (Q8)           4.9 GB        audio + vision + LLM + speech out
legacy / hybrid        9.8 GB        audio + LLM only
                                     (Gemma 7.0 GB + Whisper 2.8 GB)
```

The sequential stack needs two model stacks resident to do strictly less, and
on a 32 GB machine shared with another agent that is the difference between
running and swapping. Hybrid is the extreme case — it drives both models on
every question and lands near 62 s. Hybrid is not viable as implemented,
despite scoring 12/12.

This is a genuine argument for integration, independent of latency or
accuracy: one model with one footprint.

---

# Where this leaves the decision

```text
1. MiniCPM ties legacy on accuracy, beats it on latency, uses half the
   memory, and is the only one that speaks. It is ahead, not merely "the bar".
2. The native tower is measured transcription-shaped (0.192) and therefore
   cannot ever carry the voice work the product depends on.
3. Legacy remains fully editable and accurate, but needs two model stacks
   resident and loses prosody at the text boundary by construction.
4. The delta is still unproven -- ours scored 33% against a 135M backbone,
   which tests the backbone, not the bridge.
```

## The question to settle before spending GPU on the retrain

Whisper was trained to produce text, so **its encoder may be transcription-
shaped too**. If it is, a delta bridging whisper -> Gemma inherits the same
ceiling and the extensibility argument for it collapses.

`prosody_carry.py` already answers this shape of question; pointing it at
whisper's encoder is a small change and cheap to run. It decides whether the
next build is a bigger bridge or a second parallel branch carrying speaker
and prosody alongside the words.

Run it before the retrain, not after.
