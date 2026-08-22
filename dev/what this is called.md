# What we are building, in the field's own words

Useful because the names carry the failure modes with them, and because
several groups have already solved parts of this.

---

# The architecture

**LLaVA-style modality adapter.** The template is

```text
frozen encoder  ->  TRAINABLE PROJECTOR  ->  frozen LLM
```

named for LLaVA, which did it for vision. Every speech version copies the
shape. The middle piece has three interchangeable names:

```text
projector    most common in papers
adapter      what we have been calling it
connector    Qwen-Audio's term
```

Ours is `Whisper encoder -> linear2 (ours) -> Gemma E4B`, so: a **linear
projector**, the simplest member of the family, and the one SLAM-ASR showed
is enough.

# The category

**SpeechLLM** (also *audio-language model*, *omni model*). Siblings:
Ultravox, Qwen2-Audio, Qwen2.5-Omni, SALMONN, MiniCPM-o, Freeze-Omni.

# The distinction the whole project turns on

```text
CASCADED     audio -> ASR -> TEXT -> LLM -> TTS -> audio      legacy
END-TO-END   audio -> encoder -> projector -> LLM             this
```

Also **pipelined vs direct**, or in fusion terms **late/symbolic fusion**
(modalities meet as text) vs **early/latent fusion** (they meet as
embeddings).

"Get rid of the text transcription layer between modules" is precisely the
move from cascaded to end-to-end. The reason it matters, measured here:

```text
delivery carried by a transcript          0.000
delivery retained by whisper's encoder    0.323
```

Text is a bottleneck no amount of downstream work reopens.

# What the projector emits: SOFT TOKENS

Also *soft prompts* or *continuous prompts*. They occupy token positions but
are not drawn from the vocabulary.

**This one term explains a tool I built and had to throw away.** I wrote
`research/adapter_readout.py` on the assumption that a working projector
emits vectors near real word embeddings, so you could read them back as
words. Soft tokens carry no such requirement — they only have to be
something 42 transformer layers can use. Proof: the parallel agent's
adapter, at a measured 8.13% WER, reads out as

```text
truth    HE WAS IN A FEVERED STATE OF MIND OWING TO THE BLIGHT
readout  ) blazer tennis Nicole WERE Nicole ... endeavoured ... STATE SPO
```

A working adapter, unreadable by that test. The objective I built on top of
it (`training/align_direct.py`, CTC toward embedding rows) was optimising
token-similarity rather than usefulness, which is why it produced cosine
+0.04 while the parallel agent's cross-entropy-through-the-backbone produced
8.13% WER.

---

# Prior art that matters to us

## SLAM-ASR — the closest match

`HuBERT X-Large -> LINEAR PROJECTOR -> frozen Vicuna-7B`, only the projector
trained. Reported 17.4% / 26.9% relative WER reduction on test-clean/other
against comparable systems trained on the same 960 h.

Two numbers we should take seriously:

```text
their data      960 h LibriSpeech
our cache       5366 clips, roughly 12 h
```

Their recipe: 3 epochs with early stopping on dev CE, AdamW, **lr 1e-4**,
batch 4. We are at lr 2e-5 and batch 1 — worth an A/B once the current run
has a baseline.

The follow-up paper (*"The Good, the Bad, the Ugly, and the Way Forward"*)
is specifically about where this design breaks, and should be read before we
scale.

## Freeze-Omni — the full-duplex target

Speech-to-speech on a **frozen** text LLM, keeping the backbone's original
intelligence and avoiding the forgetting that fine-tuning induces. Streaming
in, streaming out. That is the architecture we want for the robot, and it
confirms the frozen-backbone choice is not a compromise.

## Qwen2.5-Omni

A learned projection between a **streaming-modified** Whisper encoder and the
LLM. Relevant because our streaming path needs the same modification — vanilla
Whisper is not causal.

## The standard two-stage recipe

```text
stage 1   freeze encoder AND LLM, train ONLY the projector
stage 2   freeze the projector, adapt the LLM with LoRA
```

We are in stage 1 and have never attempted stage 2. If projector-only
plateaus above the WER we need, LoRA on the backbone is the documented next
move — but it costs the frozen-LLM property Freeze-Omni argues for, so it is
a real trade, not a free upgrade.

---

# Where our numbers sit

```text
parallel agent, step 1000     13.24% WER on JFK   (1032 clips)
ours, step 1500               20.59% WER on JFK   (5366 clips)
SLAM-ASR                      competitive ASR      960 h
```

We have roughly 1.2% of SLAM-ASR's data. That is the dominant term, and it is
the cheapest one to fix.

Sources: arxiv 2402.08846 (SLAM-ASR), 2411.03866 (SLAM-ASR evaluation),
2411.00774 (Freeze-Omni), github.com/X-LANCE/SLAM-LLM
