# Hidden-conditioned Kokoro design

This is a training contract, not a claim about the current `kokoro-tts` node.
Stock Kokoro still consumes text through its phonemizer.

## Measured injection points

Kokoro's local `forward_with_tokens` graph is:

```text
phoneme IDs -> ALBERT -> bert_encoder --------------------------+
       |                                                        |
       +-> TextEncoder -> duration alignment -> linguistic ASR  |
                                                                v
voice style[128:256] -> DurationEncoder -> duration -> F0/noise predictor
voice style[0:128] + linguistic ASR + F0/noise -> iSTFT decoder -> PCM
```

The useful control variables already exist: phoneme duration, F0, excitation
noise/energy, and two 128-wide style halves. Stock inference derives all of
them from phonemes and one fixed preset voice vector.

The packaged preset is actually a `510 x 1 x 256` style table, measured across
all local voice files. Kokoro chooses row `phoneme_count - 1`; averaging two
complete tables is already supported and produces an in-between voice. A
reference encoder for arbitrary cloning must therefore predict a valid
length-conditioned table, or predict a compact identity latent that a learned
length conditioner expands to that table. Interpolating presets can approximate
voices inside their span, but is not evidence of arbitrary cloning.

## Native Gemma connection

A real embedded Kokoro variant should accept:

```text
Gemma language token IDs
Gemma cognition hidden states [tokens, 2560]
speaker identity embedding
optional performance controls (emotion, pitch contour, tempo)
```

and learn three components:

1. a monotonic phoneme planner producing Kokoro phoneme IDs and alignment from
   Gemma tensors without decoding a string;
2. a hidden-to-prosody adapter producing aligned residuals for duration, F0,
   energy/noise, pause probability, and the prosody style half;
3. a reference-speaker adapter producing Kokoro-compatible identity style
   without mixing speaker identity with momentary emotion or cadence.

Speaker identity and performance must stay separate. Otherwise asking for an
emotion changes the apparent person, and cloning a person freezes one delivery
style into every utterance.

## Scope

This design can plausibly provide phrase-level pauses, cadence, emphasis,
pitch, energy, and emotion at Kokoro's low synthesis cost. Zero-shot cloning
requires training the reference-speaker adapter; CampPlus's current 192-wide
embedding cannot simply be inserted into Kokoro's 256-wide style tensor.

Singing is a separate acceptance track. It needs note/F0 conditioning and
singing data; a conversational-speech adapter is not evidence of singing
quality. If the required generator changes materially, singing should be a
separate neural node rather than a hidden mode in the production voice node.

## Promotion gates

- no decoded string or runtime phonemizer between Gemma and the trained node;
- corpus-level spoken-word fidelity no worse than the paired Kokoro control;
- held-out speaker similarity and speaker leakage reported;
- duration, pause, F0, energy, and emotion metrics on held-out expressive data;
- free-running audio-to-audio semantics and five-minute duplex behavior;
- first-audio p50/p95, real-time factor, and barge-in/AEC survival;
- singing reported only on a held-out singing corpus.

Until those gates pass, `kokoro-tts` remains the text-bottleneck fallback and
`gemma-minicpm-hidden-talker` remains the richer experimental voice-cloning
path.
