# Reference Architecture: MiniCPM-o 4.5

We are developing a long-term neural architecture for **Jaeger OS / Jaeger Agent**.

Before designing our implementation, study **MiniCPM-o 4.5** as our primary reference architecture.

This is NOT a request to simply install MiniCPM-o or clone its repository.

The purpose is to understand **how OpenBMB combined multiple pretrained specialist neural networks into one end-to-end multimodal neural system**, and extract architectural principles that can be recreated with our own preferred components.

---

# Why MiniCPM-o is our reference

MiniCPM-o 4.5 is reported by OpenBMB as an approximately 9B-parameter end-to-end model built using:

```text
Whisper-medium
    → audio perception

SigLIP2
    → image/video perception

Qwen3-8B
    → language/cognitive backbone

CosyVoice2
    → speech generation

MiniCPM-specific components
    → adapters/projectors
    → multimodal integration
    → temporal/streaming architecture
    → training/alignment
    → full-duplex operation
```

We specifically care about the architectural idea that these are not merely separate applications passing text between one another.

They are neural subnetworks incorporated into a larger learned system.

Conceptually:

```text
                     MiniCPM-o

   AUDIO ─────► Audio Encoder ─────┐
                                   │
   IMAGE ─────► Vision Encoder ─────┤
                                   │
   VIDEO ─────► Vision Encoder ─────┤
                                   ▼
                          Multimodal adapters
                                   │
                                   ▼
                         Cognitive Backbone
                             Qwen3-8B
                                   │
                      ┌────────────┴────────────┐
                      ▼                         ▼
                    TEXT                  Speech system
                                                │
                                                ▼
                                          AUDIO OUTPUT
```

The important feature is that audio does NOT need to follow:

```text
audio
→ transcription
→ text
→ LLM
```

before the cognitive model can receive information from it.

The audio encoder can provide learned neural representations directly into the multimodal system.

Likewise, realtime speech generation is integrated into the model rather than requiring a standalone:

```text
LLM text
→ external TTS program
```

pipeline.

---

# Research questions

Study the actual MiniCPM-o architecture and answer these questions precisely.

## Audio input

Determine:

* exactly which portions of Whisper-medium are used;
* whether the Whisper decoder/ASR transcription path is required;
* what hidden-state representation comes from the audio encoder;
* its tensor dimensionality;
* temporal compression/downsampling;
* how those audio representations are projected toward the LLM;
* whether the audio encoder is frozen or trainable during different training stages;
* how streaming audio is chunked;
* how context from previous audio chunks is maintained.

We care specifically about:

```text
waveform
   ↓
audio features
   ↓
Whisper encoder
   ↓
continuous hidden representation
   ↓
projector
   ↓
LLM
```

rather than:

```text
waveform
→ Whisper transcription
→ text
→ LLM
```

---

# Multimodal projector/fusion

Study the neural components sitting between modality encoders and the language backbone.

Determine:

```text
encoder hidden size
        ↓
projection / resampling
        ↓
LLM hidden size
```

Investigate:

* projector architecture;
* number of layers;
* resampling;
* token compression;
* positional information;
* temporal information;
* modality identifiers;
* normalization;
* special modality tokens;
* how modality embeddings and ordinary text embeddings coexist.

We want to understand which parts constitute the actual **MiniCPM-specific neural interface** between pretrained models.

---

# Cognitive backbone

Study exactly how Qwen3-8B is incorporated.

Determine whether:

* stock Qwen weights are initially loaded;
* Qwen is frozen during early training;
* LoRA/adapters are used;
* portions are progressively unfrozen;
* the complete backbone is eventually jointly trained;
* hidden states used for speech generation differ from ordinary text-generation states.

The key question is:

> How does a pretrained text LLM become part of a model capable of understanding representations that did not originally exist in its training interface?

---

# Speech generation

Study the CosyVoice2-derived side carefully.

We need to understand the distinction between:

```text
LLM
→ text
→ standalone TTS
```

and MiniCPM's integrated speech architecture.

Map:

```text
Cognitive representation
        ↓
Talker / speech model
        ↓
speech/audio tokens or latent representation
        ↓
decoder / vocoder / codec
        ↓
waveform
```

Determine:

* whether text tokens remain necessary internally;
* whether speech generation consumes LLM hidden states;
* what kind of audio representation is produced;
* how voice identity is represented;
* how reference voice conditioning works;
* whether prosody/emotion/style information can bypass text;
* how speech is generated incrementally;
* what controls time-to-first-audio.

---

# Full duplex

This section is particularly important.

MiniCPM-o's behavior we want to understand is approximately:

```text
TIME ─────────────────────────────────────►

Audio Input      █████████████████████████
Video Input      █████████████████████████
Cognition           ██████████████████████
Speech Output             ████████████████

Audio Input      █████████████████████████
                     ↑
            remains active while
             model is speaking
```

Investigate how this differs from:

```text
listen
→ stop listening
→ think
→ stop
→ speak
→ stop speaking
→ listen again
```

Determine:

* how incoming and outgoing streams coexist;
* whether generation and perception share a temporal representation;
* how interruption is detected;
* what happens internally when the user interrupts;
* whether old generation is cancelled;
* whether semantic state is revised;
* whether this is model behavior or application-level audio cancellation;
* how self-generated speech is prevented from becoming new user input.

---

# What we want to learn — not copy blindly

Create a clean decomposition:

```text
OFF-THE-SHELF / PRETRAINED
├── Whisper
├── SigLIP
├── Qwen
└── CosyVoice

MINICPM-SPECIFIC
├── projectors
├── adapters
├── resampling
├── multimodal fusion
├── training strategy
├── streaming
├── full-duplex architecture
└── modality alignment
```

Determine which portions we actually need to recreate ourselves.

---

# Map MiniCPM onto Jaeger

At the end produce this mapping:

```text
MiniCPM-o                    Jaeger candidate

Whisper-medium      →        Whisper encoder / future JaegerAudio
SigLIP2             →        vision encoder / future JaegerVision
Qwen3-8B            →        Gemma cognitive backbone initially
CosyVoice2          →        expressive/clonable speech model
MiniCPM adapters    →        Jaeger Neural Adapters
Omni streaming      →        Jaeger realtime neural runtime
```

Do not assume the exact MiniCPM component is necessarily the best component for Jaeger.

We are studying the **architecture**, not committing to the exact pretrained models.

---

# Final deliverable

Produce a technical architecture report containing:

1. Component diagram.
2. Data-flow diagram.
3. Tensor/hidden-state interfaces where discoverable.
4. Which components are pretrained.
5. Which components are MiniCPM-specific.
6. Which weights are trained.
7. Training stages.
8. Audio-input architecture.
9. Speech-output architecture.
10. Full-duplex architecture.
11. How modality representations reach the LLM.
12. What would have to change to replace Qwen with Gemma.
13. What could be directly reused.
14. What would require retraining.
15. What we should reproduce first for Jaeger Agent Omni.

The final objective is to understand MiniCPM-o well enough that we could construct a similar **modular multimodal neural architecture using our own selected modules** rather than treating MiniCPM-o as a black box.
