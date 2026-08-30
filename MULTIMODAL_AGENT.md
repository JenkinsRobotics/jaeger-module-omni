# Multimodal agent boundary

Jaeger Agent Omni keeps four public capability surfaces beside one internal
runtime:

```text
jaeger_agent_omni/
  nodes/    learned tensor/audio/vision capabilities
  tools/    typed symbolic actions
  skills/   procedures that coordinate tools and nodes
  memory/   persistent agent state
  core/     cognition, streaming, policies, and integration plumbing
```

A **node** is a neural capability with its manifest, architecture code,
weights reference, preprocessing contract, and metrics in one directory. It
may transform a modality into Gemma-compatible vectors or emit learned event
state such as speaker change. A **tool** performs a symbolic action such as
reading a file. A **skill** teaches the agent how to coordinate capabilities.
Memory stores durable state; it is not a modality conversion.

## Runtime data path

```text
microphone PCM
  -> acoustic gates/AEC (authoritative deterministic path today)
  -> selected neural ear
  -> native audio embeddings
  -> Gemma session KV cache
       -> ordinary response text
       -> structured tool call
            -> tools / skills / memory
            -> trusted tool result control turn
            -> same Gemma session KV cache
  -> response text
       -> UI/log output
       -> resident Kokoro neural TTS -> audio output
```

The input invariant is strict: the live audio path does not decode an ASR
transcript and then feed it to Gemma. Speaker/scene annotations and trusted
tool results may enter between turns as explicit control data. They are not a
hidden transcription path.

The output path is not yet tensor-only. Production speech currently uses
decoded response text as Kokoro's input. This is streaming and resident, but
it cannot preserve every latent prosodic choice from Gemma. The experimental
hidden-state talker is a separate node and is not a production default until
its intelligibility, voice quality, interruption, and latency gates pass.

## Agentic integration

`python -m jaeger_agent_omni.live --agentic` adds compact tool schemas to the
system prompt before the first native audio frame. Gemma can then emit a
structured tool call from the same multimodal state it uses to answer. The
runtime executes the copied Jaeger Agent loop and appends only the trusted tool
result to the existing session.

The copied legacy `listen` and `text_to_speech` tools are denied in Omni mode:
`listen` would reintroduce transcript-as-input, while `text_to_speech` would
bypass the generation-bound duplex output router. Normal hearing and speaking
stay owned by the Omni runtime.

Agentic mode is opt-in until its physical five-minute tool-use, memory,
barge-in, and end-to-audio latency gates are measured. Tool markup is buffered
and never spoken; this safe behavior currently trades away sentence-level
speech streaming during a tool-capable response.

## Direction of travel

Future vision and richer speech nodes should enter through the same native
session boundary. Learned VAD/scene/endpoint heads may supply probabilities
and events to cognition and policy, but they replace deterministic gates only
after independent false-alarm, missed-speech, detection-delay, and compute
budgets pass. Neural capability does not remove the need for measured safety
and fallback behavior.
