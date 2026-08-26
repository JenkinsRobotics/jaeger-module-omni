# Whisper → Gemma adapter contract

The adapter exists to preserve the useful information in frozen Whisper
encoder states while expressing it in the exact representation frozen Gemma
already understands. It is not an STT decoder and must never inject a
transcript into the runtime input path.

## Pairing

An encoder and adapter are one versioned node. Changing Whisper model, encoder
revision, frame boundary, pooling rate, or Gemma boundary requires a separately
trained and evaluated adapter. Runtime validation rejects mismatched hashes,
dimensions, architecture, or pooling.

## Operational meaning of “no data loss”

Mathematical invertibility is not the product requirement. Preservation is
measured on the same held-out waveforms across complementary outputs:

1. stock Whisper decoder transcript — lexical source control;
2. adapter → Gemma verbatim transcript — information transferred to cognition;
3. adapter → Gemma direct answer — semantic use, free-running;
4. speaker retrieval/change — identity and continuity;
5. prosody, overlap, turn-end, double-talk and environment probes;
6. room/noise corpora and five-minute duplex behavior.

The lexical promotion target is an adapter-to-Gemma pooled WER within one
percentage point of the paired stock decoder on identical IDs, with no answer,
speaker, event, room, or duplex regression. Adapter-only steady-state latency
must remain below 50 ms p95; full END→AUDIO latency is reported separately.

Training loss, teacher forcing, cosine, one clip, or decoder-only WER cannot
promote a node. The three-layer free-running probe is mandatory after every
training stage. Native Gemma audio vectors are a manifold anchor, human
transcripts are lexical supervision, and speaker/event heads are independent
auxiliary signals. None becomes a text handoff at runtime.

## Current evidence

On the identical 67-word cases12 control:

| paired node | stock decoder | adapter → Gemma transcript | answers |
|---|---:|---:|---:|
| Whisper medium | 0 errors | 25 errors | 11/12 |
| Whisper large-v3-turbo | 0 errors | 63 errors | 1/12 |

The strict medium transcript count includes one formatting/reasoning failure;
excluding that failed case, it is 7 errors over 62 words. Neither adapter is
near the loss target yet, and the large node remains rejected.
