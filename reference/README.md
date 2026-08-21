# reference/

Third-party sources kept locally so architecture decisions can be grounded in
what an implementation actually does, rather than in what a paper says or what
anyone remembers.

Nothing here is ours and nothing here is imported by our code. It is read.

## Our own voice LLM implementations

Named for the taxonomy in `dev/voice pipeline taxonomy.md`:

```text
voice_sequential_halfduplex.py          A0   the control. Every latency claim
                                             is measured against this.
voice_sequential_fullduplex.py          A0-brain on the full-duplex audio stack
voice_sequential_fullduplex_variant.py  —    think() then speak(), continuous
voice_concurrent_fullduplex.py          A1   7 workers on an event bus; text
                                             concurrency taken as far as it goes
```

A1 documents the ceiling of the text column, and it is the strongest argument
for Omni in the repo: ASR cannot overlap cognition, because a transcript is
only meaningful once the utterance is finished and a revisable one has no
correction path. Everything else in that file overlaps. That single remaining
barrier is what latent interchange removes.

## What is used today

```text
~/models/MiniCPM-o-4_5-gguf     the released checkpoint. Read with
                                research/inspect_gguf.py — tensor names,
                                shapes, and metadata, header only.
~/GITHUB/llama.cpp-omni         the C++ implementation. This is where the
                                forward pass, the pooling order, the streaming
                                cache and the turn-control vocabulary were
                                read from.
```

Findings are recorded in `dev/minicpm-o measured architecture.md`, with every
claim tagged MEASURED / SOURCE / API / UNKNOWN.

## What would close the remaining UNKNOWNs

The HF Python modeling code for MiniCPM-o (`modeling_minicpmo.py` and friends
from the model repo) would answer what neither the GGUF nor the C++ port can:

```text
sequence splicing   how projected audio embeddings are inserted into the token
                    stream — placeholder-token expansion is the usual
                    mechanism but is not visible in either source we have
self-hearing        how generated audio is prevented from re-entering as user
                    input
training stages     which components were frozen when. Not recoverable from a
                    released checkpoint at all; only the paper or the training
                    code would have it.
duplex data         what corpus taught <|speak|> / <|listen|>
```

Clone it here if those become blocking:

```sh
git clone https://huggingface.co/openbmb/MiniCPM-o-4_5 reference/minicpm-o-4_5
```

`reference/` is gitignored — these are other people's repos, kept locally, not
vendored into ours.
