<h1 align="center">Jaeger Agent Omni</h1>

<p align="center">
  <em>Native multimodal neural agent for JaegerOS — audio straight to cognition, no text bridge</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-research-F0A020?style=for-the-badge" alt="Research">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2EA44F?style=for-the-badge" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11.9-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11.9">
</p>

---

> **Research repo.** Nothing here ships yet. `jaeger_agent_omni/node.py` is a
> stub; `module.yaml` declares the contract the staged build is working toward.
> See [`dev/docs/reality/STATUS.md`](dev/docs/reality/STATUS.md) for what
> actually runs today.

## What it is

The current Jaeger voice stack is sequential, and text is the transfer format
between every stage:

```text
mic → whisper → TEXT → llm → TEXT → tts → speaker
```

That works, ships today, and has a ceiling. Every conversion costs latency, and
everything not expressible as words — prosody, hesitation, emotion, who is
speaking — is discarded at the first arrow and cannot be recovered at the last.

Omni removes text as the mandatory bridge:

```text
mic → whisper encoder → JaegerAudioProjector → gemma-4-E4B
                                                    │
                                             PerformanceSpec
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                                  text        cloned TTS         avatar
                              tools / logs       audio             rig
```

Audio reaches cognition as learned representations. Text remains available as a
parallel output — it is no longer the only road.

## Parallel to jaeger-agent, not a replacement

```text
jaeger-agent          sequential pipeline, text as the transfer format
                      ships today, keeps shipping
jaeger-agent-omni     neural module integration, tensors as the transfer format
                      research, proves its own path
```

Nothing here patches, imports from, or depends on `jaeger-agent`. Omni runs its
own `whisper → text → gemma` teacher to generate training targets. The two
converge later at the **slot** level, when Omni ships a module a project can
select instead of the sequential stack.

## Layout

```text
jaeger_agent_omni/   the runtime
  neural/              modality projectors and heads
    base.py              ModalityProjector — the shared contract
    audio.py             JaegerAudioProjector (MiniCPM port)
    heads.py             TurnEndHead — our <|speak|>/<|listen|>
  capture.py           turn log format + transcript-confidence gate
  streaming.py         1 s chunking + rolling encoder cache
  performance_spec.py  voice + face intent
  outputs.py           text / speech / tool-call routing
  module.yaml          the slot contract
research/            experiment harness — stubs, wiring, GGUF inspection
training/            corpus policy, losses, run config, the training loop
dev/                 the plan, the long-term architecture, measured findings
```

**Where things live:** `jaeger_agent_omni/` is what runs — contracts and neural
modules. `research/` is the harness that proves them (stubs, end-to-end wiring,
inspection tools) and is not shipped. `training/` is everything that only
matters while learning weights.

## Neural modules

Every specialist reaches cognition the same way, and that is the point:

```text
audio    → whisper encoder → JaegerAudioProjector  ─┐
vision   → SigLIP          → JaegerVisionProjector ─┼→ shared cognition
memory   → retrieval net   → JaegerMemoryProjector ─┘
```

`neural/base.py` owns the arithmetic every modality needs — trim the encoder's
padding, compress time, project into cognition's width — so adding a modality
is a known job rather than a research question. **A neural tool is just another
projector.**

The rule for what becomes neural: text stays where the answer *is* a symbol
(`get_time`, `file_read`, `calculate`), and latents take over where text throws
the answer away (`look_at`, `listen`, episodic recall, motion).

## Run it

```sh
# voice in, cognition out — the text path works today
python -m jaeger_agent_omni --audio clip.wav --backbone <hf-id>

# every turn becomes training data for the path meant to replace it
python -m jaeger_agent_omni --audio clip.wav --backbone <hf-id> \
    --capture-to corpus/

# no transcript anywhere; nonsense until the projector is trained
python -m jaeger_agent_omni --audio clip.wav --backbone <hf-id> --path omni \
    --projector training/runs/run_stage3_warmup.pt
```

Verified end to end on real models (whisper-tiny + SmolLM2-135M, CPU, 1.3 s of
speech):

```text
text   transcribe 390ms + generate 990ms   →  a correct answer about 2+2
omni   encode 181ms + project 3ms          →  no transcript computed at all
```

Whisper's autoregressive decode (508 ms) becomes encode-only (181 ms). The rest
is generation — which is why Stage 5's incremental prefill, not the projector,
is where the conversational win lands.

No audio handy? `say --data-format=LEI16@16000 -o clip.wav "what is two plus two"`

## Train it

```sh
# turn any long audio into a corpus (segmented, transcribed, confidence-scored)
python training/import_audio.py --audio speech.wav --out corpus/

# Stage 3 warmup — audio -> transcript, projector only, backbone frozen
python training/train.py --corpus corpus/ --backbone <hf-id>

# unattended: resumes, stops on a budget, survives SIGTERM
BACKBONE=<hf-id> training/nightly.sh corpus/ 6

# how much does the projector lose against a transcript?
python training/evaluate.py --corpus corpus/ --backbone <hf-id> \
    --projector training/runs/run_stage3_warmup.pt
```

Evaluation is on a **held-out split** — deterministic, hashed by turn id, so
tonight's validation set is not tomorrow's training set. On a small corpus that
distinction is everything: the same projector scored 0.91 on data it had seen
and 0.35 on data it had not.

## Model artifacts

What we train ships in the standard adapter layout — safetensors weights, a
config naming the exact pairing, and a model card:

```text
artifacts/<name>/
    adapter_config.json         architecture + pairing + provenance
    adapter_model.safetensors   only the trained weights
    README.md                   model card
```

```sh
python -m jaeger_agent_omni.artifacts list
python -m jaeger_agent_omni.artifacts export training/runs/<ckpt>.pt artifacts/<name>
python demo.py --backbone <id> --path omni --projector artifacts/<name>
```

A training checkpoint (optimizer state + frozen passengers, 112 MB) is not a
release artifact (the 2.4M-param adapter alone, 9 MB). Frozen third-party
weights are referenced by provenance, never redistributed — MiniCPM's front end
reloads from the local GGUF at load time.

## Run the checks

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python research/check_all.py
```

Eighteen modules, ~15 seconds, fully offline, no model weights required. That
includes a real 120-step training run against stub models — a broken loop fails
here in seconds rather than an hour into a rented GPU:

```sh
python training/train.py --stub --steps 400
```

## Built on measurement, not recollection

The architecture is ported from MiniCPM-o 4.5 — read out of its weights and its
`llama.cpp` implementation, not from papers. `research/inspect_gguf.py` reads
GGUF headers without loading weights;
[`dev/minicpm-o measured architecture.md`](dev/minicpm-o%20measured%20architecture.md)
records what was found and tags every claim MEASURED / SOURCE / API / UNKNOWN.

Things that came out of that and would have been guessed wrong:

```text
projector      Linear → ReLU → Linear → AvgPool(k=5, s=5)
               ReLU not GELU; no LayerNorm; pool AFTER the linears
no resampler   no perceiver, no learned queries. Selection is the LLM's
               attention's job, not the projector's.
streaming      1 s chunks through a rolling 1500-frame encoder cache —
               Whisper's 30 s window becomes cache depth, not padding
turn-taking    <|speak|> / <|listen|> are vocabulary tokens. Turn-taking is
               learned, not a silence timeout.
```

## Documents

| Document | What it is |
|---|---|
| [Near-term plan](dev/jaeger%20agent%20omni0%20immediate%20development.md) | The staged build, tracks, and what we are explicitly not doing |
| [Long-term architecture](dev/jaeger%20long%20term%20neural%20architecture.md) | Where this is going; governs, unchanged |
| [MiniCPM-o measured](dev/minicpm-o%20measured%20architecture.md) | Reference architecture, from the artifact |
| [Reference questions](dev/reference%20architecture%20minicpm-o%204.md) | The original question set the above answers |
| [Legacy pipeline measured](dev/legacy%20pipeline%20measured.md) | What our own working stack costs, and what Omni must not rebuild |

## Ecosystem links

- [JaegerOS](https://github.com/JenkinsRobotics/JaegerOS) — framework layer
- [JaegerWhisperSTT](https://github.com/JenkinsRobotics/JaegerWhisperSTT) — the `stt` slot; its `OfflineTranscriber` batch-transcribes our corpus
- [`CONVENTIONS.md`](CONVENTIONS.md) · [`TAXONOMY.md`](TAXONOMY.md)
