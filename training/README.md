# training/

Training runs for the staged build. Data policy, run configuration, and the
train loop.

## Stock Whisper → Gemma, without Qwen or Gemma-output supervision

`stock_whisper_gemma.py distill` uses the proven MiniCPM projector structure
(1024→4096, ReLU, pool-by-5, 4096→2560) but deploys the stock OpenAI
Whisper-medium encoder. Training joins stock and tuned caches by stable
utterance ID and matches the tuned path's 2560-wide vectors *before* Gemma.
It does not use a transcript, Gemma hidden state/logit, generated answer, Qwen
model, or Whisper decoder as its distillation target.

The measured 2026-08-22 run used 4,786 LibriSpeech plus 2,466 FLEURS paired
training utterances. Step 5,500 was selected by two fixed free-running corpora:
16.34% WER on LibriSpeech (955 words) and 25.52% on FLEURS (1,403 words).
Longer training improved neither joint gate. See `STATUS.md` and the dated
`research/stock_whisper_medium_*_wer_20260822.json` artifacts.

For a future answer branch, use the ground-truth serialized target
`TRANSCRIPT: <human text>\nANSWER: <ground-truth response>`. Never train on
Gemma's generated transcript. The current local QA corpus has only 15 unique
question templates and 12 overlap `research/cases12`, so training on it would
invalidate the benchmark; broaden and split by question family first.

```text
fetch_dataset.py  pull public corpora (LibriSpeech, AMI) into capture format
data.py         tiered corpus mixing policy — which tier feeds which stage
dataset.py      capture corpus -> batches (audio prefix + teacher-forced target)
make_corpus.py  synthesise a smoke-test corpus with the system TTS
config.py       run configuration, validated, with per-stage presets
distill.py      self-distillation losses (hidden-state + KL + CE)
train.py        the loop
nightly.sh      unattended overnight runs
configs/        saved run configs, one per real run
runs/           checkpoints + logs (gitignored — big, and reproducible)
```

## A measured result

150 steps of Stage 3 warmup, 60 synthetic turns, whisper-tiny +
SmolLM2-135M, MPS:

```text
                        loss            target_acc    grad norm
without embed_norm      5.39 -> 3.17    0.76 -> 0.38  47 -> 648   diverged
with embed_norm         6.07 -> 0.268   0.66 -> 0.91  24 -> 5.6   converged
```

`target_acc` is the fraction of transcript tokens predicted from audio alone,
under teacher forcing, with no text input. On 60 synthetic turns most of that
is memorisation — the *stability* difference is the real result.

The cause: with a frozen backbone nothing constrains the projector's output
scale, and it drifted to 4.4x the backbone's own embedding norm — out of the
distribution the backbone was trained on. See `neural/audio.py`, `embed_norm`.
MiniCPM never meets this because it trains the backbone jointly.

## Get a corpus

```sh
python training/fetch_dataset.py --list
python training/fetch_dataset.py --dataset librispeech_dev --out corpus/ --hours 5

# mixing beats volume from one source — the budget splits evenly
python training/fetch_dataset.py --mix librispeech,voxpopuli,ami \
    --hours 30 --out corpus/
```

Human transcripts land with `confidence=1.0` — ground truth, no gate needed.
Streaming, so turns are written as they arrive and the job can be stopped at
any point with what it has. A corpus that fails (auth, schema drift) is
reported and skipped; the others still land.

**Adapters, not special cases.** Each corpus declares how its schema maps onto
ours; the pipeline is shared. Adding one is a few lines:

```python
Corpus(key="tedlium", hf_id="LIUM/tedlium", text_field="text",
       text_transform=strip_ted_markers)
```

Two conversions decide whether a mixed corpus is usable at all:

```text
resample + downmix   everything lands 16 kHz mono float32, matching whisper
normalise text       lowercase, markers dropped, whitespace collapsed —
                     applied to EVERY corpus, not just the ragged ones
```

On lowercase: casing carries no loudness. Delivery is a `PerformanceSpec`
field, so text does not have to encode it. And LibriSpeech (UPPERCASE, no
punctuation) mixed raw with TED-LIUM (natural) teaches the model their
disagreement rather than the speech. Punctuation restoration is a separate
task with its own data.

## Run it

```sh
python training/train.py --stub                 # full loop on tiny models, ~2s
python training/train.py --stub --steps 400     # watch the schedule work
python training/train.py --selfcheck            # assertions only
```

`--stub` is not a toy. It exercises the identical loop, optimiser, scheduler,
loss and checkpoint path against tiny random models. A broken loop fails here
in seconds instead of an hour into a rented GPU.

Real runs are blocked on Stage 0 — `train.py` refuses rather than guessing a
backbone, because a projector is only valid for the pairing it was trained
against.

## Why the tiers matter

Four data tiers are not interchangeable. Mixing them in the wrong proportion is
how a projector ends up excellent on audiobooks and useless across a room.
`data.py` owns that policy and asserts it:

```text
stage3_warmup    labeled ASR 0.8, far-field 0.2      NOT our sessions —
                                                     warmup on a tiny
                                                     in-domain corpus
                                                     memorises instead of
                                                     aligning
stage4_distill   labeled 0.4, far-field 0.25, ours 0.35
stage5_turnend   ours 1.0                            nobody else's turn
                                                     boundaries are ours
stage6_paraling  emotion 0.6, ours 0.4
```

## Overnight / unattended training

Long runs should survive being interrupted, and should stop on their own before
anyone needs the machine back.

```sh
BACKBONE=<hf-id> training/nightly.sh corpus/ 6     # 6-hour budget
```

What makes it unattended-safe:

```text
--resume        continues from the checkpoint, restoring optimiser moments and
                the LR schedule — not just weights. Without that, every resume
                boundary shows as a loss spike.
--max-hours     stops cleanly at the budget instead of being killed mid-write
SIGINT/SIGTERM  finishes the current step, saves, exits 0
save_every      periodic checkpoints, so a hard power loss costs one interval
atomic writes   checkpoint is written to .tmp and renamed, so a machine that
                sleeps mid-save cannot destroy the last good checkpoint
```

Run it every night against the same `--out` and each night continues the last.

### Scheduling it

Use the OS scheduler; do not build one. macOS launchd, nightly at 01:00:

```xml
<!-- ~/Library/LaunchAgents/com.jenkinsrobotics.omni.nightly.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict>
  <key>Label</key><string>com.jenkinsrobotics.omni.nightly</string>
  <key>ProgramArguments</key><array>
    <string>/Users/YOU/GITHUB/jaeger_agent_omni/training/nightly.sh</string>
    <string>/path/to/corpus</string>
    <string>6</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>BACKBONE</key><string>&lt;hf-id&gt;</string>
  </dict>
  <key>StartCalendarInterval</key><dict>
    <key>Hour</key><integer>1</integer><key>Minute</key><integer>0</integer>
  </dict>
  <key>StandardErrorPath</key><string>/tmp/omni-nightly.err</string>
</dict></plist>
```

```sh
launchctl load ~/Library/LaunchAgents/com.jenkinsrobotics.omni.nightly.plist
```

**One caveat that will bite:** a Mac asleep at 01:00 does not run the job.
Either `caffeinate` the run, enable Power Nap, or schedule a wake in Energy
Saver. launchd will fire a missed job at next wake, which may be mid-morning.

## Hardware

Not this Mac. M1 Max is roughly an order of magnitude slower than an A100 for
this, and bf16 on MPS falls back silently. `config.py` defaults to bf16 —
override to fp16 if a run has to happen locally.

Decide the training box before Stage 3 starts, not during.

## Before a real run

```text
backbone_id      no default, deliberately. A projector is only valid for the
                 backbone it was trained against; silently picking one is a
                 trap that costs a whole run.
cognition_dim    read from the backbone's config at load time. Never hardcoded.
licences         several corpora are research-only or need an access
                 agreement. data.needs_licence_review(stage) lists them.
```

## Checks

```sh
cd training && ../.venv/bin/python data.py
cd training && ../.venv/bin/python config.py
```

## Never disable the MPS watermark

`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` removes PyTorch's allocation ceiling.
It does not create memory; it removes the error that tells you there is none.

Measured consequence: a run that should have failed fast at 42 GiB instead
allocated past physical RAM and drove swap to 41.5 GB of 43 GB. The process
sat in uninterruptible I/O at 12% CPU for twelve minutes, producing nothing,
with no error anywhere -- indistinguishable from "training is slow".

An OOM is a **measurement**. Read it and shrink the batch:

    --batch 1 --grad-accum 32     # effective batch unchanged, 8x fewer
                                  # live activations

E4B is 42 layers of 2560 with a ~350-token audio prefix. batch 8 was fine at
135M and is not fine here.
