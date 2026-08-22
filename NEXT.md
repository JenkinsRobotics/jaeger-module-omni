# Next steps — exact commands

Run in this order. Each waits for the previous; `training/run.sh` refuses to
start a second training run, but evaluation and generation are not guarded, so
do not launch them beside a training job.

## Live demo gate

The canonical demo is now `python demo.py`. Before changing the live surface:

    ./.venv/bin/python demo.py --check
    ./.venv/bin/python -m jaeger_agent_omni.omni --selfcheck
    ./.venv/bin/python research/check_all.py

Remaining live research: continuous-double-talk AEC measures only ~0.1 dB in
the room aggregate despite 24.1 dB on the controlled echo corpus, and long
opening clauses still create occasional ~1 s END->AUDIO outliers.

The VoiceLLM Playground reciprocal review is complete for the demo surface:
keep its native MiniCPM build as a read-only optional reference; retain its
device/callback-health discipline; do not import its transcript ledger or
Gemma transcription decoder into Jaeger's input path.

## 1. Stage 3 — running now

    ./training/run.sh --corpus training/corpus_mixed \
        --backbone google/gemma-4-E4B-it --whisper openai/whisper-medium.en \
        --arch minicpm_delta --stage stage3_warmup \
        --steps 400 --batch 1 --grad-accum 4 --out runs/delta_e4b

400 micro-batches / accum 4 = 100 optimizer updates. Checkpoints every 100.

## 2. Stage 3 WER — the honest stage-3 metric

    ./.venv/bin/python training/evaluate.py \
        --corpus training/corpus_mixed \
        --backbone google/gemma-4-E4B-it \
        --projector training/runs/delta_e4b_stage3_warmup.pt \
        --whisper openai/whisper-medium.en

Compares the TEXT path (whisper decoder -> transcript) against the OMNI path
(whisper encoder -> projector -> backbone) on held-out val, same clips.

**Do not score stage 3 on the 12 questions.** Stage 3 targets TRANSCRIPTS, so
it will answer "What is the capital of France?" with "What is the capital of
France?" and score ~0. That is the stage working, not failing.

## 3. Stage 4 — the stage that makes it answer

    ./.venv/bin/python training/make_responses.py \
        --corpus training/corpus_mixed --backbone google/gemma-4-E4B-it

    ./training/run.sh --corpus training/corpus_mixed \
        --backbone google/gemma-4-E4B-it --whisper openai/whisper-medium.en \
        --arch minicpm_delta --stage stage4_distill \
        --steps 400 --batch 1 --grad-accum 4 --out runs/delta_e4b_s4

Then, and only then, the 12-question benchmark is meaningful:

    ./.venv/bin/python -m jaeger_agent_omni.omni --bench research/cases12 \
        --front-end delta --delta training/runs/delta_e4b_s4_stage4_distill.pt

## 4. Speech out

    ./.venv/bin/python research/verify_talker_text_following.py

Needs Qwen3-8B hidden states as the teacher (~16 GB) — run it with no other
GPU job. Then train SpeechOutDelta (E4B 2560 -> Qwen 4096, norm-pinned).

## 5. mlx-whisper — makes "integrated" true for MEMORY

    pip install mlx-whisper       # then port the front end off torch

Today the delta front end runs torch-whisper beside MLX-Gemma: two frameworks
resident, which is the 9.8 GB two-stack cost this architecture exists to
avoid. "More information" and "faster" are earned; "integrated" is not, until
this lands.
