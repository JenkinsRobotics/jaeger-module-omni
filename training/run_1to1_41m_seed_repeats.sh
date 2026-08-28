#!/usr/bin/env bash
# Independent short qualified-data updates. Each seed starts from the frozen
# production baseline; candidates never compound one another's forgetting.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ${JAEGER_GPU_WINDOW:-} != YES ]]; then
    echo "REFUSING: set JAEGER_GPU_WINDOW=YES only when the GPU is assigned" >&2
    exit 2
fi

DATA="$HOME/models/jaeger-training-data"
GEMMA="$HOME/models/gemma-4-E4B-it-4bit-mlx"
BASE="nodes/jaegeromni-41m-medium/weights/adapter.npz"
CONVERSATION_IDS="$DATA/whisper-medium-qualified-v1/minds14_validation/gold_under1.jsonl"

for seed in 20260830 20260831 20260832; do
    out="artifacts/jaegeromni-41m-1to1-seed${seed}.npz"
    label="research/candidate_evals/jaegeromni-41m-1to1-seed${seed}"
    training/run_gpu.sh training/cleanroom_context_ce.py \
        --resume "$BASE" --output "$out" --model "$GEMMA" \
        --train-source "$DATA/stock-whisper-medium-clean360-v1::clean360::1" \
        --holdout-source "$DATA/stock-whisper-medium-eval-v1::devclean::1" \
        --holdout-source "$DATA/stock-whisper-medium-clean360-v1::clean360_validation::1" \
        --steps 50 --learning-rate 2e-7 --warmup 20 \
        --text-teacher-weight 0.5 --anchor-weight 1.0 --accumulate 2 \
        --checkpoint-every 50 --probe-every 50 --log-every 10 --seed "$seed"

    training/run_gpu.sh training/stock_whisper_gemma.py wer \
        --cache "$DATA/stock-whisper-medium-eval-v1" --split devclean \
        --id-file "$DATA/stock-whisper-medium-eval-v1/devclean/manifest.jsonl" \
        -n 64 --adapter "$out" --model "$GEMMA" \
        --whisper openai/whisper-medium --output "${label}-devclean64.json"
    training/run_gpu.sh training/stock_whisper_gemma.py wer \
        --cache "$DATA/stock-whisper-medium-clean360-v1" \
        --split clean360_validation \
        --id-file "$DATA/stock-whisper-medium-clean360-v1/clean360_validation/manifest.jsonl" \
        -n 128 --adapter "$out" --model "$GEMMA" \
        --whisper openai/whisper-medium --output "${label}-clean360val128.json"
    training/run_gpu.sh training/stock_whisper_gemma.py wer \
        --cache "$DATA/stock-whisper-medium-eval-v1" \
        --split mindsvox_validation --id-file "$CONVERSATION_IDS" \
        -n 69 --adapter "$out" --model "$GEMMA" \
        --whisper openai/whisper-medium --output "${label}-conversation69.json"
    ./.venv/bin/python research/standard_wer.py \
        --input "devclean64=${label}-devclean64.json" \
        --input "clean360val128=${label}-clean360val128.json" \
        --input "conversation69=${label}-conversation69.json" \
        --output "${label}-standard-wer.json"
done
