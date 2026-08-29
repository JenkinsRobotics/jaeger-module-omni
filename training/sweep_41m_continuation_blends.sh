#!/usr/bin/env bash
# Recover any meeting-domain gain from the latest 41M continuation without
# promoting its broad clean-speech regressions.  These are selection gates;
# any winner still requires the complete free-running evaluation matrix.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ${JAEGER_GPU_WINDOW:-} != YES ]]; then
    echo "REFUSING: set JAEGER_GPU_WINDOW=YES only when the GPU is assigned" >&2
    exit 2
fi

DATA="$HOME/models/jaeger-training-data"
BASE="jaeger_agent_omni/nodes/jaegeromni-41m-medium/weights/adapter.npz"
CANDIDATE="artifacts/jaegeromni-41m-medium-balanced-anchor20-v2.npz"
EVAL="$DATA/stock-whisper-medium-eval-v1"
EXPANDED="$DATA/stock-whisper-medium-expanded-v1"
MODEL="$HOME/models/gemma-4-E4B-it-4bit-mlx"
OUT="artifacts/41m-continuation-blend-sweep"
RESULTS="research/candidate_evals"
mkdir -p "$OUT" "$RESULTS"

if (( $# )); then
    ALPHAS=("$@")
else
    ALPHAS=(2 5 10 15)
fi

for alpha in "${ALPHAS[@]}"; do
    [[ "$alpha" =~ ^[0-9]+$ ]] && (( alpha > 0 && alpha < 100 )) || {
        echo "alpha must be an integer percentage between 1 and 99" >&2
        exit 2
    }
    weight=$(./.venv/bin/python -c "print(${alpha}/100)")
    adapter="$OUT/jaegeromni41m-v2-alpha${alpha}.npz"
    label="jaegeromni41m-v2-alpha${alpha}-20260828"
    ./.venv/bin/python training/interpolate_adapter.py \
        --base "$BASE" --candidate "$CANDIDATE" --alpha "$weight" \
        --output "$adapter"

    training/run_gpu.sh training/stock_whisper_gemma.py wer \
        --cache "$EVAL" --split devclean \
        --id-file "$EVAL/devclean/manifest.jsonl" -n 64 \
        --adapter "$adapter" --model "$MODEL" \
        --whisper openai/whisper-medium \
        --output "$RESULTS/${label}-devclean64.json"
    [[ $(jq -r '.reference_words' \
        "$RESULTS/${label}-devclean64.json") == 955 ]]

    training/run_gpu.sh training/stock_whisper_gemma.py wer \
        --cache "$EXPANDED" --split ami5_train_validation \
        --id-file "$EXPANDED/ami5_train_validation/manifest.jsonl" -n 64 \
        --adapter "$adapter" --model "$MODEL" \
        --whisper openai/whisper-medium \
        --output "$RESULTS/${label}-ami64.json"
    [[ $(jq -r '.reference_words' \
        "$RESULTS/${label}-ami64.json") == 673 ]]
    clean=$(jq -r '.wer * 100' "$RESULTS/${label}-devclean64.json")
    ami=$(jq -r '.wer * 100' "$RESULTS/${label}-ami64.json")
    printf 'alpha=%02d%% clean=%.2f%% AMI=%.2f%%\n' \
        "$alpha" "$clean" "$ami"
done
