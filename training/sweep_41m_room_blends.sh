#!/usr/bin/env bash
# Medium-first A/B: interpolate the exact frozen 41M clean checkpoint toward
# its room/multidomain continuation, then run only the two locked selection
# gates.  The winning alpha must still pass the full free-running matrix.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ${JAEGER_GPU_WINDOW:-} != YES ]]; then
    echo "REFUSING: set JAEGER_GPU_WINDOW=YES only when the GPU is assigned" >&2
    exit 2
fi

DATA="$HOME/models/jaeger-training-data"
BASE="nodes/stock-whisper-medium-cleanroom-ctx93h-step2000/weights/adapter.npz"
ROOM="nodes/jaegeromni-41m-medium/weights/adapter.npz"
EVAL="$DATA/stock-whisper-medium-eval-v1"
EXPANDED="$DATA/stock-whisper-medium-expanded-v1"
MODEL="$HOME/models/gemma-4-E4B-it-4bit-mlx"
OUT="artifacts/41m-room-blend-sweep"
RESULTS="research/candidate_evals"
mkdir -p "$OUT" "$RESULTS"
if (( $# )); then
    ALPHAS=("$@")
else
    ALPHAS=(20 40 60 80)
fi

for alpha in "${ALPHAS[@]}"; do
    [[ "$alpha" =~ ^[0-9]+$ ]] && (( alpha > 0 && alpha < 100 )) || {
        echo "alpha must be an integer percentage between 1 and 99" >&2
        exit 2
    }
    weight=$(./.venv/bin/python -c "print(${alpha}/100)")
    adapter="$OUT/cleanroom41m-room-alpha${alpha}.npz"
    label="cleanroom41m-room-alpha${alpha}-20260827"
    ./.venv/bin/python training/interpolate_adapter.py \
        --base "$BASE" --candidate "$ROOM" --alpha "$weight" \
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
