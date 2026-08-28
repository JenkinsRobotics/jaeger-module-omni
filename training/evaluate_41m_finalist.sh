#!/usr/bin/env bash
# Broad locked finalist gates for a 41M Whisper-medium adapter. The small
# cases12 suite is intentionally excluded from checkpoint selection.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ${JAEGER_GPU_WINDOW:-} != YES || $# -ne 2 ]]; then
    echo "usage: JAEGER_GPU_WINDOW=YES $0 LABEL ADAPTER" >&2
    exit 2
fi
LABEL=$1
ADAPTER=$2
[[ "$LABEL" =~ ^[a-zA-Z0-9._-]+$ && -f "$ADAPTER" ]] || exit 2

DATA="$HOME/models/jaeger-training-data"
EVAL="$DATA/stock-whisper-medium-eval-v1"
CLEAN="$DATA/stock-whisper-medium-clean360-v1"
VOX="$DATA/stock-whisper-medium-voxpopuli33-v1"
MODEL="$HOME/models/gemma-4-E4B-it-4bit-mlx"
RESULTS="research/candidate_evals"
mkdir -p "$RESULTS"

run_gate() {
    local split=$1 cache=$2 ids=$3 count=$4 words=$5 suffix=$6
    local output="$RESULTS/${LABEL}-${suffix}.json"
    training/run_gpu.sh training/stock_whisper_gemma.py wer \
        --cache "$cache" --split "$split" --id-file "$ids" -n "$count" \
        --adapter "$ADAPTER" --model "$MODEL" \
        --whisper openai/whisper-medium --output "$output"
    [[ $(jq -r '.reference_words' "$output") == "$words" ]] || {
        echo "REFUSING: $output reference-word count changed" >&2
        exit 3
    }
}

run_gate mindsvox_validation "$EVAL" \
    "$EVAL/mindsvox_validation/manifest.jsonl" 591 9736 mindsvox591
run_gate clean360_validation "$CLEAN" \
    "$CLEAN/clean360_validation/manifest.jsonl" 128 2821 clean360val128
run_gate voxpopuli33_validation "$VOX" \
    "$VOX/voxpopuli33_validation/manifest.jsonl" 128 2826 novelvox128

for suffix in mindsvox591 clean360val128 novelvox128; do
    jq -r '"'"$suffix"' WER=" + ((.wer * 10000 | round) / 100 | tostring) + "%"' \
        "$RESULTS/${LABEL}-${suffix}.json"
done
