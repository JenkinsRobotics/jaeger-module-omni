#!/usr/bin/env bash
# Identical, locked, free-running gates for one medium or large candidate.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ${JAEGER_GPU_WINDOW:-} != YES ]]; then
    echo "REFUSING: set JAEGER_GPU_WINDOW=YES only when the GPU is assigned." >&2
    exit 2
fi
if [[ $# -ne 3 ]]; then
    echo "usage: training/evaluate_candidate.sh LABEL ADAPTER {medium|large}" >&2
    exit 2
fi
RESULT_LABEL=$1
ADAPTER_PATH=$2
FRONT_END=$3
[[ "$RESULT_LABEL" =~ ^[a-zA-Z0-9._-]+$ ]] || {
    echo "unsafe result label: $RESULT_LABEL" >&2; exit 2;
}
[[ -f "$ADAPTER_PATH" ]] || { echo "missing $ADAPTER_PATH" >&2; exit 2; }

DATA_MODELS="$HOME/models"
MEDIUM_CACHE="$DATA_MODELS/jaeger-training-data/stock-whisper-medium-expanded-v1"
MEDIUM_EVAL="$DATA_MODELS/jaeger-training-data/stock-whisper-medium-eval-v1"
LARGE_CACHE="$DATA_MODELS/jaeger-training-data/stock-whisper-large-v3-turbo"
MEDIUM_CLEAN360="$DATA_MODELS/jaeger-training-data/stock-whisper-medium-clean360-v1"
LARGE_CLEAN360="$DATA_MODELS/jaeger-training-data/stock-whisper-large-clean360-v1"
MEDIUM_VOX="$DATA_MODELS/jaeger-training-data/stock-whisper-medium-voxpopuli33-v1"
LARGE_VOX="$DATA_MODELS/jaeger-training-data/stock-whisper-large-voxpopuli33-v1"
GEMMA_RUNTIME="$DATA_MODELS/gemma-4-E4B-it-4bit-mlx"
DEV_IDS="$MEDIUM_EVAL/devclean/manifest.jsonl"
MINDS_IDS="$MEDIUM_EVAL/mindsvox_validation/manifest.jsonl"
CLEAN360_IDS="$MEDIUM_CLEAN360/clean360_validation/manifest.jsonl"
VOX_IDS="$MEDIUM_VOX/voxpopuli33_validation/manifest.jsonl"

assert_words() {
    local result_path=$1
    local expected=$2
    local observed
    observed=$(jq -r '.reference_words' "$result_path")
    [[ "$observed" == "$expected" ]] || {
        echo "REFUSING: $result_path has $observed reference words; expected $expected" >&2
        exit 3
    }
}

case "$FRONT_END" in
    medium)
        WHISPER_ID=openai/whisper-medium
        # The authoritative 64-clip/955-word gate predates the expanded
        # <=12-second training cache. Keep evaluation on its dedicated cache
        # so a training-curriculum change cannot silently change the corpus.
        EVAL_CACHE="$MEDIUM_EVAL"
        MINDS_CACHE="$MEDIUM_EVAL"
        LOCKBOX_CACHE="$DATA_MODELS/jaeger-training-data/stock-whisper-medium-lockbox-v1"
        CLEAN360_CACHE="$MEDIUM_CLEAN360"
        VOX_CACHE="$MEDIUM_VOX"
        ;;
    large)
        WHISPER_ID=openai/whisper-large-v3-turbo
        EVAL_CACHE="$LARGE_CACHE"
        MINDS_CACHE="$LARGE_CACHE"
        LOCKBOX_CACHE="$DATA_MODELS/jaeger-training-data/stock-whisper-large-lockbox-v1"
        CLEAN360_CACHE="$LARGE_CLEAN360"
        VOX_CACHE="$LARGE_VOX"
        ;;
    *) echo "front end must be medium or large" >&2; exit 2 ;;
esac

mkdir -p research/candidate_evals
training/run_gpu.sh training/stock_whisper_gemma.py wer \
    --cache "$EVAL_CACHE" --split devclean --id-file "$DEV_IDS" -n 64 \
    --adapter "$ADAPTER_PATH" --model "$GEMMA_RUNTIME" \
    --whisper "$WHISPER_ID" \
    --output "research/candidate_evals/${RESULT_LABEL}-devclean64.json"
assert_words "research/candidate_evals/${RESULT_LABEL}-devclean64.json" 955
training/run_gpu.sh training/stock_whisper_gemma.py wer \
    --cache "$MINDS_CACHE" --split mindsvox_validation \
    --id-file "$MINDS_IDS" -n 591 \
    --adapter "$ADAPTER_PATH" --model "$GEMMA_RUNTIME" \
    --whisper "$WHISPER_ID" \
    --output "research/candidate_evals/${RESULT_LABEL}-mindsvox591.json"
assert_words "research/candidate_evals/${RESULT_LABEL}-mindsvox591.json" 9736
if [[ "$FRONT_END" == medium && ${JAEGER_SKIP_TRAIN_OTHER:-NO} != YES ]]; then
    training/run_gpu.sh training/stock_whisper_gemma.py wer \
        --cache "$MEDIUM_CACHE" --split train_other_validation \
        --id-file "$MEDIUM_CACHE/train_other_validation/manifest.jsonl" -n 64 \
        --adapter "$ADAPTER_PATH" --model "$GEMMA_RUNTIME" \
        --whisper "$WHISPER_ID" \
        --output "research/candidate_evals/${RESULT_LABEL}-trainother64.json"
    assert_words "research/candidate_evals/${RESULT_LABEL}-trainother64.json" 1143
    training/run_gpu.sh training/stock_whisper_gemma.py wer \
        --cache "$MEDIUM_CACHE" --split ami5_train_validation \
        --id-file "$MEDIUM_CACHE/ami5_train_validation/manifest.jsonl" -n 64 \
        --adapter "$ADAPTER_PATH" --model "$GEMMA_RUNTIME" \
        --whisper "$WHISPER_ID" \
        --output "research/candidate_evals/${RESULT_LABEL}-ami64.json"
    assert_words "research/candidate_evals/${RESULT_LABEL}-ami64.json" 673
elif [[ "$FRONT_END" == medium ]]; then
    echo "train-other gate skipped: this candidate trained on the external "\
         "train-other pool, whose split overlaps that legacy holdout"
    training/run_gpu.sh training/stock_whisper_gemma.py wer \
        --cache "$MEDIUM_CACHE" --split ami5_train_validation \
        --id-file "$MEDIUM_CACHE/ami5_train_validation/manifest.jsonl" -n 64 \
        --adapter "$ADAPTER_PATH" --model "$GEMMA_RUNTIME" \
        --whisper "$WHISPER_ID" \
        --output "research/candidate_evals/${RESULT_LABEL}-ami64.json"
    assert_words "research/candidate_evals/${RESULT_LABEL}-ami64.json" 673
fi
training/run_gpu.sh training/stock_whisper_gemma.py wer \
    --cache "$CLEAN360_CACHE" --split clean360_validation \
    --id-file "$CLEAN360_IDS" -n 128 \
    --adapter "$ADAPTER_PATH" --model "$GEMMA_RUNTIME" \
    --whisper "$WHISPER_ID" \
    --output "research/candidate_evals/${RESULT_LABEL}-clean360val128.json"
assert_words "research/candidate_evals/${RESULT_LABEL}-clean360val128.json" 2821
training/run_gpu.sh training/stock_whisper_gemma.py wer \
    --cache "$VOX_CACHE" --split voxpopuli33_validation \
    --id-file "$VOX_IDS" -n 128 \
    --adapter "$ADAPTER_PATH" --model "$GEMMA_RUNTIME" \
    --whisper "$WHISPER_ID" \
    --output "research/candidate_evals/${RESULT_LABEL}-novelvox128.json"
assert_words "research/candidate_evals/${RESULT_LABEL}-novelvox128.json" 2826
if [[ ${JAEGER_SKIP_CASES12:-NO} == YES ]]; then
    echo "cases12 skipped: broad corpus evaluation requested"
else
    training/run_gpu.sh training/stock_whisper_gemma.py bench \
        --adapter "$ADAPTER_PATH" --model "$GEMMA_RUNTIME" \
        --whisper "$WHISPER_ID" \
        --output "research/candidate_evals/${RESULT_LABEL}-cases12.json"
fi

if [[ ${JAEGER_FINAL_LOCKBOX:-} == YES && \
      -f "$LOCKBOX_CACHE/test_clean_lockbox/manifest.jsonl" ]]; then
    training/run_gpu.sh training/stock_whisper_gemma.py wer \
        --cache "$LOCKBOX_CACHE" --split test_clean_lockbox -n 2603 \
        --adapter "$ADAPTER_PATH" --model "$GEMMA_RUNTIME" \
        --whisper "$WHISPER_ID" \
        --output "research/candidate_evals/${RESULT_LABEL}-testclean2603.json"
else
    echo "lockbox protected; set JAEGER_FINAL_LOCKBOX=YES once for the selected candidate"
fi
