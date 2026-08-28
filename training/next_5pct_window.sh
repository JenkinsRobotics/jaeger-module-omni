#!/usr/bin/env bash
# Manual, sequential medium/large campaign. Nothing runs without an explicit
# GPU-window acknowledgement from the operator.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ${JAEGER_GPU_WINDOW:-} != YES ]]; then
    echo "REFUSING: set JAEGER_GPU_WINDOW=YES only when the GPU is assigned." >&2
    exit 2
fi
if [[ $# -ne 1 ]]; then
    echo "usage: JAEGER_GPU_WINDOW=YES training/next_5pct_window.sh {precompute-room-medium|precompute-room-large|precompute-vox-medium|precompute-vox-large|precompute-clean360-medium|precompute-clean360-large|precompute-lockbox-medium|precompute-lockbox-large|medium|medium-full-cleanroom-ab|large-native|large-ce-input|large-ce-context}" >&2
    exit 2
fi

DATA_MODELS="$HOME/models"
MEDIUM_CACHE="$DATA_MODELS/jaeger-training-data/stock-whisper-medium-expanded-v1"
MEDIUM_EVAL="$DATA_MODELS/jaeger-training-data/stock-whisper-medium-eval-v1"
LARGE_CACHE="$DATA_MODELS/jaeger-training-data/stock-whisper-large-v3-turbo"
MEDIUM_ROOM_CACHE="$DATA_MODELS/jaeger-training-data/stock-whisper-medium-room-aug-v1"
LARGE_ROOM_CACHE="$DATA_MODELS/jaeger-training-data/stock-whisper-large-room-aug-v1"
ROOM_CORPUS="$DATA_MODELS/jaeger-training-audio/room-aug-v1"
VOX_CORPUS="$DATA_MODELS/jaeger-training-audio/voxpopuli-en-novel-after-stock-v1"
MEDIUM_VOX_CACHE="$DATA_MODELS/jaeger-training-data/stock-whisper-medium-voxpopuli33-v1"
LARGE_VOX_CACHE="$DATA_MODELS/jaeger-training-data/stock-whisper-large-voxpopuli33-v1"
CLEAN360_SOURCE="$DATA_MODELS/jaeger-training-data/librispeech-clean360-source-v1"
MEDIUM_CLEAN360="$DATA_MODELS/jaeger-training-data/stock-whisper-medium-clean360-v1"
LARGE_CLEAN360="$DATA_MODELS/jaeger-training-data/stock-whisper-large-clean360-v1"
LOCKBOX_SOURCE="$DATA_MODELS/jaeger-training-data/librispeech-lockbox-v1"
MEDIUM_LOCKBOX="$DATA_MODELS/jaeger-training-data/stock-whisper-medium-lockbox-v1"
LARGE_LOCKBOX="$DATA_MODELS/jaeger-training-data/stock-whisper-large-lockbox-v1"
SHARED_CACHE="$HOME/GITHUB/VoiceLLM-Playground/Omni Model/gemma-backbone/cleanroom_cache50"
GEMMA_RUNTIME="$DATA_MODELS/gemma-4-E4B-it-4bit-mlx"
GEMMA_TEACHER="$DATA_MODELS/gemma-4-e4b-it-mlx-full"

MEDIUM_BASE="artifacts/stock-whisper-gemma-medium50m-expanded-blend25.npz"
MEDIUM_CE="artifacts/stock-whisper-gemma-medium50m-cekl-anchor.npz"
MEDIUM_FULL_CE="artifacts/stock-whisper-gemma-medium50m-full-cekl-mix1000.npz"
LARGE_PORT="artifacts/stock-whisper-gemma-large-v3-turbo50m-port.npz"
LARGE_NATIVE_INPUT="artifacts/stock-whisper-gemma-large-v3-turbo50m-native-input.npz"
LARGE_NATIVE_FULL="artifacts/stock-whisper-gemma-large-v3-turbo50m-native-full.npz"
LARGE_CE_INPUT="artifacts/stock-whisper-gemma-large-v3-turbo50m-ce-input.npz"
LARGE_CE_CONTEXT="artifacts/stock-whisper-gemma-large-v3-turbo50m-ce-context.npz"

cpu_preflight() {
    ./.venv/bin/python training/test_next_window.py \
        --medium "$MEDIUM_BASE" --large "$LARGE_PORT"
}

precompute_room_medium() {
    training/run_gpu.sh training/precompute_stock_encoder.py \
        --source-corpus "$ROOM_CORPUS" --output-cache "$MEDIUM_ROOM_CACHE" \
        --corpus-split room_aug --validation-percent 0 \
        --whisper openai/whisper-medium --max-seconds 12
}

precompute_room_large() {
    training/run_gpu.sh training/precompute_stock_encoder.py \
        --source-corpus "$ROOM_CORPUS" --output-cache "$LARGE_ROOM_CACHE" \
        --corpus-split room_aug --validation-percent 0 \
        --whisper openai/whisper-large-v3-turbo --max-seconds 12
}

precompute_vox_medium() {
    training/run_gpu.sh training/precompute_stock_encoder.py \
        --source-corpus "$VOX_CORPUS" --output-cache "$MEDIUM_VOX_CACHE" \
        --corpus-split voxpopuli33 --validation-percent 10 \
        --whisper openai/whisper-medium --max-seconds 12
}

precompute_vox_large() {
    training/run_gpu.sh training/precompute_stock_encoder.py \
        --source-corpus "$VOX_CORPUS" --output-cache "$LARGE_VOX_CACHE" \
        --corpus-split voxpopuli33 --validation-percent 10 \
        --whisper openai/whisper-large-v3-turbo --max-seconds 12
}

precompute_clean360_medium() {
    training/run_gpu.sh training/precompute_stock_encoder.py \
        --source-cache "$CLEAN360_SOURCE" --split clean360 \
        --split clean360_validation --output-cache "$MEDIUM_CLEAN360" \
        --whisper openai/whisper-medium --max-seconds 12
}

precompute_clean360_large() {
    training/run_gpu.sh training/precompute_stock_encoder.py \
        --source-cache "$CLEAN360_SOURCE" --split clean360 \
        --split clean360_validation --output-cache "$LARGE_CLEAN360" \
        --whisper openai/whisper-large-v3-turbo --max-seconds 12
}

precompute_lockbox_medium() {
    training/run_gpu.sh training/precompute_stock_encoder.py \
        --source-cache "$LOCKBOX_SOURCE" --split test_clean_lockbox \
        --output-cache "$MEDIUM_LOCKBOX" \
        --whisper openai/whisper-medium --max-seconds 28
}

precompute_lockbox_large() {
    training/run_gpu.sh training/precompute_stock_encoder.py \
        --source-cache "$LOCKBOX_SOURCE" --split test_clean_lockbox \
        --output-cache "$LARGE_LOCKBOX" \
        --whisper openai/whisper-large-v3-turbo --max-seconds 28
}

medium() {
    cpu_preflight
    [[ -f "$MEDIUM_ROOM_CACHE/room_aug/manifest.jsonl" ]] || {
        echo "missing medium room cache; run precompute-room-medium first" >&2
        exit 2
    }
    [[ -f "$MEDIUM_VOX_CACHE/voxpopuli33/manifest.jsonl" ]] || {
        echo "missing medium novel-Vox cache; run precompute-vox-medium first" >&2
        exit 2
    }
    [[ -f "$MEDIUM_CLEAN360/clean360/manifest.jsonl" ]] || {
        echo "missing medium clean360 cache; run precompute-clean360-medium first" >&2
        exit 2
    }
    training/run_gpu.sh training/context_ce.py \
        --train-source "$SHARED_CACHE::train100::0.15" \
        --train-source "$MEDIUM_CLEAN360::clean360::0.15" \
        --train-source "$MEDIUM_CACHE::train_other::0.15" \
        --train-source "$MEDIUM_CACHE::ami5_train::0.10" \
        --train-source "$SHARED_CACHE::fleurs_train::0.10" \
        --train-source "$SHARED_CACHE::mindsvox_train::0.10" \
        --train-source "$MEDIUM_ROOM_CACHE::room_aug::0.15" \
        --train-source "$MEDIUM_VOX_CACHE::voxpopuli33::0.10" \
        --holdout-source "$MEDIUM_CACHE::devclean::1" \
        --holdout-source "$MEDIUM_CACHE::train_other_validation::1" \
        --holdout-source "$MEDIUM_CACHE::ami5_train_validation::1" \
        --holdout-source "$MEDIUM_EVAL::mindsvox_validation::1" \
        --holdout-source "$MEDIUM_VOX_CACHE::voxpopuli33_validation::1" \
        --holdout-source "$MEDIUM_CLEAN360::clean360_validation::1" \
        --resume "$MEDIUM_BASE" --output "$MEDIUM_CE" \
        --model "$GEMMA_RUNTIME" --train-branch context \
        --steps 2500 --learning-rate 1e-5 --warmup 100 \
        --text-teacher-weight 0.5 --anchor-weight 0.05 \
        --accumulate 2 --checkpoint-every 250 --log-every 25
}

# Controlled architecture/training A/B against the clean-room 41M recipe.
# It never overwrites or promotes MEDIUM_BASE. The data, optimizer, objective,
# schedule, and checkpoint cadence match the successful clean-room round; the
# adapter architecture is the only intentional model difference.
medium_full_cleanroom_ab() {
    cpu_preflight
    [[ -f "$SHARED_CACHE/mix1000/manifest.jsonl" ]] || {
        echo "missing audited clean-room mix1000 manifest" >&2
        exit 2
    }
    ./.venv/bin/python training/audit_feature_manifest.py \
        --manifest "$SHARED_CACHE/mix1000/manifest.jsonl" \
        --finite-samples 1024
    training/run_gpu.sh training/context_ce.py \
        --train-source "$SHARED_CACHE::mix1000::1.0" \
        --holdout-source "$MEDIUM_EVAL::devclean::1" \
        --holdout-source "$MEDIUM_CACHE::ami5_train_validation::1" \
        --holdout-source "$MEDIUM_EVAL::mindsvox_validation::1" \
        --holdout-source "$MEDIUM_VOX_CACHE::voxpopuli33_validation::1" \
        --holdout-source "$MEDIUM_CLEAN360::clean360_validation::1" \
        --resume "$MEDIUM_BASE" --output "$MEDIUM_FULL_CE" \
        --model "$GEMMA_RUNTIME" --train-branch all \
        --steps 2000 --learning-rate 2e-5 --warmup 200 \
        --schedule constant --text-teacher-weight 0.5 \
        --anchor-weight 0 --speaker-weight 0 --accumulate 1 \
        --allow-acoustic-variants \
        --checkpoint-every 1000 --log-every 50
}

large_native() {
    cpu_preflight
    [[ -f "$LARGE_ROOM_CACHE/room_aug/manifest.jsonl" ]] || {
        echo "missing large room cache; run precompute-room-large first" >&2
        exit 2
    }
    [[ -f "$LARGE_VOX_CACHE/voxpopuli33/manifest.jsonl" ]] || {
        echo "missing large novel-Vox cache; run precompute-vox-large first" >&2
        exit 2
    }
    [[ -f "$LARGE_CLEAN360/clean360/manifest.jsonl" ]] || {
        echo "missing large clean360 cache; run precompute-clean360-large first" >&2
        exit 2
    }
    training/run_gpu.sh training/context_distill.py \
        --cache "$LARGE_CACHE" --split train100 \
        --extra-split fleurs_train --extra-split mindsvox_train \
        --extra-data "$LARGE_ROOM_CACHE" room_aug \
        --extra-data "$LARGE_VOX_CACHE" voxpopuli33 \
        --extra-data "$LARGE_CLEAN360" clean360 \
        --model "$GEMMA_TEACHER" --resume "$LARGE_PORT" \
        --output "$LARGE_NATIVE_INPUT" --train-branch input \
        --lexical-head-arch bigru --speaker-weight 0 \
        --steps 2000 --learning-rate 5e-5 --warmup 150 \
        --balance-sources --checkpoint-every 250 --probe-every 250
    training/run_gpu.sh training/context_distill.py \
        --cache "$LARGE_CACHE" --split train100 \
        --extra-split fleurs_train --extra-split mindsvox_train \
        --extra-data "$LARGE_ROOM_CACHE" room_aug \
        --extra-data "$LARGE_VOX_CACHE" voxpopuli33 \
        --extra-data "$LARGE_CLEAN360" clean360 \
        --model "$GEMMA_TEACHER" --resume "$LARGE_NATIVE_INPUT" \
        --output "$LARGE_NATIVE_FULL" --train-branch input_full \
        --lexical-head-arch bigru --speaker-weight 0 \
        --steps 1000 --learning-rate 1e-5 --warmup 50 \
        --balance-sources --checkpoint-every 250 --probe-every 250
}

large_ce_input() {
    cpu_preflight
    [[ -f "$LARGE_NATIVE_FULL" ]] || {
        echo "missing $LARGE_NATIVE_FULL; finish large-native first" >&2
        exit 2
    }
    training/run_gpu.sh training/context_ce.py \
        --train-source "$LARGE_CACHE::train100::0.20" \
        --train-source "$LARGE_CLEAN360::clean360::0.20" \
        --train-source "$LARGE_CACHE::fleurs_train::0.10" \
        --train-source "$LARGE_CACHE::mindsvox_train::0.10" \
        --train-source "$LARGE_ROOM_CACHE::room_aug::0.20" \
        --train-source "$LARGE_VOX_CACHE::voxpopuli33::0.20" \
        --holdout-source "$LARGE_CACHE::devclean::1" \
        --holdout-source "$LARGE_CACHE::mindsvox_validation::1" \
        --holdout-source "$LARGE_VOX_CACHE::voxpopuli33_validation::1" \
        --holdout-source "$LARGE_CLEAN360::clean360_validation::1" \
        --resume "$LARGE_NATIVE_FULL" --output "$LARGE_CE_INPUT" \
        --model "$GEMMA_RUNTIME" --train-branch input_full \
        --steps 1500 --learning-rate 1e-5 --warmup 75 \
        --text-teacher-weight 0.5 --anchor-weight 0.05 \
        --accumulate 2 --checkpoint-every 250 --log-every 25
}

large_ce_context() {
    cpu_preflight
    [[ -f "$LARGE_CE_INPUT" ]] || {
        echo "missing $LARGE_CE_INPUT; finish large-ce-input first" >&2
        exit 2
    }
    training/run_gpu.sh training/context_ce.py \
        --train-source "$LARGE_CACHE::train100::0.20" \
        --train-source "$LARGE_CLEAN360::clean360::0.20" \
        --train-source "$LARGE_CACHE::fleurs_train::0.10" \
        --train-source "$LARGE_CACHE::mindsvox_train::0.10" \
        --train-source "$LARGE_ROOM_CACHE::room_aug::0.20" \
        --train-source "$LARGE_VOX_CACHE::voxpopuli33::0.20" \
        --holdout-source "$LARGE_CACHE::devclean::1" \
        --holdout-source "$LARGE_CACHE::mindsvox_validation::1" \
        --holdout-source "$LARGE_VOX_CACHE::voxpopuli33_validation::1" \
        --holdout-source "$LARGE_CLEAN360::clean360_validation::1" \
        --resume "$LARGE_CE_INPUT" --output "$LARGE_CE_CONTEXT" \
        --model "$GEMMA_RUNTIME" --train-branch context \
        --steps 1500 --learning-rate 1e-5 --warmup 75 \
        --text-teacher-weight 0.5 --anchor-weight 0.05 \
        --accumulate 2 --checkpoint-every 250 --log-every 25
}

case "$1" in
    precompute-room-medium) precompute_room_medium ;;
    precompute-room-large) precompute_room_large ;;
    precompute-vox-medium) precompute_vox_medium ;;
    precompute-vox-large) precompute_vox_large ;;
    precompute-clean360-medium) precompute_clean360_medium ;;
    precompute-clean360-large) precompute_clean360_large ;;
    precompute-lockbox-medium) precompute_lockbox_medium ;;
    precompute-lockbox-large) precompute_lockbox_large ;;
    medium) medium ;;
    medium-full-cleanroom-ab) medium_full_cleanroom_ab ;;
    large-native) large_native ;;
    large-ce-input) large_ce_input ;;
    large-ce-context) large_ce_context ;;
    *) echo "unknown stage: $1" >&2; exit 2 ;;
esac
