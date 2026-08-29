#!/usr/bin/env bash
# Controlled medium-encoder A/B: GemmaOmni recipe applied to JaegerOmni 50M.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ${JAEGER_GPU_WINDOW:-} != YES ]]; then
    echo "REFUSING: set JAEGER_GPU_WINDOW=YES only for an assigned GPU window" >&2
    exit 2
fi

DATA="$HOME/models/jaeger-training-data"
SUPPLEMENT="$DATA/stock-whisper-medium-supplemental-v1"
LARGE_BASE="$DATA/stock-whisper-large-v3-turbo"
MEDIUM_CLEAN="$DATA/stock-whisper-medium-clean360-v1"
MEDIUM_EXPANDED="$DATA/stock-whisper-medium-expanded-v1"
MEDIUM_ROOM="$DATA/stock-whisper-medium-room-aug-v1"
MEDIUM_VOX="$DATA/stock-whisper-medium-voxpopuli33-v1"
BASE="jaeger_agent_omni/nodes/stock-whisper-medium-cekl-blend50/weights/adapter.npz"
OUTPUT="artifacts/jaegeromni-medium50m-balanced-all-step1000.npz"

echo "phase 1/3: restore missing medium feature caches from common audio"
training/run_gpu.sh training/precompute_stock_encoder.py \
    --source-cache "$LARGE_BASE" --output-cache "$SUPPLEMENT" \
    --split mindsvox_train --split fleurs_train \
    --whisper openai/whisper-medium --device mps --max-seconds 12 \
    --log-every 100

echo "phase 2/3: exact-objective all-parameter JaegerOmni 50M continuation"
training/run_gpu.sh training/context_ce.py \
    --resume "$BASE" --output "$OUTPUT" \
    --train-source "$MEDIUM_CLEAN::clean360::0.20" \
    --train-source "$MEDIUM_EXPANDED::ami5_train::0.15" \
    --train-source "$SUPPLEMENT::mindsvox_train::0.15" \
    --train-source "$MEDIUM_ROOM::room_aug::0.20" \
    --train-source "$MEDIUM_VOX::voxpopuli33::0.20" \
    --train-source "$SUPPLEMENT::fleurs_train::0.10" \
    --holdout-source "$MEDIUM_EXPANDED::devclean::1" \
    --holdout-source "$MEDIUM_EXPANDED::ami_train_validation::1" \
    --holdout-source "$MEDIUM_CLEAN::clean360_validation::1" \
    --holdout-source "$MEDIUM_VOX::voxpopuli33_validation::1" \
    --allow-acoustic-variants --steps 1000 --learning-rate 5e-6 \
    --warmup 100 --accumulate 1 --train-branch all --speaker-weight 0 \
    --checkpoint-every 250 --probe-every 250 --log-every 25

echo "phase 3/3: identical free-running promotion gates"
JAEGER_SKIP_TRAIN_OTHER=YES training/evaluate_candidate.sh \
    jaegeromni-medium50m-balanced-all-20260826 "$OUTPUT" medium
echo "medium architecture A/B complete"
