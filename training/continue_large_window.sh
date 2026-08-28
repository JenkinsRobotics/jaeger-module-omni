#!/usr/bin/env bash
# Finish the assigned GPU window sequentially after the active medium eval.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ${JAEGER_GPU_WINDOW:-} != YES ]]; then
    echo "REFUSING: set JAEGER_GPU_WINDOW=YES only for an assigned GPU window" >&2
    exit 2
fi

WAIT_PID=${1:-}
if [[ -n "$WAIT_PID" ]]; then
    echo "waiting for medium benchmark pid $WAIT_PID"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
fi

# The balanced continuation already crossed a predeclared rejection boundary:
# dev-clean 6.60% versus 4.92% for this checkpoint, and its running MINDS/Vox
# score was also worse. Keep it for analysis, but port from the verified node.
MEDIUM_BASE="nodes/stock-whisper-medium-cleanroom-ctx93h-step2000/weights/adapter.npz"
LARGE_PORT="artifacts/cleanroom41m-large-v3-turbo-port750.npz"
LARGE_FINAL="artifacts/cleanroom41m-large-v3-turbo-balanced-cekl1000.npz"
SHARED="$HOME/GITHUB/VoiceLLM-Playground/Omni Model/gemma-backbone/cleanroom_cache50"
MED_CLEAN="$HOME/models/jaeger-training-data/stock-whisper-medium-clean360-v1"
MED_ROOM="$HOME/models/jaeger-training-data/stock-whisper-medium-room-aug-v1"
MED_VOX="$HOME/models/jaeger-training-data/stock-whisper-medium-voxpopuli33-v1"
LARGE_BASE="$HOME/models/jaeger-training-data/stock-whisper-large-v3-turbo"
LARGE_CLEAN="$HOME/models/jaeger-training-data/stock-whisper-large-clean360-v1"
LARGE_ROOM="$HOME/models/jaeger-training-data/stock-whisper-large-room-aug-v1"
LARGE_VOX="$HOME/models/jaeger-training-data/stock-whisper-large-voxpopuli33-v1"

echo "phase 1/4: paired large-v3-turbo input seam"
training/run_gpu.sh training/port_cleanroom_large.py \
    --medium-checkpoint "$MEDIUM_BASE" \
    --pair-source "$SHARED::$LARGE_BASE::train100" \
    --pair-source "$SHARED::$LARGE_BASE::fleurs_train" \
    --pair-source "$SHARED::$LARGE_BASE::mindsvox_train" \
    --pair-source "$MED_CLEAN::$LARGE_CLEAN::clean360" \
    --pair-source "$MED_ROOM::$LARGE_ROOM::room_aug" \
    --pair-source "$MED_VOX::$LARGE_VOX::voxpopuli33" \
    --output "$LARGE_PORT" --steps 750 --learning-rate 2e-5 \
    --warmup 75 --accumulate 2 --checkpoint-every 250

echo "phase 2/4: human-label CE/KL refinement of every deployed parameter"
training/run_gpu.sh training/cleanroom_context_ce.py \
    --resume "$LARGE_PORT" --output "$LARGE_FINAL" \
    --train-source "$LARGE_CLEAN::clean360::0.20" \
    --train-source "$LARGE_ROOM::room_aug::0.20" \
    --train-source "$LARGE_VOX::voxpopuli33::0.20" \
    --train-source "$LARGE_BASE::mindsvox_train::0.15" \
    --train-source "$LARGE_BASE::fleurs_train::0.10" \
    --train-source "$LARGE_BASE::train100::0.15" \
    --holdout-source "$LARGE_BASE::devclean::1" \
    --holdout-source "$LARGE_BASE::mindsvox_validation::1" \
    --holdout-source "$LARGE_CLEAN::clean360_validation::1" \
    --holdout-source "$LARGE_VOX::voxpopuli33_validation::1" \
    --allow-acoustic-variants --steps 1000 --learning-rate 5e-6 \
    --warmup 100 --accumulate 1 --checkpoint-every 250 \
    --probe-every 250 --log-every 25

echo "phase 3/4: identical free-running large promotion gates"
JAEGER_SKIP_TRAIN_OTHER=YES training/evaluate_candidate.sh \
    cleanroom41m-large-balanced-20260826 "$LARGE_FINAL" large

echo "phase 4/4: invariant and repository gates"
./.venv/bin/python -m jaeger_agent_omni.omni --selfcheck
./.venv/bin/python research/check_all.py
echo "large window complete"
