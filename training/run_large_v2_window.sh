#!/usr/bin/env bash
# Large-v3-turbo v2: conservative full-adapter refinement, then broad gates.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ${JAEGER_GPU_WINDOW:-} != YES ]]; then
    echo "REFUSING: set JAEGER_GPU_WINDOW=YES only for an assigned GPU window" >&2
    exit 2
fi

DATA="$HOME/models/jaeger-training-data"
BASE="artifacts/cleanroom41m-large-v3-turbo-port750.npz"
OUT="artifacts/cleanroom41m-large-v3-turbo-balanced-anchor10-v2.npz"
LABEL="cleanroom41m-large-balanced-anchor10-v2-20260827"

[[ -f "$BASE" ]] || { echo "missing paired Large port: $BASE" >&2; exit 2; }

training/run_gpu.sh training/cleanroom_context_ce.py \
    --resume "$BASE" --output "$OUT" \
    --train-source "$DATA/stock-whisper-large-clean360-v1::clean360::0.20" \
    --train-source "$DATA/stock-whisper-large-room-aug-v1::room_aug::0.20" \
    --train-source "$DATA/stock-whisper-large-voxpopuli33-v1::voxpopuli33::0.20" \
    --train-source "$DATA/stock-whisper-large-v3-turbo::mindsvox_train::0.15" \
    --train-source "$DATA/stock-whisper-large-v3-turbo::fleurs_train::0.10" \
    --train-source "$DATA/stock-whisper-large-v3-turbo::train100::0.15" \
    --holdout-source "$DATA/stock-whisper-large-v3-turbo::devclean::1" \
    --holdout-source "$DATA/stock-whisper-large-v3-turbo::mindsvox_validation::1" \
    --holdout-source "$DATA/stock-whisper-large-clean360-v1::clean360_validation::1" \
    --holdout-source "$DATA/stock-whisper-large-voxpopuli33-v1::voxpopuli33_validation::1" \
    --allow-acoustic-variants --steps 1000 --learning-rate 3e-6 \
    --warmup 100 --text-teacher-weight 0.5 --anchor-weight 0.10 \
    --accumulate 1 --checkpoint-every 250 --probe-every 250 \
    --log-every 25 --seed 20260827

JAEGER_SKIP_CASES12=YES JAEGER_SKIP_TRAIN_OTHER=YES \
    training/evaluate_candidate.sh "$LABEL" "$OUT" large

./.venv/bin/python -m jaeger_agent_omni.omni --selfcheck
./.venv/bin/python research/check_all.py
echo "Large v2 window complete: $OUT"
