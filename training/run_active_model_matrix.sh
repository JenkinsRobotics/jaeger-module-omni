#!/usr/bin/env bash
# Sequentially improve every trainable active Jaeger audio-ear candidate.
# Gemma Native and imported GemmaOmni remain immutable controls.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ${JAEGER_GPU_WINDOW:-} != YES ]]; then
    echo "REFUSING: set JAEGER_GPU_WINDOW=YES only for an assigned GPU window" >&2
    exit 2
fi

WAIT_PID=${1:-}
if [[ -n "$WAIT_PID" ]]; then
    echo "waiting for active Large v2 matrix member pid $WAIT_PID"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
fi

DATA="$HOME/models/jaeger-training-data"
GEMMA="$HOME/models/gemma-4-E4B-it-4bit-mlx"
M41_BASE="jaeger_agent_omni/nodes/jaegeromni-41m-medium/weights/adapter.npz"
M41_OUT="artifacts/jaegeromni-41m-medium-balanced-anchor20-v2.npz"
M50_BASE="jaeger_agent_omni/nodes/stock-whisper-medium-cekl-blend50/weights/adapter.npz"
M50_OUT="artifacts/jaegeromni-50m-medium-balanced-anchor20-v2.npz"

common_sources=(
    --train-source "$DATA/stock-whisper-medium-clean360-v1::clean360::0.25"
    --train-source "$DATA/stock-whisper-medium-room-aug-v1::room_aug::0.20"
    --train-source "$DATA/stock-whisper-medium-voxpopuli33-v1::voxpopuli33::0.20"
    --train-source "$DATA/stock-whisper-medium-expanded-v1::ami5_train::0.15"
    --train-source "$DATA/stock-whisper-medium-supplemental-v1::mindsvox_train::0.10"
    --train-source "$DATA/stock-whisper-medium-supplemental-v1::fleurs_train::0.10"
    --holdout-source "$DATA/stock-whisper-medium-eval-v1::devclean::1"
    --holdout-source "$DATA/stock-whisper-medium-expanded-v1::ami5_train_validation::1"
    --holdout-source "$DATA/stock-whisper-medium-eval-v1::mindsvox_validation::1"
    --holdout-source "$DATA/stock-whisper-medium-clean360-v1::clean360_validation::1"
    --holdout-source "$DATA/stock-whisper-medium-voxpopuli33-v1::voxpopuli33_validation::1"
)

echo "matrix 2/3: conservative JaegerOmni 41M Medium continuation"
training/run_gpu.sh training/cleanroom_context_ce.py \
    --resume "$M41_BASE" --output "$M41_OUT" "${common_sources[@]}" \
    --model "$GEMMA" --allow-acoustic-variants --steps 500 \
    --learning-rate 1e-6 --warmup 50 --text-teacher-weight 0.5 \
    --anchor-weight 0.20 --accumulate 1 --checkpoint-every 250 \
    --probe-every 250 --log-every 25 --seed 20260828
JAEGER_SKIP_CASES12=YES JAEGER_SKIP_TRAIN_OTHER=YES \
    training/evaluate_candidate.sh \
    jaegeromni-41m-medium-balanced-anchor20-v2-20260828 "$M41_OUT" medium

echo "matrix 3/3: same transferable curriculum on JaegerOmni 50M Medium"
training/run_gpu.sh training/context_ce.py \
    --resume "$M50_BASE" --output "$M50_OUT" "${common_sources[@]}" \
    --model "$GEMMA" --train-branch all --allow-acoustic-variants \
    --steps 500 --learning-rate 1e-6 --warmup 50 \
    --text-teacher-weight 0.5 --anchor-weight 0.20 --speaker-weight 0 \
    --accumulate 1 --checkpoint-every 250 --log-every 25 --seed 20260828
JAEGER_SKIP_CASES12=YES JAEGER_SKIP_TRAIN_OTHER=YES \
    training/evaluate_candidate.sh \
    jaegeromni-50m-medium-balanced-anchor20-v2-20260828 "$M50_OUT" medium

./.venv/bin/python -m jaeger_agent_omni.omni --selfcheck
./.venv/bin/python research/check_all.py
echo "active model matrix complete"
