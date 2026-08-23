#!/usr/bin/env bash
# Launch a training run, refusing to start if one is already going.
#
# Concurrent GPU jobs have produced four wrong measurements in this project
# and one silently starved training run. `sweep.sh` enforces this for
# benchmarks; launching training with a bare `&` bypassed it, and I then ran
# two E4B trainings at once on a 32 GB machine (free memory: 0.10 GB).
#
#   training/run.sh --corpus ... --backbone ... [args]
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_PATTERN='training/train\.py|training/stock_whisper_gemma\.py (precompute|pretrain-ctc|distill|train|probe|geometry|wer|bench)|cleanroom_(train|distill)\.py|compare_tracks\.py|train_audio_adapter\.py'
if pgrep -f "$GPU_PATTERN" >/dev/null; then
    echo "REFUSING: a training run is already active —" >&2
    ps -o pid=,etime=,command= $(pgrep -f "$GPU_PATTERN") | cut -c1-100 >&2
    echo "kill it first, or wait. Two E4B runs do not fit in 32 GB." >&2
    exit 1
fi

LOG="training/runs/$(date +%Y%m%d-%H%M%S).log"
mkdir -p training/runs
echo "logging to $LOG"
# tee, never `| tail` — tail buffers the whole stream until exit, which made
# three live runs look dead.
exec ./.venv/bin/python -u training/train.py "$@" 2>&1 | tee "$LOG"
