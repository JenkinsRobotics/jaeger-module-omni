#!/usr/bin/env bash
# Run one explicitly named repo-local GPU program with contention protection.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
    echo "usage: training/run_gpu.sh training/program.py [args...]" >&2
    exit 2
fi
ENTRYPOINT=$1
shift
case "$ENTRYPOINT" in
    training/cleanroom_context_ce.py|training/port_cleanroom_large.py|training/context_distill.py|training/context_ce.py|training/precompute_stock_encoder.py|training/stock_whisper_gemma.py|research/transfer_probe.py|research/direct_whisper_wer.py|research/latency_breakdown.py)
        ;;
    *)
        echo "REFUSING: unapproved GPU entrypoint $ENTRYPOINT" >&2
        exit 2
        ;;
esac

GPU_PATTERN='training/train\.py|training/(cleanroom_context_ce|port_cleanroom_large|context_(distill|ce)|context_train)\.py|training/precompute_stock_encoder\.py|training/stock_whisper_gemma\.py (precompute|pretrain-ctc|distill|train|probe|geometry|wer|bench)|research/(transfer_probe|direct_whisper_wer|latency_breakdown)\.py|eval/benchmark_suite\.py|cleanroom_(train|distill)\.py|compare_tracks\.py|train_audio_adapter\.py'
CONFLICTS=()
while IFS= read -r PID; do
    [[ -n "$PID" ]] || continue
    COMMAND=$(ps -p "$PID" -o comm=)
    [[ "$COMMAND" == *python* ]] && CONFLICTS+=("$PID")
done < <(pgrep -f "$GPU_PATTERN" || true)
if (( ${#CONFLICTS[@]} )); then
    echo "REFUSING: a known GPU job is already active —" >&2
    ps -o pid=,etime=,command= "${CONFLICTS[@]}" | cut -c1-140 >&2
    exit 1
fi

mkdir -p training/runs
NAME=$(basename "$ENTRYPOINT" .py)
LOG="training/runs/$(date +%Y%m%d-%H%M%S)-${NAME}.log"
echo "logging to $LOG"
exec ./.venv/bin/python -u "$ENTRYPOINT" "$@" 2>&1 | tee "$LOG"
