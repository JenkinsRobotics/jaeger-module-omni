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
    research/paired_omni_benchmark.py|training/pretrained_overlap.py|training/wavlm_streaming_features.py|training/parallel_overlap.py|training/cleanroom_context_ce.py|training/port_cleanroom_large.py|training/context_distill.py|training/context_ce.py|training/precompute_stock_encoder.py|training/stock_whisper_gemma.py|jaeger_agent_omni/nodes/neural-acoustic-controller/train.py|jaeger_agent_omni/tests/converse_sim.py|research/converse_ear_wer.py|research/converse_context_eval.py|research/converse_answer_eval.py|research/benchmark_agentic_corpus.py|research/benchmark_agentic_omni.py|research/physical_omni_trial.py|research/transfer_probe.py|research/direct_whisper_wer.py|research/latency_breakdown.py|research/streaming_stability.py|research/causal_adapter_check.py)
        ;;
    *)
        echo "REFUSING: unapproved GPU entrypoint $ENTRYPOINT" >&2
        exit 2
        ;;
esac

GPU_PATTERN='converse_answer_eval\.py|converse_context_eval\.py|physical_omni_trial\.py|parallel_overlap\.py train|converse_sim\.py|converse_ear_wer\.py|training/train\.py|training/(cleanroom_context_ce|port_cleanroom_large|context_(distill|ce)|context_train)\.py|training/precompute_stock_encoder\.py|training/stock_whisper_gemma\.py (precompute|pretrain-ctc|distill|train|probe|geometry|wer|bench)|neural-acoustic-controller/train\.py|jaeger_agent_omni\.benchmark|research/(transfer_probe|direct_whisper_wer|latency_breakdown|streaming_stability|causal_adapter_check|benchmark_agentic_(omni|corpus)|duplex_check)\.py|eval/benchmark_suite\.py|cleanroom_(train|distill)\.py|compare_tracks\.py|train_audio_adapter\.py'
CONFLICTS=()
GPU_PATTERN="wavlm_streaming_features\.py cache|paired_omni_benchmark\.py run|pretrained_overlap\.py|$GPU_PATTERN"
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
