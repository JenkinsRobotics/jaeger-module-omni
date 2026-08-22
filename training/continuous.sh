#!/usr/bin/env bash
# Continuous adapter training. Runs rounds until stopped, resuming each time.
#
#   training/continuous.sh [steps_per_round] [rounds]
#
# Each round resumes the previous checkpoint, so this is one long run split
# into inspectable pieces -- a round that goes wrong can be discarded without
# losing the ones before it, and validation CE is recorded per round so the
# trend is visible instead of inferred.
#
# grep BLOCK-BUFFERS when stdout is not a terminal, so a live run writes
# nothing to the log until 4 KB accumulates -- round 2 ran 3:45 in total
# silence and looked hung. --line-buffered is not optional here.
#
# One job at a time: concurrent GPU work has produced four wrong measurements
# in this project, including a 635 ms pipeline that measured 48 s.
set -uo pipefail
cd "$(dirname "$0")/.."

SRC="$HOME/GITHUB/VoiceLLM-Playground/Omni Model/gemma-backbone"
PY="$PWD/.venv/bin/python"
WORK="$HOME/models/gemma-audio-adapter"
CKPT="$WORK/adapter_jaeger.npz"
STEPS="${1:-1000}"
ROUNDS="${2:-24}"
LOG="training/runs/continuous.log"

if pgrep -f "train_audio_adapter.py train" >/dev/null; then
    echo "REFUSING: a training round is already active" >&2; exit 1
fi
mkdir -p training/runs
echo "continuous: ${ROUNDS} rounds x ${STEPS} steps -> $CKPT" | tee -a "$LOG"

for r in $(seq 1 "$ROUNDS"); do
    # "${RESUME[@]}" on an EMPTY array is an unbound-variable error under
    # set -u on bash 3.2 (macOS ships 3.2). Use a plain string instead.
    RESUME=""
    [ -f "$CKPT" ] && RESUME="--resume $CKPT"
    # --max-steps is ABSOLUTE, not per-round: the trainer resumes at the
    # checkpoint's step and stops when it reaches max_steps. Passing the same
    # 1000 every round means round 2 onward do NOTHING -- measured: round 2
    # finished in 11 s with validation CE identical to round 1.
    TARGET=$(( r * STEPS ))
    echo "=== round $r/$ROUNDS -> step $TARGET  $(date +%H:%M:%S) ===" | tee -a "$LOG"
    ( cd "$SRC" && "$PY" -u train_audio_adapter.py train \
        --cache-dir "$WORK/cache" \
        --train-split train.100 \
        --validation-split validation \
        --max-steps "$TARGET" \
        --learning-rate 2e-5 \
        --save-every 250 \
        --output "$CKPT" \
        $RESUME ) 2>&1 | grep --line-buffered -viE "Malloc|it/s\]" | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "round $r FAILED rc=$rc — stopping rather than burning the budget" \
            | tee -a "$LOG"
        exit "$rc"
    fi
done
echo "continuous: all rounds done" | tee -a "$LOG"
