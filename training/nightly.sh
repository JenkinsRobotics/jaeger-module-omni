#!/usr/bin/env bash
# Overnight training. Runs against a time budget, resumes where it left off,
# and saves cleanly when interrupted or when the budget expires.
#
#   training/nightly.sh <corpus-dir> [hours]
#
# Safe to run every night against the same --out: each run continues the
# previous one, restoring optimiser moments and the LR schedule. Without that
# restore every resume boundary would show up as a loss spike.
#
# Schedule it with launchd (macOS) — see training/README.md. Do not build a
# scheduler; the OS has one.

set -euo pipefail

CORPUS="${1:?usage: nightly.sh <corpus-dir> [hours]}"
HOURS="${2:-6}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$REPO/training/runs/nightly-$STAMP.log"

: "${BACKBONE:=google/gemma-4-E4B-it}"
: "${WHISPER:=openai/whisper-medium}"

# E4B is the ONLY supported backbone. A projector is valid solely for the
# backbone it was trained against, and training into a small stand-in measures
# the stand-in: our delta scored 33% free-running against SmolLM2-135M, which
# said nothing about the bridge because a 135M brain cannot answer the
# questions regardless. If BACKBONE is overridden, say so loudly.
case "$BACKBONE" in
  *E4B*|*e4b*) ;;
  *) echo "WARNING: backbone is $BACKBONE, not gemma-4-E4B-it."           "Results will NOT be comparable to any measured pipeline." >&2 ;;
esac
: "${STAGE:=stage3_warmup}"
: "${STEPS:=100000}"

mkdir -p "$REPO/training/runs"

echo "nightly: stage=$STAGE budget=${HOURS}h corpus=$CORPUS" | tee "$LOG"
echo "nightly: backbone=$BACKBONE whisper=$WHISPER" | tee -a "$LOG"

# -u so the log is readable while it runs, not only after it exits.
exec "$REPO/.venv/bin/python" -u "$REPO/training/train.py" \
    --corpus "$CORPUS" \
    --backbone "$BACKBONE" \
    --whisper "$WHISPER" \
    --stage "$STAGE" \
    --steps "$STEPS" \
    --resume \
    --max-hours "$HOURS" \
    --out "runs/nightly" 2>&1 | tee -a "$LOG"
