#!/bin/bash
# One job at a time. Concurrent GPU jobs made a 635ms pipeline measure 48514ms
# -- every number from a contended run is worthless, so this never overlaps.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
LOG=research/sweep.log
: > "$LOG"

step () {
  echo "=== $1 ===" | tee -a "$LOG"
  shift
  "$@" 2>&1 | grep -viE "^Fetching|it/s\]|^\s*$" | tee -a "$LOG"
  echo "=== done ===" | tee -a "$LOG"
}

step "pipelines: legacy/native/hybrid on cases12" \
  $PY research/close_the_gap.py --cases research/cases12 \
  --modes legacy,native,hybrid --out research/pipelines12.json
step "prosody: does the tower carry delivery" \
  $PY research/prosody_carry.py
step "minicpm: stage A/B/C on the same 12 wavs" \
  $PY research/minicpm_stages.py --n 12
echo "SWEEP COMPLETE" | tee -a "$LOG"
