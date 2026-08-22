#!/usr/bin/env bash
# Cache several corpora in sequence. One at a time -- concurrent GPU work has
# produced four wrong measurements in this project.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
for c in "$@"; do
    echo "=== $c  $(date +%H:%M:%S) ==="
    $PY -u training/precompute_multi.py --corpus "$c" --max-examples 4000 \
        2>&1 | grep --line-buffered -viE "Malloc|it/s\]"
done
echo "ALL CACHED"
