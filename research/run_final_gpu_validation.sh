#!/usr/bin/env bash
# Final frozen-checkpoint GPU queue. No training and no automatic promotion.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=${1:-$(date +%Y%m%d-%H%M%S)}
OUT="research/results/final-gpu-${STAMP}"
mkdir -p "$OUT"

# Establish a genuinely apples-to-apples 41M/50M latency result and test the
# checkpoint-preserving packed-QKV implementation first.
research/run_latency_ab.sh "$OUT/latency"

# Measure how much right context the frozen Whisper+adapter needs before its
# vectors stop revising. Diagnostic only: this cannot promote causal prefill
# without a later free-running answer/WER gate.
training/run_gpu.sh research/streaming_stability.py \
  --front-end jaegeromni-41m-medium \
  --output "$OUT/streaming-stability-41m.json"

echo "FINAL GPU VALIDATION COMPLETE: $OUT"
echo "No checkpoint was trained or promoted. Review reports before changing defaults."
