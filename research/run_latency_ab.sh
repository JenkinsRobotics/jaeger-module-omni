#!/usr/bin/env bash
# Identical, sequential latency A/B. Each model exits before the next loads.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${1:-research/results/latency-ab-20260828}
mkdir -p "$OUT"

training/run_gpu.sh research/latency_breakdown.py \
  --front-end jaegeromni-41m-medium --repeats 2 \
  --output "$OUT/jaeger41-baseline.json"
training/run_gpu.sh research/latency_breakdown.py \
  --front-end jaegeromni-41m-medium --fused-cleanroom-attention --repeats 2 \
  --output "$OUT/jaeger41-fused-qkv.json"
training/run_gpu.sh research/latency_breakdown.py \
  --front-end jaegeromni-50m-medium --repeats 2 \
  --output "$OUT/jaeger50.json"

./.venv/bin/python research/compare_latency_breakdowns.py \
  "$OUT/jaeger41-baseline.json" \
  "$OUT/jaeger41-fused-qkv.json" \
  "$OUT/jaeger50.json" \
  --output "$OUT/comparison.json"
