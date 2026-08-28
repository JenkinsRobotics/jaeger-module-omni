#!/usr/bin/env bash
# Product-focused 1:1 lexical window. Only a source corpus whose matched
# Whisper-medium decoder is below 5% WER is allowed to train this candidate.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ${JAEGER_GPU_WINDOW:-} != YES ]]; then
    echo "REFUSING: set JAEGER_GPU_WINDOW=YES only when the GPU is assigned" >&2
    exit 2
fi

DATA="$HOME/models/jaeger-training-data"
GEMMA="$HOME/models/gemma-4-E4B-it-4bit-mlx"
BASE="nodes/jaegeromni-41m-medium/weights/adapter.npz"
OUT="artifacts/jaegeromni-41m-1to1-clean-gated-v1.npz"
RUN="research/candidate_evals/jaegeromni-41m-1to1-clean-gated-v1"

# Measured source control, same first 128 Clean-360 validation waveforms:
# strict WER 4.32%; standard-normalized WER 3.27%.
training/run_gpu.sh training/cleanroom_context_ce.py \
    --resume "$BASE" --output "$OUT" --model "$GEMMA" \
    --train-source "$DATA/stock-whisper-medium-clean360-v1::clean360::1" \
    --holdout-source "$DATA/stock-whisper-medium-eval-v1::devclean::1" \
    --holdout-source "$DATA/stock-whisper-medium-clean360-v1::clean360_validation::1" \
    --steps 200 --learning-rate 2e-7 --warmup 20 \
    --text-teacher-weight 0.5 --anchor-weight 1.0 --accumulate 2 \
    --checkpoint-every 50 --probe-every 50 --log-every 10 \
    --seed 20260828

for step in 000050 000100 000150 000200; do
    adapter="${OUT%.npz}.step${step}.npz"
    label="${RUN}-step${step}"
    training/run_gpu.sh training/stock_whisper_gemma.py wer \
        --cache "$DATA/stock-whisper-medium-eval-v1" --split devclean \
        --id-file "$DATA/stock-whisper-medium-eval-v1/devclean/manifest.jsonl" \
        -n 64 --adapter "$adapter" --model "$GEMMA" \
        --whisper openai/whisper-medium --output "${label}-devclean64.json"
    training/run_gpu.sh training/stock_whisper_gemma.py wer \
        --cache "$DATA/stock-whisper-medium-clean360-v1" \
        --split clean360_validation \
        --id-file "$DATA/stock-whisper-medium-clean360-v1/clean360_validation/manifest.jsonl" \
        -n 128 --adapter "$adapter" --model "$GEMMA" \
        --whisper openai/whisper-medium --output "${label}-clean360val128.json"
    ./.venv/bin/python research/standard_wer.py \
        --input "devclean64=${label}-devclean64.json" \
        --input "clean360val128=${label}-clean360val128.json" \
        --output "${label}-standard-wer.json"
done

./.venv/bin/python - <<'PY'
import json
from pathlib import Path
paths = sorted(Path('research/candidate_evals').glob(
    'jaegeromni-41m-1to1-clean-gated-v1-step*-standard-wer.json'))
rows = [(json.loads(path.read_text()), path) for path in paths]
passing = [(row, path) for row, path in rows if row['gate_pass']]
if passing:
    row, path = min(passing, key=lambda item: item[0]['aggregate']['normalized_wer'])
    print(f"1:1 WINNER {path}: {100*row['aggregate']['normalized_wer']:.3f}%")
else:
    print("1:1 NO PROMOTION: no checkpoint passed both <5% gates")
PY

./.venv/bin/python -m jaeger_agent_omni.omni --selfcheck
./.venv/bin/python research/check_all.py
