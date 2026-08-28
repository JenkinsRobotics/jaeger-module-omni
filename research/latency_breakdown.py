#!/usr/bin/env python3
"""Apples-to-apples latency breakdown for one packaged audio node.

Run each candidate in a separate process through ``training/run_gpu.sh`` so
Metal memory is released between models and the repository's one-job guard is
enforced.  The measured interval starts after the waveform has been buffered,
matching the cases12 turn-end latency definition.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def median(rows: list[dict], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-end", required=True)
    parser.add_argument("--cases", type=Path,
                        default=ROOT / "research/cases12")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--fused-cleanroom-attention", action="store_true",
                        help="A/B the equivalent packed-QKV inference path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    if args.fused_cleanroom_attention:
        os.environ["JAEGER_FUSED_CLEANROOM_ATTN"] = "1"

    from jaeger_agent_omni.app import read_wav
    from jaeger_agent_omni.omni import build
    from grading import graded

    cases = json.loads((args.cases / "cases.json").read_text())
    session, meta = build(front_end=args.front_end)
    if meta["front_end"] != "stock":
        raise SystemExit("latency breakdown currently requires a stock node")

    # Compile the exact encoder -> adapter -> Gemma path before collecting any
    # sample.  Historical 50M measurements included a 1.25 s compile outlier;
    # it must not make either candidate look faster or slower.
    warmup_ms = session.warmup()
    encoder_ms = [0.0]
    original_front_end = session.front_end

    def measured_front_end(samples):
        started = time.monotonic()
        result = original_front_end(samples)
        encoder_ms[0] += (time.monotonic() - started) * 1000
        return result

    session.front_end = measured_front_end
    rows = []
    for repeat in range(args.repeats):
        for case in cases:
            audio = read_wav(args.cases / f"{case['key']}.wav")
            session.reset()
            session.open_turn()
            for offset in range(0, len(audio), 1600):
                session.feed_audio(audio[offset:offset + 1600])

            encoder_ms[0] = 0.0
            prefill0 = session.total_prefill_ms
            started = time.monotonic()
            session.end_of_turn(close=False)
            tail_ms = (time.monotonic() - started) * 1000
            prefill1 = session.total_prefill_ms

            started = time.monotonic()
            session.close_turn()
            close_ms = (time.monotonic() - started) * 1000
            prefill2 = session.total_prefill_ms

            first_token_at = [None]
            generation_started = time.monotonic()

            def on_token(_token, _count):
                if first_token_at[0] is None:
                    first_token_at[0] = time.monotonic()

            reply = session.respond(
                max_tokens=args.max_tokens, eos_ids=meta["eos"],
                on_token=on_token)
            generation_ms = (time.monotonic() - generation_started) * 1000
            prefill3 = session.total_prefill_ms
            first_token_ms = (
                (first_token_at[0] - generation_started) * 1000
                if first_token_at[0] is not None else generation_ms)
            total_ms = tail_ms + close_ms + generation_ms
            row = {
                "repeat": repeat,
                "case": case["key"],
                "ok": graded(reply, case["answers"]),
                "reply": reply,
                "output_tokens": len(session.output_ids),
                "encoder_ms": encoder_ms[0],
                "adapter_and_audio_prefill_ms": max(
                    0.0, tail_ms - encoder_ms[0]),
                "audio_tail_total_ms": tail_ms,
                "audio_prefill_counter_ms": prefill1 - prefill0,
                "prompt_close_ms": close_ms,
                "prompt_close_prefill_counter_ms": prefill2 - prefill1,
                "first_token_ms": first_token_ms,
                "first_token_prefill_counter_ms": prefill3 - prefill2,
                "decode_after_first_ms": max(0.0, generation_ms - first_token_ms),
                "generation_ms": generation_ms,
                "turn_end_to_reply_ms": total_ms,
            }
            rows.append(row)
            print(
                f"{case['key']} r{repeat + 1} "
                f"{total_ms:6.1f} ms = encoder {row['encoder_ms']:5.1f} + "
                f"audio/prefill {row['adapter_and_audio_prefill_ms']:5.1f} + "
                f"close {close_ms:5.1f} + first {first_token_ms:5.1f} + "
                f"decode {row['decode_after_first_ms']:5.1f}", flush=True)

    summary_keys = (
        "encoder_ms", "adapter_and_audio_prefill_ms", "audio_tail_total_ms",
        "audio_prefill_counter_ms", "prompt_close_ms",
        "prompt_close_prefill_counter_ms", "first_token_ms",
        "first_token_prefill_counter_ms", "decode_after_first_ms",
        "generation_ms", "turn_end_to_reply_ms")
    payload = {
        "schema_version": 1,
        "kind": "adapter_latency_breakdown",
        "measurement": "free_running_turn_end_to_complete_reply",
        "front_end": args.front_end,
        "model_id": meta["model_id"],
        "checkpoint": meta["checkpoint"],
        "warmup_ms": warmup_ms,
        "fused_cleanroom_attention": args.fused_cleanroom_attention,
        "repeats": args.repeats,
        "correct": sum(bool(row["ok"]) for row in rows),
        "n": len(rows),
        "median": {key: median(rows, key) for key in summary_keys},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["median"], indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
