"""Close the audio-comprehension gap to 90%+, measured against legacy.

Established: native E4B audio scores 9/12 on spoken questions while the SAME
backbone answers all three failures correctly from text. The loss is in audio
comprehension, not reasoning. This measures the candidates.

    legacy      whisper medium.en -> transcript -> E4B      the reference
    native      E4B audio tower -> E4B                      current, 75%
    hybrid      audio tower AND transcript, both in context  belt and braces

The hybrid is the interesting one and costs no training: give the model the
audio (which carries prosody and timing) AND the transcript (which carries
reliable words). If it recovers the legacy score while keeping the audio in
context, we get accuracy without giving up what a transcript throws away.

    python research/close_the_gap.py --cases <dir> --mode legacy,native,hybrid
"""

from __future__ import annotations

import argparse
import re
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SYSTEM = "You are a helpful assistant. Answer in one short sentence."


from grading import graded                                       # noqa: E402


def run(mode, cases, cases_dir, model_path, whisper_id="openai/whisper-medium.en",
        whisper_device="mps"):
    import mlx.core as mx
    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.utils import load

    from jaeger_agent_omni.app import read_wav
    from jaeger_agent_omni.runtime import GemmaAudioPrefiller, StreamingSession

    m, proc = load(model_path)
    cfg = json.load(open(Path(model_path) / "config.json"))
    eos = {proc.tokenizer.eos_token_id}

    transcribe = None
    if mode in ("legacy", "hybrid"):
        from transformers import (WhisperForConditionalGeneration,
                                  WhisperProcessor)
        wproc = WhisperProcessor.from_pretrained(whisper_id)
        # MPS, not CPU: HF whisper on CPU is ~5x slower and made an earlier
        # legacy measurement read 11.8s when it is really ~0.6s.
        wmodel = WhisperForConditionalGeneration.from_pretrained(
            whisper_id).to(whisper_device).eval()

        def transcribe(audio):
            import torch
            feats = wproc(audio, sampling_rate=16000,
                          return_tensors="pt").input_features.to(whisper_device)
            with torch.no_grad():
                ids = wmodel.generate(feats)
            return wproc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def fresh():
        return StreamingSession(m, proc, GemmaAudioPrefiller(m, proc),
                                cache_factory=KVCache, config=cfg, system=SYSTEM)

    fresh().warmup()
    rows = []

    for c in cases:
        audio = read_wav(Path(cases_dir) / f"{c['key']}.wav")
        s = fresh()
        s.reset()

        if mode == "legacy":
            # No audio in context at all — the reference architecture.
            t_asr0 = time.monotonic()
            text = transcribe(audio)
            asr_ms = (time.monotonic() - t_asr0) * 1000
            s.system = SYSTEM
            s.open_turn()          # opens with an audio slot we simply skip
            s.feed_text(text)
            t0 = time.monotonic()
            s.end_of_turn(close=True)
            reply = s.respond(max_tokens=48, eos_ids=eos)
            wait = (time.monotonic() - t0) * 1000 + asr_ms
            extra = f"asr {asr_ms:.0f}ms · {text[:40]!r}"
        else:
            s.open_turn()
            for i in range(0, len(audio), 1600):
                s.feed_audio(audio[i:i + 1600])
            asr_ms = 0.0
            if mode == "hybrid":
                # Audio is already in the cache; add the transcript after it so
                # the model has reliable words AND the original audio.
                t_asr0 = time.monotonic()
                text = transcribe(audio)
                asr_ms = (time.monotonic() - t_asr0) * 1000
                s.feed_text(f" (transcript: {text})")
            t0 = time.monotonic()
            s.end_of_turn()
            reply = s.respond(max_tokens=48, eos_ids=eos)
            wait = (time.monotonic() - t0) * 1000 + asr_ms
            extra = f"asr {asr_ms:.0f}ms" if mode == "hybrid" else ""

        ok = graded(reply, c["answers"])
        rows.append({"case": c["key"], "ok": ok, "wait_ms": wait,
                     "rt": wait / 1000 / c["duration_s"], "reply": reply.strip()[:60]})
        print(f"  {c['key']:4s} {'OK' if ok else '..'} {wait:6.0f}ms  "
              f"{reply.strip()[:46]!r} {extra}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--modes", default="legacy,native,hybrid")
    ap.add_argument("--model",
                    default="/Users/jonathanjenkins/models/gemma-4-e4b-it-mlx-full")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cases = json.load(open(Path(args.cases) / "cases.json"))
    all_rows = {}
    for mode in args.modes.split(","):
        print(f"--- {mode} ---")
        try:
            all_rows[mode] = run(mode, cases, args.cases, args.model)
        except Exception as exc:                       # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:100]}")
            continue

    print(f"\n{'mode':10s} {'accuracy':>10s} {'median ms':>11s} {'realtime':>9s}")
    for mode, rows in all_rows.items():
        n = sum(r["ok"] for r in rows)
        print(f"{mode:10s} {n:>4}/{len(rows):<5} {100*n/len(rows):>3.0f}% "
              f"{statistics.median(r['wait_ms'] for r in rows):>10.0f} "
              f"{statistics.median(r['rt'] for r in rows):>8.2f}x")
    if args.out:
        Path(args.out).write_text(json.dumps(all_rows, indent=1))
    return 0


def demo():
    assert graded("Seven continents.", ["7", "seven"])
    assert not graded("No continents.", ["7", "seven"])
    print("close_the_gap: grader OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
