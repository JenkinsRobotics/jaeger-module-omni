"""Warm-session sweep of global Talker sampling settings on one audio case.

This produces WAVs only. Score them with ``evaluate_omni_audio.py`` from the
independent ASR environment, exactly like the strict end-to-end benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from jaeger_agent_omni.app import read_wav
from jaeger_agent_omni.core.neural.speech import GemmaQwenHiddenBridge, TokenIdBridge
from jaeger_agent_omni.omni import (build, condition_session_token_bridge,
                                    one_turn)
from research.load_minicpm_talker import SAMPLING, load as load_talker
from research.t2w_client import (Token2WavClient, _clip_reference,
                                 synthesize_code_stream_with_client)


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="q05")
    ap.add_argument("--cases", type=Path, default=REPO / "research/cases12")
    ap.add_argument("--token-bridge", type=Path, required=True)
    ap.add_argument("--hidden-bridge", type=Path, required=True)
    ap.add_argument("--ref-audio", type=Path,
                    default=REPO / "training/samples/jfk_moon.wav")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-codes", type=int, default=180)
    ap.add_argument("--talker-device", choices=("cpu", "mps"), default="cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = json.loads((args.cases / "cases.json").read_text())
    case = next((x for x in cases if x["key"] == args.case), None)
    if case is None:
        raise SystemExit(f"unknown case {args.case}")

    token_bridge = TokenIdBridge.load(args.token_bridge)
    hidden_ck = torch.load(args.hidden_bridge, map_location="cpu",
                           weights_only=False)
    hidden_bridge = GemmaQwenHiddenBridge.from_checkpoint(hidden_ck)
    session, meta = build()
    session.warmup()
    samples = read_wav(args.cases / f"{case['key']}.wav")
    reply, reply_ms = one_turn(session, meta, samples, collect_hidden=True)

    talker = load_talker()
    if args.talker_device == "mps":
        talker = talker.to(device="mps", dtype=torch.float16)
    t0 = time.monotonic()
    condition = condition_session_token_bridge(
        session, talker, token_bridge, hidden_bridge=hidden_bridge)
    condition = condition.to(device=next(talker.parameters()).device,
                             dtype=next(talker.parameters()).dtype)
    condition_ms = (time.monotonic() - t0) * 1000

    # Each candidate is a deployable global policy. The full benchmark still
    # decides whether a candidate generalises beyond this diagnostic case.
    candidates = [
        (seed, temperature, top_p, top_k)
        for seed in range(5)
        for temperature, top_p, top_k in ((0.8, 0.85, 25), (0.5, 0.85, 25))
    ]
    rows = []
    t2w = Token2WavClient()
    t2w.init(device="cpu")
    t2w.set_ref_audio(_clip_reference(args.ref_audio))
    try:
        for seed, temperature, top_p, top_k in candidates:
            SAMPLING.update(temperature=temperature, top_p=top_p, top_k=top_k)
            torch.manual_seed(seed)
            name = f"{case['key']}_s{seed}_t{temperature:g}"
            out = args.out_dir / f"{name}.wav"
            with torch.no_grad():
                stream = talker.iter_code_chunks(
                    [(condition, True)], max_codes=args.max_codes)
                _, timing = synthesize_code_stream_with_client(t2w, stream, out)
            row = {
                "case": name, "answers": case["answers"],
                "input_pcm": str((args.cases / f"{case['key']}.wav").resolve()),
                "output_pcm": str(out.resolve()),
                "gemma_reply_log_only": reply,
                "seed": seed, "temperature": temperature,
                "top_p": top_p, "top_k": top_k,
                "speech_codes": timing["speech_codes"],
                "output_seconds": round(wav_seconds(out), 3),
                "reply_ms": round(reply_ms, 1),
                "condition_ms": round(condition_ms, 1),
                "talker_vocoder_ms": timing["total_ms"],
                "first_audio_ms": round(reply_ms + condition_ms +
                                        timing["first_chunk_ms"], 1),
                "vocoder_first_chunk_ms": timing["first_chunk_ms"],
                "stream_chunks": timing["chunks"],
                "talker_streaming": True,
                "transcript_handoff": False,
            }
            rows.append(row)
            print(f"{name}: {timing['speech_codes']} codes, "
                  f"{timing['chunks']} chunks, first {row['first_audio_ms']:.0f} ms",
                  flush=True)
            (args.out_dir / "results.json").write_text(json.dumps({
                "format": "jaeger-talker-global-sampling-sweep-v1",
                "architecture": {"transcript_or_decoded_text_handoff": False},
                "rows": rows}, indent=2))
    finally:
        t2w.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
