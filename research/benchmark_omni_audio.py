"""Run the strict Jaeger graph and emit auditable audio artifacts.

No ASR is imported here. A decoded reply is logged for diagnosis, but the
speech path receives only Gemma token embeddings and hidden-state tensors.
Use evaluate_omni_audio.py in an independent environment to score the WAVs.
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
from jaeger_agent_omni.neural.speech import (GemmaQwenHiddenBridge,
                                             GemmaTalkerConditioner,
                                             TokenIdBridge)
from jaeger_agent_omni.omni import (build, condition_session,
                                    condition_session_token_bridge, one_turn)
from research.load_minicpm_talker import SAMPLING, load as load_talker
from research.t2w_client import (Token2WavClient, _clip_reference,
                                 synthesize_code_stream_with_client,
                                 synthesize_with_client)

def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", type=Path, default=REPO / "research/cases12")
    ap.add_argument("--conditioner", type=Path, default=None)
    ap.add_argument("--token-bridge", type=Path, default=None)
    ap.add_argument("--hidden-map", type=Path, default=None)
    ap.add_argument("--hidden-bridge", type=Path, default=None)
    ap.add_argument("--ref-audio", type=Path,
                    default=REPO / "training/samples/jfk_moon.wav")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--keys", default=None,
                    help="comma-separated case keys; applied before --n")
    ap.add_argument("--max-codes", type=int, default=180)
    ap.add_argument("--talker-device", choices=("cpu", "mps"), default="mps")
    ap.add_argument("--seed", type=int, default=4,
                    help="global deterministic policy validated on all 12 cases")
    ap.add_argument("--temperature", type=float, default=SAMPLING["temperature"])
    ap.add_argument("--top-p", type=float, default=SAMPLING["top_p"])
    ap.add_argument("--top-k", type=int, default=SAMPLING["top_k"])
    ap.add_argument("--stream-output", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="render speech codes while Talker is still decoding")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.conditioner) == bool(args.token_bridge):
        raise SystemExit("choose exactly one of --conditioner or --token-bridge")
    if args.token_bridge and bool(args.hidden_map) == bool(args.hidden_bridge):
        raise SystemExit("--token-bridge requires exactly one of --hidden-map "
                         "or --hidden-bridge")
    conditioner = None
    bridge = None
    if args.conditioner:
        checkpoint = torch.load(args.conditioner, map_location="cpu",
                                weights_only=False)
        conditioner = GemmaTalkerConditioner.from_checkpoint(checkpoint)
    else:
        bridge = TokenIdBridge.load(args.token_bridge)
    hidden_bridge = None
    if args.hidden_bridge:
        hidden_ck = torch.load(args.hidden_bridge, map_location="cpu",
                               weights_only=False)
        hidden_bridge = GemmaQwenHiddenBridge.from_checkpoint(hidden_ck)
    session, meta = build()
    session.warmup()
    talker = load_talker()
    if args.talker_device == "mps":
        talker = talker.to(device="mps", dtype=torch.float16)
    cases = json.loads((args.cases / "cases.json").read_text())
    if args.keys:
        wanted = set(args.keys.split(","))
        cases = [case for case in cases if case["key"] in wanted]
        missing = wanted - {case["key"] for case in cases}
        if missing:
            raise SystemExit(f"unknown case keys: {','.join(sorted(missing))}")
    cases = cases[:args.n]
    SAMPLING.update(temperature=args.temperature, top_p=args.top_p,
                    top_k=args.top_k)
    rows = []
    t2w = Token2WavClient()
    t2w.init(device="cpu")
    t2w.set_ref_audio(_clip_reference(args.ref_audio))
    for i, case in enumerate(cases):
        source = args.cases / f"{case['key']}.wav"
        samples = read_wav(source)
        reply, reply_ms = one_turn(session, meta, samples, collect_hidden=True)
        t0 = time.monotonic()
        if conditioner is not None:
            condition = condition_session(session, conditioner)
        else:
            condition = condition_session_token_bridge(
                session, talker, bridge, args.hidden_map, hidden_bridge)
        condition = condition.to(device=next(talker.parameters()).device,
                                 dtype=next(talker.parameters()).dtype)
        condition_ms = (time.monotonic() - t0) * 1000
        torch.manual_seed(args.seed)
        out = args.out_dir / f"{case['key']}.wav"
        if args.stream_output:
            with torch.no_grad():
                code_stream = talker.iter_code_chunks(
                    [(condition, True)], max_codes=args.max_codes)
                _, stream = synthesize_code_stream_with_client(
                    t2w, code_stream, out)
            speech_codes = stream["speech_codes"]
            talker_ms = None
            vocoder_ms = None
            talker_vocoder_ms = stream["total_ms"]
        else:
            t0 = time.monotonic()
            with torch.no_grad():
                codes = talker.generate_code_chunks(
                    [(condition, True)], max_codes=args.max_codes)
            talker_ms = (time.monotonic() - t0) * 1000
            speech_codes = int(codes.shape[1])
            t0 = time.monotonic()
            _, stream = synthesize_with_client(t2w, codes[0].tolist(), out)
            vocoder_ms = (time.monotonic() - t0) * 1000
            talker_vocoder_ms = talker_ms + vocoder_ms
        row = {"case": case["key"], "answers": case["answers"],
               "input_pcm": str(source.resolve()),
               "output_pcm": str(out.resolve()),
               "input_samples_received": len(samples),
               "input_samples_consumed": len(samples),
               "gemma_reply_log_only": reply,
               "gemma_tokens": len(session.output_ids),
               "gemma_hidden_shape": [len(session.hidden),
                                      int(session.hidden[0].shape[-1])],
               "talker_condition_shape": list(condition.shape),
               "speech_codes": speech_codes,
               "output_seconds": round(wav_seconds(out), 3),
               "reply_ms": round(reply_ms, 1),
               "condition_ms": round(condition_ms, 1),
               "talker_ms": (round(talker_ms, 1)
                              if talker_ms is not None else None),
               "vocoder_ms": (round(vocoder_ms, 1)
                               if vocoder_ms is not None else None),
               "talker_vocoder_ms": round(talker_vocoder_ms, 1),
               "first_audio_ms": round(reply_ms + condition_ms +
                                       stream["first_chunk_ms"], 1),
               "stream_chunks": stream["chunks"],
               "talker_streaming": args.stream_output,
               "vocoder_first_chunk_ms": stream["first_chunk_ms"],
               "wall_clock_paced_input": False,
               "barge_in_tested": False, "transcript_handoff": False}
        rows.append(row)
        print(f"{case['key']} {reply_ms:.0f}ms + talker/vocoder "
              f"{talker_vocoder_ms:.0f}ms; {speech_codes} codes; "
              f"{row['output_seconds']:.2f}s WAV", flush=True)
        report = {"format": "jaeger-omni-audio-bench-v1",
                  "architecture": {
                      "graph": "PCM->audio_encoder->Gemma->conditioner->Talker->Token2Wav->PCM",
                      "transcript_or_decoded_text_handoff": False,
                      "conditioner": (str(args.conditioner.resolve())
                                      if args.conditioner else
                                      "exact-token-bridge+hidden-map"),
                      "token_bridge": (str(args.token_bridge.resolve())
                                       if args.token_bridge else None),
                      "hidden_map": (str(args.hidden_map.resolve())
                                     if args.hidden_map else None),
                      "hidden_bridge": (str(args.hidden_bridge.resolve())
                                        if args.hidden_bridge else None)},
                  "generation": {"seed": args.seed,
                                 "temperature": args.temperature,
                                 "top_p": args.top_p,
                                 "top_k": args.top_k,
                                 "stream_output": args.stream_output},
                  "rows": rows}
        (args.out_dir / "results.json").write_text(json.dumps(report, indent=2))
    t2w.close()
    print(f"wrote {args.out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
