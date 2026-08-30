"""What is Gemma actually receiving, right now? A live readout of the feed.

Separates two questions that are easy to confuse:

    is the AUDIO PATH delivering clean information?   <- this tool
    is GEMMA reasoning well about it?                  <- not this tool

The LLM never runs here. We read the adapter's output vectors back against
Gemma's embedding table and print the words they land on, continuously, the
way a transcription display would — plus the signals that ride alongside:
level, turn state, latency, slot count.

If this view is clean and keeps up, the input side is done and any remaining
problem is the model's. If it is garbage, no amount of prompt work helps.

    python research/audio_feed_monitor.py --ckpt <adapter.pt> --wav clip.wav
    python research/audio_feed_monitor.py --ckpt <adapter.pt> --corpus DIR -n 6
"""

from __future__ import annotations

import argparse, sys, time, wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

SR = 16000
GEMMA = "google/gemma-4-E4B-it"
CHUNK_S = 1.0            # the runtime's streaming cadence


def rms_db(x) -> float:
    import numpy as np
    r = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    return 20.0 * np.log10(max(r, 1e-6))


def bar(db: float, width: int = 12) -> str:
    lo, hi = -60.0, 0.0
    n = int(max(0.0, min(1.0, (db - lo) / (hi - lo))) * width)
    return "#" * n + "." * (width - n)


def main() -> int:
    import numpy as np
    import torch
    from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperModel

    from fit_alignment import embedding_table
    from jaeger_agent_omni.app import read_wav
    from jaeger_agent_omni.core.neural.whisper_to_gemma import WhisperToGemma
    from jaeger_agent_omni.core.runtime.endpoint import Endpoint

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav")
    ap.add_argument("--corpus")
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--whisper", default="openai/whisper-medium.en")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(GEMMA)
    table = embedding_table(GEMMA)
    table = table * (table.shape[1] ** 0.5)
    tnorm = table / table.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = WhisperToGemma(cognition_dim=ck["cognition_dim"],
                           embed_norm=ck["embed_norm"])
    model.load_state_dict(ck["model"])
    model.eval()
    enc = WhisperModel.from_pretrained(a.whisper).encoder.eval()
    fx = WhisperFeatureExtractor.from_pretrained(a.whisper)

    clips = []
    if a.wav:
        clips = [(Path(a.wav).name, read_wav(Path(a.wav)))]
    elif a.corpus:
        import json
        rows = [json.loads(l) for l in
                (Path(a.corpus) / "turns.jsonl").read_text().splitlines() if l.strip()]
        for r in rows[:a.n]:
            clips.append((r["transcript"][:34],
                          read_wav(Path(a.corpus) / r["audio_path"])))
    else:
        raise SystemExit("--wav or --corpus")

    print(f"{'level':14s} {'slots':>5} {'ms':>6}  what Gemma receives")
    print("-" * 78)
    for label, audio in clips:
        ep = Endpoint()
        n_frame = int(SR * ep.frame_ms / 1000)
        for i in range(0, len(audio) - n_frame + 1, n_frame):
            ep.push(audio[i:i + n_frame])

        for start in range(0, len(audio), int(SR * CHUNK_S)):
            chunk = audio[start:start + int(SR * CHUNK_S)]
            if len(chunk) < SR // 4:
                continue
            t0 = time.perf_counter()
            feats = fx(chunk, sampling_rate=SR, return_tensors="pt").input_features
            with torch.no_grad():
                h = enc(feats).last_hidden_state
                out = model(h.float(), duration_s=len(chunk) / SR)[0]
            ms = (time.perf_counter() - t0) * 1000

            v = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            ids = (tnorm @ v.T).argmax(dim=0)
            words = [tok.decode([int(i)]).strip() or "." for i in ids]
            # collapse CTC-style repeats so the readout is legible
            seen = [w for i, w in enumerate(words) if i == 0 or w != words[i - 1]]

            db = rms_db(chunk)
            state = "END" if ep.ended else ("...." if ep.speaking else "quiet")
            print(f"{bar(db)} {db:6.1f} {out.shape[0]:>5} {ms:>6.0f}  "
                  f"[{state}] {' '.join(seen[:10])}")
        print(f"{'':14s} {'':5s} {'':6s}  ^ truth: {label!r}\n")
    return 0


def demo() -> None:
    assert bar(-60.0).startswith("."), bar(-60.0)
    assert bar(0.0).startswith("#"), bar(0.0)
    assert rms_db([]) == -120.0
    assert abs(rms_db([1.0, -1.0]) - 0.0) < 1e-6
    print("audio_feed_monitor: level metering OK")


if __name__ == "__main__":
    raise SystemExit(demo() if "--demo" in sys.argv else main())
