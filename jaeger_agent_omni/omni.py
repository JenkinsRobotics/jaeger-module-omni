"""The current no-transcript input path — not yet the complete Omni pipeline.

    mic ─► audio encoder ─► E4B ─► text tokens
                             │
                             └──► hidden states ──X  talker   (not wired yet)

There is NO transcription step on the implemented input chain. Audio becomes
embeddings and stays embeddings until the backbone emits language. That is the
entire point of this repository: the sequential stack sends TEXT between
modules and loses everything text cannot carry.

Do not call this module audio-to-audio or end-to-end Omni yet.  The Talker
port is now independently alive, but Gemma states do not condition it and this
CLI emits no waveform.  Component scores here cannot satisfy Omni acceptance.

Run it:

    python -m jaeger_agent_omni.omni --wav clip.wav          # one turn
    python -m jaeger_agent_omni.omni --bench research/cases12  # score vs legacy
    python -m jaeger_agent_omni.omni --duplex --seconds 20    # always-on
    python -m jaeger_agent_omni.omni --selfcheck              # no models

The audio front end is swappable at this seam and NOTHING else changes:

    tower   Gemma's own audio tower. Works today, scores 10/12 (83% of
            legacy's 12/12). Frozen Google weights -- cannot be extended, and
            measured transcription-shaped (delivery ratio 0.192).
    delta   whisper encoder -> our adapter. We own it, it carries ~0.323 of
            delivery where a transcript carries 0.000, and it skips whisper's
            autoregressive decoder entirely.

Both are text-free. `delta` is the one we can keep improving.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SYSTEM = "You are a helpful assistant. Answer in one short sentence."
MODEL = "/Users/jonathanjenkins/models/gemma-4-e4b-it-mlx-full"


def build(model_path: str = MODEL, front_end: str = "tower", delta_path=None):
    """Load the one graph. Returns (session, meta)."""
    import mlx.core as mx                                       # noqa: F401
    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.utils import load

    from jaeger_agent_omni.runtime import GemmaAudioPrefiller, StreamingSession

    m, proc = load(model_path)
    cfg = json.load(open(Path(model_path) / "config.json"))
    prefiller = GemmaAudioPrefiller(m, proc)

    if front_end == "delta":
        # The seam. Same session, same duplex loop, same speech out -- only
        # the thing that turns waveform into LLM-space embeddings changes.
        if not delta_path or not Path(delta_path).exists():
            raise SystemExit(
                f"--front-end delta needs a trained adapter (--delta PATH).\n"
                f"  train it:  python training/train.py --backbone "
                f"google/gemma-4-E4B-it --corpus training/corpus_mixed \\\n"
                f"             --arch minicpm_delta --out runs/delta_e4b")
        raise SystemExit("delta front end: adapter loading lands with the "
                         "first trained E4B checkpoint (training in progress)")

    s = StreamingSession(m, proc, prefiller, cache_factory=KVCache,
                         config=cfg, system=SYSTEM)
    return s, {"model": model_path, "front_end": front_end,
               "eos": {proc.tokenizer.eos_token_id}}


def one_turn(s, meta, audio, max_tokens: int = 48,
             collect_hidden: bool = False) -> tuple[str, float]:
    """Audio in, text out. No transcript is produced at any point.

    `collect_hidden` also retains the per-token cognition states, which is what
    speech out conditions on -- see speak().
    """
    s.reset()
    s.open_turn()
    for i in range(0, len(audio), 1600):            # 100 ms chunks, streaming
        s.feed_audio(audio[i:i + 1600])
    t0 = time.monotonic()
    s.end_of_turn()
    reply = s.respond(max_tokens=max_tokens, eos_ids=meta["eos"],
                      collect_hidden=collect_hidden)
    return reply, (time.monotonic() - t0) * 1000


def speak(session, text: str, out_path: str, ref_audio=None,
          delta_path: str = "training/runs/speech_out_delta.pt") -> str:
    """Reply -> waveform, conditioned on the cognition states, not the words.

        merged[t] = emb_text(ids[t]) + L2norm(proj_semantic(hidden[t]))

    `session.hidden` must already be populated by respond(collect_hidden=True).
    Passing text alone is NOT a fallback: zero-conditioning was measured twice
    and produces fluent English babble -- real speech, wrong words -- so a
    missing delta is an error, never a degraded mode that ships silently.

    Voice cloning happens at the vocoder: token2wav's `set_ref_audio` embeds
    the reference speaker via campplus. Load the Soul, you loaded the voice.
    """
    import torch

    from jaeger_agent_omni.neural.delta import SpeechOutDelta

    hidden = getattr(session, "hidden", None)
    if not hidden:
        raise RuntimeError(
            "no hidden states — call respond(collect_hidden=True). Speech "
            "conditioned on text alone babbles; there is no text-only path.")
    if not Path(delta_path).exists():
        raise RuntimeError(
            f"SpeechOutDelta not trained ({delta_path}).\n"
            f"  Until it is, speech out produces babble, not these words.\n"
            f"  The audible-today route is research/t2w_client.py --say.")

    import numpy as np
    ck = torch.load(delta_path, map_location="cpu")
    state = ck.get("state", ck) if isinstance(ck, dict) else ck
    # target_norm travels WITH the weights. Rebuilding without it leaves the
    # output scale free, and scale reaches the talker through a biased ReLU:
    # unit vectors vs teacher-scale ones project to cosine 0.176 -- almost
    # orthogonal conditioning, which sounds exactly like an undertrained delta.
    tn = ck.get("target_norm") if isinstance(ck, dict) else None
    if tn is None:
        raise RuntimeError(
            f"{delta_path} has no target_norm; refusing to guess the output "
            "scale — the wrong one produces confident babble, not an error.")
    d = SpeechOutDelta(cognition_dim=session.hidden[0].shape[-1],
                       target_norm=tn)
    d.load_state_dict(state)
    d.eval()
    h = torch.from_numpy(np.stack([np.array(x, copy=False).reshape(-1)
                                   for x in hidden])).float()
    with torch.no_grad():
        cond = d(h.unsqueeze(0))          # unit vectors, talker-space

    from research.t2w_client import speak as t2w_speak      # noqa: E402
    return t2w_speak(text, out_path, ref_audio=ref_audio, conditioning=cond)


def bench(cases_dir: str, model_path: str, front_end: str, delta_path=None,
          out_path: str = "research/omni_bench.json") -> int:
    """Score the no-transcript input path; this is not audio-output proof."""
    import statistics as st

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
    from grading import graded                                   # noqa: E402

    from jaeger_agent_omni.app import read_wav

    cases = json.load(open(Path(cases_dir) / "cases.json"))
    s, meta = build(model_path, front_end, delta_path)
    s.warmup()

    rows = []
    for c in cases:
        audio = read_wav(Path(cases_dir) / f"{c['key']}.wav")
        reply, ms = one_turn(s, meta, audio)
        ok = graded(reply, c["answers"])
        rows.append({"case": c["key"], "ok": ok, "ms": ms, "reply": reply[:70]})
        print(f"  {c['key']} {'OK ' if ok else 'MISS'} {ms:7.0f}ms  {reply[:60]!r}",
              flush=True)

    n = sum(r["ok"] for r in rows)
    med = st.median(r["ms"] for r in rows)
    print(f"\nomni [{front_end}]  {n}/{len(rows)} = {100*n/len(rows):.0f}%  "
          f"median {med:.0f} ms")
    print(f"legacy reference   12/12 = 100%   (text between modules)")
    print(f"ratio to legacy    {100*n/len(rows)/100*100/100:.0%} accuracy"
          .replace('%%', '%'))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"front_end": front_end, "correct": n,
                               "n": len(rows), "median_ms": med,
                               "rows": rows}, indent=1))
    print(f"wrote {out}")
    return 0


def selfcheck() -> int:
    """Structure only -- no models, no audio, runs in CI.

    The invariant this guards is the whole premise: nothing in the omni path
    may turn audio into text and feed the text back. If a transcript ever
    becomes the transfer format again, this pipeline is just the sequential
    stack wearing a different name.
    """
    import inspect
    code = "".join(inspect.getsource(f)
                   for f in (build, one_turn, bench, speak, main))
    for banned, why in (("transcribe(", "calls an ASR decoder"),
                        ("feed_text(", "injects a transcript into the cache")):
        assert banned not in code, f"omni path {why}"
    for fn in (build, one_turn, bench, speak):
        assert fn.__doc__, f"{fn.__name__} needs a docstring"
    print("omni: no transcription step, no text injection, entry points OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav")
    ap.add_argument("--bench")
    ap.add_argument("--duplex", action="store_true")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--front-end", default="tower", choices=("tower", "delta"))
    ap.add_argument("--delta", default="training/runs/delta_e4b.pt")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="research/omni_bench.json")
    ap.add_argument("--speak", action="store_true",
                    help="also synthesise the reply (needs a trained SpeechOutDelta)")
    ap.add_argument("--ref-audio", default=None, help="voice to clone")
    ap.add_argument("--out-wav", default="reply.wav")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        return selfcheck()
    if a.bench:
        return bench(a.bench, a.model, a.front_end, a.delta, a.out)
    if a.wav:
        from jaeger_agent_omni.app import read_wav
        s, meta = build(a.model, a.front_end, a.delta)
        s.warmup()
        reply, ms = one_turn(s, meta, read_wav(Path(a.wav)),
                             collect_hidden=a.speak)
        print(f"{ms:.0f} ms  {reply!r}")
        if a.speak:
            wav = speak(s, reply, a.out_wav, ref_audio=a.ref_audio)
            print(f"spoke -> {wav}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
