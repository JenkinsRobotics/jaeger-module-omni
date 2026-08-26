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

The assistant audio front end is swappable at this seam and NOTHING else
changes:

    tower   Gemma's own audio tower. Works today, scores 10/12 (83% of
            legacy's 12/12). Frozen Google weights -- cannot be extended, and
            measured transcription-shaped (delivery ratio 0.192).
    stock   stock OpenAI Whisper encoder -> fresh Jaeger adapter -> Gemma.
            The promoted owned checkpoint distils Gemma's native audio tower
            and contains no MiniCPM/Qwen model or ASR decoder.

Both are text-free. `tower` remains the live default; `stock` is the owned,
extensible development path. The MiniCPM-tuned encoder remains isolated in
meeting observer mode while it still owns that mode's best measured WER.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

SYSTEM = (
    "You are a voice assistant. Answer the spoken question directly in one "
    "short sentence. Do not mention images unless the user explicitly asks "
    "about an image."
)
DUAL_SYSTEM = (
    "Listen to the spoken audio. Return exactly two lines. Line 1 must begin "
    "TRANSCRIPT: and contain the exact spoken words. Line 2 must begin "
    "ANSWER: and directly answer the spoken question. Do not repeat the "
    "question as the answer."
)
MODEL = "/Users/jonathanjenkins/models/gemma-4-e4b-it-mlx-full"
STOCK = (Path(__file__).resolve().parent.parent / "artifacts"
         / "stock-whisper-gemma-medium50m-expanded-blend25.npz")


def build(model_path: str = MODEL, front_end: str = "tower", adapter_path=None,
          system: str | None = SYSTEM):
    """Load the one graph. Returns (session, meta)."""
    if front_end == "stock":
        from jaeger_agent_omni.runtime.stock_whisper import (
            DEFAULT_GEMMA, StockWhisperGemmaSession)
        selected_adapter = Path(adapter_path or STOCK)
        if not selected_adapter.exists():
            raise SystemExit(
                "--front-end stock needs a trained adapter (--adapter PATH).")
        stock_model = DEFAULT_GEMMA if model_path == MODEL else model_path
        if selected_adapter.is_dir():
            from jaeger_agent_omni.model_nodes import (
                load_audio_node,
                load_node_factory,
            )
            node = load_audio_node(selected_adapter)
            session = load_node_factory(node)(
                model_path=stock_model, system=system)
        else:
            session = StockWhisperGemmaSession(
                selected_adapter, model_path=stock_model, system=system)
        configured_eos = getattr(session.tokenizer, "eos_token_id", 1)
        eos = {1, 106, int(configured_eos)}
        return session, {
            "model": str(stock_model), "front_end": front_end,
            "adapter": str(session.adapter_path),
            "node": session.node.node_id if session.node else None,
            "eos": eos,
        }

    import mlx.core as mx                                       # noqa: F401
    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.utils import load

    from jaeger_agent_omni.runtime import GemmaAudioPrefiller, StreamingSession

    m, proc = load(model_path)
    cfg = json.load(open(Path(model_path) / "config.json"))
    prefiller = GemmaAudioPrefiller(m, proc)

    s = StreamingSession(m, proc, prefiller, cache_factory=KVCache,
                         config=cfg, system=system)
    eos = cfg.get("eos_token_id", proc.tokenizer.eos_token_id)
    eos = {int(x) for x in (eos if isinstance(eos, list) else [eos])}
    return s, {"model": model_path, "front_end": front_end, "eos": eos}


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


def split_dual_output(reply: str) -> tuple[str, str]:
    """Separate Gemma's diagnostic transcript and answer output.

    Both strings were generated by the same Gemma pass. The transcript is not
    fed back into the session and is never an input-path transfer format.
    """
    transcript_match = re.search(
        r"TRANSCRIPT:\s*(.*?)(?:\n+ANSWER:|$)", reply,
        flags=re.IGNORECASE | re.DOTALL)
    answer_match = re.search(
        r"ANSWER:\s*(.*)", reply, flags=re.IGNORECASE | re.DOTALL)
    transcript = transcript_match.group(1).strip() if transcript_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    return transcript, answer


def condition_session(session, conditioner):
    """Convert a completed Gemma generation to Talker conditions.

    This is the architectural seam under test: tensors in, tensors out. It
    cannot accept a decoded answer string.
    """
    import mlx.core as mx
    import numpy as np
    import torch

    hidden = getattr(session, "hidden", None)
    if not hidden:
        raise RuntimeError("no hidden states — call respond(collect_hidden=True)")
    output_ids = getattr(session, "output_ids", ())
    if len(output_ids) != len(hidden):
        raise RuntimeError(f"speech tensors desynchronised: {len(output_ids)} ids "
                           f"for {len(hidden)} hidden states")
    h = torch.from_numpy(np.stack([np.array(
        x.astype(mx.float32), copy=False).reshape(-1) for x in hidden])).float()
    token_emb, _ = session._embed_ids(list(output_ids))
    token_emb = torch.from_numpy(np.array(
        token_emb.astype(mx.float32), copy=False)).float()
    with torch.no_grad():
        return conditioner(token_emb, h.unsqueeze(0))


def condition_session_token_bridge(session, talker, token_bridge,
                                   hidden_map_path=None, hidden_bridge=None,
                                   teacher_hidden_norm: float = 72.6281):
    """Exact Talker lexical embeddings plus mapped Gemma hidden tensors.

    The token bridge is compiled offline. Runtime performs integer lookup and
    tensor math only: no tokenizer, decode, string, or TTS text request.
    """
    import mlx.core as mx
    import numpy as np
    import torch

    hidden = getattr(session, "hidden", None)
    ids = getattr(session, "output_ids", ())
    if not hidden or len(ids) != len(hidden):
        raise RuntimeError("Gemma IDs and hidden states are missing or misaligned")
    h = torch.from_numpy(np.stack([np.array(
        x.astype(mx.float32), copy=False).reshape(-1) for x in hidden])).float()
    qwen_ids, aligned = token_bridge.expand(ids, h.unsqueeze(0))
    device = next(talker.parameters()).device
    dtype = next(talker.parameters()).dtype
    qwen_ids = qwen_ids.to(device)
    aligned = aligned.to(device=device, dtype=dtype)
    if hidden_bridge is not None:
        hidden_bridge = hidden_bridge.to(device=device, dtype=dtype)
        qwen_hidden = hidden_bridge(aligned)
    else:
        if hidden_map_path is None:
            raise ValueError("a learned hidden bridge or linear map is required")
        mapping = torch.from_numpy(np.load(hidden_map_path)).to(
            device=device, dtype=dtype)
        if mapping.shape != (aligned.shape[-1], 4096):
            raise ValueError(f"hidden map {mapping.shape} != "
                             f"({aligned.shape[-1]}, 4096)")
        qwen_hidden = aligned @ mapping
        qwen_hidden = qwen_hidden / qwen_hidden.norm(
            dim=-1, keepdim=True).clamp(min=1e-6) * teacher_hidden_norm
    with torch.no_grad():
        return talker.emb_text(qwen_ids) + talker.proj_semantic(qwen_hidden)


def speak(session, out_path: str, ref_audio=None,
          conditioner_path: str = "training/runs/gemma_talker_conditioner.pt") -> str:
    """Gemma tensors -> Talker codes -> waveform, with no text handoff.

        (Gemma token embeddings, Gemma hidden states) -> merged condition
        merged condition -> Talker -> speech codes -> Token2Wav -> PCM

    `session.hidden` must already be populated by respond(collect_hidden=True).
    Passing text alone is NOT a fallback: zero-conditioning was measured twice
    and produces fluent English babble -- real speech, wrong words -- so a
    missing delta is an error, never a degraded mode that ships silently.

    Voice cloning happens at the vocoder: token2wav's `set_ref_audio` embeds
    the reference speaker via campplus. Load the Soul, you loaded the voice.
    """
    import torch

    if not Path(conditioner_path).exists():
        raise RuntimeError(
            f"GemmaTalkerConditioner not trained ({conditioner_path}).\n"
            f"  Until it is, speech out produces babble, not these words.\n"
            f"  Component-only Talker proof: research/verify_talker_port.py")

    from jaeger_agent_omni.neural.speech import GemmaTalkerConditioner

    ck = torch.load(conditioner_path, map_location="cpu", weights_only=False)
    conditioner = GemmaTalkerConditioner.from_checkpoint(ck)
    cond = condition_session(session, conditioner)

    from research.load_minicpm_talker import load as load_talker
    from research.t2w_client import synthesize
    talker = load_talker()
    with torch.no_grad():
        codes = talker.generate_code_chunks([(cond, True)])
    synthesize(codes[0].tolist(), out_path, ref_audio)
    return str(out_path)


def bench(cases_dir: str, model_path: str, front_end: str, adapter_path=None,
          out_path: str = "research/omni_bench.json",
          dual_output: bool = False) -> int:
    """Score the no-transcript input path; this is not audio-output proof."""
    import statistics as st

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
    from grading import graded                                   # noqa: E402

    from jaeger_agent_omni.app import read_wav

    cases = json.load(open(Path(cases_dir) / "cases.json"))
    s, meta = build(model_path, front_end, adapter_path,
                    system=DUAL_SYSTEM if dual_output else SYSTEM)
    s.warmup()

    rows = []
    for c in cases:
        audio = read_wav(Path(cases_dir) / f"{c['key']}.wav")
        reply, ms = one_turn(s, meta, audio, max_tokens=96 if dual_output else 48)
        transcript, answer = split_dual_output(reply) if dual_output else ("", reply)
        ok = graded(answer, c["answers"])
        row = {"case": c["key"], "ok": ok, "ms": ms,
               "reply": reply if dual_output else reply[:70]}
        if dual_output:
            row.update({"transcript": transcript, "answer": answer})
        rows.append(row)
        shown = answer if dual_output else reply
        print(f"  {c['key']} {'OK ' if ok else 'MISS'} {ms:7.0f}ms  {shown[:60]!r}",
              flush=True)
        if dual_output:
            print(f"      transcript {transcript[:100]!r}", flush=True)

    n = sum(r["ok"] for r in rows)
    med = st.median(r["ms"] for r in rows)
    transcript_wer = None
    transcript_words = 0
    if dual_output and all(c.get("transcript") for c in cases):
        from training.evaluate import corpus_wer
        transcript_wer, transcript_words = corpus_wer([
            (case["transcript"], row["transcript"])
            for case, row in zip(cases, rows)])
    print(f"\nomni [{front_end}]  {n}/{len(rows)} = {100*n/len(rows):.0f}%  "
          f"median {med:.0f} ms")
    if transcript_wer is not None:
        print(f"generated transcript WER {100*transcript_wer:.2f}% "
              f"over {transcript_words} words")
    print(f"legacy reference   12/12 = 100%   (text between modules)")
    print(f"ratio to legacy    {100*n/len(rows)/100*100/100:.0%} accuracy"
          .replace('%%', '%'))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"front_end": front_end, "dual_output": dual_output,
               "correct": n, "n": len(rows), "median_ms": med,
               "rows": rows}
    if transcript_wer is not None:
        payload.update({"transcript_wer": transcript_wer,
                        "transcript_words": transcript_words})
    out.write_text(json.dumps(payload, indent=1))
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
                   for f in (build, one_turn, split_dual_output, bench,
                             condition_session,
                             condition_session_token_bridge, speak, main))
    for banned, why in (("transcribe(", "calls an ASR decoder"),
                        ("feed_text(", "injects a transcript into the cache")):
        assert banned not in code, f"omni path {why}"
    for fn in (build, one_turn, bench, speak):
        assert fn.__doc__, f"{fn.__name__} needs a docstring"
    assert split_dual_output(
        "TRANSCRIPT: What is two plus two?\nANSWER: Four.") == (
            "What is two plus two?", "Four.")
    print("omni: no transcription step, no text injection, entry points OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav")
    ap.add_argument("--bench")
    ap.add_argument("--duplex", action="store_true")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--front-end", default="tower",
                    choices=("tower", "stock"))
    ap.add_argument("--adapter", default=None,
                    help="adapter checkpoint for the stock front end")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="research/omni_bench.json")
    ap.add_argument("--dual-output", action="store_true",
                    help="emit and separately score TRANSCRIPT plus ANSWER")
    ap.add_argument("--speak", action="store_true",
                    help="also synthesise the reply (needs a trained SpeechOutDelta)")
    ap.add_argument("--ref-audio", default=None, help="voice to clone")
    ap.add_argument("--out-wav", default="reply.wav")
    ap.add_argument("--speech-conditioner",
                    default="training/runs/gemma_talker_conditioner.pt")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    adapter_path = a.adapter or str(STOCK)

    if a.selfcheck:
        return selfcheck()
    if a.bench:
        return bench(a.bench, a.model, a.front_end, adapter_path, a.out,
                     a.dual_output)
    if a.wav:
        from jaeger_agent_omni.app import read_wav
        s, meta = build(
            a.model, a.front_end, adapter_path,
            system=DUAL_SYSTEM if a.dual_output else SYSTEM)
        s.warmup()
        reply, ms = one_turn(s, meta, read_wav(Path(a.wav)),
                             collect_hidden=a.speak)
        print(f"{ms:.0f} ms  {reply!r}")
        if a.speak:
            wav = speak(s, a.out_wav, ref_audio=a.ref_audio,
                        conditioner_path=a.speech_conditioner)
            print(f"spoke -> {wav}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
