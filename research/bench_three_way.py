"""Three voice pipelines, identical spoken questions, measured side by side.

    ours       native Gemma 4 E4B tower -> streaming KV-cached prefill (MLX)
    ultravox   frozen Whisper -> trained projector -> Llama 3.2 1B (HF)
    minicpm    Whisper-derived tower -> projector -> Qwen3-8B (llama.cpp)

Three genuinely different answers to the same problem — native audio tower,
trained bridge onto a text LLM, and purpose-built omni model — so this is the
comparison that says which shape we should be in.

What is measured, and why each:

    correct     does it actually answer the spoken question? The only
                metric that matters first; everything else is tie-breaking.
    latency     time from "audio ends" to "reply exists". Streaming systems
                should win here by prefilling during speech.
    realtime x  latency relative to clip duration. Under 1.0 means the
                pipeline is faster than the speech it consumed.

## MiniCPM status: harness mismatch, not a model result

`llama-omni-cli --test` treats the input clip as a VOICE-CLONING REFERENCE and
replies with its Chinese persona introduction ("你好，我是面壁智能小钢炮") —
it never reads the audio as a question. Measured latencies from that path are
real (median 5.3 s wall clock including process start and model load), but the
0/5 correctness is the harness, NOT the model. A fair read needs
`llama-omni-server` with an English system prompt, or its Python API.

Do not quote the CLI numbers as a MiniCPM capability result.

## Ultravox status: BLOCKED, not slow

Its checkpoint ships the ADAPTER ONLY — `model.safetensors` is the projector,
and the base LLM is fetched from `config.text_model_id` at load time. That id
is Meta's gated `Llama-3.2-1B-Instruct`, and `UltravoxConfig.__init__` resolves
it before any override can apply (tried: config object patch, `AutoConfig`
kwarg, direct `AutoModel` construction). Running it needs Meta access on this
machine, not more code.

Two things that attempt did establish:

  * their distribution model is adapter-only with the base referenced by id —
    exactly what `jaeger_agent_omni.artifacts` does. Independent confirmation
    that our artifact design matches field practice.
  * v0.5-1b is Llama-based; the Gemma variants are Gemma-2/3, i.e. models with
    no native audio. Ultravox exists to give audio to LLMs that lack it —
    which Gemma 4 already has.

The architectural comparison we actually wanted (stack-8 + SwiGLU + RMSNorm vs
MiniCPM's pool-5 + ReLU) is covered by `training/ab_projectors.py`, which
implements both designs against one fixed baseline and needs no gated weights.

Fairness notes, since it would be easy to cheat by accident:
    * same wav files, same order, warmed before timing
    * each pipeline gets its own natural prompting — handicapping one with
      another's template would measure the template, not the pipeline
    * ours is the ONLY one measured in streaming mode, because it is the only
      one that has one; that advantage is the point, not a thumb on the scale

    python research/bench_three_way.py --cases <dir> [--only ours,ultravox]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = []


def judge(reply: str, key: str) -> bool:
    """Cheap keyword grading — these questions have unambiguous answers."""
    r = reply.lower()
    return {
        "q1": lambda: "4" in r or "four" in r,
        "q2": lambda: "paris" in r,
        "q3": lambda: "7" in r or "seven" in r,
        "q4": lambda: sum(c in r for c in ("red", "blue", "yellow", "green")) >= 2,
        "q5": lambda: "1969" in r,
    }[key]()


def record(pipeline: str, key: str, reply: str, latency_ms: float, dur_s: float):
    ok = judge(reply, key)
    RESULTS.append({"pipeline": pipeline, "case": key, "ok": ok,
                    "latency_ms": latency_ms, "rt": latency_ms / 1000 / dur_s,
                    "reply": reply.strip()[:90]})
    print(f"  [{pipeline:8s}] {key} {'OK ' if ok else '.. '} "
          f"{latency_ms:6.0f}ms ({latency_ms/1000/dur_s:.2f}x) {reply.strip()[:60]!r}")


# -- ours: native Gemma, streaming ------------------------------------------
def run_ours(cases, cases_dir, model_path):
    import json as _json

    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.utils import load

    from jaeger_agent_omni.app import read_wav
    from jaeger_agent_omni.runtime import GemmaAudioPrefiller, StreamingSession

    m, proc = load(model_path)
    cfg = _json.load(open(Path(model_path) / "config.json"))
    eos = {proc.tokenizer.eos_token_id}

    def session():
        s = StreamingSession(m, proc, GemmaAudioPrefiller(m, proc),
                             cache_factory=KVCache, config=cfg,
                             system="You are a helpful assistant. Answer in one short sentence.")
        return s

    session().warmup()          # compile graphs before timing anything

    for c in cases:
        s = session()
        s.reset()
        s.open_turn()
        audio = read_wav(Path(cases_dir) / f"{c['key']}.wav")
        # Stream it in as a mic would, so prefill overlaps "speech".
        for i in range(0, len(audio), 1600):
            s.feed_audio(audio[i:i + 1600])
        t0 = time.monotonic()           # clock starts when the user STOPS
        s.end_of_turn()
        reply = s.respond(max_tokens=40, eos_ids=eos)
        record("ours", c["key"], reply, (time.monotonic() - t0) * 1000,
               c["duration_s"])


# -- ultravox: frozen whisper -> projector -> llama --------------------------
def run_ultravox(cases, cases_dir, repo="fixie-ai/ultravox-v0_5-llama-3_2-1b",
                 base="unsloth/Llama-3.2-1B-Instruct"):
    import transformers

    from jaeger_agent_omni.app import read_wav

    # Ultravox ships ONLY the adapter — model.safetensors is the projector and
    # the base LLM is pulled from its text_model_id at load time. That id is
    # Meta's gated repo, so we point it at an ungated mirror of the same
    # weights. (The adapter-only distribution model is itself a finding: it is
    # exactly what jaeger_agent_omni.artifacts does.)
    from transformers import AutoConfig, AutoModel, AutoProcessor
    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
    if base:
        cfg.text_model_id = base
    # Build the model ourselves so the patched config is the one used at
    # construction; the pipeline helper re-reads config from the repo.
    model = AutoModel.from_pretrained(repo, config=cfg, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
    pipe = transformers.pipeline("audio-text-to-text", model=model,
                                 processor=processor, trust_remote_code=True)
    warm = read_wav(Path(cases_dir) / f"{cases[0]['key']}.wav")
    turns = [{"role": "system",
              "content": "You are a helpful assistant. Answer in one short sentence."}]
    pipe({"audio": warm, "turns": list(turns), "sampling_rate": 16000},
         max_new_tokens=8)

    for c in cases:
        audio = read_wav(Path(cases_dir) / f"{c['key']}.wav")
        t0 = time.monotonic()
        out = pipe({"audio": audio, "turns": list(turns), "sampling_rate": 16000},
                   max_new_tokens=40)
        reply = out if isinstance(out, str) else str(out)
        record("ultravox", c["key"], reply, (time.monotonic() - t0) * 1000,
               c["duration_s"])


# -- minicpm: the purpose-built omni model ----------------------------------
def run_minicpm(cases, cases_dir, binary, model):
    import re
    import subprocess
    import tempfile

    for c in cases:
        with tempfile.TemporaryDirectory() as tmp:
            # its --test harness wants <prefix>0000.wav
            src = Path(cases_dir) / f"{c['key']}.wav"
            (Path(tmp) / "c_0000.wav").write_bytes(src.read_bytes())
            t0 = time.monotonic()
            p = subprocess.run([binary, "-m", model, "--no-tts", "--test",
                                f"{tmp}/c_", "1"],
                               capture_output=True, text=True, timeout=600)
            dt = (time.monotonic() - t0) * 1000
        # Its reply is streamed to the TTS consumer, not stdout; the text
        # chunks are logged though.
        # NOTE: with --test this scrapes the model's PERSONA intro, not an
        # answer — see the module note. Kept so the latency column is real.
        chunks = re.findall(r"text='([^']*)'", p.stdout + p.stderr)
        record("minicpm", c["key"], "".join(dict.fromkeys(chunks)), dt,
               c["duration_s"])


def summarise():
    if not RESULTS:
        return
    print(f"\n{'pipeline':10s} {'correct':>9s} {'median ms':>10s} {'median x':>9s}")
    for name in sorted({r["pipeline"] for r in RESULTS}):
        rs = [r for r in RESULTS if r["pipeline"] == name]
        lat = sorted(r["latency_ms"] for r in rs)
        rts = sorted(r["rt"] for r in rs)
        mid = len(lat) // 2
        print(f"{name:10s} {sum(r['ok'] for r in rs):>4}/{len(rs):<4} "
              f"{lat[mid]:>10.0f} {rts[mid]:>9.2f}")
    print("\ncorrect = answered the spoken question · x = latency / clip duration")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--only", default="ours,ultravox,minicpm")
    ap.add_argument("--gemma", default="/Users/jonathanjenkins/models/gemma-4-e4b-it-mlx-full")
    ap.add_argument("--minicpm-bin",
                    default=str(Path.home() / "GITHUB/llama.cpp-omni/build/bin/llama-omni-cli"))
    ap.add_argument("--minicpm-model",
                    default=str(Path.home() / "models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q8_0.gguf"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cases = json.load(open(Path(args.cases) / "cases.json"))
    want = set(args.only.split(","))
    print(f"{len(cases)} spoken questions, "
          f"{sum(c['duration_s'] for c in cases):.1f}s of audio\n")

    for name, fn in (("ours", lambda: run_ours(cases, args.cases, args.gemma)),
                     ("ultravox", lambda: run_ultravox(cases, args.cases)),
                     ("minicpm", lambda: run_minicpm(cases, args.cases,
                                                     args.minicpm_bin,
                                                     args.minicpm_model))):
        if name not in want:
            continue
        print(f"--- {name} ---")
        try:
            fn()
        except Exception as exc:                       # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:110]}")

    summarise()
    if args.out:
        Path(args.out).write_text(json.dumps(RESULTS, indent=1))
    return 0


def demo():
    RESULTS.clear()
    record("x", "q1", "The answer is four.", 100.0, 1.0)
    record("x", "q2", "It is Berlin.", 200.0, 2.0)
    assert RESULTS[0]["ok"] and not RESULTS[1]["ok"], "grader must discriminate"
    assert abs(RESULTS[1]["rt"] - 0.1) < 1e-9
    # every case must have a grader, or a pipeline silently scores zero
    for k in ("q1", "q2", "q3", "q4", "q5"):
        judge("test", k)
    assert judge("red, blue and yellow", "q4")
    assert not judge("red only", "q4"), "needs 2+ colours"
    RESULTS.clear()
    print("bench_three_way: grader + accounting OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
