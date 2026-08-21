"""Stage-wise baseline of MiniCPM-o 4.5 — its real engine, our 12 questions.

End-to-end duplex benchmarking stayed blocked on the harness, so measure the
model at every boundary that IS reachable instead:

    STAGE A  pre-LLM     wav -> whisper enc -> projector -> LLM-space
    STAGE B  LLM         LLM-space -> text          <- the accuracy number
    STAGE C  post-LLM    hidden -> talker -> token2wav -> waveform

Two details cost the parallel agent (and us) several dead runs, both fixed here:

  * ctypes needs argtypes. Without them the 64-bit ctx pointer is truncated to
    int and the engine "segfaults" for no visible reason.
  * prefill takes a COUNT. The first clip (0) is consumed as the voice-clone
    reference, not as a question. Passing one clip means you never asked
    anything -- which is why `--test` kept reporting force_listen.

    python research/minicpm_stages.py [--n 12]
"""

from __future__ import annotations

import argparse, ctypes, json, os, re, sys, time, wave
from pathlib import Path

PLAYGROUND = Path.home() / "GITHUB/VoiceLLM-Playground/Omni Model"
SHIM = next(PLAYGROUND.glob("*/native/libomni_shim.dylib"))
REF_AUDIO = next(PLAYGROUND.glob("*/assets/ref_english.wav"))
LLM_GGUF = Path(os.environ.get("OMNI_LLM_GGUF",
                Path.home() / "models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q8_0.gguf"))
MEDIA_AUDIO = 1     # audio only -- skips the vision encoder entirely
HERE = Path(__file__).resolve().parent
AUDIO = HERE / "bench_audio"
EXPECTED = ["four", "blue", "seven", "paris", "seven", "eight", "jupiter",
            "twenty five", "autumn", "twenty four", "zero", "shakespeare"]


sys.path.insert(0, str(Path(__file__).resolve().parent))
from grading import hit as scores                                  # noqa: E402


def load_shim() -> ctypes.CDLL:
    """argtypes are NOT optional: unbound, ctypes truncates the 64-bit ctx
    pointer to int and every call 'segfaults' with no explanation."""
    lib = ctypes.CDLL(str(SHIM))
    V, C, I = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int
    lib.omni_c_init.restype = V
    lib.omni_c_init.argtypes = [C, I, I, I, I, I, C]
    for fn, res, sig in [("omni_c_stream_prefill", I, [V, C, I]),
                         ("omni_c_stream_decode", I, [V, C, I]),
                         ("omni_c_pop_text", I, [V, C, I]),
                         ("omni_c_set_language", None, [V, C]),
                         ("omni_c_free", None, [V])]:
        f = getattr(lib, fn); f.restype = res; f.argtypes = sig
    return lib


def wav_seconds(p: Path) -> float:
    with wave.open(str(p), "rb") as w:
        return w.getnframes() / w.getframerate()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=str(HERE / "minicpm_stages.json"))
    ap.add_argument("--work", type=Path, default=HERE / "stage_bench_out",
                    help="fresh per-round engine artifacts; never reuse the "
                         "evidence directory for a different run")
    args = ap.parse_args()

    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    enginelog = HERE / "stage_bench_engine.log"

    # The engine is C++ and writes straight to fd 1/2, so Python-level capture
    # cannot see it. dup2 the real descriptors into a file, keep a private copy
    # of the console for our own progress lines, and parse the log afterwards
    # for STAGE C -- the engine times its own vocoder better than we can.
    console = os.fdopen(os.dup(1), "w", buffering=1)
    fd = os.open(enginelog, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1); os.dup2(fd, 2); os.close(fd)

    def say(m=""):
        print(m, file=console, flush=True)

    lib = load_shim()

    t0 = time.perf_counter()
    # n_ctx must be non-zero: at 0 the engine loads, accepts audio, and
    # silently generates nothing. duplex=0 for the simplex prefill/decode path.
    ctx = lib.omni_c_init(str(LLM_GGUF).encode(), MEDIA_AUDIO, 1, 0,
                          8192, 0, str(work).encode())
    if not ctx:
        print("omni_c_init failed", file=sys.stderr); return 1
    lib.omni_c_set_language(ctx, b"en")   # engine defaults to CHINESE prompts
    load_s = time.perf_counter() - t0
    say(f"load {load_s:.2f}s  ({LLM_GGUF.name})")

    # STAGE A(0): the reference clip. cnt=0 makes it the voice to clone, and
    # it is NOT a question -- the engine will not answer it.
    t0 = time.perf_counter()
    lib.omni_c_stream_prefill(ctx, str(REF_AUDIO).encode(), 0)
    ref_s = time.perf_counter() - t0
    say(f"ref prefill {ref_s:.2f}s  ({wav_seconds(REF_AUDIO):.1f}s audio)")

    rows, buf = [], ctypes.create_string_buffer(8192)
    for i in range(args.n):
        q = AUDIO / f"q{i:02d}.wav"
        secs = wav_seconds(q)
        round_start = time.time()
        time.sleep(0.02)                       # mtime granularity

        t0 = time.perf_counter()                       # ---- STAGE A: pre-LLM
        lib.omni_c_stream_prefill(ctx, str(q).encode(), i + 1)
        prefill_s = time.perf_counter() - t0

        t0 = time.perf_counter()                       # ---- STAGE B: the LLM
        # The final argument is the artifact round, not a boolean.  Passing
        # zero for every question silently overwrote the previous teacher
        # hidden states and left us with one training example despite a 12
        # case benchmark.  Keep every answer independently addressable.
        lib.omni_c_stream_decode(ctx, str(work).encode(), i)
        decode_s = time.perf_counter() - t0

        chunks = []
        while lib.omni_c_pop_text(ctx, buf, len(buf)):
            chunks.append(buf.value.decode(errors="replace"))
        text = " ".join(chunks).strip()

        t0 = time.perf_counter()                       # ---- STAGE C: speech
        first_wav, n_wav = None, 0
        while time.perf_counter() - t0 < 12.0:
            now = {w: w.stat().st_mtime for w in work.rglob("wav_*.wav")}
            fresh = [w for w, m in now.items() if m > round_start]
            if fresh and first_wav is None:
                first_wav = time.perf_counter() - t0
            n_wav = len(fresh)
            if any(f.name == "generation_done.flag"
                   for f in work.rglob("generation_done.flag")
                   if f.stat().st_mtime > round_start):
                break
            time.sleep(0.05)

        hit = scores(EXPECTED[i], text)
        rows.append(dict(i=i, audio_s=round(secs, 2), expected=EXPECTED[i],
                         text=text[:200], correct=hit,
                         prefill_s=round(prefill_s, 3),
                         decode_s=round(decode_s, 3),
                         first_wav_s=round(first_wav, 3) if first_wav else None,
                         n_wav=n_wav,
                         prefill_ratio=round(prefill_s / secs, 3)))
        say(f"q{i:02d} {'OK ' if hit else 'MISS'} "
            f"pre {prefill_s:.2f}s dec {decode_s:.2f}s "
            f"wav {n_wav}@{first_wav or float('nan'):.2f}s  {text[:70]!r}")

    # STAGE C from the engine's own instrumentation -- it times the vocoder
    # per chunk and reports RTF, which we cannot see from outside.
    import statistics as st
    logtxt = enginelog.read_text(errors="replace")
    first_ms = [int(m) for m in re.findall(r"First Audio Response\D*?(\d+)ms", logtxt)]
    chunks = re.findall(r"wav_\d+\.wav \| ([\d.]+)s audio \| ([\d.]+)ms inference"
                        r" \| RTF=([\d.]+)", logtxt)
    stage_c = {}
    if first_ms:
        stage_c["first_audio_ms_median"] = round(st.median(first_ms), 1)
        stage_c["first_audio_n"] = len(first_ms)
    if chunks:
        stage_c["vocoder_ms_median"] = round(st.median(float(c[1]) for c in chunks), 1)
        stage_c["rtf_median"] = round(st.median(float(c[2]) for c in chunks), 3)
        stage_c["chunk_audio_s_median"] = round(st.median(float(c[0]) for c in chunks), 3)
        stage_c["chunks_n"] = len(chunks)
    say("")
    for k, v in stage_c.items():
        say(f"  STAGE C  {k:26s} {v}")

    ok = sum(r["correct"] for r in rows)
    summary = dict(model=LLM_GGUF.name, load_s=round(load_s, 2),
                   ref_prefill_s=round(ref_s, 2), n=len(rows), correct=ok,
                   accuracy=round(ok / max(len(rows), 1), 4),
                   stage_c=stage_c, engine_log=str(enginelog), rows=rows)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    say(f"\nSTAGE B accuracy {ok}/{len(rows)} = {100*ok/max(len(rows),1):.0f}%")
    say(f"wrote {args.out}")
    try: lib.omni_c_free(ctx)
    except Exception: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
