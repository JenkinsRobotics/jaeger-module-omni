"""Capture MiniCPM speech-condition activations from an audio manifest.

This is an offline teacher-data tool. It drives the known-good native model,
keeps a distinct round directory for every clip, and records the exact answer
whose hidden/merged tensors the engine wrote. It does not score MiniCPM and it
does not participate in Jaeger's runtime.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

from minicpm_stages import (LLM_GGUF, MEDIA_AUDIO, REF_AUDIO, load_shim)

REPO = Path(__file__).resolve().parent.parent


def corpus_rows(root: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in (root / "turns.jsonl").read_text().splitlines()
            if line.strip()]
    # Stable diversity: one pass through the manifest, unique transcript only.
    seen, out = set(), []
    for row in rows:
        text = row.get("transcript", "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        audio = (root / row["audio_path"]).resolve()
        if audio.exists():
            out.append({"key": row["turn_id"], "audio": str(audio)})
        if len(out) >= limit:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=REPO / "training/corpus_real")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = corpus_rows(args.corpus, args.n)
    if not rows:
        raise SystemExit(f"no usable audio in {args.corpus}")

    # Native logs are valuable evidence but far too verbose for a terminal.
    log = args.log or args.work.with_suffix(".engine.log")
    console = os.fdopen(os.dup(1), "w", buffering=1)
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1); os.dup2(fd, 2); os.close(fd)

    lib = load_shim()
    ctx = lib.omni_c_init(str(LLM_GGUF).encode(), MEDIA_AUDIO, 1, 0,
                          8192, 0, str(args.work).encode())
    if not ctx:
        raise SystemExit("MiniCPM init failed")
    lib.omni_c_set_language(ctx, b"en")
    lib.omni_c_stream_prefill(ctx, str(REF_AUDIO).encode(), 0)
    buf = ctypes.create_string_buffer(8192)
    captured = []
    try:
        for i, row in enumerate(rows):
            t0 = time.perf_counter()
            lib.omni_c_stream_prefill(ctx, row["audio"].encode(), i + 1)
            lib.omni_c_stream_decode(ctx, str(args.work).encode(), i)
            pieces = []
            while lib.omni_c_pop_text(ctx, buf, len(buf)):
                pieces.append(buf.value.decode(errors="replace"))
            text = " ".join(pieces).strip()
            flag = args.work / f"round_{i:03d}/tts_wav/generation_done.flag"
            deadline = time.monotonic() + 20
            while not flag.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            chunks = list((args.work / f"round_{i:03d}/llm_debug").glob(
                "chunk_*/merged_embeddings.bin"))
            result = {**row, "round": i, "teacher_text": text,
                      "condition_chunks": len(chunks),
                      "seconds": round(time.perf_counter() - t0, 3)}
            captured.append(result)
            print(f"{i + 1:3d}/{len(rows)} {row['key']}  {len(chunks)} chunks  "
                  f"{result['seconds']:.2f}s  {text[:60]!r}",
                  file=console, flush=True)
            # Incremental manifest survives an interrupted long capture.
            args.manifest.write_text("".join(json.dumps(x) + "\n"
                                             for x in captured))
    finally:
        lib.omni_c_free(ctx)
    print(f"wrote {args.manifest}: {len(captured)} turns; engine log {log}",
          file=console)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
