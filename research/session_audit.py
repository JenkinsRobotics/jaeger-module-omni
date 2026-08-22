"""What did the room say, and how much of it reached the model?

Whisper-transcribes both recordings a `--record` session produces and diffs
them. The delta IS the lost context -- speech under the RMS threshold, audio
zeroed during playback, chunks dropped under lag.

    python research/session_audit.py --dir SESSION_DIR
"""
from __future__ import annotations

import argparse, difflib, re, sys
from pathlib import Path


def words(t): return re.findall(r"[a-z0-9']+", t.lower())


def transcribe(path, proc, model):
    import numpy as np
    import soundfile as sf
    import torch
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    out = []
    step = 16000 * 28                       # stay under whisper's 30 s window
    for i in range(0, len(audio), step):
        seg = audio[i:i + step]
        if float(np.sqrt((seg ** 2).mean())) < 1e-4:
            continue                        # pure silence segment
        f = proc(seg, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(f, max_new_tokens=220)
        out.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return " ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)

    from transformers import (WhisperForConditionalGeneration, WhisperProcessor)
    proc = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-medium.en").eval()

    raw = transcribe(d / "raw_mic.wav", proc, model)
    fed = transcribe(d / "fed_model.wav", proc, model)
    rw, fw = words(raw), words(fed)

    sm = difflib.SequenceMatcher(a=rw, b=fw)
    kept = sum(b.size for b in sm.get_matching_blocks())
    print(f"room said        {len(rw)} words")
    print(f"model received   {len(fw)} words")
    print(f"survived         {kept}/{len(rw)} = "
          f"{100 * kept / max(1, len(rw)):.1f}%\n")

    print("--- what the room said (whisper) ---")
    print(raw[:600] or "(nothing)")
    print("\n--- what the model was fed (whisper) ---")
    print(fed[:600] or "(nothing)")

    lost = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("delete", "replace") and i2 - i1 >= 3:
            lost.append(" ".join(rw[i1:i2]))
    if lost:
        print("\n--- LOST (3+ word runs the model never saw) ---")
        for chunk in lost[:12]:
            print(f"  · {chunk[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
