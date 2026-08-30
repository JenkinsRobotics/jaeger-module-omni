"""The modality matrix — test every direction against the text anchor.

Text in/out works natively, so it is the baseline every other direction gets
checked against, one at a time:

    text  -> text     the anchor. If this fails nothing else means anything.
    audio -> text     transcribe path (works) and omni-delta path (measured)
    text  -> codes    talker, zero-conditioning — speech codes, not yet sound
    text  -> speech   STAND-IN today (say/Kokoro/browser); token2wav unported
    audio -> speech   composition; lights up when the pieces above do

    python research/matrix.py --backbone <id> [--whisper id] [--artifact dir]
                              [--talker] [--device cpu]

Each row runs independently and reports PASS / MARGINAL / SKIP / STAND-IN with
evidence. A SKIP names what is missing — the matrix is also the map of what to
build next.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = []


def row(direction: str, status: str, evidence: str):
    RESULTS.append((direction, status, evidence))
    print(f"  {direction:16s} {status:9s} {evidence}")


def synth_speech(text: str) -> np.ndarray | None:
    """Real speech via the system TTS; None where unavailable."""
    import shutil
    import subprocess
    import tempfile
    import wave

    if not shutil.which("say"):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.wav"
        subprocess.run(["say", "--data-format=LEI16@16000", "-o", str(p), text],
                       check=True, capture_output=True)
        with wave.open(str(p)) as w:
            raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--whisper", default="openai/whisper-tiny")
    ap.add_argument("--artifact", default=None,
                    help="delta artifact dir for the omni row")
    ap.add_argument("--talker", action="store_true",
                    help="run the text->codes row (loads 445M talker)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from jaeger_agent_omni.app import OmniApp, load_models

    print(f"matrix: backbone={args.backbone} whisper={args.whisper} "
          f"device={args.device}\n")
    processor, whisper, tokenizer, backbone = load_models(
        args.whisper, args.backbone, device=args.device, need_decoder=True)
    app = OmniApp(processor, whisper, tokenizer, backbone, path="text",
                  device=args.device, max_new_tokens=32)

    # -- text -> text (the anchor) ------------------------------------------
    t0 = time.monotonic()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is two plus two? One word."}],
        add_generation_prompt=True, tokenize=False)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    import torch
    with torch.no_grad():
        out = backbone.generate(ids, max_new_tokens=8,
                                pad_token_id=tokenizer.pad_token_id
                                or tokenizer.eos_token_id)
    reply = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    ok = "4" in reply or "four" in reply.lower()
    row("text -> text", "PASS" if ok else "MARGINAL",
        f"{(time.monotonic()-t0)*1000:.0f}ms  {reply.strip()[:40]!r}")

    # -- audio -> text, transcribe path -------------------------------------
    audio = synth_speech("What is two plus two?")
    if audio is None:
        row("audio -> text", "SKIP", "no system TTS to synthesise test speech")
    else:
        t0 = time.monotonic()
        transcript = app.transcribe(audio)
        got = "2" in transcript or "two" in transcript.lower()
        row("audio -> text", "PASS" if got else "MARGINAL",
            f"{(time.monotonic()-t0)*1000:.0f}ms  {transcript.strip()[:40]!r}")

    # -- audio -> text, omni delta path -------------------------------------
    if not args.artifact:
        row("audio -> omni", "SKIP",
            "no --artifact; export one from the current delta run")
    elif audio is None:
        row("audio -> omni", "SKIP", "no test speech")
    else:
        try:
            from jaeger_agent_omni.core.artifacts import load_audio_path
            enc, projector = load_audio_path(args.artifact)
            omni = OmniApp(processor, enc or whisper, tokenizer, backbone,
                           path="omni", projector=projector,
                           device=args.device, max_new_tokens=24)
            t0 = time.monotonic()
            o, _ = omni.listen(audio)
            row("audio -> omni", "MEASURED",
                f"{(time.monotonic()-t0)*1000:.0f}ms  {o.text.strip()[:40]!r} "
                f"(quality = evaluate.py's WER, not this smoke test)")
        except Exception as exc:                       # noqa: BLE001
            row("audio -> omni", "SKIP", f"{type(exc).__name__}: {str(exc)[:50]}")

    # -- text -> speech codes (talker, zero conditioning) --------------------
    if not args.talker:
        row("text -> codes", "SKIP", "pass --talker to load the 445M talker")
    else:
        try:
            from load_minicpm_talker import N_TEXT_VOCAB, load as load_talker
            from transformers import AutoTokenizer
            talker = load_talker()
            qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            qids = qtok("We choose to go to the moon.",
                        return_tensors="pt").input_ids.clamp(max=N_TEXT_VOCAB - 1)
            t0 = time.monotonic()
            codes = talker.generate_codes(talker.condition(qids), max_codes=80)
            uniq = codes.unique().numel()
            frac = (codes == codes.mode().values).float().mean().item()
            status = "MARGINAL" if (uniq < 10 or frac > 0.7) else "PASS"
            row("text -> codes", status,
                f"{(time.monotonic()-t0)*1000:.0f}ms  {uniq} distinct codes, "
                f"mode {frac*100:.0f}% — audible only after token2wav")
        except Exception as exc:                       # noqa: BLE001
            row("text -> codes", "SKIP", f"{type(exc).__name__}: {str(exc)[:50]}")

    # -- audible speech ------------------------------------------------------
    row("text -> speech", "STAND-IN",
        "say/Kokoro/browser TTS today; their token2wav (flow+hifigan) unported")
    row("audio -> speech", "BLOCKED",
        "composes audio->text/omni with text->speech once the above lands")

    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    print(f"\n{n_pass}/{len(RESULTS)} PASS · "
          f"{sum(1 for _, s, _ in RESULTS if s == 'SKIP')} skipped · "
          "the SKIP column is the build queue")
    return 0


def demo():
    RESULTS.clear()
    row("a -> b", "PASS", "x")
    row("c -> d", "SKIP", "y")
    assert RESULTS == [("a -> b", "PASS", "x"), ("c -> d", "SKIP", "y")]
    RESULTS.clear()
    print("matrix: table plumbing OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
