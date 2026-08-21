"""Score the AUDIO PATH, not the model's cleverness.

Answer-correctness conflates two things: how well the audio was understood, and
how good the LLM is. That is why our delta's 4/12 was unreadable — a 135M brain
and a possibly-fine audio path, tangled.

This measures only what reaches cognition. Every pipeline gets the same
instruction — "repeat exactly what was said" — and is scored on word error rate
against the known text. Same backbone, same prompt, only the audio path varies.

    legacy   whisper decoder -> transcript          the reference
    native   Gemma's own audio tower                zero training
    delta    whisper encoder -> trained adapter     no text bottleneck

WER is the right metric here because the task has one correct output. A
pipeline that transcribes at 3% but answers questions badly has a good audio
path and a weak brain — and that is a completely different problem to fix.

    python research/audio_fidelity.py --cases <dir> --modes legacy,native
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPEAT = ("Repeat the user's words exactly, verbatim. "
          "Output only their words, nothing else.")


def norm(t: str) -> list:
    # Cut the model's internal reasoning channel before scoring. Gemma leaks
    # "<|channel>thought\nThinking Process..." after a correct answer, and
    # those insertions dominate WER on short clips — 224% pooled, where the
    # transcribed WORDS were right. That measures a formatting quirk, not
    # audio fidelity.
    t = re.split(r"<\|channel|Thinking Process|<\|turn\|>u\b", t)[0]
    t = re.sub(r"<[^>]*>", " ", t)            # strip <turn|> and friends
    t = re.sub(r"[^a-z0-9' ]", " ", t.lower())
    words = t.split()
    # A bare trailing "thought" is the same leak surviving tag-stripping.
    while words and words[-1] in ("thought", "thoughts"):
        words.pop()
    return words


def wer(ref: list, hyp: list) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--modes", default="legacy,native")
    ap.add_argument("--model",
                    default="/Users/jonathanjenkins/models/gemma-4-e4b-it-mlx-full")
    ap.add_argument("--whisper", default="openai/whisper-medium.en")
    ap.add_argument("--whisper-device", default="mps",
                    help="HF whisper on CPU is ~5x slower than MPS and made an "
                         "earlier legacy measurement look 5x worse than it is")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.utils import load

    from jaeger_agent_omni.app import read_wav
    from jaeger_agent_omni.runtime import GemmaAudioPrefiller, StreamingSession

    cases = json.load(open(Path(args.cases) / "cases.json"))
    m, proc = load(args.model)
    cfg = json.load(open(Path(args.model) / "config.json"))
    eos = {proc.tokenizer.eos_token_id}
    results = {}

    for mode in args.modes.split(","):
        print(f"--- {mode} ---")
        pairs, times = [], []

        if mode == "legacy":
            import torch
            from transformers import (WhisperForConditionalGeneration,
                                      WhisperProcessor)
            wp = WhisperProcessor.from_pretrained(args.whisper)
            wm = WhisperForConditionalGeneration.from_pretrained(
                args.whisper).to(args.whisper_device).eval()

        for c in cases:
            audio = read_wav(Path(args.cases) / f"{c['key']}.wav")
            t0 = time.monotonic()
            if mode == "legacy":
                feats = wp(audio, sampling_rate=16000,
                           return_tensors="pt").input_features.to(
                               args.whisper_device)
                with torch.no_grad():
                    ids = wm.generate(feats)
                heard = wp.batch_decode(ids, skip_special_tokens=True)[0]
            else:
                s = StreamingSession(m, proc, GemmaAudioPrefiller(m, proc),
                                     cache_factory=KVCache, config=cfg,
                                     system=REPEAT)
                s.reset()
                s.open_turn()
                for i in range(0, len(audio), 1600):
                    s.feed_audio(audio[i:i + 1600])
                s.end_of_turn()
                heard = s.respond(max_tokens=48, eos_ids=eos)
            ms = (time.monotonic() - t0) * 1000
            e = wer(norm(c["text"]), norm(heard))
            pairs.append((c["text"], heard, e))
            times.append(ms)
            print(f"  {c['key']:4s} WER {e*100:5.1f}%  {ms:6.0f}ms  "
                  f"{' '.join(norm(heard))[:52]!r}")

        # Pooled WER: weight by reference length, so a short clip getting one
        # word wrong does not dominate the mean.
        edits = sum(e * len(norm(r)) for r, _, e in pairs)
        words = sum(len(norm(r)) for r, _, _ in pairs)
        results[mode] = {"wer": edits / words, "median_ms": statistics.median(times)}
        print(f"  pooled WER {100*edits/words:.1f}%\n")

    print(f"{'mode':10s} {'pooled WER':>11s} {'median ms':>11s}")
    for k, v in results.items():
        print(f"{k:10s} {v['wer']*100:>10.1f}% {v['median_ms']:>10.0f}")
    print("\nWER measures what reaches the LLM. Low WER + wrong answers = the "
          "audio path is fine and the brain is the problem.")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
    return 0


def demo():
    assert wer(norm("what is two plus two"), norm("What is two plus two?")) == 0.0
    assert wer(norm("a b c d"), norm("a b x d")) == 0.25
    # markup must not count as errors
    assert wer(norm("hello there"), norm("hello there<turn|><turn|>")) == 0.0
    # the reasoning leak must not count as transcription errors
    assert wer(norm("name three primary colors"),
               norm("name three primary colors<|channel>thought\nThinking Process 1")) == 0.0
    assert wer([], []) == 0.0 and wer([], ["x"]) == 1.0
    print("audio_fidelity: WER + normalisation OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
