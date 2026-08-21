"""Compare the two paths on held-out audio. One number each, same clips.

    python training/evaluate.py --corpus corpus/ --backbone <id> \
        --projector training/runs/jfk_stage3_warmup.pt

The question this answers: **how much does the projector lose relative to a
transcript?**

    text path   whisper decoder -> transcript -> its own WER vs the corpus
    omni path   whisper encoder -> projector -> backbone -> WER vs the same

The corpus transcript is the reference for both, so this measures agreement
with the pipeline we already trust, not absolute correctness. Whisper's own WER
against human ground truth is measured separately — 2.75-3.33% on the JFK
speech (`dev/legacy pipeline measured.md`). That is the ceiling; this measures
the gap to it.

Evaluated on the `val` split only. Scoring on training turns reports
memorisation, and a small corpus memorises fast.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def normalize(text: str) -> list:
    """Lowercase, drop bracketed markers and punctuation. Standard WER prep."""
    text = re.sub(r"\[[^\]]*\]", " ", text.lower())
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return text.split()


def wer(ref: list, hyp: list) -> float:
    """Levenshtein over words, normalised by reference length."""
    if not ref:
        return 1.0 if hyp else 0.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def corpus_wer(pairs: list) -> tuple:
    """Aggregate WER over (reference, hypothesis) pairs.

    Pooled, not averaged per utterance: a one-word clip getting one word wrong
    is 100% WER and would dominate a mean. Pooling weights by length, which is
    what the standard metric does.
    """
    edits = words = 0
    for ref_text, hyp_text in pairs:
        ref, hyp = normalize(ref_text), normalize(hyp_text)
        edits += wer(ref, hyp) * len(ref)
        words += len(ref)
    return (edits / words if words else 1.0), words


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--projector", required=True)
    ap.add_argument("--whisper", default="openai/whisper-medium")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    from jaeger_agent_omni.app import OmniApp, load_models
    from jaeger_agent_omni.capture import CaptureReader
    from jaeger_agent_omni.neural import JaegerAudioProjector
    from dataset import CaptureDataset

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    processor, whisper, tokenizer, backbone = load_models(
        args.whisper, args.backbone, device=device, need_decoder=True)

    if Path(args.projector).is_dir():
        # Artifact directory. For a delta artifact this also swaps in THEIR
        # fine-tuned encoder — the pairing the delta was trained against;
        # evaluating on stock whisper would silently break it.
        from jaeger_agent_omni.artifacts import load_audio_path
        enc, projector = load_audio_path(args.projector)
        if enc is not None:
            class _W:                      # transcription still needs the
                pass                       # full whisper; omni path needs
            omni_whisper = enc             # their encoder only
        else:
            omni_whisper = whisper
        print(f"artifact: {args.projector}")
    else:
        ck = torch.load(args.projector, map_location=device, weights_only=False)
        projector = JaegerAudioProjector(audio_dim=whisper.config.d_model,
                                         cognition_dim=ck["cognition_dim"],
                                         embed_norm=ck.get("embed_norm") or None)
        if not ck.get("chat_wrapped", False):
            print("NOTE this checkpoint predates chat-template wrapping.")
        projector.load_state_dict(ck["projector"])
        omni_whisper = whisper
        print(f"projector: step {ck['step']}, dim {ck['cognition_dim']}, "
              f"embed_norm {ck.get('embed_norm', 0):.2f}")

    # Held-out turns only — the same split the run trained against.
    val = CaptureDataset(args.corpus, processor, tokenizer, split="val",
                         min_confidence=0.0)
    turns = val.turns[: args.limit]
    reader = CaptureReader(args.corpus)
    print(f"evaluating {len(turns)} held-out turns\n")

    text_app = OmniApp(processor, whisper, tokenizer, backbone, path="text",
                       device=device, max_new_tokens=48)
    omni_app = OmniApp(processor, omni_whisper, tokenizer, backbone, path="omni",
                       projector=projector, device=device, max_new_tokens=48)
    # Evaluate under the prompt the artifact was trained with, read from its
    # config — a train/eval prompt mismatch reads as a worse model. None on
    # legacy artifacts means the old assistant default, which OmniApp already
    # uses.
    import json as _json
    from jaeger_agent_omni.prompt import ChatWrapper
    if Path(args.projector).is_dir():
        _cfg = _json.loads((Path(args.projector) / "adapter_config.json").read_text())
        trained_prompt = _cfg.get("training", {}).get("system_prompt")
        if trained_prompt:
            omni_app.wrapper = ChatWrapper(tokenizer, backbone,
                                           system=trained_prompt, device=device)
            print(f"eval prompt: from artifact ({trained_prompt[:40]!r}...)")

    text_pairs, omni_pairs = [], []
    for i, turn in enumerate(turns):
        audio = reader.audio(turn)
        text_pairs.append((turn.transcript, text_app.transcribe(audio)))
        omni_out, _ = omni_app.listen(audio)
        omni_pairs.append((turn.transcript, omni_out.text))
        if i < 3:
            print(f"  ref : {turn.transcript[:64]}")
            print(f"  text: {text_pairs[-1][1][:64]}")
            print(f"  omni: {omni_pairs[-1][1][:64]}\n")

    text_score, n_words = corpus_wer(text_pairs)
    omni_score, _ = corpus_wer(omni_pairs)
    print(f"{n_words} reference words over {len(turns)} held-out turns")
    print(f"  text path   WER {text_score*100:6.2f}%")
    print(f"  omni path   WER {omni_score*100:6.2f}%")
    print(f"  gap         {(omni_score-text_score)*100:+6.2f} points")
    return 0


def demo():
    # Identical strings are free of error; disjoint ones are total.
    assert wer(normalize("hello world"), normalize("hello world")) == 0.0
    assert wer(normalize("a b c"), normalize("x y z")) == 1.0

    # Punctuation, case and [APPLAUSE] markers must not count as errors —
    # otherwise a correct transcript scores badly for cosmetic reasons.
    assert wer(normalize("Hello, World!"), normalize("[APPLAUSE] hello world")) == 0.0

    # One substitution in four words.
    assert abs(wer(normalize("a b c d"), normalize("a b x d")) - 0.25) < 1e-9

    # Insertions and deletions both count.
    assert abs(wer(normalize("a b c"), normalize("a b")) - 1/3) < 1e-9
    assert abs(wer(normalize("a b"), normalize("a b c")) - 0.5) < 1e-9

    # Pooling weights by length. Averaging per-utterance would let a one-word
    # clip getting one word wrong (100%) outweigh a 50-word clip at 2%.
    pairs = [("one", "wrong"), (" ".join(["w"] * 50), " ".join(["w"] * 50))]
    pooled, words = corpus_wer(pairs)
    assert words == 51
    assert abs(pooled - 1/51) < 1e-9, pooled
    naive = sum(wer(normalize(r), normalize(h)) for r, h in pairs) / len(pairs)
    assert naive > pooled * 20, "pooling must not degenerate to the mean"

    # Empty reference is fully wrong only if something was said.
    assert wer([], []) == 0.0 and wer([], ["x"]) == 1.0

    print("evaluate: WER, normalisation, pooling OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
