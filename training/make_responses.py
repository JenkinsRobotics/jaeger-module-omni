"""Give the 549 LibriSpeech turns responses, so Stage 4 has real data.

Stage 4 teaches the audio path to reach the same conclusion the TEXT path
reaches. Its target is `response`, and only the 300 synthetic turns have one --
LibriSpeech ships transcripts, not answers. Training Stage 4 on 300 turns of
15 fixed prompts read by 5 `say` voices would learn those 15 prompts, not
speech.

The fix needs no new data: run the backbone on each transcript and keep what
it says. That is exactly the teacher Stage 4 distils against, so the labels
are consistent with the distillation target by construction rather than by
luck.

    python training/make_responses.py --corpus training/corpus_mixed \
        --backbone google/gemma-4-E4B-it

Writes turns.jsonl in place (atomically) with `response` filled in and
`meta.response_source = "backbone"` so generated targets stay distinguishable
from authored ones -- a distinction that matters the moment someone reports
an accuracy number and has to say what it was measured against.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SYSTEM = "You are a helpful assistant. Answer in one short sentence."


def generate(backbone_id: str, transcripts: list, device: str = "mps",
             max_new_tokens: int = 32, verbose: bool = True) -> list:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(backbone_id)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    m = AutoModelForCausalLM.from_pretrained(backbone_id, dtype=dtype)
    m = m.to(device).eval()

    out = []
    for i, t in enumerate(transcripts):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": t}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=False)
        ids = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            # input_ids, NOT inputs_embeds: Gemma E4B builds per_layer_inputs
            # from the ids itself. Passing embeds sends it into the 29 GiB
            # vocab-recovery branch (see train.py::_per_layer).
            gen = m.generate(**ids, max_new_tokens=max_new_tokens,
                             do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        reply = tok.decode(gen[0][ids["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()
        out.append(reply)
        if verbose and (i % 25 == 0 or i == len(transcripts) - 1):
            print(f"  {i+1}/{len(transcripts)}  {t[:38]!r} -> {reply[:44]!r}",
                  flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--backbone", default="google/gemma-4-E4B-it")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    path = Path(a.corpus) / "turns.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    todo = [i for i, r in enumerate(rows) if not (r.get("response") or "").strip()]
    if a.limit:
        todo = todo[:a.limit]
    if not todo:
        print("every turn already has a response — nothing to do")
        return 0
    print(f"{len(todo)} of {len(rows)} turns need responses")

    replies = generate(a.backbone, [rows[i]["transcript"] for i in todo],
                       device=a.device)
    for i, reply in zip(todo, replies):
        rows[i]["response"] = reply
        meta = rows[i].setdefault("meta", {})
        # Generated targets must stay distinguishable from authored ones.
        meta["response_source"] = "backbone"
        meta["response_backbone"] = a.backbone

    # Atomic: a half-written corpus is worse than none, and this file is the
    # only copy of the alignment between audio and targets.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)

    have = sum(1 for r in rows if (r.get("response") or "").strip())
    print(f"\n{have}/{len(rows)} turns now carry a response -> {path}")
    return 0


def demo() -> None:
    """Round-trip without a model: the file surgery is the risky part."""
    import tempfile as tf

    with tf.TemporaryDirectory() as td:
        p = Path(td) / "turns.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in [
            {"transcript": "a", "response": "kept"},
            {"transcript": "b", "response": ""},
            {"transcript": "c"},
        ]) + "\n")
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        todo = [i for i, r in enumerate(rows)
                if not (r.get("response") or "").strip()]
        assert todo == [1, 2], todo
        for i, rep in zip(todo, ["B", "C"]):
            rows[i]["response"] = rep
            rows[i].setdefault("meta", {})["response_source"] = "backbone"
        fd, tmp = tempfile.mkstemp(dir=td, suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, p)
        back = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        assert back[0]["response"] == "kept", "authored responses must survive"
        assert back[1]["response"] == "B" and back[2]["response"] == "C"
        assert "response_source" not in back[0].get("meta", {})
        assert back[1]["meta"]["response_source"] == "backbone"
    print("make_responses: fills only empties, tags source, atomic write OK")


if __name__ == "__main__":
    raise SystemExit(demo() if "--demo" in sys.argv else main())
