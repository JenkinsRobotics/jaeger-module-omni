"""Compile Gemma token IDs into Qwen/Talker token IDs for tensor-only use."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gemma", default=str(
        Path.home() / "models/gemma-4-e4b-it-mlx-full"))
    ap.add_argument("--qwen", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=2048)
    args = ap.parse_args()
    gemma = AutoTokenizer.from_pretrained(args.gemma)
    qwen = AutoTokenizer.from_pretrained(args.qwen)
    special = set(gemma.all_special_ids)
    values, offsets = [], [0]
    for start in range(0, len(gemma), args.batch):
        stop = min(start + args.batch, len(gemma))
        ids = list(range(start, stop))
        texts = gemma.batch_decode([[i] for i in ids],
                                   skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        mapped = qwen(texts, add_special_tokens=False).input_ids
        for source_id, target in zip(ids, mapped):
            # Generated structural tokens are stopping conditions, never
            # lexical speech. Empty byte-fallback fragments are also omitted.
            if source_id in special:
                target = []
            values.extend(int(x) for x in target)
            offsets.append(len(values))
        if start == 0 or start % (args.batch * 16) == 0:
            print(f"compiled {stop}/{len(gemma)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, offsets=np.asarray(offsets, dtype=np.int64),
                        values=np.asarray(values, dtype=np.int32),
                        source="gemma4", target="qwen2")
    print(f"wrote {args.out}: {len(offsets)-1} source IDs -> "
          f"{len(values)} target IDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
