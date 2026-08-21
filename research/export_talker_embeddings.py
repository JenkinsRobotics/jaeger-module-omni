"""Export only MiniCPM Talker's lexical table from its GGUF.

Run with the VoiceLLM environment, which carries the GGUF reader. This avoids
loading the entire 500M-parameter Talker merely to fit the lexical bridge.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=(
        Path.home() / "models/MiniCPM-o-4_5-gguf/tts/"
        "MiniCPM-o-4_5-tts-F16.gguf"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    reader = GGUFReader(str(args.gguf), "r")
    tensor = next((t for t in reader.tensors if t.name == "emb_text.weight"), None)
    if tensor is None:
        raise SystemExit("emb_text.weight not found")
    table = dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
    if table.shape != (152064, 768):
        raise ValueError(f"unexpected lexical table {table.shape}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, table)
    print(f"wrote {args.out} {table.shape} {table.nbytes / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
