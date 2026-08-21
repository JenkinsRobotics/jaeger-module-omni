"""Verify our Talker port from exact MiniCPM engine conditioning artifacts.

This removes the LLM and semantic projector from the experiment.  The working
MiniCPM C++ reference saved its post-projector merged embeddings for two text
chunks.  We feed those exact tensors through our PyTorch Talker port, preserve
one cache across both chunks, render through Token2Wav, and transcribe the WAV.

If this fails, the Talker port is wrong.  If it passes while a different LLM's
states fail, the missing component is the LLM-state adapter.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DEBUG = REPO / "research/stage_bench_out/round_000/llm_debug"


def load_merged(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        n_tokens, width = struct.unpack("<ii", f.read(8))
        values = np.frombuffer(f.read(), dtype="<f4").copy()
    if values.size != n_tokens * width:
        raise ValueError(f"{path}: {values.size} values != {n_tokens}x{width}")
    return torch.from_numpy(values.reshape(1, n_tokens, width))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG)
    ap.add_argument("--out", type=Path, default=REPO / "training/runs/talker_port.wav")
    ap.add_argument("--ref-audio", type=Path,
                    default=REPO / "training/samples/jfk_moon.wav")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(args.debug_dir.glob("chunk_*/merged_embeddings.bin"))
    if not paths:
        raise SystemExit(f"no merged embedding artifacts under {args.debug_dir}")
    chunks = [(load_merged(p), i == len(paths) - 1) for i, p in enumerate(paths)]

    torch.manual_seed(args.seed)
    from load_minicpm_talker import load
    talker = load()
    codes = talker.generate_code_chunks(chunks)
    print(f"talker: {codes.shape[1]} codes, {codes.unique().numel()} distinct")

    from t2w_client import synthesize
    synthesize(codes[0].tolist(), args.out, args.ref_audio)

    from jaeger_agent_omni.app import OmniApp, load_models, read_wav
    p, w, t, b = load_models("openai/whisper-tiny",
                             "HuggingFaceTB/SmolLM2-135M-Instruct", device="cpu")
    heard = OmniApp(p, w, t, b, path="text", device="cpu").transcribe(read_wav(args.out))
    expected = "Romeo and Juliet were written by William Shakespeare"
    want = set(expected.lower().split())
    got = set(heard.lower().replace(".", "").replace(",", "").split())
    overlap = len(want & got) / len(want)
    print(f"expected: {expected!r}\nheard   : {heard!r}\noverlap : {overlap:.0%}")
    return 0 if overlap >= 0.6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
