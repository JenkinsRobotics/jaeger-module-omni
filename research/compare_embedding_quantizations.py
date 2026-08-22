"""Compare dequantized Gemma embedding rows from MLX 4-bit and GGUF Q4_K_M."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_GGUF = (Path.home() / ".lmstudio/models/lmstudio-community/"
                "gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf")
DEFAULT_MLX = Path.home() / "models/gemma-4-E4B-it-4bit-mlx/model.safetensors"


def stats(x: np.ndarray) -> dict:
    return {"min": float(x.min()), "p0_1": float(np.percentile(x, 0.1)),
            "p1": float(np.percentile(x, 1)),
            "p5": float(np.percentile(x, 5)),
            "median": float(np.median(x)), "mean": float(x.mean()),
            "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)),
            "p99_9": float(np.percentile(x, 99.9)), "max": float(x.max())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--mlx", type=Path, default=DEFAULT_MLX)
    ap.add_argument("--chunk-rows", type=int, default=1024)
    ap.add_argument("--out", type=Path,
                    default=Path("research/embedding_quantization_comparison.json"))
    args = ap.parse_args()

    import mlx.core as mx
    from gguf import GGUFReader
    from gguf.quants import dequantize
    from safetensors import safe_open

    reader = GGUFReader(str(args.gguf), "r")
    gguf = next(t for t in reader.tensors if t.name == "token_embd.weight")
    with safe_open(str(args.mlx), framework="pt") as f:
        packed = f.get_tensor("language_model.model.embed_tokens.weight")
        scales = f.get_tensor("language_model.model.embed_tokens.scales")
        biases = f.get_tensor("language_model.model.embed_tokens.biases")

    n_rows = int(packed.shape[0])
    width = int(scales.shape[1] * 64)
    if tuple(gguf.shape) != (width, n_rows):
        raise ValueError(f"GGUF shape {tuple(gguf.shape)} != ({width}, {n_rows})")
    cosine = np.empty(n_rows, dtype=np.float32)
    norm_ratio_mlx_over_gguf = np.empty(n_rows, dtype=np.float32)
    relative_l2 = np.empty(n_rows, dtype=np.float32)

    for start in range(0, n_rows, args.chunk_rows):
        end = min(n_rows, start + args.chunk_rows)
        m = mx.dequantize(
            mx.array(packed[start:end].numpy()),
            mx.array(scales[start:end].float().numpy()),
            mx.array(biases[start:end].float().numpy()),
            group_size=64, bits=4)
        m = np.array(m, dtype=np.float32)
        g = dequantize(gguf.data[start:end], gguf.tensor_type).astype(
            np.float32, copy=False)
        mn = np.linalg.norm(m, axis=1)
        gn = np.linalg.norm(g, axis=1)
        cosine[start:end] = np.einsum("ij,ij->i", m, g) / np.maximum(mn * gn, 1e-12)
        norm_ratio_mlx_over_gguf[start:end] = mn / np.maximum(gn, 1e-12)
        relative_l2[start:end] = np.linalg.norm(m - g, axis=1) / np.maximum(gn, 1e-12)
        if start == 0 or end == n_rows or end % (32 * args.chunk_rows) == 0:
            print(f"compared {end}/{n_rows} rows", flush=True)

    report = {
        "format": "gemma-embedding-quantization-comparison-v1",
        "mlx": str(args.mlx.resolve()), "gguf": str(args.gguf.resolve()),
        "rows": n_rows, "width": width,
        "cosine": stats(cosine),
        "cosine_fraction_gt_0_999": float((cosine > 0.999).mean()),
        "cosine_fraction_gt_0_99": float((cosine > 0.99).mean()),
        "norm_ratio_mlx_over_gguf": stats(norm_ratio_mlx_over_gguf),
        "relative_l2_mlx_vs_gguf": stats(relative_l2),
        "per_row": str(args.out.with_suffix(".npz").resolve()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix(".npz"), cosine=cosine,
                        norm_ratio_mlx_over_gguf=norm_ratio_mlx_over_gguf,
                        relative_l2_mlx_vs_gguf=relative_l2)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
