#!/usr/bin/env python3
"""Make an isolated post-training ternary copy of a Jaeger 41M adapter.

This is a quality gate for easy conversion, not quantization-aware training.
All learned rank-2 ``*.weight`` matrices use per-output-row absmean scaling
and {-1, 0, +1} codes. Norms, biases, metadata, and the source checkpoint are
left untouched. The output stores reconstructed floats so the existing runtime
can exercise the exact real pipeline; packed-runtime latency is a later gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ternarize_rows(value: np.ndarray, scheme: str
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(value, dtype=np.float32)
    scale = np.maximum(np.mean(np.abs(source), axis=1), 1e-12)
    if scheme == "absmean":
        codes = np.clip(
            np.rint(source / scale[:, None]), -1, 1).astype(np.int8)
    else:
        # Lloyd-style coordinate descent for the row-wise least-squares
        # ternary approximation. Given a scale, the optimal boundary is
        # scale/2; given assignments, the optimal scale is mean |w| over the
        # non-zero set. Eight iterations are stable for these adapter rows.
        magnitude = np.abs(source)
        for _ in range(8):
            active = magnitude > scale[:, None] / 2
            scale = (np.sum(magnitude * active, axis=1) /
                     np.maximum(np.sum(active, axis=1), 1))
        codes = (np.sign(source) *
                 (magnitude > scale[:, None] / 2)).astype(np.int8)
    reconstructed = codes.astype(np.float32) * scale[:, None]
    return reconstructed, codes, scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scheme", choices=("absmean", "least-squares"),
                        default="least-squares")
    parser.add_argument("--include", default=r".*",
                        help="regex selecting matrix names to ternarize")
    parser.add_argument("--exclude", default=r"$^",
                        help="regex excluding matrix names after --include")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output:
        parser.error("--output must not replace the source checkpoint")

    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}

    eligible = {
        key: value for key, value in arrays.items()
        if key.endswith(".weight") and value.ndim == 2 and
        value.dtype.kind == "f"
    }
    total_matrix_parameters = sum(value.size for value in eligible.values())
    rows = []
    quantized_parameters = 0
    fp32_bytes = packed_bytes = 0
    for key, value in list(arrays.items()):
        if not (key.endswith(".weight") and value.ndim == 2 and
                value.dtype.kind == "f" and re.search(args.include, key) and
                not re.search(args.exclude, key)):
            continue
        reconstructed, codes, scale = ternarize_rows(value, args.scheme)
        arrays[key] = reconstructed
        quantized_parameters += value.size
        fp32_bytes += value.size * 4
        packed_bytes += (value.size * 2 + 7) // 8 + scale.size * 2
        source32 = np.asarray(value, dtype=np.float32)
        denominator = float(np.linalg.norm(source32))
        cosine_denominator = denominator * float(np.linalg.norm(reconstructed))
        cosine = (float(np.sum(source32 * reconstructed)) /
                  max(cosine_denominator, 1e-12))
        rows.append({
            "tensor": key,
            "shape": list(value.shape),
            "parameters": int(value.size),
            "zero_fraction": float(np.mean(codes == 0)),
            "cosine": cosine,
            "relative_l2": float(np.linalg.norm(source32 - reconstructed) /
                                 max(denominator, 1e-12)),
            "scale_min": float(scale.min()),
            "scale_max": float(scale.max()),
        })

    if not rows:
        raise RuntimeError("no rank-2 *.weight matrices found")
    # Compression here is safe and convenient for the temporary fake-quant
    # artifact. It expands to float32 at runtime; only the theoretical packed
    # byte count describes a future packed checkpoint/runtime.
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    report = {
        "kind": "jaegeromni_41m_post_training_ternary_conversion",
        "quality_claim": False,
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "scheme": f"per_output_row_{args.scheme}_ternary",
        "include": args.include,
        "exclude": args.exclude,
        "scope": "rank-2 learned *.weight matrices; norms and biases retained",
        "runtime_storage_warning": (
            "fake-quant output expands reconstructed weights to float32; "
            "packed size is theoretical until a packed runtime exists"),
        "quantized_parameters": quantized_parameters,
        "total_matrix_parameters": total_matrix_parameters,
        "quantized_matrix_fraction": (
            quantized_parameters / total_matrix_parameters),
        "source_matrix_bytes_fp32": fp32_bytes,
        "theoretical_packed_matrix_bytes_w2_scale_fp16": packed_bytes,
        "theoretical_reduction_vs_fp32": 1.0 - packed_bytes / fp32_bytes,
        "theoretical_reduction_vs_fp16": 1.0 - packed_bytes / (fp32_bytes / 2),
        "theoretical_whole_matrix_bytes_mixed_fp16_w2": (
            packed_bytes + (total_matrix_parameters - quantized_parameters) * 2),
        "theoretical_whole_matrix_reduction_vs_fp16": 1.0 - (
            packed_bytes + (total_matrix_parameters - quantized_parameters) * 2
        ) / (total_matrix_parameters * 2),
        "tensors": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"ternarized {quantized_parameters:,} parameters in {len(rows)} tensors")
    print(f"matrix storage if packed: {fp32_bytes / 2**20:.2f} MiB FP32 -> "
          f"{packed_bytes / 2**20:.2f} MiB "
          f"(-{100 * report['theoretical_reduction_vs_fp32']:.1f}%)")
    print(f"fake-quant checkpoint: {output} ({output.stat().st_size / 2**20:.2f} MiB)")
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
