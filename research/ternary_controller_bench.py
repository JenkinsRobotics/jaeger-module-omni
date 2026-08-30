#!/usr/bin/env python3
"""Go/no-go latency benchmark for a tiny always-on ternary audio controller.

This is an inference-kernel test, not an acoustic-quality result.  It measures
batch-one causal controller layers, because benchmarking a large batched GEMM
would answer the wrong question for a 10 ms streaming audio loop.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


THREADS = 32
PACK = 16  # sixteen two-bit ternary values per uint32


TERNARY_GEMV = mx.fast.metal_kernel(
    name="jaeger_ternary_gemv_w2a16",
    input_names=["x", "packed", "scales"],
    output_names=["out"],
    source=r"""
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_threadgroup;
        uint words = packed_shape[1];
        uint length = x_shape[0];
        float total = 0.0f;
        for (uint column = lane; column < length; column += 32) {
            uint word = packed[row * words + column / 16];
            uint code = (word >> (2 * (column % 16))) & 3;
            float weight = code == 0 ? -1.0f : (code == 2 ? 1.0f : 0.0f);
            total += float(x[column]) * weight;
        }
        total = simd_sum(total);
        if (lane == 0) {
            out[row] = T(total * float(scales[row]));
        }
    """,
    ensure_row_contiguous=True,
    compile_options={"math_mode": "fast"},
)


def pack_ternary(weight: np.ndarray) -> tuple[mx.array, mx.array, mx.array]:
    """Row-scale and pack an FP weight matrix into {-1, 0, +1}."""
    weight = np.asarray(weight, dtype=np.float32)
    scale = np.maximum(np.mean(np.abs(weight), axis=1), 1e-8).astype(np.float16)
    ternary = np.clip(np.rint(weight / scale[:, None]), -1, 1).astype(np.int8)
    rows, columns = ternary.shape
    words = (columns + PACK - 1) // PACK
    packed = np.zeros((rows, words), dtype=np.uint32)
    codes = (ternary + 1).astype(np.uint32)
    for offset in range(PACK):
        values = codes[:, offset::PACK]
        packed[:, :values.shape[1]] |= values << (2 * offset)
    reconstructed = ternary.astype(np.float16) * scale[:, None]
    return mx.array(packed), mx.array(scale), mx.array(reconstructed)


def ternary_gemv(x: mx.array, packed: mx.array, scales: mx.array,
                 output_width: int) -> mx.array:
    return TERNARY_GEMV(
        inputs=[x, packed, scales],
        template=[("T", mx.float16)],
        grid=(output_width * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(output_width,)],
        output_dtypes=[mx.float16],
    )[0]


def timed(callable_, iterations: int, warmup: int) -> dict:
    for _ in range(warmup):
        mx.eval(callable_())
    mx.synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        mx.eval(callable_())
        mx.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "mean_ms": statistics.mean(samples),
        "frames_per_second": 1000.0 / statistics.mean(samples),
    }


def benchmark(width: int, layers: int, iterations: int, warmup: int) -> dict:
    rng = np.random.default_rng(width * 100 + layers)
    raw_weights = [
        (rng.standard_normal((width, width)) / np.sqrt(width)).astype(np.float16)
        for _ in range(layers)
    ]
    fp_weights = [mx.array(value) for value in raw_weights]
    packed_layers = [pack_ternary(value) for value in raw_weights]
    ternary_fp_weights = [item[2] for item in packed_layers]

    quantized_layers = []
    for value in fp_weights:
        layer = nn.Linear(width, width, bias=False)
        layer.weight = value
        quantized_layers.append(nn.QuantizedLinear.from_linear(
            layer, group_size=32, bits=2, mode="affine"))

    seed = mx.array(rng.standard_normal(width).astype(np.float16))

    def fp16_frame():
        value = seed
        for weight in fp_weights:
            value = mx.maximum(value @ weight.T, 0)
        return value

    def ternary_emulated_frame():
        value = seed
        for weight in ternary_fp_weights:
            value = mx.maximum(value @ weight.T, 0)
        return value

    def mlx_two_bit_frame():
        value = seed
        for layer in quantized_layers:
            value = mx.maximum(layer(value), 0)
        return value

    def packed_ternary_frame():
        value = seed
        for packed, scales, _ in packed_layers:
            value = mx.maximum(
                ternary_gemv(value, packed, scales, width), 0)
        return value

    # Prove the custom kernel computes the intended ternary model before its
    # speed is allowed into the report.
    reference = ternary_emulated_frame()
    candidate = packed_ternary_frame()
    mx.eval(reference, candidate)
    max_error = float(mx.max(mx.abs(reference - candidate)))
    if max_error > 0.05:
        raise RuntimeError(f"packed ternary correctness failure: {max_error}")

    fp_bytes = layers * width * width * 2
    ternary_bytes = layers * (
        width * ((width + PACK - 1) // PACK) * 4 + width * 2)
    result = {
        "width": width,
        "layers": layers,
        "parameters": layers * width * width,
        "fp16_weight_bytes": fp_bytes,
        "packed_ternary_weight_bytes": ternary_bytes,
        "packed_weight_reduction": 1.0 - ternary_bytes / fp_bytes,
        "packed_max_abs_error_vs_fp16_ternary": max_error,
        "fp16": timed(fp16_frame, iterations, warmup),
        "ternary_emulated_fp16_matmul": timed(
            ternary_emulated_frame, iterations, warmup),
        "mlx_affine_w2a16": timed(mlx_two_bit_frame, iterations, warmup),
        "packed_ternary_w2a16_custom_metal": timed(
            packed_ternary_frame, iterations, warmup),
    }
    fp_median = result["fp16"]["median_ms"]
    for name in ("ternary_emulated_fp16_matmul", "mlx_affine_w2a16",
                 "packed_ternary_w2a16_custom_metal"):
        result[name]["speedup_vs_fp16_median"] = (
            fp_median / result[name]["median_ms"])
    return result


def aggregate(trials: list[dict]) -> dict:
    """Keep trial measurements visible and use medians for the decision."""
    first = trials[0]
    result = {
        key: first[key] for key in (
            "width", "layers", "parameters", "fp16_weight_bytes",
            "packed_ternary_weight_bytes", "packed_weight_reduction")
    }
    result["packed_max_abs_error_vs_fp16_ternary"] = max(
        trial["packed_max_abs_error_vs_fp16_ternary"] for trial in trials)
    methods = (
        "fp16", "ternary_emulated_fp16_matmul", "mlx_affine_w2a16",
        "packed_ternary_w2a16_custom_metal")
    for method in methods:
        medians = [trial[method]["median_ms"] for trial in trials]
        p95s = [trial[method]["p95_ms"] for trial in trials]
        result[method] = {
            "median_ms_across_trials": statistics.median(medians),
            "p95_ms_across_trials": statistics.median(p95s),
            "trial_medians_ms": medians,
        }
    fp16 = result["fp16"]["median_ms_across_trials"]
    for method in methods[1:]:
        result[method]["speedup_vs_fp16_median"] = (
            fp16 / result[method]["median_ms_across_trials"])
    return result


def chip_name() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--widths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--go-speedup", type=float, default=1.5,
                        help="minimum representative speedup worth QAT work")
    parser.add_argument("--out", type=Path,
                        default=Path("research/results/ternary-controller.json"))
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise SystemExit("packed ternary benchmark requires Apple Metal")
    mx.set_default_device(mx.gpu)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    trials_by_width = [
        [benchmark(width, args.layers, args.iterations, args.warmup)
         for _ in range(args.repeats)]
        for width in args.widths
    ]
    results = [aggregate(trials) for trials in trials_by_width]
    representative = min(results, key=lambda row: abs(row["width"] - 256))
    observed_speedup = representative[
        "packed_ternary_w2a16_custom_metal"]["speedup_vs_fp16_median"]
    decision = (
        "proceed_to_ternary_qat" if observed_speedup >= args.go_speedup
        else "defer_ternary_use_fp16_or_supported_quantization")
    report = {
        "kind": "jaeger_ternary_controller_kernel_benchmark",
        "quality_claim": False,
        "workload": "batch-one causal frame; one invocation per 10 ms hop",
        "machine": platform.platform(),
        "chip": chip_name(),
        "mlx_version": version("mlx"),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "decision": {
            "representative_width": representative["width"],
            "minimum_speedup": args.go_speedup,
            "observed_speedup": observed_speedup,
            "outcome": decision,
            "scope": "kernel viability only; no acoustic-quality claim",
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for row in results:
        custom = row["packed_ternary_w2a16_custom_metal"]
        builtin = row["mlx_affine_w2a16"]
        print(f"width {row['width']:4d} × {row['layers']} · "
              f"FP16 {row['fp16']['median_ms_across_trials']:.3f} ms · "
              f"MLX2 {builtin['median_ms_across_trials']:.3f} ms "
              f"({builtin['speedup_vs_fp16_median']:.2f}x) · "
              f"ternary {custom['median_ms_across_trials']:.3f} ms "
              f"({custom['speedup_vs_fp16_median']:.2f}x) · "
              f"weights -{100*row['packed_weight_reduction']:.1f}%")
    print(f"decision: {decision} ({observed_speedup:.2f}x at width "
          f"{representative['width']}; gate {args.go_speedup:.2f}x)")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
