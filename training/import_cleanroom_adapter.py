#!/usr/bin/env python3
"""Import a measured Gemma-native cleanroom checkpoint into Jaeger format.

The source contains only the owned LayerNorm/MLP tensors.  This importer adds
strict runtime metadata; it does not transform, quantize, or retrain weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    with np.load(args.source, allow_pickle=False) as archive:
        source_arch = (str(archive["adapter_arch"])
                       if "adapter_arch" in archive.files
                       else "mean_pool_gelu_v1")
        if source_arch == "frame_stack_swiglu_residual_temporal2_context_v1":
            required = ("rms_pre_gamma", "stack_w1", "rms_mid_gamma",
                        "stack_w2", "residual_hidden", "res_rms_pre_gamma",
                        "res_w1", "res_rms_mid_gamma", "res_w2",
                        "temporal_hidden", "temp_rms_pre_gamma", "temp_w1",
                        "temp_dw", "temp_rms_mid_gamma", "temp_w2",
                        "temporal2_hidden", "temporal2_kernel",
                        "temp2_rms_pre_gamma", "temp2_w1", "temp2_dw",
                        "temp2_rms_mid_gamma", "temp2_w2",
                        "context_hidden", "context_heads", "context_kernel",
                        "ctx_rms_pre_gamma", "ctx_w1",
                        "ctx_rms_attn_gamma", "ctx_qkv_w", "ctx_attn_w",
                        "ctx_rms_conv_gamma", "ctx_dw",
                        "ctx_rms_mid_gamma", "ctx_w2")
        elif source_arch == "frame_stack_swiglu_residual_temporal2_v1":
            required = ("rms_pre_gamma", "stack_w1", "rms_mid_gamma",
                        "stack_w2", "residual_hidden", "res_rms_pre_gamma",
                        "res_w1", "res_rms_mid_gamma", "res_w2",
                        "temporal_hidden", "temp_rms_pre_gamma", "temp_w1",
                        "temp_dw", "temp_rms_mid_gamma", "temp_w2",
                        "temporal2_hidden", "temporal2_kernel",
                        "temp2_rms_pre_gamma", "temp2_w1", "temp2_dw",
                        "temp2_rms_mid_gamma", "temp2_w2")
        elif source_arch == "frame_stack_swiglu_residual_temporal_v1":
            required = ("rms_pre_gamma", "stack_w1", "rms_mid_gamma",
                        "stack_w2", "residual_hidden", "res_rms_pre_gamma",
                        "res_w1", "res_rms_mid_gamma", "res_w2",
                        "temporal_hidden", "temp_rms_pre_gamma", "temp_w1",
                        "temp_dw", "temp_rms_mid_gamma", "temp_w2")
        elif source_arch == "frame_stack_swiglu_residual_v1":
            required = ("rms_pre_gamma", "stack_w1", "rms_mid_gamma",
                        "stack_w2", "residual_hidden", "res_rms_pre_gamma",
                        "res_w1", "res_rms_mid_gamma", "res_w2")
        elif source_arch == "frame_stack_swiglu_v1":
            required = ("rms_pre_gamma", "stack_w1", "rms_mid_gamma",
                        "stack_w2")
        else:
            required = ("ln_gamma", "ln_beta", "w1", "b1", "w2", "b2")
        arrays = {key: archive[key] for key in required}
    if source_arch in ("frame_stack_swiglu_residual_v1",
                       "frame_stack_swiglu_residual_temporal_v1",
                       "frame_stack_swiglu_residual_temporal2_v1",
                       "frame_stack_swiglu_residual_temporal2_context_v1"):
        hidden = 2560
        residual_hidden = int(arrays["residual_hidden"])
        runtime_arch = {
            "frame_stack_swiglu_residual_v1":
                "gemma_native_stack_swiglu_residual_v1",
            "frame_stack_swiglu_residual_temporal_v1":
                "gemma_native_stack_swiglu_residual_temporal_v1",
            "frame_stack_swiglu_residual_temporal2_v1":
                "gemma_native_stack_swiglu_residual_temporal2_v1",
            "frame_stack_swiglu_residual_temporal2_context_v1":
                "gemma_native_stack_swiglu_residual_temporal2_context_v1",
        }[source_arch]
        expected = {
            "rms_pre_gamma": (2048,), "stack_w1": (5120, 2048),
            "rms_mid_gamma": (2560,), "stack_w2": (2560, 2560),
            "residual_hidden": (), "res_rms_pre_gamma": (2560,),
            "res_w1": (residual_hidden * 2, 2560),
            "res_rms_mid_gamma": (residual_hidden,),
            "res_w2": (2560, residual_hidden),
        }
        if source_arch in ("frame_stack_swiglu_residual_temporal_v1",
                           "frame_stack_swiglu_residual_temporal2_v1",
                           "frame_stack_swiglu_residual_temporal2_context_v1"):
            temporal_hidden = int(arrays["temporal_hidden"])
            expected.update({
                "temporal_hidden": (),
                "temp_rms_pre_gamma": (2560,),
                "temp_w1": (temporal_hidden * 2, 2560),
                "temp_dw": (temporal_hidden, 3, 1),
                "temp_rms_mid_gamma": (temporal_hidden,),
                "temp_w2": (2560, temporal_hidden),
            })
        if source_arch in ("frame_stack_swiglu_residual_temporal2_v1",
                           "frame_stack_swiglu_residual_temporal2_context_v1"):
            temporal2_hidden = int(arrays["temporal2_hidden"])
            temporal2_kernel = int(arrays["temporal2_kernel"])
            if temporal2_kernel <= 0 or temporal2_kernel % 2 == 0:
                raise ValueError(f"{args.source}: temporal2_kernel must be "
                                 "a positive odd value")
            expected.update({
                "temporal2_hidden": (), "temporal2_kernel": (),
                "temp2_rms_pre_gamma": (2560,),
                "temp2_w1": (temporal2_hidden * 2, 2560),
                "temp2_dw": (temporal2_hidden, temporal2_kernel, 1),
                "temp2_rms_mid_gamma": (temporal2_hidden,),
                "temp2_w2": (2560, temporal2_hidden),
            })
        if source_arch == "frame_stack_swiglu_residual_temporal2_context_v1":
            context_hidden = int(arrays["context_hidden"])
            context_heads = int(arrays["context_heads"])
            context_kernel = int(arrays["context_kernel"])
            if context_hidden <= 0 or context_heads <= 0:
                raise ValueError(f"{args.source}: invalid context shape")
            if context_hidden % context_heads:
                raise ValueError(f"{args.source}: context hidden must divide "
                                 "evenly across heads")
            if context_kernel <= 0 or context_kernel % 2 == 0:
                raise ValueError(f"{args.source}: context kernel must be a "
                                 "positive odd value")
            expected.update({
                "context_hidden": (), "context_heads": (),
                "context_kernel": (),
                "ctx_rms_pre_gamma": (2560,),
                "ctx_w1": (context_hidden * 2, 2560),
                "ctx_rms_attn_gamma": (context_hidden,),
                "ctx_qkv_w": (context_hidden * 3, context_hidden),
                "ctx_attn_w": (context_hidden, context_hidden),
                "ctx_rms_conv_gamma": (context_hidden,),
                "ctx_dw": (context_hidden, context_kernel, 1),
                "ctx_rms_mid_gamma": (context_hidden,),
                "ctx_w2": (2560, context_hidden),
            })
    elif source_arch == "frame_stack_swiglu_v1":
        hidden = 2560
        runtime_arch = "gemma_native_stack_swiglu_v1"
        expected = {
            "rms_pre_gamma": (2048,), "stack_w1": (5120, 2048),
            "rms_mid_gamma": (2560,), "stack_w2": (2560, 2560),
        }
    elif source_arch == "mean_pool_gelu_v1":
        hidden = int(arrays["w1"].shape[0])
        runtime_arch = "gemma_native_distill_v1"
        expected = {
            "ln_gamma": (1024,), "ln_beta": (1024,),
            "w1": (hidden, 1024), "b1": (hidden,),
            "w2": (2560, hidden), "b2": (2560,),
        }
    else:
        raise ValueError(f"{args.source}: unsupported adapter_arch "
                         f"{source_arch!r}")
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{args.source}: {key} is {arrays[key].shape}, "
                             f"expected {shape}")
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{args.source}: {key} contains non-finite values")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.destination, **arrays,
        adapter_arch=np.array(runtime_arch),
        hidden_dim=np.array(hidden, dtype=np.int32),
        input_boundary=np.array("stock_whisper_encoder_50hz"),
        encoder=np.array("openai/whisper-medium"),
        lineage=np.array("qwen_free"),
        teacher_lineage=np.array("google_gemma_native_audio_tower"),
        pool_step=np.array(2, dtype=np.int32),
        output_mode=np.array("raw"),
    )
    print(f"imported {args.source} -> {args.destination} "
          f"({runtime_arch=}, {hidden=}, pool_step=2, output_mode=raw)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
