#!/usr/bin/env python3
"""CPU-only structural tests for the medium/large training window.

No Whisper encoder or Gemma model is loaded. The test proves checkpoint
lineage, tensor widths, runtime forward shape, and the exact parameter leaves
exposed by each optimization stage before scarce GPU time is authorized.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jaeger_agent_omni.core.runtime.stock_whisper import (  # noqa: E402
    INPUT_DIM,
    LARGE_WHISPER,
    DEFAULT_WHISPER,
    OUTPUT_DIM,
    validate_checkpoint,
)


EXPECTED = {
    "all": {
        "pre.weight", "up.weight", "mid.weight", "out.weight",
        "res_pre.weight", "res_up.weight", "res_mid.weight",
        "res_out.weight", "temp_pre.weight", "temp_up.weight",
        "temp_conv.weight", "temp_mid.weight", "temp_out.weight",
        "temp2_pre.weight", "temp2_up.weight", "temp2_conv.weight",
        "temp2_mid.weight", "temp2_out.weight", "ctx_pre.weight",
        "ctx_up.weight", "ctx_attn_norm.weight", "ctx_qkv.weight",
        "ctx_attn_out.weight", "ctx_conv_norm.weight", "ctx_conv.weight",
        "ctx_mid.weight", "ctx_out.weight",
    },
    "context": {
        "ctx_pre.weight", "ctx_up.weight", "ctx_attn_norm.weight",
        "ctx_qkv.weight", "ctx_attn_out.weight", "ctx_conv_norm.weight",
        "ctx_conv.weight", "ctx_mid.weight", "ctx_out.weight",
    },
    "input": {"pre.weight", "up.weight"},
    "input_full": {"pre.weight", "up.weight", "mid.weight", "out.weight"},
}


def leaves(tree, prefix=""):
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from leaves(value, name)
        else:
            yield name, value


def model_arrays(values):
    suffixes = ("gamma", "_w", "_w1", "_w2", "_dw")
    return {key for key in values
            if key.endswith(suffixes) or key in ("stack_w1", "stack_w2")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--medium", type=Path, required=True)
    parser.add_argument("--large", type=Path, required=True)
    args = parser.parse_args()

    # Select CPU before constructing MLX arrays. This test must remain safe
    # while another agent owns the GPU window.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    mx.set_default_device(mx.cpu)
    from training.context_ce import (
        _tree_add, _tree_scale, configure_trainable,
    )
    from training.context_distill import build_adapter, freeze_for_branch

    medium = validate_checkpoint(args.medium.expanduser().resolve())
    large = validate_checkpoint(args.large.expanduser().resolve())
    assert str(medium["encoder"]) == DEFAULT_WHISPER
    assert str(large["encoder"]) == LARGE_WHISPER
    assert int(medium.get("input_dim", INPUT_DIM)) == 1024
    assert int(large["input_dim"]) == 1280
    assert large["stack_w1"].shape[1] == 1280 * int(large["pool_step"])

    # The port may only replace the encoder-facing norm/projection. Every
    # proven downstream numerical array must remain byte-identical.
    changed_allowed = {"rms_pre_gamma", "stack_w1"}
    for key in model_arrays(medium) - changed_allowed:
        if key not in large or not np.array_equal(medium[key], large[key]):
            raise AssertionError(f"large port unexpectedly changed {key}")

    rng = np.random.default_rng(20260824)
    for name, values, width in (
            ("medium", medium, 1024), ("large", large, 1280)):
        adapter = build_adapter(values, 0, "bigru", 384)
        encoded = mx.array(rng.standard_normal((8, width)).astype(np.float32))
        output = adapter(encoded)
        mx.eval(output)
        assert tuple(output.shape) == (4, OUTPUT_DIM), tuple(output.shape)
        assert math.isfinite(float(mx.max(mx.abs(output))))
        for branch, expected in EXPECTED.items():
            candidate = build_adapter(values, 0, "bigru", 384)
            configure_trainable(candidate, branch)
            actual = {key for key, _ in leaves(
                candidate.trainable_parameters())}
            if actual != expected:
                raise AssertionError(
                    f"{name}/{branch} leaves {sorted(actual)} != "
                    f"{sorted(expected)}")
            if branch == "all":
                count = sum(int(value.size) for _, value in leaves(
                    candidate.trainable_parameters()))
                expected_count = {
                    "medium": 50_146_816,
                    "large": 52_768_768,
                }[name]
                if count != expected_count:
                    raise AssertionError(
                        f"{name}/all exposes {count:,} parameters, expected "
                        f"{expected_count:,}")

    # One real CPU backward pass proves the new large seam participates in
    # autodiff and that both exposed input tensors receive finite gradients.
    adapter = build_adapter(large, 0, "bigru", 384)
    freeze_for_branch(adapter, "input")
    encoded = mx.array(rng.standard_normal((8, 1280)).astype(np.float32))

    def loss_fn(module, features):
        return mx.mean(mx.square(module(features)))

    loss, gradients = nn.value_and_grad(adapter, loss_fn)(adapter, encoded)
    mx.eval(loss, gradients)
    gradient_leaves = dict(leaves(gradients))
    assert set(gradient_leaves) == EXPECTED["input"]
    for key, value in gradient_leaves.items():
        norm = float(mx.sqrt(mx.sum(mx.square(value))))
        if not math.isfinite(norm) or norm <= 0:
            raise AssertionError(f"invalid {key} gradient norm {norm}")

    # Context gradients include sequence containers inside MLX's PyTree. The
    # CE trainer accumulates two microbatches, so exercise the exact add/scale
    # path that previously handled dictionaries only and crashed on lists.
    context_adapter = build_adapter(medium, 0, "bigru", 384)
    freeze_for_branch(context_adapter, "context")
    context_encoded = mx.array(
        rng.standard_normal((8, 1024)).astype(np.float32))
    context_loss, context_gradients = nn.value_and_grad(
        context_adapter, loss_fn)(context_adapter, context_encoded)
    accumulated = _tree_add(context_gradients, context_gradients)
    averaged = _tree_scale(accumulated, 0.5)
    mx.eval(context_loss, averaged)
    original_leaves = tree_flatten(context_gradients)
    averaged_leaves = tree_flatten(averaged)
    assert [name for name, _ in original_leaves] == [
        name for name, _ in averaged_leaves]
    for (name, original), (_, restored) in zip(
            original_leaves, averaged_leaves):
        if not np.array_equal(np.asarray(original), np.asarray(restored)):
            raise AssertionError(
                f"gradient accumulation changed context leaf {name}")

    # The full-adapter A/B must update every deployed medium tensor, not just
    # expose it in a parameter tree. Prove all 27 leaves receive finite,
    # nonzero gradients without loading Whisper or Gemma.
    full_adapter = build_adapter(medium, 0, "bigru", 384)
    configure_trainable(full_adapter, "all")
    full_loss, full_gradients = nn.value_and_grad(
        full_adapter, loss_fn)(full_adapter, context_encoded)
    mx.eval(full_loss, full_gradients)
    full_leaves = dict(leaves(full_gradients))
    assert set(full_leaves) == EXPECTED["all"]
    for key, value in full_leaves.items():
        norm = float(mx.sqrt(mx.sum(mx.square(value))))
        if not math.isfinite(norm) or norm <= 0:
            raise AssertionError(f"invalid full-adapter {key} gradient {norm}")
    print("PASS CPU medium/large structure, freeze sets, forward, and all-50M "
          "gradient")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
