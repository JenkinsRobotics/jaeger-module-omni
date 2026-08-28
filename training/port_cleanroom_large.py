#!/usr/bin/env python3
"""Create a paired large-v3-turbo clean-room adapter from the medium node.

The downstream context block is copied exactly.  Only the encoder-specific
LayerNorm and 1280->hidden projection are freshly initialized and trained on
paired features from the same waveform to reproduce the medium connector's
pre-context representation.  The resulting checkpoint is an initialization,
not a promoted model; it must subsequently pass human-label CE/KL training and
the same free-running gates as medium.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jaeger_agent_omni.runtime.stock_whisper import (  # noqa: E402
    ARCH_CLEANROOM_CONTEXT,
    INPUT_BOUNDARY,
    LARGE_WHISPER,
    OUTPUT_MODE_RAW,
    validate_checkpoint,
)
from training.cleanroom_context_ce import _build_adapter  # noqa: E402
from training.context_ce import _tree_add, _tree_scale  # noqa: E402
from training.context_distill import atomic_save, refuse_gpu_contention  # noqa: E402


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(cache: Path, split: str) -> dict[str, dict]:
    root = cache / split
    manifest = root / "manifest.jsonl"
    result = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("bucket", "train") != "train":
            continue
        feature = Path(str(row["feature"])).expanduser()
        if not feature.is_absolute():
            feature = root / feature
        if not feature.is_file():
            raise FileNotFoundError(feature)
        row["_feature"] = feature
        result[str(row["id"])] = row
    return result


def _initial_large(base: dict[str, np.ndarray], seed: int):
    values = {name: np.asarray(value) for name, value in base.items()}
    hidden = int(np.asarray(base["hidden"]).reshape(-1)[0])
    rng = np.random.default_rng(seed)
    values.update({
        "adapter_arch": np.array(ARCH_CLEANROOM_CONTEXT),
        "encoder": np.array(LARGE_WHISPER),
        "input_dim": np.array(1280, dtype=np.int32),
        "input_boundary": np.array(INPUT_BOUNDARY),
        "lineage": np.array("qwen_free"),
        "output_mode": np.array(OUTPUT_MODE_RAW),
        "ln.weight": np.ones(1280, np.float32),
        "ln.bias": np.zeros(1280, np.float32),
        "up.weight": (rng.standard_normal((hidden, 1280))
                      * math.sqrt(2.0 / 1280)).astype(np.float32),
        "up.bias": np.zeros(hidden, np.float32),
    })
    return values


def _save(adapter, base, args, step, pairs):
    from mlx.utils import tree_flatten

    values = {name: np.asarray(value) for name, value in
              tree_flatten(adapter.parameters())}
    for name in ("arch", "hidden", "adapter_arch", "encoder", "input_dim",
                 "input_boundary", "lineage", "pool_step", "output_mode"):
        values[name] = np.asarray(base[name])
    values.update({
        "stage": np.array("paired_medium_to_large_input_port"),
        "trainer": np.array("training/port_cleanroom_large.py"),
        "port_step": np.array(step, np.int32),
        "source_medium_sha256": np.array(_hash(args.medium_checkpoint)),
        "paired_rows": np.array(len(pairs), np.int32),
        "learning_rate": np.array(args.learning_rate, np.float32),
    })
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--medium-checkpoint", type=Path, required=True)
    parser.add_argument("--medium-cache", type=Path)
    parser.add_argument("--large-cache", type=Path)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument(
        "--pair-source", action="append", default=[],
        metavar="MEDIUM_CACHE::LARGE_CACHE::SPLIT",
        help="repeat for paired domains stored in different cache roots")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=75)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("medium_checkpoint", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if bool(args.medium_cache) != bool(args.large_cache):
        parser.error("--medium-cache and --large-cache must be used together")
    if args.medium_cache:
        args.medium_cache = args.medium_cache.expanduser().resolve()
        args.large_cache = args.large_cache.expanduser().resolve()
    if not args.pair_source and not (
            args.medium_cache and args.large_cache and args.split):
        parser.error("provide --pair-source or both cache roots plus --split")

    medium = validate_checkpoint(args.medium_checkpoint)
    if str(medium["adapter_arch"]) != ARCH_CLEANROOM_CONTEXT:
        raise ValueError("medium checkpoint is not cleanroom_context_v2")
    pair_specs = []
    if args.medium_cache:
        pair_specs.extend((args.medium_cache, args.large_cache, split)
                          for split in args.split)
    for value in args.pair_source:
        parts = value.rsplit("::", 2)
        if len(parts) != 3:
            parser.error(f"bad --pair-source {value!r}")
        pair_specs.append((Path(parts[0]).expanduser().resolve(),
                           Path(parts[1]).expanduser().resolve(), parts[2]))

    pairs = []
    split_counts = {}
    for medium_cache, large_cache, split in pair_specs:
        medium_rows = _rows(medium_cache, split)
        large_rows = _rows(large_cache, split)
        shared = sorted(medium_rows.keys() & large_rows.keys())
        key = f"{medium_cache.name}->{large_cache.name}/{split}"
        split_counts[key] = len(shared)
        pairs.extend((medium_rows[key]["_feature"],
                      large_rows[key]["_feature"], split) for key in shared)
    if not pairs:
        raise RuntimeError("no paired feature IDs")
    # Physically verify the seam rather than trusting directory names.
    medium_width = np.load(pairs[0][0], mmap_mode="r").shape[-1]
    large_width = np.load(pairs[0][1], mmap_mode="r").shape[-1]
    print(json.dumps({
        "mode": "dry-run-no-gpu" if args.dry_run else "training",
        "medium_checkpoint": str(args.medium_checkpoint),
        "medium_sha256": _hash(args.medium_checkpoint),
        "paired_rows": len(pairs), "split_counts": split_counts,
        "medium_width": medium_width, "large_width": large_width,
        "steps": args.steps, "learning_rate": args.learning_rate,
        "trainable_boundary": "large ln + up only",
    }, indent=2), flush=True)
    if medium_width != 1024 or large_width != 1280:
        raise ValueError("paired caches do not contain medium/large features")
    if args.dry_run:
        return

    refuse_gpu_contention()
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    mx.set_cache_limit(2 * 1024 ** 3)
    mx.reset_peak_memory()
    large_values = _initial_large(medium, args.seed)
    teacher = _build_adapter(medium)
    teacher.freeze()
    candidate = _build_adapter(large_values)
    candidate.freeze()
    candidate.ln.unfreeze()
    candidate.up.unfreeze()
    trainable = sum(int(np.prod(value.shape)) for _, value in
                    tree_flatten(candidate.trainable_parameters()))
    print(f"training {trainable:,} encoder-specific parameters", flush=True)

    def projected(module, features):
        return nn.gelu(module.up(module.ln(features)))

    def loss_fn(module, pair):
        medium_path, large_path, _ = pair
        left = mx.array(np.load(medium_path).astype(np.float32))
        right = mx.array(np.load(large_path).astype(np.float32))
        target = mx.stop_gradient(projected(teacher, left))
        current = projected(module, right)
        count = min(target.shape[0], current.shape[0])
        target, current = target[:count], current[:count]
        mse = mx.mean(mx.square(current - target))
        target_n = target / (mx.linalg.norm(target, axis=-1,
                                            keepdims=True) + 1e-6)
        current_n = current / (mx.linalg.norm(current, axis=-1,
                                              keepdims=True) + 1e-6)
        cosine = 1.0 - mx.mean(mx.sum(target_n * current_n, axis=-1))
        return mse + args.cosine_weight * cosine

    value_and_grad = nn.value_and_grad(candidate, loss_fn)
    warmup = max(1, args.warmup)
    schedule = optim.join_schedules([
        optim.linear_schedule(0.0, args.learning_rate, warmup),
        optim.cosine_decay(args.learning_rate, max(1, args.steps - warmup)),
    ], [warmup])
    optimizer = optim.AdamW(schedule, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    started = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        gradients = None
        value = 0.0
        for _ in range(args.accumulate):
            pair = pairs[int(rng.integers(len(pairs)))]
            loss, one_grad = value_and_grad(candidate, pair)
            gradients = _tree_add(gradients, one_grad)
            value += float(loss)
        value /= args.accumulate
        gradients = _tree_scale(gradients, 1.0 / args.accumulate)
        gradients, norm = optim.clip_grad_norm(gradients, 1.0)
        optimizer.update(candidate, gradients)
        mx.eval(candidate.parameters(), optimizer.state)
        mx.clear_cache()
        losses.append(value)
        if step == 1 or step % 25 == 0:
            print(f"step {step:4d}/{args.steps} loss={value:.5f} "
                  f"recent={np.mean(losses[-25:]):.5f} grad={float(norm):.4f} "
                  f"{(time.time()-started)/step:.2f}s/step", flush=True)
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            path = args.output.with_name(
                f"{args.output.stem}.step{step:06d}{args.output.suffix}")
            atomic_save(path, _save(candidate, large_values, args, step, pairs))
            validate_checkpoint(path)
            print(f"checkpoint {path}", flush=True)
    atomic_save(args.output, _save(
        candidate, large_values, args, args.steps, pairs))
    validate_checkpoint(args.output)
    print(f"saved {args.output}; first10={np.mean(losses[:10]):.5f} "
          f"last10={np.mean(losses[-10:]):.5f}", flush=True)


if __name__ == "__main__":
    main()
