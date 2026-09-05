#!/usr/bin/env python3
"""Continue the Qwen-free clean-room context adapter through frozen Gemma.

This is the repository-owned trainer for ``cleanroom_context_v2``.  It keeps
Whisper and Gemma frozen, trains every deployed connector parameter, and uses
only human transcript labels: next-token CE plus same-Gemma text-equivalence
KL.  A frozen-start output anchor protects the already measured acoustic map
while a balanced multi-domain curriculum improves room robustness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jaeger_agent_omni.core.runtime.stock_whisper import (  # noqa: E402
    ARCH_CLEANROOM_CONTEXT,
    DEFAULT_GEMMA,
    OUTPUT_DIM,
    OUTPUT_MODE_RAW,
    validate_checkpoint,
)
from training.context_ce import (  # noqa: E402
    SourceSpec,
    TRANSCRIPT_SYSTEM,
    _prompt_parts,
    _tree_add,
    _tree_scale,
    audit_holdout,
    load_sources,
)
from training.context_distill import atomic_save, refuse_gpu_contention  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_adapter(values: dict[str, np.ndarray], *, causal: bool = False):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    input_dim = int(values["input_dim"])
    hidden = int(np.asarray(values["hidden"]).reshape(-1)[0])
    pool_step = int(values["pool_step"])

    class CleanroomContext(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(input_dim)
            self.up = nn.Linear(input_dim, hidden)
            self.attn = nn.MultiHeadAttention(hidden, 8)
            self.ln_a = nn.LayerNorm(hidden)
            self.ff1 = nn.Linear(hidden, hidden * 2)
            self.ff2 = nn.Linear(hidden * 2, hidden)
            self.ln_f = nn.LayerNorm(hidden)
            self.out = nn.Linear(hidden, OUTPUT_DIM)

        def __call__(self, features, return_context=False):
            projected = nn.gelu(self.up(self.ln(features)))
            usable = (projected.shape[0] // pool_step) * pool_step
            if usable:
                pooled = mx.mean(projected[:usable].reshape(
                    -1, pool_step, hidden), axis=1)
            else:
                pooled = mx.mean(projected, axis=0, keepdims=True)
            context = pooled[None, :]
            mask = None
            if causal:
                n = context.shape[1]
                mask = mx.where(mx.tril(mx.ones((n, n))) > 0, 0.0, -1e9)
            context = self.ln_a(
                context + self.attn(context, context, context, mask=mask))
            context = self.ln_f(
                context + self.ff2(nn.gelu(self.ff1(context))))
            output = self.out(context[0])
            return (output, context[0]) if return_context else output

    adapter = CleanroomContext()
    parameter_names = {name for name, _ in __import__(
        "mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
            adapter.parameters())}
    adapter.update(tree_unflatten([
        (name, mx.array(values[name])) for name in parameter_names
    ]))
    mx.eval(adapter.parameters())
    return adapter


def _checkpoint_values(adapter, base, args, step, sources, holdouts):
    from mlx.utils import tree_flatten

    values = {name: np.asarray(value) for name, value in
              tree_flatten(adapter.parameters())}
    # Explicit metadata lets the same architecture support both medium and a
    # future 1280-wide large-v3-turbo front end without implicit defaults.
    for name in ("arch", "hidden", "adapter_arch", "encoder", "input_dim",
                 "input_boundary", "lineage", "pool_step", "output_mode"):
        if name in base:
            values[name] = np.asarray(base[name])
    values.update({
        "adapter_arch": np.array(ARCH_CLEANROOM_CONTEXT),
        "output_mode": np.array(OUTPUT_MODE_RAW),
        "stage": np.array("human_ce_text_kl_full_adapter_balanced"),
        "trainer": np.array("training/cleanroom_context_ce.py"),
        "ce_step": np.array(step, dtype=np.int32),
        "learning_rate": np.array(args.learning_rate, np.float32),
        "text_teacher_weight": np.array(args.text_teacher_weight, np.float32),
        "anchor_weight": np.array(args.anchor_weight, np.float32),
        "accumulate": np.array(args.accumulate, np.int32),
        "trainable_parameters": np.array(args.trainable_parameters,
                                          dtype=np.int64),
        "ce_train_branch": np.array(args.train_branch),
        "resume_checkpoint": np.array(str(args.resume)),
        "resume_sha256": np.array(_sha256(args.resume)),
        "training_sources_json": np.array(json.dumps(sources, sort_keys=True)),
        "holdout_sources_json": np.array(json.dumps(holdouts, sort_keys=True)),
        "transcript_objective": np.array(TRANSCRIPT_SYSTEM),
    })
    if getattr(args, "causal", False):
        # Heads-free causal port: never mislabel these weights as bidirectional
        # or as the legacy causal architecture with attached perception heads.
        values["arch"] = np.frombuffer(b"context_causal_v1", dtype=np.uint8)
        values["adapter_arch"] = np.array("cleanroom_context_causal_v1")
        values["causal"] = np.array([1], dtype=np.int32)
        values["stage"] = np.array("causal_port_human_ce_text_kl_bidir_anchor")
    return values


def command_train(args) -> None:
    train_specs = [SourceSpec.parse(value) for value in args.train_source]
    holdout_specs = [SourceSpec.parse(value) for value in args.holdout_source]
    rows, probabilities, sources = load_sources(
        train_specs, args.max_seconds, args.allow_acoustic_variants)
    holdouts = audit_holdout(rows, holdout_specs, args.max_seconds)
    base = validate_checkpoint(args.resume)
    if str(base["adapter_arch"]) != ARCH_CLEANROOM_CONTEXT:
        raise ValueError("resume checkpoint is not cleanroom_context_v2")
    expected_width = int(base["input_dim"])
    widths = Counter(row["_feature_width"] for row in rows)
    if set(widths) != {expected_width}:
        raise ValueError(f"feature widths {dict(widths)} != {expected_width}")

    source_mass = {}
    for row, probability in zip(rows, probabilities):
        source_mass[row["_source_key"]] = (
            source_mass.get(row["_source_key"], 0.0) + float(probability))
    audit = {
        "mode": "dry-run-no-gpu" if args.dry_run else "training",
        "resume": str(args.resume), "resume_sha256": _sha256(args.resume),
        "architecture": str(base["adapter_arch"]),
        "encoder": str(base["encoder"]), "training_rows": len(rows),
        "feature_widths": dict(widths),
        "sampling_mass": {key: round(value, 6)
                          for key, value in source_mass.items()},
        "sources": sources, "holdouts": holdouts, "steps": args.steps,
        "learning_rate": args.learning_rate,
        "text_teacher_weight": args.text_teacher_weight,
        "anchor_weight": args.anchor_weight, "accumulate": args.accumulate,
        "allow_acoustic_variants": args.allow_acoustic_variants,
        "causal_port": args.causal,
    }
    print(json.dumps(audit, indent=2), flush=True)
    if args.dry_run:
        return

    refuse_gpu_contention()
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load
    from mlx.utils import tree_flatten

    mx.random.seed(args.seed)
    cache_limit = int(args.metal_cache_limit_gb * 1024 ** 3)
    mx.set_cache_limit(cache_limit)
    mx.reset_peak_memory()
    print(f"Metal cache ceiling: {args.metal_cache_limit_gb:.2f} GiB",
          flush=True)
    print(f"loading frozen Gemma: {args.model}", flush=True)
    backbone, tokenizer = load(str(args.model), lazy=False)
    backbone.freeze()
    backbone.eval()
    adapter = _build_adapter(base, causal=args.causal)
    if args.train_branch == "out":
        adapter.freeze()
        adapter.out.unfreeze()
    anchor = _build_adapter(base)
    anchor.freeze()
    prefix, suffix, closing = _prompt_parts(tokenizer)

    class Alignment(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.adapter = adapter
            self.backbone.freeze()

        def target_logits(self, ids, embeddings, first, count):
            lm = self.backbone.language_model
            hidden = lm.model(ids, input_embeddings=embeddings)
            hidden = hidden[:, first - 1:first - 1 + count]
            logits = (lm.model.embed_tokens.as_linear(hidden)
                      if lm.tie_word_embeddings else lm.lm_head(hidden))
            cap = lm.final_logit_softcapping
            return mx.tanh(logits / cap) * cap if cap is not None else logits

    alignment = Alignment()
    mx.eval(alignment.parameters(), anchor.parameters())
    args.trainable_parameters = sum(
        int(np.prod(value.shape)) for _, value in
        tree_flatten(adapter.trainable_parameters()))
    print(f"training {args.trainable_parameters:,} deployed adapter "
          f"parameters ({args.train_branch})", flush=True)

    probe_rows = []
    probe_sources = set()
    for row in rows:
        if row["_source_key"] not in probe_sources:
            probe_rows.append(row)
            probe_sources.add(row["_source_key"])
        if len(probe_rows) >= args.probe_examples:
            break

    def geometry(step):
        embedding = backbone.language_model.model.embed_tokens
        hits = total = 0
        diversities = []
        for row in probe_rows:
            features = mx.array(np.load(row["feature"]).astype(np.float32))
            output = alignment.adapter(features) / math.sqrt(OUTPUT_DIM)
            scores = embedding.as_linear(output)
            top = mx.argpartition(-scores, args.probe_topk, axis=-1)[
                :, :args.probe_topk]
            top1 = mx.argmax(scores, axis=-1)
            mx.eval(top, top1)
            wanted = set(tokenizer.encode(row["target"],
                                          add_special_tokens=False))
            found = {int(token) for frame in np.asarray(top) for token in frame}
            hits += len(wanted & found)
            total += len(wanted)
            values = np.asarray(top1)
            diversities.append(len(set(map(int, values))) / max(1, len(values)))
        recall = hits / max(1, total)
        diversity = float(np.mean(diversities))
        print(f"    geometry@{step}: recall@{args.probe_topk}={recall:.2%} "
              f"frame-diversity={diversity:.2%}", flush=True)
        return recall, diversity

    baseline_recall, baseline_diversity = geometry(0)
    warmup = max(1, args.warmup)
    schedule = optim.join_schedules([
        optim.linear_schedule(0.0, args.learning_rate, warmup),
        optim.cosine_decay(args.learning_rate, max(1, args.steps - warmup)),
    ], [warmup])
    optimizer = optim.AdamW(learning_rate=schedule,
                            weight_decay=args.weight_decay)

    def loss_fn(module, row):
        cached = mx.array(np.load(row["feature"]).astype(np.float32))
        current = module.adapter(cached)
        audio = current / math.sqrt(OUTPUT_DIM)
        text_ids = tokenizer.encode(
            row["target"], add_special_tokens=False)[:args.max_target_tokens]
        target_ids = text_ids + closing
        n_audio = audio.shape[0]
        ids = mx.array(prefix + [0] * n_audio + suffix + target_ids,
                       dtype=mx.int32)[None, :]
        token_embeddings = module.backbone.language_model.model.embed_tokens(ids)
        start = len(prefix)
        embeddings = mx.concatenate([
            token_embeddings[:, :start], audio[None, :],
            token_embeddings[:, start + n_audio:],
        ], axis=1)
        first = start + n_audio + len(suffix)
        logits = module.target_logits(ids, embeddings, first, len(target_ids))
        targets = mx.array(target_ids, dtype=mx.int32)[None, :]
        ce = nn.losses.cross_entropy(logits, targets, reduction="mean")

        teacher_ids = mx.array(prefix + text_ids + suffix + target_ids,
                               dtype=mx.int32)[None, :]
        teacher_first = len(prefix) + len(text_ids) + len(suffix)
        teacher_logits = mx.stop_gradient(module.target_logits(
            teacher_ids,
            module.backbone.language_model.model.embed_tokens(teacher_ids),
            teacher_first, len(target_ids)))
        log_teacher = teacher_logits - mx.logsumexp(
            teacher_logits, axis=-1, keepdims=True)
        log_student = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        text_kl = mx.mean(mx.sum(
            mx.exp(log_teacher) * (log_teacher - log_student), axis=-1))

        original = mx.stop_gradient(anchor(cached))
        count = min(current.shape[0], original.shape[0])
        delta = current[:count] - original[:count]
        absolute = mx.abs(delta)
        huber = mx.mean(mx.where(
            absolute < 1.0, 0.5 * mx.square(delta), absolute - 0.5))
        left = current[:count] / (
            mx.linalg.norm(current[:count], axis=-1, keepdims=True) + 1e-6)
        right = original[:count] / (
            mx.linalg.norm(original[:count], axis=-1, keepdims=True) + 1e-6)
        anchor_loss = huber + 0.1 * (
            1.0 - mx.mean(mx.sum(left * right, axis=-1)))
        return ce + args.text_teacher_weight * text_kl + \
            args.anchor_weight * anchor_loss

    value_and_grad = nn.value_and_grad(alignment, loss_fn)
    rng = np.random.default_rng(args.seed)
    losses = []
    started = time.time()
    for step in range(1, args.steps + 1):
        gradients = None
        value = 0.0
        for _ in range(args.accumulate):
            row = rows[int(rng.choice(len(rows), p=probabilities))]
            loss, one_grad = value_and_grad(alignment, row)
            gradients = _tree_add(gradients, one_grad)
            value += float(loss)
        value /= args.accumulate
        gradients = _tree_scale(gradients, 1.0 / args.accumulate)
        gradients, grad_norm = optim.clip_grad_norm(gradients, args.clip)
        optimizer.update(alignment, gradients)
        mx.eval(alignment.parameters(), optimizer.state)
        # Variable-length Gemma graphs otherwise let MLX's reusable buffer
        # cache grow to ~24 GB on a 32 GB host, forcing tens of GB of swap and
        # turning seconds into minutes. Trainable weights and optimizer state
        # remain live; only completed graph buffers are released.
        mx.clear_cache()
        losses.append(value)
        if step == 1 or step % args.log_every == 0:
            active_gb = mx.get_active_memory() / 1024 ** 3
            cache_gb = mx.get_cache_memory() / 1024 ** 3
            peak_gb = mx.get_peak_memory() / 1024 ** 3
            print(f"step {step:5d}/{args.steps} loss={value:.5f} "
                  f"recent={np.mean(losses[-args.log_every:]):.5f} "
                  f"grad={float(grad_norm):.4f} "
                  f"{(time.time()-started)/step:.2f}s/step "
                  f"metal={active_gb:.1f}G cache={cache_gb:.1f}G "
                  f"peak={peak_gb:.1f}G", flush=True)
        if args.probe_every and step % args.probe_every == 0:
            recall, diversity = geometry(step)
            if recall < baseline_recall * args.warn_recall_ratio:
                print(f"    WARN recall {recall:.2%} below "
                      f"{args.warn_recall_ratio:.0%} of baseline", flush=True)
            if diversity < baseline_diversity * args.abort_diversity_ratio:
                raise SystemExit(
                    f"ABORT: diversity {diversity:.2%} collapsed below "
                    f"{args.abort_diversity_ratio:.0%} of baseline")
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            path = args.output.with_name(
                f"{args.output.stem}.step{step:06d}{args.output.suffix}")
            atomic_save(path, _checkpoint_values(
                alignment.adapter, base, args, step, sources, holdouts))
            validate_checkpoint(path)
            print(f"checkpoint {path}", flush=True)

    atomic_save(args.output, _checkpoint_values(
        alignment.adapter, base, args, args.steps, sources, holdouts))
    validate_checkpoint(args.output)
    print(f"saved {args.output}; first10={np.mean(losses[:10]):.5f} "
          f"last10={np.mean(losses[-10:]):.5f}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-source", action="append", required=True,
                        metavar="CACHE::SPLIT::WEIGHT")
    parser.add_argument("--holdout-source", action="append", default=[],
                        metavar="CACHE::SPLIT::WEIGHT")
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_GEMMA)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--causal", action="store_true",
                        help="train a NEW heads-free causal port against the "
                             "frozen bidirectional anchor; encoder stays frozen")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--train-branch", choices=("all", "out"),
                        default="all")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--text-teacher-weight", type=float, default=0.5)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--max-target-tokens", type=int, default=160)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--allow-acoustic-variants", action="store_true")
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--probe-every", type=int, default=250)
    parser.add_argument("--probe-examples", type=int, default=6)
    parser.add_argument("--probe-topk", type=int, default=10)
    parser.add_argument("--warn-recall-ratio", type=float, default=0.5)
    parser.add_argument("--abort-diversity-ratio", type=float, default=0.6)
    parser.add_argument("--metal-cache-limit-gb", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    args.resume = args.resume.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    if args.steps <= 0 or args.accumulate <= 0:
        raise ValueError("steps and accumulate must be positive")
    if min(args.text_teacher_weight, args.anchor_weight) < 0:
        raise ValueError("loss weights cannot be negative")
    if args.metal_cache_limit_gb <= 0:
        raise ValueError("metal cache limit must be positive")
    command_train(args)


if __name__ == "__main__":
    main()
