#!/usr/bin/env python3
"""Truth-supervise Jaeger's owned contextual connector through frozen Gemma.

This is the post-distillation stage.  It resumes the measured 50M contextual
runtime checkpoint, trains only its context branch, and combines three signals:

* human-transcript cross entropy through frozen Gemma;
* text-equivalence KL from the same frozen Gemma reading that transcript;
* a frozen-start output anchor, protecting broad acoustic behavior while CE
  learns past the native audio-tower teacher's WER ceiling.

The Whisper encoder and Gemma remain frozen.  Human text is a training label;
it is never handed to Gemma by the deployed audio runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jaeger_agent_omni.core.runtime.stock_whisper import (  # noqa: E402
    AUDIO_MARKER,
    ARCH_GEMMA_NATIVE_CONTEXT,
    DEFAULT_GEMMA,
    INPUT_DIM,
    OUTPUT_DIM,
    validate_checkpoint,
)
from training.context_distill import (  # noqa: E402
    adapter_values,
    atomic_save,
    build_adapter,
    freeze_for_branch,
    refuse_gpu_contention,
)

TRANSCRIPT_SYSTEM = (
    "Repeat the spoken words exactly. Output only what was said, with no "
    "commentary or explanation."
)


def configure_trainable(adapter, branch: str) -> None:
    """Select the deployed adapter weights optimized by this run.

    ``all`` deliberately excludes the training-only lexical probe. Every
    parameter used by the runtime remains trainable.
    """
    if branch == "all":
        adapter.lexical_head.freeze()
        return
    freeze_for_branch(adapter, branch)


@dataclass(frozen=True)
class SourceSpec:
    cache: Path
    split: str
    weight: float

    @classmethod
    def parse(cls, value: str) -> "SourceSpec":
        parts = value.rsplit("::", 2)
        if len(parts) != 3:
            raise ValueError(
                f"source must be CACHE::SPLIT::WEIGHT, got {value!r}")
        cache, split, weight = parts
        result = cls(Path(cache).expanduser().resolve(), split, float(weight))
        if result.weight <= 0:
            raise ValueError(f"source weight must be positive: {value!r}")
        return result


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _manifest_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    return _manifest_hash(path)


def _license_allowed(value: str) -> bool:
    normalized = str(value or "").upper().replace("_", "-")
    blocked = ("NONCOMMERCIAL", "NON-COMMERCIAL", "CC-BY-NC", "RESEARCH")
    return not any(marker in normalized for marker in blocked)


def load_source(spec: SourceSpec, max_seconds: float, *,
                training: bool = True) -> tuple[list[dict], dict]:
    root = spec.cache / spec.split
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    rows = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # Merged imported corpora can retain their original validation rows.
        # A training source must never silently absorb those rows.
        if training and row.get("bucket", "train") != "train":
            continue
        text = str(row.get("transcript") or row.get("text") or "").strip()
        if not text:
            continue
        seconds = float(row.get("seconds", row.get("duration_s", 0.0)))
        if max_seconds and seconds > max_seconds:
            continue
        confidence = row.get("confidence")
        if row.get("human_label") is False or (
                confidence is not None and float(confidence) < 1.0):
            continue
        if not _license_allowed(str(row.get("license") or
                                    (row.get("meta") or {}).get("license") or "")):
            raise ValueError(f"non-shippable license in {manifest}: {row.get('license')}")
        feature = _resolve(root, str(row.get("feature") or ""))
        if not feature.is_file():
            raise FileNotFoundError(f"{manifest}: missing feature for {row.get('id')}")
        row.update({
            "target": text,
            "feature": feature,
            "seconds": seconds,
            "_source_key": f"{spec.cache}/{spec.split}",
            "_source_weight": spec.weight,
            "_feature_width": (int(row["input_dim"])
                               if row.get("input_dim") is not None else None),
        })
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no human-labelled rows in {manifest}")
    # Modern manifests declare every width. Imported clean-room manifests
    # predate that field, so physically inspect one tensor and apply the
    # source-level width after still verifying that every file exists.
    declared = {row["_feature_width"] for row in rows
                if row["_feature_width"] is not None}
    sample_shape = np.load(rows[0]["feature"], mmap_mode="r").shape
    if len(sample_shape) != 2 or sample_shape[0] < 1:
        raise ValueError(
            f"{manifest}: bad sample feature shape {sample_shape}")
    if declared and declared != {int(sample_shape[1])}:
        raise ValueError(
            f"{manifest}: declared widths {declared} != sample {sample_shape}")
    for row in rows:
        if row["_feature_width"] is None:
            row["_feature_width"] = int(sample_shape[1])
    return rows, {
        "cache": str(spec.cache), "split": spec.split, "weight": spec.weight,
        "rows": len(rows), "manifest": str(manifest),
        "manifest_sha256": _manifest_hash(manifest),
    }


def _canonical_label(value: str) -> str:
    return re.sub(r"[^a-z0-9']+", " ", value.lower()).strip()


def load_sources(specs: list[SourceSpec], max_seconds: float,
                 allow_acoustic_variants: bool = False):
    rows, metadata = [], []
    seen: dict[str, dict] = {}
    seen_features = set()
    acoustic_variant_rows = 0
    for spec in specs:
        source_rows, source_meta = load_source(spec, max_seconds)
        for row in source_rows:
            item_id = str(row["id"])
            feature = str(row["feature"])
            if feature in seen_features:
                raise ValueError(f"duplicate training feature path: {feature}")
            seen_features.add(feature)
            if item_id in seen:
                if not allow_acoustic_variants:
                    raise ValueError(
                        f"duplicate training ID across sources: {item_id}")
                if _canonical_label(row["target"]) != _canonical_label(
                        seen[item_id]["target"]):
                    raise ValueError(
                        f"conflicting labels for acoustic variant {item_id}")
                acoustic_variant_rows += 1
            else:
                seen[item_id] = row
            rows.append(row)
        metadata.append(source_meta)
    # First normalize within each declared source, then apply the requested
    # mass. Dataset size cannot silently override the explicit mixture.
    counts = Counter(row["_source_key"] for row in rows)
    total_weight = sum(spec.weight for spec in specs)
    by_key = {f"{spec.cache}/{spec.split}": spec.weight / total_weight
              for spec in specs}
    probabilities = np.array([
        by_key[row["_source_key"]] / counts[row["_source_key"]]
        for row in rows
    ], dtype=np.float64)
    probabilities /= probabilities.sum()
    for item in metadata:
        item["allow_acoustic_variants"] = allow_acoustic_variants
    if metadata:
        metadata[0]["acoustic_variant_extra_rows_total"] = \
            acoustic_variant_rows
    return rows, probabilities, metadata


def audit_holdout(train_rows: list[dict], specs: list[SourceSpec],
                  max_seconds: float) -> list[dict]:
    def speaker_key(row: dict) -> tuple[str, str] | None:
        speaker = str(row.get("speaker") or "").strip()
        if not speaker:
            return None
        # Speaker labels are dataset-local (LibriSpeech and VoxPopuli both
        # use small numeric IDs). Comparing the bare value creates false
        # leakage reports across unrelated corpora. Acoustic augmentation
        # retains the source speaker, so normalize only that lineage prefix.
        lineage = str(row.get("source") or row.get("dataset") or "").strip()
        while lineage.startswith("augmented:"):
            lineage = lineage.removeprefix("augmented:")
        return lineage, speaker

    train_ids = {str(row["id"]) for row in train_rows}
    train_speakers = {key for row in train_rows
                      if (key := speaker_key(row)) is not None}
    metadata = []
    for spec in specs:
        rows, meta = load_source(spec, max_seconds, training=False)
        overlap = train_ids & {str(row["id"]) for row in rows}
        if overlap:
            raise ValueError(
                f"training/holdout ID overlap in {spec.cache}/{spec.split}: "
                f"{sorted(overlap)[:3]}")
        speakers = {key for row in rows
                    if (key := speaker_key(row)) is not None}
        meta["speaker_overlap"] = len(train_speakers & speakers)
        meta["speakers"] = len(speakers)
        metadata.append(meta)
    return metadata


def _prompt_parts(tokenizer):
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": TRANSCRIPT_SYSTEM},
         {"role": "user", "content": AUDIO_MARKER}],
        tokenize=False, add_generation_prompt=True)
    prefix, suffix = rendered.split(AUDIO_MARKER, 1)
    return (tokenizer.encode(prefix, add_special_tokens=False),
            tokenizer.encode(suffix, add_special_tokens=False),
            tokenizer.encode("<turn|>\n", add_special_tokens=False))


def _tree_add(left, right):
    if left is None:
        return right
    from mlx.utils import tree_map
    return tree_map(lambda value, other: value + other, left, right)


def _tree_scale(tree, scale):
    from mlx.utils import tree_map
    return tree_map(lambda value: value * scale, tree)


def checkpoint_values(adapter, base, step: int, args, sources, holdouts):
    values = adapter_values(adapter, base, step, args)
    # This stage is not another native-distillation continuation. Preserve
    # those counters and give truth supervision its own auditable counter.
    for key in ("step", "context_step"):
        if key in base:
            values[key] = np.array(base[key])
    values.update({
        "stage": np.array("human_ce_text_kl_full_adapter" if
                          args.train_branch == "all" else
                          "human_ce_text_kl_with_frozen_start_anchor"),
        "trainer": np.array("training/context_ce.py"),
        "ce_step": np.array(step, dtype=np.int32),
        f"{args.train_branch}_ce_step": np.array(step, dtype=np.int32),
        "ce_train_branch": np.array(args.train_branch),
        "text_teacher_weight": np.array(args.text_teacher_weight, np.float32),
        "anchor_weight": np.array(args.anchor_weight, np.float32),
        "speaker_margin": np.array(args.speaker_margin, np.float32),
        "accumulate": np.array(args.accumulate, np.int32),
        "learning_rate_schedule": np.array(args.schedule),
        "trainable_parameters": np.array(
            args.trainable_parameters, dtype=np.int64),
        "allow_acoustic_variants": np.array(args.allow_acoustic_variants),
        "resume_checkpoint": np.array(str(args.resume.expanduser().resolve())),
        "resume_sha256": np.array(_file_hash(args.resume.expanduser().resolve())),
        "training_sources_json": np.array(json.dumps(sources, sort_keys=True)),
        "holdout_sources_json": np.array(json.dumps(holdouts, sort_keys=True)),
    })
    return values


def dry_run(args, rows, probabilities, sources, holdouts, resume):
    mass = {}
    for row, probability in zip(rows, probabilities):
        key = row["_source_key"]
        mass[key] = mass.get(key, 0.0) + float(probability)
    widths = Counter(row["_feature_width"] for row in rows)
    print(json.dumps({
        "mode": "dry-run-no-gpu", "resume": str(args.resume),
        "resume_sha256": _file_hash(args.resume.expanduser().resolve()),
        "architecture": str(resume["adapter_arch"]),
        "training_rows": len(rows), "feature_widths_all_rows": dict(widths),
        "sampling_mass": {key: round(value, 6) for key, value in mass.items()},
        "sources": sources, "holdouts": holdouts,
        "steps": args.steps, "learning_rate": args.learning_rate,
        "schedule": args.schedule, "train_branch": args.train_branch,
        "text_teacher_weight": args.text_teacher_weight,
        "anchor_weight": args.anchor_weight, "accumulate": args.accumulate,
        "speaker_weight": args.speaker_weight,
        "speaker_margin": args.speaker_margin,
        "allow_acoustic_variants": args.allow_acoustic_variants,
    }, indent=2))


def command_train(args) -> None:
    train_specs = [SourceSpec.parse(value) for value in args.train_source]
    holdout_specs = [SourceSpec.parse(value) for value in args.holdout_source]
    rows, probabilities, sources = load_sources(
        train_specs, args.max_seconds, args.allow_acoustic_variants)
    holdouts = audit_holdout(rows, holdout_specs, args.max_seconds)
    resume = validate_checkpoint(args.resume)
    if str(resume["adapter_arch"]) != ARCH_GEMMA_NATIVE_CONTEXT:
        raise ValueError("context CE requires the owned contextual runtime checkpoint")
    expected_width = int(resume.get("input_dim", INPUT_DIM))
    widths = Counter(row["_feature_width"] for row in rows)
    if set(widths) != {expected_width}:
        raise ValueError(
            f"training feature widths {dict(widths)} != checkpoint "
            f"input_dim {expected_width}")
    speaker_rows = {}
    for row in rows:
        speaker = str(row.get("speaker") or "").strip()
        if speaker:
            lineage = str(row.get("source", row["_source_key"]))
            while lineage.startswith("augmented:"):
                lineage = lineage.removeprefix("augmented:")
            key = f"{lineage}:{speaker}"
            row["_speaker_key"] = key
            speaker_rows.setdefault(key, []).append(row)
    speaker_keys = [key for key, group in speaker_rows.items()
                    if len(group) >= 2]
    if args.speaker_weight and len(speaker_keys) < 2:
        raise ValueError("speaker loss needs two repeated labelled speakers")
    print(f"speaker groups with repeated clips: {len(speaker_keys)}")
    if args.dry_run:
        dry_run(args, rows, probabilities, sources, holdouts, resume)
        return

    refuse_gpu_contention()
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load

    mx.random.seed(args.seed)
    cache_limit = int(args.metal_cache_limit_gb * 1024 ** 3)
    mx.set_cache_limit(cache_limit)
    mx.reset_peak_memory()
    print(f"Metal cache ceiling: {args.metal_cache_limit_gb:.2f} GiB",
          flush=True)
    print(f"loading frozen Gemma: {args.model}", flush=True)
    backbone, tokenizer = load(str(args.model.expanduser().resolve()), lazy=False)
    backbone.freeze()
    backbone.eval()
    prefix, suffix, closing = _prompt_parts(tokenizer)
    adapter = build_adapter(resume, args.seed, args.lexical_head_arch,
                            args.lexical_head_hidden)
    configure_trainable(adapter, args.train_branch)
    anchor = None
    if args.anchor_weight:
        anchor = build_adapter(resume, args.seed, args.lexical_head_arch,
                               args.lexical_head_hidden)
        anchor.freeze()

    class Alignment(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.adapter = adapter
            self.backbone.freeze()

        def target_logits(self, ids, embeds, first, count):
            lm = self.backbone.language_model
            hidden = lm.model(ids, input_embeddings=embeds)
            hidden = hidden[:, first - 1:first - 1 + count]
            logits = (lm.model.embed_tokens.as_linear(hidden)
                      if lm.tie_word_embeddings else lm.lm_head(hidden))
            cap = lm.final_logit_softcapping
            return mx.tanh(logits / cap) * cap if cap is not None else logits

    alignment = Alignment()
    mx.eval(alignment.parameters())
    if anchor is not None:
        mx.eval(anchor.parameters())
    from mlx.utils import tree_flatten
    args.trainable_parameters = sum(
        int(np.prod(value.shape)) for _, value in
        tree_flatten(adapter.trainable_parameters()))
    print(f"training {args.trainable_parameters:,} deployed adapter "
          f"parameters ({args.train_branch})", flush=True)

    # One deterministic clip per source makes the collapse probe broad without
    # letting the largest corpus own it. Nearest-token recall is diagnostic;
    # frame diversity is the hard abort signal for an ignore-the-audio basin.
    probe_rows = []
    probe_sources = set()
    for row in rows:
        if row["_source_key"] not in probe_sources:
            probe_rows.append(row)
            probe_sources.add(row["_source_key"])
        if len(probe_rows) >= args.probe_examples:
            break

    def lexical_geometry(step):
        embedding = backbone.language_model.model.embed_tokens
        hits = total = 0
        diversity = []
        for row in probe_rows:
            features = mx.array(np.load(row["feature"]).astype(np.float32))
            audio = alignment.adapter(features) / math.sqrt(OUTPUT_DIM)
            scores = embedding.as_linear(audio)
            top = mx.argpartition(-scores, args.probe_topk, axis=-1)[
                :, :args.probe_topk]
            top1 = mx.argmax(scores, axis=-1)
            mx.eval(top, top1)
            wanted = set(tokenizer.encode(
                row["target"], add_special_tokens=False))
            found = {int(token) for frame in np.asarray(top) for token in frame}
            hits += len(wanted & found)
            total += len(wanted)
            values = np.asarray(top1)
            diversity.append(len(set(map(int, values))) / max(1, len(values)))
        recall = hits / max(1, total)
        frame_diversity = float(np.mean(diversity))
        print(f"    geometry@{step}: recall@{args.probe_topk}="
              f"{recall:.2%} frame-diversity={frame_diversity:.2%}",
              flush=True)
        return recall, frame_diversity

    baseline_recall, baseline_diversity = lexical_geometry(0)
    warmup = max(1, args.warmup)
    tail = (optim.step_decay(args.learning_rate, 1.0, 10 ** 9)
            if args.schedule == "constant" else
            optim.cosine_decay(
                args.learning_rate, max(1, args.steps - warmup)))
    schedule = optim.join_schedules([
        optim.linear_schedule(0.0, args.learning_rate, warmup), tail,
    ], [warmup])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=args.weight_decay)

    def loss_fn(module, example):
        row, peer, relation = example
        cached = mx.array(np.load(row["feature"]).astype(np.float32))
        current, context = module.adapter(cached, return_context=True)
        audio = current / math.sqrt(OUTPUT_DIM)
        text_ids = tokenizer.encode(
            row["target"], add_special_tokens=False)[:args.max_target_tokens]
        target_ids = text_ids + closing
        n_audio = audio.shape[0]
        ids = mx.array(prefix + [0] * n_audio + suffix + target_ids,
                       dtype=mx.int32)[None, :]
        token_embeds = module.backbone.language_model.model.embed_tokens(ids)
        start = len(prefix)
        embeds = mx.concatenate([
            token_embeds[:, :start], audio[None, :],
            token_embeds[:, start + n_audio:],
        ], axis=1)
        first = start + n_audio + len(suffix)
        logits = module.target_logits(ids, embeds, first, len(target_ids))
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

        anchor_loss = mx.array(0.0, dtype=current.dtype)
        if anchor is not None:
            original = mx.stop_gradient(anchor(cached))
            count = min(current.shape[0], original.shape[0])
            anchored, original = current[:count], original[:count]
            delta = anchored - original
            absolute = mx.abs(delta)
            huber = mx.mean(mx.where(
                absolute < 1.0, 0.5 * mx.square(delta), absolute - 0.5))
            left = anchored / (
                mx.linalg.norm(anchored, axis=-1, keepdims=True) + 1e-6)
            right = original / (
                mx.linalg.norm(original, axis=-1, keepdims=True) + 1e-6)
            anchor_loss = huber + 0.1 * (
                1.0 - mx.mean(mx.sum(left * right, axis=-1)))
        speaker_loss = mx.array(0.0, dtype=current.dtype)
        if relation:
            peer_features = mx.array(
                np.load(peer["feature"]).astype(np.float32))
            _, peer_context = module.adapter(
                peer_features, return_context=True)
            summary = mx.mean(context, axis=0)
            other = mx.mean(peer_context, axis=0)
            similarity = mx.sum(summary * other) / (
                (mx.linalg.norm(summary) + 1e-6)
                * (mx.linalg.norm(other) + 1e-6))
            speaker_loss = (1.0 - similarity if relation > 0 else
                            mx.square(mx.maximum(
                                similarity - args.speaker_margin, 0.0)))
        return ce + args.text_teacher_weight * text_kl + \
            args.anchor_weight * anchor_loss + \
            args.speaker_weight * speaker_loss

    value_and_grad = nn.value_and_grad(alignment, loss_fn)
    rng = np.random.default_rng(args.seed)
    losses = []
    started = time.time()
    for step in range(1, args.steps + 1):
        gradients = None
        value = 0.0
        for _ in range(args.accumulate):
            row = rows[int(rng.choice(len(rows), p=probabilities))]
            peer = row
            relation = 0
            speaker = row.get("_speaker_key")
            if args.speaker_weight and speaker in speaker_rows \
                    and len(speaker_rows[speaker]) >= 2:
                if rng.random() < 0.5:
                    group = speaker_rows[speaker]
                    peer = group[int(rng.integers(len(group)))]
                    while peer["id"] == row["id"]:
                        peer = group[int(rng.integers(len(group)))]
                    relation = 1
                else:
                    other_key = speaker
                    while other_key == speaker:
                        other_key = speaker_keys[int(rng.integers(
                            len(speaker_keys)))]
                    group = speaker_rows[other_key]
                    peer = group[int(rng.integers(len(group)))]
                    relation = -1
            loss, one_grad = value_and_grad(
                alignment, (row, peer, relation))
            gradients = _tree_add(gradients, one_grad)
            value += float(loss)
        value /= args.accumulate
        gradients = _tree_scale(gradients, 1.0 / args.accumulate)
        gradients, grad_norm = optim.clip_grad_norm(gradients, args.clip)
        optimizer.update(alignment, gradients)
        mx.eval(alignment.parameters(), optimizer.state)
        # Variable sequence lengths otherwise allow MLX's reusable Metal
        # buffers to consume most unified memory and force swap. Parameters
        # and optimizer state remain live; completed graph buffers do not.
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
            recall, diversity = lexical_geometry(step)
            if recall < baseline_recall * args.warn_recall_ratio:
                print(f"    WARN recall {recall:.2%} is below "
                      f"{args.warn_recall_ratio:.0%} of baseline "
                      f"{baseline_recall:.2%}", flush=True)
            if diversity < baseline_diversity * args.abort_diversity_ratio:
                path = args.output.with_name(
                    f"{args.output.stem}.collapsed-step{step:06d}"
                    f"{args.output.suffix}")
                atomic_save(path, checkpoint_values(
                    alignment.adapter, resume, step, args, sources, holdouts))
                raise SystemExit(
                    f"ABORT at {step}: frame diversity {diversity:.2%} < "
                    f"{args.abort_diversity_ratio:.0%} of baseline "
                    f"{baseline_diversity:.2%}; saved {path}")
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            path = args.output.with_name(
                f"{args.output.stem}.step{step:06d}{args.output.suffix}")
            atomic_save(path, checkpoint_values(
                alignment.adapter, resume, step, args, sources, holdouts))
            validate_checkpoint(path)
            print(f"checkpoint {path}", flush=True)
    atomic_save(args.output, checkpoint_values(
        alignment.adapter, resume, args.steps, args, sources, holdouts))
    validate_checkpoint(args.output)
    print(f"saved {args.output}; first10={np.mean(losses[:10]):.5f} "
          f"last10={np.mean(losses[-10:]):.5f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-source", action="append", required=True,
                        metavar="CACHE::SPLIT::WEIGHT")
    parser.add_argument("--holdout-source", action="append", default=[],
                        metavar="CACHE::SPLIT::WEIGHT")
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_GEMMA)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--schedule", choices=("cosine", "constant"),
                        default="cosine")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--text-teacher-weight", type=float, default=0.5)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--speaker-weight", type=float, default=0.005)
    parser.add_argument("--speaker-margin", type=float, default=0.3)
    parser.add_argument("--max-target-tokens", type=int, default=160)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument(
        "--allow-acoustic-variants", action="store_true",
        help="allow clean/augmented rows sharing an ID only when labels "
             "normalize identically and feature paths are distinct")
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--probe-every", type=int, default=250)
    parser.add_argument("--probe-examples", type=int, default=6)
    parser.add_argument("--probe-topk", type=int, default=10)
    parser.add_argument("--warn-recall-ratio", type=float, default=0.5)
    parser.add_argument("--abort-diversity-ratio", type=float, default=0.6)
    parser.add_argument("--metal-cache-limit-gb", type=float, default=2.0)
    parser.add_argument("--lexical-head-arch", choices=("linear", "bigru"),
                        default="bigru")
    parser.add_argument("--lexical-head-hidden", type=int, default=384)
    parser.add_argument("--train-branch",
                        choices=("all", "context", "input", "input_full"),
                        default="context",
                        help="all trains every deployed adapter weight; "
                             "medium historically continued context only; "
                             "a fresh large port starts with input, then "
                             "input_full")
    parser.add_argument("--dry-run", action="store_true")
    # adapter_values consumes the standard trainer metadata fields.
    parser.set_defaults(cosine_weight=0.1, lexical_weight=0.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.resume = args.resume.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.steps <= 0 or args.accumulate <= 0:
        raise ValueError("--steps and --accumulate must be positive")
    if args.anchor_weight < 0 or args.text_teacher_weight < 0 \
            or args.speaker_weight < 0:
        raise ValueError("loss weights cannot be negative")
    if args.metal_cache_limit_gb <= 0:
        raise ValueError("metal cache limit must be positive")
    if args.probe_examples <= 0 or args.probe_topk <= 0:
        raise ValueError("probe sizes must be positive")
    if not 0 < args.warn_recall_ratio <= 1 or not 0 < \
            args.abort_diversity_ratio <= 1:
        raise ValueError("probe ratios must be in (0, 1]")
    command_train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
