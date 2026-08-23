#!/usr/bin/env python3
"""Train and evaluate a wholly Qwen-free stock-Whisper -> Gemma adapter.

Only the fresh Jaeger MLP is optimized.  Stock Whisper-medium and the 4-bit
MLX Gemma backbone remain frozen.  Cached features make each training step one
small MLP plus a frozen Gemma forward pass.

Typical pilot::

    python training/stock_whisper_gemma.py precompute-corpus \
        --corpus training/corpus_mixed
    python training/stock_whisper_gemma.py train --task transcript --steps 1000
    python training/stock_whisper_gemma.py train --task answer --steps 400 \
        --resume artifacts/stock-whisper-gemma-transcript.npz
    python training/stock_whisper_gemma.py bench --adapter artifacts/stock-whisper-gemma-answer.npz
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from jaeger_agent_omni.runtime.stock_whisper import (
    AUDIO_MARKER,
    DEFAULT_GEMMA,
    DEFAULT_WHISPER,
    HIDDEN_DIM,
    INPUT_BOUNDARY,
    INPUT_DIM,
    OUTPUT_DIM,
    POOL_STEP,
    SAMPLE_RATE,
    TARGET_NORM,
    StockWhisperFrontEnd,
    StockWhisperGemmaSession,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = Path.home() / "models/stock-whisper-gemma/cache-raw-v1"
DEFAULT_TRANSCRIPT = REPO / "artifacts/stock-whisper-gemma-transcript.npz"
DEFAULT_ANSWER = REPO / "artifacts/stock-whisper-gemma-answer.npz"
TRANSCRIPT_SYSTEM = (
    "Repeat the spoken words exactly. Output only what was said, with no "
    "commentary or explanation."
)
ANSWER_SYSTEM = (
    "You are a voice assistant. Answer the spoken question directly in one "
    "short sentence."
)

GPU_JOB_PATTERNS = (
    r"training/train\.py",
    r"training/stock_whisper_gemma\.py\s+"
    r"(?:precompute|pretrain-ctc|train|probe|wer|bench)",
    r"cleanroom_train\.py\s+(?:precompute|train|bench)",
    r"compare_tracks\.py",
    r"train_audio_adapter\.py",
)


def _refuse_gpu_contention() -> None:
    """Catch known Jaeger/VoiceLLM jobs even when another repo launched them."""
    listing = subprocess.check_output(
        ["ps", "-Ao", "pid=,comm=,args="], text=True)
    conflicts = []
    for line in listing.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3 or int(fields[0]) == os.getpid():
            continue
        _, command = fields[1:]
        executable = command.split(maxsplit=1)[0]
        if "python" not in Path(executable).name.lower():
            continue
        if any(re.search(pattern, command) for pattern in GPU_JOB_PATTERNS):
            conflicts.append(line.strip())
    if conflicts:
        joined = "\n  ".join(conflicts)
        raise SystemExit(
            "REFUSING: another known GPU job is active:\n  " + joined)


def _read_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    wav, rate = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly
        divisor = math.gcd(int(rate), SAMPLE_RATE)
        wav = resample_poly(
            wav, SAMPLE_RATE // divisor, int(rate) // divisor).astype(np.float32)
    return np.asarray(wav, dtype=np.float32)


def _safe_id(value: str) -> str:
    return hashlib.sha1(value.encode()).hexdigest()[:16]


def _split(item_id: str, val_percent: int) -> str:
    bucket = int(hashlib.sha1(item_id.encode()).hexdigest()[:8], 16) % 100
    return "validation" if bucket < val_percent else "train"


def _write_feature(root: Path, split: str, item_id: str, features,
                   record: dict) -> None:
    split_root = root / split
    feature_dir = split_root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_id(item_id) + ".npy"
    np.save(feature_dir / name, np.asarray(features, dtype=np.float16))
    payload = dict(record)
    payload.update({
        "id": item_id,
        "frames": int(features.shape[0]),
        "feature": f"features/{name}",
        "encoder": DEFAULT_WHISPER,
        "boundary": INPUT_BOUNDARY,
    })
    with (split_root / "manifest.jsonl").open("a", buffering=1) as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _completed(root: Path) -> set[str]:
    result = set()
    for manifest in root.glob("*/manifest.jsonl"):
        for line in manifest.read_text().splitlines():
            if line.strip():
                result.add(json.loads(line)["id"])
    return result


def command_precompute_corpus(args) -> None:
    corpus = args.corpus.resolve()
    rows = [json.loads(line) for line in
            (corpus / "turns.jsonl").read_text().splitlines() if line.strip()]
    done = _completed(args.cache)
    front = StockWhisperFrontEnd(args.whisper, args.device)
    written = 0
    started = time.time()
    for index, row in enumerate(rows, 1):
        item_id = str(row["turn_id"])
        if item_id in done:
            continue
        wav_path = corpus / row["audio_path"]
        waveform = _read_wav(wav_path)
        seconds = len(waveform) / SAMPLE_RATE
        if seconds > args.max_seconds:
            continue
        features = front(waveform)
        split = _split(item_id, args.validation_percent)
        _write_feature(args.cache, split, item_id, features, {
            "transcript": str(row.get("transcript") or "").strip(),
            "response": str(row.get("response") or "").strip(),
            "seconds": seconds,
            "source_pcm": str(wav_path),
            "source": str((row.get("meta") or {}).get("source", "local")),
        })
        written += 1
        if written == 1 or written % args.log_every == 0:
            print(f"cached {written}: {item_id} {seconds:.2f}s -> "
                  f"{features.shape} ({time.time()-started:.1f}s)", flush=True)
        if args.max_examples and written >= args.max_examples:
            break
    print(f"cached {written} new corpus examples at {args.cache}", flush=True)
    if getattr(front.device, "type", "") == "mps":
        os._exit(0)


def _decode_audio(row: dict) -> np.ndarray:
    import soundfile as sf

    audio = row["audio"]
    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    wav, rate = sf.read(source, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly
        divisor = math.gcd(int(rate), SAMPLE_RATE)
        wav = resample_poly(
            wav, SAMPLE_RATE // divisor, int(rate) // divisor).astype(np.float32)
    return np.asarray(wav, dtype=np.float32)


def command_precompute_open(args) -> None:
    from datasets import Audio, load_dataset

    done = _completed(args.cache)
    dataset = load_dataset(
        "openslr/librispeech_asr", args.config, split=args.split,
        streaming=True).cast_column("audio", Audio(decode=False))
    if args.shuffle_buffer:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    front = StockWhisperFrontEnd(args.whisper, args.device)
    written = 0
    started = time.time()
    for index, row in enumerate(dataset, 1):
        item_id = "librispeech:" + str(row.get("id") or index)
        if item_id in done:
            continue
        wav = _decode_audio(row)
        seconds = len(wav) / SAMPLE_RATE
        if seconds < args.min_seconds or seconds > args.max_seconds:
            continue
        features = front(wav)
        partition = _split(item_id, args.validation_percent)
        _write_feature(args.cache, partition, item_id, features, {
            "transcript": str(row["text"]).strip(),
            "response": "",
            "seconds": seconds,
            "source": f"openslr/librispeech_asr:{args.config}",
            "license": "CC-BY-4.0",
        })
        written += 1
        if written == 1 or written % args.log_every == 0:
            print(f"cached {written}/{index}: {seconds:.2f}s -> {features.shape} "
                  f"({time.time()-started:.1f}s)", flush=True)
        if args.max_examples and written >= args.max_examples:
            break
    print(f"cached {written} new open examples at {args.cache}", flush=True)
    if getattr(front.device, "type", "") == "mps":
        os._exit(0)


def _rows(cache: Path, split: str, task: str) -> list[dict]:
    root = cache / split
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    result = []
    field = "transcript" if task == "transcript" else "response"
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        target = str(row.get(field) or "").strip()
        if not target:
            continue
        row["target"] = target
        row["feature"] = root / row["feature"]
        if row.get("encoder") != DEFAULT_WHISPER:
            raise ValueError(f"mixed encoder cache in {manifest}")
        if row.get("boundary") != INPUT_BOUNDARY:
            raise ValueError(
                f"stale cache boundary {row.get('boundary')!r} in {manifest}; "
                f"expected {INPUT_BOUNDARY!r}")
        result.append(row)
    return result


def _prompt_parts(tokenizer, system: str) -> tuple[list[int], list[int], list[int]]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": AUDIO_MARKER}],
        tokenize=False, add_generation_prompt=True)
    if AUDIO_MARKER not in rendered:
        raise RuntimeError("audio marker disappeared from Gemma template")
    prefix, suffix = rendered.split(AUDIO_MARKER, 1)
    return (tokenizer.encode(prefix, add_special_tokens=False),
            tokenizer.encode(suffix, add_special_tokens=False),
            tokenizer.encode("<turn|>\n", add_special_tokens=False))


def _atomic_checkpoint(path: Path, adapter, **metadata) -> None:
    _atomic_checkpoint_arrays(path, {
        "linear1_weight": np.array(adapter.linear1.weight),
        "linear1_bias": np.array(adapter.linear1.bias),
        "linear2_weight": np.array(adapter.linear2.weight),
        "linear2_bias": np.array(adapter.linear2.bias),
    }, **metadata)


def _atomic_checkpoint_arrays(path: Path, weights: dict, **metadata) -> None:
    payload = {
        **weights,
        "input_boundary": np.array(INPUT_BOUNDARY),
        "encoder": np.array(DEFAULT_WHISPER),
        "lineage": np.array("qwen_free"),
        "target_norm": np.array(TARGET_NORM, dtype=np.float32),
        "format_version": np.array(1, dtype=np.int32),
    }
    payload.update({key: np.asarray(value) for key, value in metadata.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
                dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def command_pretrain_ctc(args) -> None:
    """Audio-ground the fresh adapter without loading or running Gemma.

    CTC supplies the missing monotonic alignment between 50 Hz audio states
    and transcript tokens. A sampled Gemma vocabulary keeps the objective
    cheap while still training toward Gemma's own embedding rows.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from transformers import AutoTokenizer

    from fit_alignment import embedding_table

    rows = _rows(args.cache, "train", "transcript")
    validation_rows = _rows(args.cache, "validation", "transcript")
    if args.max_examples:
        rows = rows[:args.max_examples]
    device_name = args.device
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
    table = embedding_table("google/gemma-4-E4B-it")
    table = functional.normalize(table, dim=-1).to(device)

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
            self.linear2 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

        def forward(self, states):
            values = self.linear2(functional.relu(self.linear1(states)))
            usable = (values.shape[0] // POOL_STEP) * POOL_STEP
            if usable:
                values = values[:usable].reshape(
                    -1, POOL_STEP, OUTPUT_DIM).mean(dim=1)
            else:
                values = values.mean(dim=0, keepdim=True)
            return functional.normalize(values, dim=-1)

    adapter = Adapter().to(device)
    if args.resume:
        from jaeger_agent_omni.runtime.stock_whisper import validate_checkpoint
        saved = validate_checkpoint(args.resume)
        with torch.no_grad():
            adapter.linear1.weight.copy_(torch.from_numpy(saved["linear1_weight"]))
            adapter.linear1.bias.copy_(torch.from_numpy(saved["linear1_bias"]))
            adapter.linear2.weight.copy_(torch.from_numpy(saved["linear2_weight"]))
            adapter.linear2.bias.copy_(torch.from_numpy(saved["linear2_bias"]))
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    def one(row):
        features = torch.from_numpy(
            np.load(row["feature"]).astype(np.float32)).to(device)
        token_ids = tokenizer.encode(
            row["target"], add_special_tokens=False)[:args.max_target_tokens]
        if not token_ids:
            return None
        minimum = len(token_ids) + sum(
            left == right for left, right in zip(token_ids, token_ids[1:]))
        emitted = adapter(features)
        if emitted.shape[0] < minimum:
            return None
        positive = list(dict.fromkeys(token_ids))
        excluded = set(positive) | {0}
        negatives = []
        while len(negatives) < args.negative_tokens:
            candidate = rng.randrange(1, table.shape[0])
            if candidate not in excluded:
                excluded.add(candidate)
                negatives.append(candidate)
        candidates = [0] + positive + negatives
        remap = {token: index + 1 for index, token in enumerate(positive)}
        targets = torch.tensor(
            [remap[token] for token in token_ids], dtype=torch.long)
        subset = table[torch.tensor(candidates, device=device)]
        logits = emitted @ subset.T / args.temperature
        log_probs = functional.log_softmax(logits.float(), dim=-1).unsqueeze(1)
        # PyTorch has no native MPS CTC kernel. The reduced vocabulary keeps
        # this explicit differentiable CPU transfer small (~100 KB/turn).
        return ctc(
            log_probs.cpu(), targets,
            torch.tensor([emitted.shape[0]], dtype=torch.long),
            torch.tensor([len(targets)], dtype=torch.long))

    step = 0
    rolling = []
    started = time.time()
    while step < args.steps:
        epoch = list(rows)
        rng.shuffle(epoch)
        for row in epoch:
            loss = one(row)
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            step += 1
            value = float(loss.detach())
            rolling.append(value)
            rolling = rolling[-args.log_every:]
            if step == 1 or step % args.log_every == 0:
                print(f"ctc {step}/{args.steps} loss={value:.4f} "
                      f"avg={sum(rolling)/len(rolling):.4f} "
                      f"elapsed={time.time()-started:.1f}s", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                weights = {
                    "linear1_weight": adapter.linear1.weight.detach().cpu().numpy(),
                    "linear1_bias": adapter.linear1.bias.detach().cpu().numpy(),
                    "linear2_weight": adapter.linear2.weight.detach().cpu().numpy(),
                    "linear2_bias": adapter.linear2.bias.detach().cpu().numpy(),
                }
                _atomic_checkpoint_arrays(
                    args.output, weights, step=step, task="transcript",
                    stage="sampled_ctc", learning_rate=args.learning_rate,
                    negative_tokens=args.negative_tokens,
                    temperature=args.temperature, train_loss=value)
                print(f"checkpoint {args.output}", flush=True)
            if step >= args.steps:
                break

    adapter.eval()
    eval_losses = []
    with torch.no_grad():
        for row in validation_rows[:args.eval_examples]:
            loss = one(row)
            if loss is not None:
                eval_losses.append(float(loss))
    if eval_losses:
        print(f"validation sampled CTC ({len(eval_losses)}): "
              f"{sum(eval_losses)/len(eval_losses):.4f}")


def command_train(args) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load

    mx.random.seed(args.seed)

    train_rows = _rows(args.cache, "train", args.task)
    validation_rows = _rows(args.cache, "validation", args.task)
    if args.max_examples:
        train_rows = train_rows[:args.max_examples]
    if not train_rows or not validation_rows:
        raise RuntimeError("need non-empty train and validation manifests")
    print(f"task={args.task}: {len(train_rows)} train / "
          f"{len(validation_rows)} validation", flush=True)
    print(f"loading frozen Gemma from {args.model}", flush=True)
    backbone, tokenizer = load(str(args.model), lazy=False)
    backbone.freeze()
    backbone.eval()
    system = TRANSCRIPT_SYSTEM if args.task == "transcript" else ANSWER_SYSTEM
    prefix, suffix, closing = _prompt_parts(tokenizer, system)

    saved = None
    start_step = 0
    if args.resume:
        from jaeger_agent_omni.runtime.stock_whisper import validate_checkpoint
        saved = validate_checkpoint(args.resume)
        start_step = int(saved.get("step", 0))

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(INPUT_DIM, HIDDEN_DIM, bias=True)
            self.linear2 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM, bias=True)
            if saved is not None:
                self.linear1.weight = mx.array(saved["linear1_weight"])
                self.linear1.bias = mx.array(saved["linear1_bias"])
                self.linear2.weight = mx.array(saved["linear2_weight"])
                self.linear2.bias = mx.array(saved["linear2_bias"])

        def __call__(self, x):
            from jaeger_agent_omni.runtime.stock_whisper import _pool_mlx
            y = self.linear2(nn.relu(self.linear1(x)))
            y = _pool_mlx(mx, y)
            norms = mx.sqrt(mx.sum(y * y, axis=-1, keepdims=True) + 1e-8)
            return y / norms * TARGET_NORM

    class Alignment(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.adapter = Adapter()
            self.backbone.freeze()

        def target_logits(self, ids, embeds, first_target, count):
            lm = self.backbone.language_model
            hidden = lm.model(ids, input_embeddings=embeds)
            hidden = hidden[:, first_target - 1:first_target - 1 + count]
            logits = (lm.model.embed_tokens.as_linear(hidden)
                      if lm.tie_word_embeddings else lm.lm_head(hidden))
            if lm.final_logit_softcapping is not None:
                cap = lm.final_logit_softcapping
                logits = mx.tanh(logits / cap) * cap
            return logits

    alignment = Alignment()
    mx.eval(alignment.parameters())
    optimizer = optim.AdamW(
        learning_rate=args.learning_rate, weight_decay=args.weight_decay)

    def example(row):
        cached = np.load(row["feature"]).astype(np.float32)
        if cached.ndim != 2 or cached.shape[1] != INPUT_DIM:
            raise ValueError(f"bad feature {row['feature']}: {cached.shape}")
        audio = alignment.adapter(mx.array(cached)) / math.sqrt(OUTPUT_DIM)
        n_audio = audio.shape[0]
        target = tokenizer.encode(
            row["target"], add_special_tokens=False)[:args.max_target_tokens]
        available = (args.sequence_length - len(prefix) - n_audio
                     - len(suffix) - len(closing))
        if available < 1:
            raise ValueError(f"sequence length too short for {row['id']}")
        target = target[:available] + closing
        all_ids = prefix + [0] * n_audio + suffix + target
        all_ids += [0] * (args.sequence_length - len(all_ids))
        ids = mx.array(all_ids, dtype=mx.int32)[None, :]
        token_embeds = backbone.language_model.model.embed_tokens(ids)
        start = len(prefix)
        embeds = mx.concatenate(
            [token_embeds[:, :start], audio[None, :],
             token_embeds[:, start + n_audio:]], axis=1)
        first_target = len(prefix) + n_audio + len(suffix)
        return (ids, embeds, mx.array(target, dtype=mx.int32)[None, :],
                first_target)

    def loss_fn(module, row):
        ids, embeds, targets, first = example(row)
        logits = module.target_logits(ids, embeds, first, targets.shape[1])
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    loss_and_grad = nn.value_and_grad(alignment, loss_fn)
    rng = random.Random(args.seed)
    step = start_step
    started = time.time()
    rolling = []
    while step < args.steps:
        epoch = list(train_rows)
        rng.shuffle(epoch)
        for row in epoch:
            loss, grads = loss_and_grad(alignment, row)
            optimizer.update(alignment, grads)
            mx.eval(alignment.parameters(), optimizer.state, loss)
            step += 1
            value = float(loss.item())
            rolling.append(value)
            rolling = rolling[-args.log_every:]
            if step == 1 or step % args.log_every == 0:
                print(f"step {step}/{args.steps} loss={value:.4f} "
                      f"avg={sum(rolling)/len(rolling):.4f} "
                      f"elapsed={time.time()-started:.1f}s", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                _atomic_checkpoint(
                    args.output, alignment.adapter, step=step, task=args.task,
                    learning_rate=args.learning_rate, train_loss=value,
                    model=str(args.model))
                print(f"checkpoint {args.output}", flush=True)
            if step >= args.steps:
                break

    eval_losses = []
    for row in validation_rows[:args.eval_examples]:
        value = loss_fn(alignment, row)
        mx.eval(value)
        eval_losses.append(float(value.item()))
    validation_ce = sum(eval_losses) / len(eval_losses)
    print(f"validation CE ({len(eval_losses)}): {validation_ce:.4f}")
    _atomic_checkpoint(
        args.output, alignment.adapter, step=step, task=args.task,
        learning_rate=args.learning_rate, train_loss=value,
        validation_ce=validation_ce, model=str(args.model))


def command_probe(args) -> None:
    rows = _rows(args.cache, args.split, args.task)
    row = rows[args.index]
    session = StockWhisperGemmaSession(
        args.adapter, args.model, args.whisper,
        TRANSCRIPT_SYSTEM if args.task == "transcript" else ANSWER_SYSTEM)
    features = np.load(row["feature"]).astype(np.float32)
    session._features = features
    session.n_prefilled = len(features)
    answer = session.respond(max_tokens=args.max_tokens, eos_ids={1, 106})
    print(f"id: {row['id']}")
    print(f"expected: {row['target']}")
    print(f"generated: {answer}")


def command_wer(args) -> None:
    """Free-run held-out clips and report corpus-level, pooled WER."""
    from evaluate import normalize, wer

    rows = _rows(args.cache, args.split, "transcript")[:args.examples]
    session = StockWhisperGemmaSession(
        args.adapter, args.model, args.whisper, TRANSCRIPT_SYSTEM)
    total_edits = 0.0
    total_words = 0
    outputs = []
    for index, row in enumerate(rows, 1):
        features = np.load(row["feature"]).astype(np.float32)
        session._features = features
        session.n_prefilled = max(1, len(features) // POOL_STEP)
        budget = max(8, int(float(row["seconds"]) * args.words_per_second * 1.6))
        hypothesis = session.respond(max_tokens=budget, eos_ids={1, 106})
        reference_words = normalize(row["target"])
        hypothesis_words = normalize(hypothesis)
        edits = wer(reference_words, hypothesis_words) * len(reference_words)
        total_edits += edits
        total_words += len(reference_words)
        outputs.append({
            "id": row["id"], "reference": row["target"],
            "hypothesis": hypothesis,
            "wer": edits / max(1, len(reference_words)),
        })
        if index <= 3:
            print(f"ref: {row['target']}")
            print(f"hyp: {hypothesis}\n")
        if index % 16 == 0:
            print(f"{index}/{len(rows)} running WER "
                  f"{100*total_edits/max(1,total_words):.2f}%", flush=True)
    score = total_edits / max(1, total_words)
    print(f"CORPUS WER {score*100:.2f}% over {len(rows)} held-out clips, "
          f"{total_words} reference words")
    if args.output:
        args.output.write_text(json.dumps({
            "adapter": str(args.adapter), "split": args.split,
            "n": len(rows), "wer": score, "reference_words": total_words,
            "rows": outputs,
        }, indent=2))


def command_bench(args) -> None:
    import statistics
    from jaeger_agent_omni.app import read_wav
    from research.grading import graded

    cases = json.loads((args.cases / "cases.json").read_text())
    session = StockWhisperGemmaSession(
        args.adapter, args.model, args.whisper, ANSWER_SYSTEM)
    rows = []
    for case in cases:
        session.reset()
        session.open_turn()
        wav = read_wav(args.cases / f"{case['key']}.wav")
        session.feed_audio(wav)
        started = time.monotonic()
        session.end_of_turn()
        reply = session.respond(max_tokens=args.max_tokens, eos_ids={1, 106})
        elapsed = (time.monotonic() - started) * 1000
        ok = graded(reply, case["answers"])
        rows.append({"case": case["key"], "ok": ok, "ms": elapsed,
                     "reply": reply})
        print(f"{case['key']} {'OK' if ok else 'MISS'} {elapsed:.0f}ms "
              f"{reply[:80]!r}", flush=True)
    correct = sum(row["ok"] for row in rows)
    median = statistics.median(row["ms"] for row in rows)
    print(f"stock-whisper Qwen-free: {correct}/{len(rows)} "
          f"({100*correct/len(rows):.0f}%), median {median:.0f}ms")
    if args.output:
        args.output.write_text(json.dumps({
            "front_end": "stock_whisper_qwen_free", "correct": correct,
            "n": len(rows), "median_ms": median, "rows": rows}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def common_pre(p):
        p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
        p.add_argument("--whisper", default=DEFAULT_WHISPER)
        p.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
        p.add_argument("--max-examples", type=int, default=0)
        p.add_argument("--min-seconds", type=float, default=0.3)
        p.add_argument("--max-seconds", type=float, default=20.0)
        p.add_argument("--validation-percent", type=int, default=10)
        p.add_argument("--log-every", type=int, default=25)

    pre = commands.add_parser("precompute-corpus")
    common_pre(pre)
    pre.add_argument("--corpus", type=Path, required=True)
    pre.set_defaults(func=command_precompute_corpus)

    opened = commands.add_parser("precompute-open")
    common_pre(opened)
    opened.add_argument("--config", default="clean", choices=("clean", "other"))
    opened.add_argument("--split", default="train.100")
    opened.add_argument("--shuffle-buffer", type=int, default=10_000)
    opened.add_argument("--seed", type=int, default=7)
    opened.set_defaults(func=command_precompute_open)

    ctc = commands.add_parser("pretrain-ctc")
    ctc.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ctc.add_argument("--output", type=Path, default=DEFAULT_TRANSCRIPT)
    ctc.add_argument("--resume", type=Path)
    ctc.add_argument("--steps", type=int, default=400)
    ctc.add_argument("--max-examples", type=int, default=0)
    ctc.add_argument("--max-target-tokens", type=int, default=96)
    ctc.add_argument("--negative-tokens", type=int, default=4096)
    ctc.add_argument("--temperature", type=float, default=0.05)
    ctc.add_argument("--learning-rate", type=float, default=3e-4)
    ctc.add_argument("--weight-decay", type=float, default=0.0)
    ctc.add_argument("--log-every", type=int, default=10)
    ctc.add_argument("--save-every", type=int, default=50)
    ctc.add_argument("--eval-examples", type=int, default=16)
    ctc.add_argument("--seed", type=int, default=7)
    ctc.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    ctc.set_defaults(func=command_pretrain_ctc)

    train = commands.add_parser("train")
    train.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    train.add_argument("--task", choices=("transcript", "answer"), required=True)
    train.add_argument("--model", type=Path, default=DEFAULT_GEMMA)
    train.add_argument("--output", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument("--steps", type=int, default=1000)
    train.add_argument("--max-examples", type=int, default=0)
    train.add_argument("--max-target-tokens", type=int, default=96)
    train.add_argument("--sequence-length", type=int, default=384)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--save-every", type=int, default=50)
    train.add_argument("--eval-examples", type=int, default=16)
    train.add_argument("--seed", type=int, default=7)
    train.set_defaults(func=command_train)

    for name, func in (("probe", command_probe),):
        probe = commands.add_parser(name)
        probe.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
        probe.add_argument("--split", choices=("train", "validation"),
                           default="validation")
        probe.add_argument("--task", choices=("transcript", "answer"), required=True)
        probe.add_argument("--adapter", type=Path, required=True)
        probe.add_argument("--model", type=Path, default=DEFAULT_GEMMA)
        probe.add_argument("--whisper", default=DEFAULT_WHISPER)
        probe.add_argument("--index", type=int, default=0)
        probe.add_argument("--max-tokens", type=int, default=96)
        probe.set_defaults(func=func)

    wer_parser = commands.add_parser("wer")
    wer_parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    wer_parser.add_argument("--split", choices=("train", "validation"),
                            default="validation")
    wer_parser.add_argument("--adapter", type=Path, required=True)
    wer_parser.add_argument("--model", type=Path, default=DEFAULT_GEMMA)
    wer_parser.add_argument("--whisper", default=DEFAULT_WHISPER)
    wer_parser.add_argument("-n", "--examples", type=int, default=64)
    wer_parser.add_argument("--words-per-second", type=float, default=2.8)
    wer_parser.add_argument("--output", type=Path,
                            default=REPO / "research/stock_whisper_wer.json")
    wer_parser.set_defaults(func=command_wer)

    bench = commands.add_parser("bench")
    bench.add_argument("--cases", type=Path, default=REPO / "research/cases12")
    bench.add_argument("--adapter", type=Path, required=True)
    bench.add_argument("--model", type=Path, default=DEFAULT_GEMMA)
    bench.add_argument("--whisper", default=DEFAULT_WHISPER)
    bench.add_argument("--max-tokens", type=int, default=64)
    bench.add_argument("--output", type=Path,
                       default=REPO / "research/stock_whisper_bench.json")
    bench.set_defaults(func=command_bench)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _refuse_gpu_contention()
    if args.command == "train" and args.output is None:
        args.output = DEFAULT_TRANSCRIPT if args.task == "transcript" else DEFAULT_ANSWER
    args.func(args)


if __name__ == "__main__":
    main()
