#!/usr/bin/env python3
"""Train Jaeger's contextual stock-Whisper adapter against Gemma audio.

All executable code for this path is owned by this repository.  Model and
dataset paths may point elsewhere, but they are treated as read-only inputs.
The output is written directly in :mod:`jaeger_agent_omni` runtime format.

The native-vector stages never see a transcript, Gemma token, logit, or
generated answer:

    stock Whisper encoder states -> Jaeger adapter -> (T, 2560)
    same waveform -> Gemma audio_tower -> embed_audio -> (T, 2560)

An optional training-only character CTC head can read the contextual
bottleneck. Human transcripts supervise that disposable head rather than
pulling deployed Gemma-space vectors toward vocabulary rows. The head is
ignored by the runtime and exists to measure and improve lexical retention.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from jaeger_agent_omni.runtime.stock_whisper import (
    ARCH_GEMMA_NATIVE_CONTEXT,
    ARCH_GEMMA_NATIVE_TEMPORAL2,
    DEFAULT_GEMMA,
    ENCODER_DIMS,
    INPUT_DIM,
    OUTPUT_DIM,
    SAMPLE_RATE,
    validate_checkpoint,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO / "artifacts/stock-whisper-gemma-context1536.npz"
LEXICAL_ALPHABET = " abcdefghijklmnopqrstuvwxyz'0123456789"
LEXICAL_SIZE = len(LEXICAL_ALPHABET) + 1  # CTC blank is zero.
GPU_JOB_PATTERNS = (
    r"training/context_distill\.py",
    r"training/precompute_stock_encoder\.py",
    r"training/stock_whisper_gemma\.py\s+"
    r"(?:precompute|pretrain-ctc|distill|train|probe|geometry|wer|bench)",
    r"training/train\.py",
    r"cleanroom_(?:train|distill)\.py",
    r"train_audio_adapter\.py",
)


def refuse_gpu_contention() -> None:
    """Refuse concurrent Python GPU work known to invalidate measurements."""
    listing = subprocess.check_output(
        ["ps", "-Ao", "pid=,ppid=,comm=,args="], text=True)
    mine = {os.getpid(), os.getppid()}
    conflicts = []
    for line in listing.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) != 4 or int(fields[0]) in mine:
            continue
        command = fields[3]
        if "python" not in Path(fields[2]).name.lower():
            continue
        if any(re.search(pattern, command) for pattern in GPU_JOB_PATTERNS):
            conflicts.append(line.strip())
    if conflicts:
        raise SystemExit(
            "REFUSING: another known GPU job is active:\n  "
            + "\n  ".join(conflicts))


def _resolve_data_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def read_rows(cache: Path, splits: list[str], max_seconds: float) -> list[dict]:
    """Read feature/audio manifests without importing code from the cache."""
    rows = []
    for split in splits:
        root = cache.expanduser().resolve() / split
        manifest = root / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        count = 0
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seconds = float(row.get("seconds", 0.0))
            if max_seconds and seconds > max_seconds:
                continue
            if not row.get("feature") or not row.get("audio"):
                continue
            row["feature"] = _resolve_data_path(root, row["feature"])
            row["audio"] = _resolve_data_path(root, row["audio"])
            if not row["feature"].is_file() or not row["audio"].is_file():
                raise FileNotFoundError(
                    f"{manifest}: missing data for {row.get('id', '<unknown>')}")
            row["split"] = split
            rows.append(row)
            count += 1
        print(f"  data split {split}: {count} usable rows", flush=True)
    if not rows:
        raise RuntimeError("no rows contain both stock features and waveform")
    return rows


def read_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    pcm, rate = sf.read(path, dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly
        divisor = math.gcd(int(rate), SAMPLE_RATE)
        pcm = resample_poly(
            pcm, SAMPLE_RATE // divisor, int(rate) // divisor)
    return np.asarray(pcm, dtype=np.float32)


def lexical_text(value: str) -> str:
    """Return the deliberately small, stable character-CTC target alphabet."""
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lexical_ids(value: str) -> list[int]:
    lookup = {char: index + 1 for index, char in enumerate(LEXICAL_ALPHABET)}
    return [lookup[char] for char in lexical_text(value)]


def row_lexical_text(row: dict) -> str:
    """Read a human label across the owned and imported cache schemas."""
    return lexical_text(row.get("transcript") or row.get("text") or "")


def ctc_minimum_frames(targets: list[int]) -> int:
    return len(targets) + sum(
        left == right for left, right in zip(targets, targets[1:]))


def ctc_loss(logits, targets):
    """Differentiable single-example CTC for MLX."""
    import mlx.core as mx

    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    target_values = [int(value) for value in np.asarray(targets).tolist()]
    extended = [0]
    for value in target_values:
        extended.extend((value, 0))
    states = len(extended)
    negative = mx.array(-1e9, dtype=log_probs.dtype)
    initial = [log_probs[0, 0]]
    if states > 1:
        initial.append(log_probs[0, extended[1]])
    initial.extend([negative] * (states - len(initial)))
    alpha = mx.stack(initial)
    skip_mask = mx.array([
        index >= 2 and extended[index] != 0
        and extended[index] != extended[index - 2]
        for index in range(states)
    ])
    emissions = mx.array(extended, dtype=mx.int32)
    for frame in range(1, logits.shape[0]):
        previous_one = mx.concatenate((negative[None], alpha[:-1]))
        previous_two = (mx.concatenate((
            mx.broadcast_to(negative, (2,)), alpha[:-2]))
            if states > 2 else mx.broadcast_to(negative, (states,)))
        candidates = mx.stack((
            alpha, previous_one,
            mx.where(skip_mask, previous_two, negative),
        ))
        alpha = (mx.logsumexp(candidates, axis=0)
                 + mx.take(log_probs[frame], emissions))
    terminal = alpha[-1:] if states == 1 else alpha[-2:]
    return -mx.logsumexp(terminal) / max(1, len(target_values))


def greedy_lexical_text(logits) -> str:
    """Collapse a CTC character stream for a probe; never used at runtime."""
    ids = np.asarray(logits.argmax(axis=-1)).tolist()
    collapsed = []
    previous = None
    for value in ids:
        if value and value != previous:
            collapsed.append(LEXICAL_ALPHABET[int(value) - 1])
        previous = value
    return re.sub(r"\s+", " ", "".join(collapsed)).strip()


def word_distance(reference: str, hypothesis: str) -> tuple[int, int]:
    left, right = reference.split(), hypothesis.split()
    row = list(range(len(right) + 1))
    for index, word in enumerate(left, 1):
        nxt = [index]
        for other_index, other in enumerate(right, 1):
            nxt.append(min(
                nxt[-1] + 1, row[other_index] + 1,
                row[other_index - 1] + (word != other)))
        row = nxt
    return row[-1], len(left)


def load_teacher(model_path: Path):
    """Load Gemma's native tower/projector from an external model artifact."""
    from mlx_vlm.utils import load

    model, processor = load(str(model_path.expanduser().resolve()), lazy=False)
    if getattr(model, "audio_tower", None) is None:
        raise RuntimeError(f"{model_path} has no audio_tower")
    if getattr(model, "embed_audio", None) is None:
        raise RuntimeError(f"{model_path} has no embed_audio projector")
    return model, processor


def teacher_embeddings(model, processor, pcm: np.ndarray):
    """Return Gemma-native pre-language-model audio vectors, shape T x 2560."""
    import mlx.core as mx

    extractor = getattr(processor, "feature_extractor", processor)
    features = extractor(
        np.asarray(pcm, dtype=np.float32), sampling_rate=SAMPLE_RATE,
        return_tensors="np")
    keys = list(features.keys())
    if len(keys) < 2:
        raise RuntimeError("Gemma feature extractor returned no validity mask")
    mel = mx.array(np.asarray(features[keys[0]]))
    valid = mx.array(np.asarray(features[keys[1]]))
    tower = model.audio_tower(mel, ~valid)[0]
    embedded = model.embed_audio(tower)
    result = embedded[0] if embedded.ndim == 3 else embedded
    if result.shape[-1] != OUTPUT_DIM:
        raise RuntimeError(
            f"Gemma teacher width is {result.shape[-1]}, expected {OUTPUT_DIM}")
    return result


def add_context_arrays(values: dict[str, np.ndarray], seed: int,
                       hidden: int, heads: int,
                       kernel: int) -> dict[str, np.ndarray]:
    """Add an exact-zero contextual residual to a validated temporal2 model."""
    architecture = str(values["adapter_arch"])
    if architecture != ARCH_GEMMA_NATIVE_TEMPORAL2:
        raise ValueError(
            "context upgrade requires a temporal2 runtime checkpoint; got "
            f"{architecture!r}")
    if hidden <= 0 or heads <= 0 or hidden % heads:
        raise ValueError("context hidden must divide evenly across heads")
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("context kernel must be a positive odd value")
    rng = np.random.default_rng(seed)

    def he(shape, fan_in):
        return (rng.standard_normal(shape) * math.sqrt(2.0 / fan_in)).astype(
            np.float32)

    upgraded = dict(values)
    upgraded.update({
        "adapter_arch": np.array(ARCH_GEMMA_NATIVE_CONTEXT),
        "context_hidden": np.array(hidden, dtype=np.int32),
        "context_heads": np.array(heads, dtype=np.int32),
        "context_kernel": np.array(kernel, dtype=np.int32),
        "ctx_rms_pre_gamma": np.ones(OUTPUT_DIM, np.float32),
        "ctx_w1": he((hidden * 2, OUTPUT_DIM), OUTPUT_DIM),
        "ctx_rms_attn_gamma": np.ones(hidden, np.float32),
        "ctx_qkv_w": he((hidden * 3, hidden), hidden),
        "ctx_attn_w": he((hidden, hidden), hidden),
        "ctx_rms_conv_gamma": np.ones(hidden, np.float32),
        "ctx_dw": he((hidden, kernel, 1), kernel),
        "ctx_rms_mid_gamma": np.ones(hidden, np.float32),
        # This is the preservation guarantee: step zero equals the resume.
        "ctx_w2": np.zeros((OUTPUT_DIM, hidden), np.float32),
    })
    return upgraded


def prepare_context_arrays(values: dict[str, np.ndarray], seed: int,
                           hidden: int, heads: int,
                           kernel: int) -> dict[str, np.ndarray]:
    """Upgrade temporal2 once, or continue a matching context checkpoint."""
    architecture = str(values["adapter_arch"])
    if architecture == ARCH_GEMMA_NATIVE_TEMPORAL2:
        return add_context_arrays(values, seed, hidden, heads, kernel)
    if architecture != ARCH_GEMMA_NATIVE_CONTEXT:
        raise ValueError(
            "resume must be a temporal2 or context runtime checkpoint; got "
            f"{architecture!r}")
    actual = (int(values["context_hidden"]), int(values["context_heads"]),
              int(values["context_kernel"]))
    requested = (hidden, heads, kernel)
    if actual != requested:
        raise ValueError(
            f"resume context shape {actual} != requested {requested}")
    return dict(values)


def retarget_encoder_arrays(values: dict[str, np.ndarray], encoder: str,
                            seed: int) -> dict[str, np.ndarray]:
    """Create a fresh input stack for another stock Whisper encoder.

    The proven Gemma-width output, residual, temporal, and context branches
    are retained. Only the encoder-facing norm/projection is reinitialized;
    it is trained first while every downstream branch remains frozen.
    """
    if encoder not in ENCODER_DIMS:
        raise ValueError(f"unsupported encoder {encoder!r}")
    input_dim = ENCODER_DIMS[encoder]
    pool_step = int(values["pool_step"])
    fan_in = input_dim * pool_step
    rng = np.random.default_rng(seed)
    retargeted = dict(values)
    retargeted.update({
        "encoder": np.array(encoder),
        "input_dim": np.array(input_dim, dtype=np.int32),
        "rms_pre_gamma": np.ones(fan_in, np.float32),
        "stack_w1": (rng.standard_normal((OUTPUT_DIM * 2, fan_in))
                     * math.sqrt(2.0 / fan_in)).astype(np.float32),
    })
    return retargeted


def build_adapter(values: dict[str, np.ndarray], seed: int = 0,
                  lexical_arch: str = "linear",
                  lexical_hidden: int = 384):
    """Build the exact architecture consumed by the owned runtime."""
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(seed)

    if str(values["adapter_arch"]) != ARCH_GEMMA_NATIVE_CONTEXT:
        raise ValueError("context trainer requires contextual runtime arrays")
    pool_step = int(values["pool_step"])
    input_dim = int(values.get("input_dim", INPUT_DIM))
    residual_hidden = int(values["residual_hidden"])
    temporal_hidden = int(values["temporal_hidden"])
    temporal2_hidden = int(values["temporal2_hidden"])
    temporal2_kernel = int(values["temporal2_kernel"])
    context_hidden = int(values["context_hidden"])
    context_kernel = int(values["context_kernel"])
    context_heads = int(values["context_heads"])

    saved_lexical_arch = str(values.get("lexical_head_arch", ""))
    if saved_lexical_arch == lexical_arch:
        lexical_hidden = int(values.get("lexical_head_hidden", lexical_hidden))

    class LexicalHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch = lexical_arch
            if lexical_arch == "linear":
                self.output = nn.Linear(
                    context_hidden, LEXICAL_SIZE, bias=True)
            elif lexical_arch == "bigru":
                self.forward_gru = nn.GRU(context_hidden, lexical_hidden)
                self.backward_gru = nn.GRU(context_hidden, lexical_hidden)
                self.output = nn.Linear(
                    lexical_hidden * 2, LEXICAL_SIZE, bias=True)
            else:
                raise ValueError(f"unknown lexical head {lexical_arch!r}")

        def __call__(self, context):
            if self.arch == "linear":
                return self.output(context)
            forward = self.forward_gru(context)
            backward = self.backward_gru(context[::-1])[::-1]
            return self.output(mx.concatenate((forward, backward), axis=-1))

        @staticmethod
        def _load_gru(module, prefix):
            module.Wx = mx.array(values[f"{prefix}_wx"])
            module.Wh = mx.array(values[f"{prefix}_wh"])
            module.b = mx.array(values[f"{prefix}_b"])
            module.bhn = mx.array(values[f"{prefix}_bhn"])

        def load_saved(self):
            if saved_lexical_arch != self.arch:
                return
            if self.arch == "linear":
                self.output.weight = mx.array(values["lexical_head_w"])
                self.output.bias = mx.array(values["lexical_head_b"])
                return
            self._load_gru(self.forward_gru, "lexical_forward")
            self._load_gru(self.backward_gru, "lexical_backward")
            self.output.weight = mx.array(values["lexical_output_w"])
            self.output.bias = mx.array(values["lexical_output_b"])

    class ContextAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.context_heads = context_heads
            self.pool_step = pool_step
            self.input_dim = input_dim
            self.pre = nn.RMSNorm(input_dim * pool_step, eps=1e-6)
            self.up = nn.Linear(
                input_dim * pool_step, OUTPUT_DIM * 2, bias=False)
            self.mid = nn.RMSNorm(OUTPUT_DIM, eps=1e-6)
            self.out = nn.Linear(OUTPUT_DIM, OUTPUT_DIM, bias=False)
            self.res_pre = nn.RMSNorm(OUTPUT_DIM, eps=1e-6)
            self.res_up = nn.Linear(
                OUTPUT_DIM, residual_hidden * 2, bias=False)
            self.res_mid = nn.RMSNorm(residual_hidden, eps=1e-6)
            self.res_out = nn.Linear(
                residual_hidden, OUTPUT_DIM, bias=False)
            self.temp_pre = nn.RMSNorm(OUTPUT_DIM, eps=1e-6)
            self.temp_up = nn.Linear(
                OUTPUT_DIM, temporal_hidden * 2, bias=False)
            self.temp_conv = nn.Conv1d(
                temporal_hidden, temporal_hidden, 3, padding=1,
                groups=temporal_hidden, bias=False)
            self.temp_mid = nn.RMSNorm(temporal_hidden, eps=1e-6)
            self.temp_out = nn.Linear(
                temporal_hidden, OUTPUT_DIM, bias=False)
            self.temp2_pre = nn.RMSNorm(OUTPUT_DIM, eps=1e-6)
            self.temp2_up = nn.Linear(
                OUTPUT_DIM, temporal2_hidden * 2, bias=False)
            self.temp2_conv = nn.Conv1d(
                temporal2_hidden, temporal2_hidden, temporal2_kernel,
                padding=temporal2_kernel // 2, groups=temporal2_hidden,
                bias=False)
            self.temp2_mid = nn.RMSNorm(temporal2_hidden, eps=1e-6)
            self.temp2_out = nn.Linear(
                temporal2_hidden, OUTPUT_DIM, bias=False)
            self.ctx_pre = nn.RMSNorm(OUTPUT_DIM, eps=1e-6)
            self.ctx_up = nn.Linear(
                OUTPUT_DIM, context_hidden * 2, bias=False)
            self.ctx_attn_norm = nn.RMSNorm(context_hidden, eps=1e-6)
            self.ctx_qkv = nn.Linear(
                context_hidden, context_hidden * 3, bias=False)
            self.ctx_attn_out = nn.Linear(
                context_hidden, context_hidden, bias=False)
            self.ctx_conv_norm = nn.RMSNorm(context_hidden, eps=1e-6)
            self.ctx_conv = nn.Conv1d(
                context_hidden, context_hidden, context_kernel,
                padding=context_kernel // 2, groups=context_hidden,
                bias=False)
            self.ctx_mid = nn.RMSNorm(context_hidden, eps=1e-6)
            self.ctx_out = nn.Linear(context_hidden, OUTPUT_DIM, bias=False)
            self.lexical_head = LexicalHead()

            assignments = (
                (self.pre, "rms_pre_gamma"),
                (self.up, "stack_w1"),
                (self.mid, "rms_mid_gamma"),
                (self.out, "stack_w2"),
                (self.res_pre, "res_rms_pre_gamma"),
                (self.res_up, "res_w1"),
                (self.res_mid, "res_rms_mid_gamma"),
                (self.res_out, "res_w2"),
                (self.temp_pre, "temp_rms_pre_gamma"),
                (self.temp_up, "temp_w1"),
                (self.temp_conv, "temp_dw"),
                (self.temp_mid, "temp_rms_mid_gamma"),
                (self.temp_out, "temp_w2"),
                (self.temp2_pre, "temp2_rms_pre_gamma"),
                (self.temp2_up, "temp2_w1"),
                (self.temp2_conv, "temp2_dw"),
                (self.temp2_mid, "temp2_rms_mid_gamma"),
                (self.temp2_out, "temp2_w2"),
                (self.ctx_pre, "ctx_rms_pre_gamma"),
                (self.ctx_up, "ctx_w1"),
                (self.ctx_attn_norm, "ctx_rms_attn_gamma"),
                (self.ctx_qkv, "ctx_qkv_w"),
                (self.ctx_attn_out, "ctx_attn_w"),
                (self.ctx_conv_norm, "ctx_rms_conv_gamma"),
                (self.ctx_conv, "ctx_dw"),
                (self.ctx_mid, "ctx_rms_mid_gamma"),
                (self.ctx_out, "ctx_w2"),
            )
            for module, key in assignments:
                module.weight = mx.array(values[key])
            if saved_lexical_arch == lexical_arch:
                self.lexical_head.load_saved()
            elif lexical_arch == "linear":
                rng = np.random.default_rng(seed)
                self.lexical_head.output.weight = mx.array(
                    (rng.standard_normal((LEXICAL_SIZE, context_hidden))
                     * math.sqrt(1.0 / context_hidden)).astype(np.float32))
                self.lexical_head.output.bias = mx.zeros((LEXICAL_SIZE,))

        def __call__(self, features, return_context=False):
            usable = (features.shape[0] // self.pool_step) * self.pool_step
            if not usable:
                features = mx.concatenate([features, features], axis=0)
                usable = self.pool_step
            stacked = features[:usable].reshape(
                -1, self.input_dim * self.pool_step)
            gate, value = mx.split(self.up(self.pre(stacked)), 2, axis=-1)
            result = self.out(self.mid(nn.silu(gate) * value))

            gate, value = mx.split(
                self.res_up(self.res_pre(result)), 2, axis=-1)
            result = result + self.res_out(
                self.res_mid(nn.silu(gate) * value))
            gate, value = mx.split(
                self.temp_up(self.temp_pre(result)), 2, axis=-1)
            temporal = self.temp_conv(
                (nn.silu(gate) * value)[None, :, :])[0]
            result = result + self.temp_out(self.temp_mid(temporal))
            gate, value = mx.split(
                self.temp2_up(self.temp2_pre(result)), 2, axis=-1)
            temporal = self.temp2_conv(
                (nn.silu(gate) * value)[None, :, :])[0]
            result = result + self.temp2_out(self.temp2_mid(temporal))

            gate, value = mx.split(
                self.ctx_up(self.ctx_pre(result)), 2, axis=-1)
            context = nn.silu(gate) * value
            normalized = self.ctx_attn_norm(context)
            query, key, value = mx.split(
                self.ctx_qkv(normalized), 3, axis=-1)
            head_dim = query.shape[-1] // self.context_heads
            query = query.reshape(-1, self.context_heads, head_dim)
            key = key.reshape(-1, self.context_heads, head_dim)
            value = value.reshape(-1, self.context_heads, head_dim)
            scores = mx.einsum(
                "thd,shd->hts", query, key) / math.sqrt(head_dim)
            attended = mx.einsum(
                "hts,shd->thd", mx.softmax(scores, axis=-1), value)
            context = context + self.ctx_attn_out(
                attended.reshape(context.shape))
            convolved = self.ctx_conv(
                self.ctx_conv_norm(context)[None, :, :])[0]
            context = context + nn.silu(convolved)
            result = result + self.ctx_out(self.ctx_mid(context))
            return (result, context) if return_context else result

    adapter = ContextAdapter()
    mx.eval(adapter.parameters())
    return adapter


def freeze_for_branch(adapter, branch: str) -> None:
    """Expose only the requested branch to MLX's parameter traversal."""
    if branch in ("context", "context_lexical"):
        modules = (
            adapter.pre, adapter.up, adapter.mid, adapter.out,
            adapter.res_pre, adapter.res_up, adapter.res_mid, adapter.res_out,
            adapter.temp_pre, adapter.temp_up, adapter.temp_conv,
            adapter.temp_mid, adapter.temp_out, adapter.temp2_pre,
            adapter.temp2_up, adapter.temp2_conv, adapter.temp2_mid,
            adapter.temp2_out)
        if branch == "context":
            modules += (adapter.lexical_head,)
    elif branch == "lexical_head":
        modules = (
            adapter.pre, adapter.up, adapter.mid, adapter.out,
            adapter.res_pre, adapter.res_up, adapter.res_mid, adapter.res_out,
            adapter.temp_pre, adapter.temp_up, adapter.temp_conv,
            adapter.temp_mid, adapter.temp_out, adapter.temp2_pre,
            adapter.temp2_up, adapter.temp2_conv, adapter.temp2_mid,
            adapter.temp2_out, adapter.ctx_pre, adapter.ctx_up,
            adapter.ctx_attn_norm, adapter.ctx_qkv, adapter.ctx_attn_out,
            adapter.ctx_conv_norm, adapter.ctx_conv, adapter.ctx_mid,
            adapter.ctx_out)
    elif branch == "input":
        modules = (
            adapter.mid, adapter.out, adapter.res_pre, adapter.res_up,
            adapter.res_mid, adapter.res_out, adapter.temp_pre,
            adapter.temp_up, adapter.temp_conv, adapter.temp_mid,
            adapter.temp_out, adapter.temp2_pre, adapter.temp2_up,
            adapter.temp2_conv, adapter.temp2_mid, adapter.temp2_out,
            adapter.ctx_pre, adapter.ctx_up, adapter.ctx_attn_norm,
            adapter.ctx_qkv, adapter.ctx_attn_out, adapter.ctx_conv_norm,
            adapter.ctx_conv, adapter.ctx_mid, adapter.ctx_out)
        modules += (adapter.lexical_head,)
    elif branch == "input_full":
        modules = (
            adapter.res_pre, adapter.res_up, adapter.res_mid, adapter.res_out,
            adapter.temp_pre, adapter.temp_up, adapter.temp_conv,
            adapter.temp_mid, adapter.temp_out, adapter.temp2_pre,
            adapter.temp2_up, adapter.temp2_conv, adapter.temp2_mid,
            adapter.temp2_out, adapter.ctx_pre, adapter.ctx_up,
            adapter.ctx_attn_norm, adapter.ctx_qkv, adapter.ctx_attn_out,
            adapter.ctx_conv_norm, adapter.ctx_conv, adapter.ctx_mid,
            adapter.ctx_out)
        modules += (adapter.lexical_head,)
    else:
        raise ValueError(f"unknown train branch {branch!r}")
    for module in modules:
        module.freeze()


def adapter_values(adapter, base: dict[str, np.ndarray], step: int,
                   args) -> dict[str, np.ndarray]:
    """Serialize only model values and auditable training metadata."""
    values = dict(base)
    values.update({
        "rms_pre_gamma": np.array(adapter.pre.weight),
        "stack_w1": np.array(adapter.up.weight),
        "rms_mid_gamma": np.array(adapter.mid.weight),
        "stack_w2": np.array(adapter.out.weight),
        "res_rms_pre_gamma": np.array(adapter.res_pre.weight),
        "res_w1": np.array(adapter.res_up.weight),
        "res_rms_mid_gamma": np.array(adapter.res_mid.weight),
        "res_w2": np.array(adapter.res_out.weight),
        "temp_rms_pre_gamma": np.array(adapter.temp_pre.weight),
        "temp_w1": np.array(adapter.temp_up.weight),
        "temp_dw": np.array(adapter.temp_conv.weight),
        "temp_rms_mid_gamma": np.array(adapter.temp_mid.weight),
        "temp_w2": np.array(adapter.temp_out.weight),
        "temp2_rms_pre_gamma": np.array(adapter.temp2_pre.weight),
        "temp2_w1": np.array(adapter.temp2_up.weight),
        "temp2_dw": np.array(adapter.temp2_conv.weight),
        "temp2_rms_mid_gamma": np.array(adapter.temp2_mid.weight),
        "temp2_w2": np.array(adapter.temp2_out.weight),
        "ctx_rms_pre_gamma": np.array(adapter.ctx_pre.weight),
        "ctx_w1": np.array(adapter.ctx_up.weight),
        "ctx_rms_attn_gamma": np.array(adapter.ctx_attn_norm.weight),
        "ctx_qkv_w": np.array(adapter.ctx_qkv.weight),
        "ctx_attn_w": np.array(adapter.ctx_attn_out.weight),
        "ctx_rms_conv_gamma": np.array(adapter.ctx_conv_norm.weight),
        "ctx_dw": np.array(adapter.ctx_conv.weight),
        "ctx_rms_mid_gamma": np.array(adapter.ctx_mid.weight),
        "ctx_w2": np.array(adapter.ctx_out.weight),
        "lexical_alphabet": np.array(LEXICAL_ALPHABET),
        "lexical_head_arch": np.array(args.lexical_head_arch),
        "lexical_head_hidden": np.array(
            args.lexical_head_hidden, dtype=np.int32),
        "step": np.array(step, dtype=np.int32),
        "stage": np.array("native_context_distillation"),
        "learning_rate": np.array(args.learning_rate, dtype=np.float32),
        "cosine_weight": np.array(args.cosine_weight, dtype=np.float32),
        "speaker_weight": np.array(args.speaker_weight, dtype=np.float32),
        "lexical_weight": np.array(args.lexical_weight, dtype=np.float32),
        "teacher_lineage": np.array("google_gemma_native_audio_tower"),
        "trainer": np.array("training/context_distill.py"),
        "train_branch": np.array(args.train_branch),
    })
    # Training-only diagnostic head. The runtime ignores these arrays and
    # never instantiates this layer.
    head = adapter.lexical_head
    if args.lexical_head_arch == "linear":
        values.update({
            "lexical_head_w": np.array(head.output.weight),
            "lexical_head_b": np.array(head.output.bias),
        })
    else:
        values.update({
            "lexical_forward_wx": np.array(head.forward_gru.Wx),
            "lexical_forward_wh": np.array(head.forward_gru.Wh),
            "lexical_forward_b": np.array(head.forward_gru.b),
            "lexical_forward_bhn": np.array(head.forward_gru.bhn),
            "lexical_backward_wx": np.array(head.backward_gru.Wx),
            "lexical_backward_wh": np.array(head.backward_gru.Wh),
            "lexical_backward_b": np.array(head.backward_gru.b),
            "lexical_backward_bhn": np.array(head.backward_gru.bhn),
            "lexical_output_w": np.array(head.output.weight),
            "lexical_output_b": np.array(head.output.bias),
        })
    values[f"{args.train_branch}_step"] = np.array(step, dtype=np.int32)
    return values


def atomic_save(path: Path, values: dict[str, np.ndarray]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
                dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def probe(model, processor, adapter, rows: list[dict], step: int) -> None:
    """Measure teacher cosine and cross-clip collapse on a fixed mini-corpus."""
    import mlx.core as mx

    cosines = []
    summaries = []
    for row in rows[:4]:
        features = mx.array(np.load(row["feature"]).astype(np.float32))
        target = teacher_embeddings(model, processor, read_audio(row["audio"]))
        prediction = adapter(features)
        length = min(prediction.shape[0], target.shape[0])
        left = prediction[:length]
        right = target[:length]
        left = left / (mx.linalg.norm(left, axis=-1, keepdims=True) + 1e-6)
        right = right / (mx.linalg.norm(right, axis=-1, keepdims=True) + 1e-6)
        cosines.append(float(mx.mean(mx.sum(left * right, axis=-1))))
        summaries.append(np.asarray(mx.mean(prediction, axis=0)))
    normalized = np.stack(summaries)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True) + 1e-9
    similarity = normalized @ normalized.T
    cross_clip = float(similarity[
        np.triu_indices(len(normalized), 1)].mean())
    health = "collapsed" if cross_clip > 0.995 else (
        "suspect" if cross_clip > 0.98 else "healthy")
    print(f"    probe@{step}: cosine={np.mean(cosines):.4f} "
          f"cross_clip={cross_clip:.4f} {health}", flush=True)


def lexical_probe(adapter, rows: list[dict], step: int) -> None:
    """Report held-out greedy CTC WER from the disposable lexical head."""
    import mlx.core as mx

    edits = words = usable = 0
    examples = []
    for row in rows:
        target = row_lexical_text(row)
        targets = lexical_ids(target)
        features = mx.array(np.load(row["feature"]).astype(np.float32))
        _, context = adapter(features, return_context=True)
        if not targets or context.shape[0] < ctc_minimum_frames(targets):
            continue
        hypothesis = greedy_lexical_text(adapter.lexical_head(context))
        distance, count = word_distance(target, hypothesis)
        edits += distance
        words += count
        usable += 1
        if len(examples) < 2:
            examples.append((target, hypothesis))
    print(f"    lexical@{step}: {usable} clips {edits}/{words} "
          f"WER={edits / max(1, words):.2%}", flush=True)
    for target, hypothesis in examples:
        print(f"      ref={target!r} hyp={hypothesis!r}", flush=True)


def sampling_probabilities(rows: list[dict], short_weight: float,
                           short_seconds: float,
                           balance_sources: bool) -> np.ndarray:
    """Return clip probabilities, optionally giving every source equal mass."""
    weights = np.array([
        short_weight if float(row.get("seconds", 0.0)) <= short_seconds else 1.0
        for row in rows
    ], dtype=np.float64)
    if balance_sources:
        source_totals: dict[str, float] = {}
        for row, weight in zip(rows, weights):
            source = str(row.get("source", "unknown"))
            source_totals[source] = source_totals.get(source, 0.0) + float(weight)
        for index, row in enumerate(rows):
            source = str(row.get("source", "unknown"))
            weights[index] /= source_totals[source]
    return weights / weights.sum()


def command_train(args) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    refuse_gpu_contention()
    rows = read_rows(args.cache, [args.split, *args.extra_split],
                     args.max_seconds)
    for cache, split in args.extra_data:
        rows.extend(read_rows(Path(cache), [split], args.max_seconds))
    lexical_mode = args.train_branch in ("lexical_head", "context_lexical")
    if lexical_mode:
        rows = [row for row in rows if lexical_ids(row_lexical_text(row))]
        rows = [
            row for row in rows
            if max(1, np.load(row["feature"], mmap_mode="r").shape[0] // 2)
            >= ctc_minimum_frames(lexical_ids(row_lexical_text(row)))
        ]
        if not rows:
            raise RuntimeError("lexical stage needs human transcripts")
        lexical_validation = read_rows(
            args.cache, args.lexical_probe_split, args.max_seconds)
        lexical_validation = [
            row for row in lexical_validation
            if lexical_ids(row_lexical_text(row))
        ][:args.lexical_probe_examples]
    else:
        lexical_validation = []
    sources = Counter(str(row.get("source", "unknown")) for row in rows)
    print(f"{len(rows)} training clips; sources={dict(sources)}", flush=True)

    speaker_rows: dict[str, list[dict]] = {}
    for row in rows:
        speaker = str(row.get("speaker") or "").strip()
        if speaker:
            speaker_rows.setdefault(speaker, []).append(row)
    speaker_keys = [key for key, group in speaker_rows.items()
                    if len(group) >= 2]
    if args.speaker_weight and len(speaker_keys) < 2:
        raise ValueError(
            "speaker loss needs two repeated labeled speakers")
    print(f"  repeated labeled speakers: {len(speaker_keys)}", flush=True)

    resume = validate_checkpoint(args.resume)
    values = prepare_context_arrays(
        resume, args.seed, args.context_hidden, args.context_heads,
        args.context_kernel)
    if args.encoder:
        values = retarget_encoder_arrays(values, args.encoder, args.seed)
    adapter = build_adapter(
        values, args.seed, args.lexical_head_arch, args.lexical_head_hidden)
    freeze_for_branch(adapter, args.train_branch)
    input_dim = int(values.get("input_dim", INPUT_DIM))
    first_shape = np.load(rows[0]["feature"], mmap_mode="r").shape
    if len(first_shape) != 2 or first_shape[1] != input_dim:
        raise ValueError(
            f"feature width {first_shape} does not match input_dim {input_dim}")
    parameter_count = sum(int(np.prod(value.shape)) for key, value in values.items()
                          if key.endswith(("gamma", "_w", "_w1", "_w2", "_dw"))
                          or key in ("stack_w1", "stack_w2"))
    print(f"  architecture: {ARCH_GEMMA_NATIVE_CONTEXT}", flush=True)
    print(f"  adapter parameters: {parameter_count:,}; training "
          f"{args.train_branch!r} branch only",
          flush=True)
    if args.train_branch == "lexical_head":
        model = processor = None
        print("  lexical-head diagnostic: native adapter frozen; "
              "Gemma teacher not loaded", flush=True)
    else:
        print(f"  loading read-only Gemma teacher: {args.model}", flush=True)
        model, processor = load_teacher(args.model)

    probabilities = sampling_probabilities(
        rows, args.short_weight, args.short_seconds, args.balance_sources)
    source_mass: dict[str, float] = {}
    for row, probability in zip(rows, probabilities):
        source = str(row.get("source", "unknown"))
        source_mass[source] = source_mass.get(source, 0.0) + float(probability)
    print("  sampling source mass: " + str({
        source: round(mass, 4) for source, mass in source_mass.items()
    }), flush=True)
    rng = np.random.default_rng(args.seed)
    schedule = optim.join_schedules(
        [optim.linear_schedule(0.0, args.learning_rate, max(1, args.warmup)),
         optim.step_decay(args.learning_rate, 1.0, 10 ** 9)],
        [max(1, args.warmup)])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.0)

    def loss_fn(module, encoded, target, peer, relation, character_targets):
        prediction, context = module(encoded, return_context=True)
        if args.train_branch == "lexical_head":
            loss = ctc_loss(module.lexical_head(context), character_targets)
        else:
            length = min(prediction.shape[0], target.shape[0])
            prediction = prediction[:length]
            target = target[:length]
            difference = prediction - target
            absolute = mx.abs(difference)
            huber = mx.mean(mx.where(
                absolute < 1.0, 0.5 * mx.square(difference), absolute - 0.5))
            left = prediction / (
                mx.linalg.norm(prediction, axis=-1, keepdims=True) + 1e-6)
            right = target / (
                mx.linalg.norm(target, axis=-1, keepdims=True) + 1e-6)
            loss = huber + args.cosine_weight * (
                1.0 - mx.mean(mx.sum(left * right, axis=-1)))
            if lexical_mode:
                loss = loss + args.lexical_weight * ctc_loss(
                    module.lexical_head(context), character_targets)
        if relation:
            _, peer_context = module(peer, return_context=True)
            summary = mx.mean(context, axis=0)
            other = mx.mean(peer_context, axis=0)
            similarity = mx.sum(summary * other) / (
                (mx.linalg.norm(summary) + 1e-6)
                * (mx.linalg.norm(other) + 1e-6))
            speaker_loss = (1.0 - similarity if relation > 0 else
                            mx.square(mx.maximum(
                                similarity - args.speaker_margin, 0.0)))
            loss = loss + args.speaker_weight * speaker_loss
        return loss

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)
    losses = []
    started = time.time()
    inferred_start = int(resume.get(
        f"{args.train_branch}_step",
        resume.get("context_step", 0) if args.train_branch == "context" else 0))
    start_step = args.start_step if args.start_step is not None else inferred_start
    end_step = start_step + args.steps
    print(f"  optimization updates: {args.steps}; context steps "
          f"{start_step + 1}..{end_step}", flush=True)
    for update in range(1, args.steps + 1):
        step = start_step + update
        row = rows[int(rng.choice(len(rows), p=probabilities))]
        encoded = np.load(row["feature"]).astype(np.float32)
        characters = lexical_ids(row_lexical_text(row))
        target = (mx.zeros((1, OUTPUT_DIM), dtype=mx.float32)
                  if model is None else mx.stop_gradient(teacher_embeddings(
                      model, processor, read_audio(row["audio"]))))
        peer = encoded
        relation = 0
        speaker = str(row.get("speaker") or "").strip()
        if args.train_branch != "lexical_head" and args.speaker_weight \
                and speaker in speaker_rows \
                and len(speaker_rows[speaker]) >= 2:
            if rng.random() < 0.5:
                group = speaker_rows[speaker]
                candidate = group[int(rng.integers(len(group)))]
                while candidate.get("id") == row.get("id"):
                    candidate = group[int(rng.integers(len(group)))]
                relation = 1
            else:
                other_keys = [key for key in speaker_keys if key != speaker]
                other_speaker = other_keys[int(rng.integers(len(other_keys)))]
                group = speaker_rows[other_speaker]
                candidate = group[int(rng.integers(len(group)))]
                relation = -1
            peer = np.load(candidate["feature"]).astype(np.float32)
        loss, gradients = loss_and_grad(
            adapter, mx.array(encoded), target, mx.array(peer), relation,
            mx.array(characters, dtype=mx.int32))
        gradients, _ = optim.clip_grad_norm(gradients, args.clip)
        optimizer.update(adapter, gradients)
        mx.eval(adapter.parameters(), optimizer.state)
        losses.append(float(loss))
        if step == 1 or step % args.log_every == 0:
            recent = float(np.mean(losses[-args.log_every:]))
            print(f"step {step:5d}/{end_step} loss={float(loss):.5f} "
                  f"recent={recent:.5f} "
                  f"{(time.time() - started) / update:.2f}s/step", flush=True)
        if args.probe_every and step % args.probe_every == 0:
            if model is not None:
                probe(model, processor, adapter, rows, step)
            if lexical_mode:
                lexical_probe(adapter, lexical_validation, step)
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            snapshot = args.output.with_name(
                f"{args.output.stem}.step{step:06d}{args.output.suffix}")
            atomic_save(snapshot, adapter_values(adapter, values, step, args))
            print(f"    checkpoint: {snapshot}", flush=True)
    final_values = adapter_values(adapter, values, end_step, args)
    atomic_save(args.output, final_values)
    validate_checkpoint(args.output)
    print(f"saved runtime checkpoint: {args.output}", flush=True)
    print(f"loss first10={np.mean(losses[:10]):.5f} "
          f"last10={np.mean(losses[-10:]):.5f}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True,
                        help="read-only dataset/feature cache")
    parser.add_argument("--split", default="train100")
    parser.add_argument("--extra-split", action="append", default=[])
    parser.add_argument("--extra-data", nargs=2, action="append", default=[],
                        metavar=("CACHE", "SPLIT"),
                        help="additional read-only cache/split for replay")
    parser.add_argument("--model", type=Path, default=DEFAULT_GEMMA,
                        help="read-only Gemma MLX model with audio tower")
    parser.add_argument("--resume", type=Path, required=True,
                        help="owned temporal2 runtime checkpoint")
    parser.add_argument("--encoder", choices=tuple(ENCODER_DIMS),
                        help="retarget the encoder-facing stack to this model")
    parser.add_argument("--train-branch",
                        choices=("context", "input", "input_full",
                                 "lexical_head", "context_lexical"),
                        default="context")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--start-step", type=int,
                        help="override inferred context step when importing a "
                             "checkpoint without training metadata")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--speaker-weight", type=float, default=0.01)
    parser.add_argument("--speaker-margin", type=float, default=0.3)
    parser.add_argument("--lexical-weight", type=float, default=0.05,
                        help="character CTC weight for context_lexical")
    parser.add_argument("--lexical-head-arch", choices=("linear", "bigru"),
                        default="linear")
    parser.add_argument("--lexical-head-hidden", type=int, default=384)
    parser.add_argument("--lexical-probe-split", action="append",
                        default=None,
                        help="held-out split for lexical probes (default: "
                             "devclean)")
    parser.add_argument("--lexical-probe-examples", type=int, default=32)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--short-seconds", type=float, default=3.0)
    parser.add_argument("--short-weight", type=float, default=2.0)
    parser.add_argument(
        "--balance-sources", action="store_true",
        help="give each manifest source equal sampling mass")
    parser.add_argument("--context-hidden", type=int, default=1536)
    parser.add_argument("--context-heads", type=int, default=12)
    parser.add_argument("--context-kernel", type=int, default=31)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--probe-every", type=int, default=1000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.lexical_probe_split is None:
        args.lexical_probe_split = ["devclean"]
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    command_train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
