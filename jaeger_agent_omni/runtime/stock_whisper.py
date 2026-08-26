"""Qwen-free stock-Whisper audio front end for frozen Gemma.

The runtime boundary is deliberately small and explicit::

    PCM -> stock OpenAI Whisper encoder
        -> fresh Jaeger MLP -> temporal pool -> Gemma 4 E4B -> answer tokens

Whisper's decoder is never run. Hugging Face currently constructs the complete
``WhisperModel`` while loading the published checkpoint, after which this
runtime retains only ``model.encoder``; this experimental control is therefore
Qwen-free and transcript-free, but not an encoder-only model-load optimization.
The adapter checkpoint contains only our two affine layers; it cannot silently
carry MiniCPM or Qwen weights.

This first implementation is turn-bounded: Whisper is bidirectional, so a
completed turn is encoded once.  The microphone queue can continue collecting
while that worker runs, but causal incremental Whisper is a separate measured
milestone.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
POOL_STEP = 5
INPUT_DIM = 1024
HIDDEN_DIM = 4096
OUTPUT_DIM = 2560
TARGET_NORM = 61.0
OUTPUT_MODE_NORMALIZED = "unit_norm_scaled"
OUTPUT_MODE_RAW = "raw"
OUTPUT_MODE_DIRECT = "direct"
AUDIO_MARKER = "__JAEGER_STOCK_WHISPER_AUDIO__"
DEFAULT_WHISPER = "openai/whisper-medium"
LARGE_WHISPER = "openai/whisper-large-v3-turbo"
ENCODER_DIMS = {
    "openai/whisper-medium": 1024,
    "openai/whisper-medium.en": 1024,
    LARGE_WHISPER: 1280,
}
SUPPORTED_WHISPER = frozenset(ENCODER_DIMS)
DEFAULT_GEMMA = Path.home() / "models/gemma-4-E4B-it-4bit-mlx"
INPUT_BOUNDARY = "stock_whisper_encoder_50hz"
ARCH_PAIRED = "paired_mlp_v1"
ARCH_GEMMA_NATIVE = "gemma_native_distill_v1"
ARCH_GEMMA_NATIVE_STACK = "gemma_native_stack_swiglu_v1"
ARCH_GEMMA_NATIVE_RESIDUAL = "gemma_native_stack_swiglu_residual_v1"
ARCH_GEMMA_NATIVE_TEMPORAL = (
    "gemma_native_stack_swiglu_residual_temporal_v1")
ARCH_GEMMA_NATIVE_TEMPORAL2 = (
    "gemma_native_stack_swiglu_residual_temporal2_v1")
ARCH_GEMMA_NATIVE_CONTEXT = (
    "gemma_native_stack_swiglu_residual_temporal2_context_v1")
WHISPER_WINDOW_S = 28
WHISPER_OVERLAP_S = 1


class StockWhisperFrontEnd:
    """Frozen stock Whisper-medium encoder with no decoder.

    Hugging Face's public encoder rejects variable-length mel inputs even
    though every constituent layer supports them.  Driving those layers
    directly avoids padding every short utterance to 30 seconds and matches
    the measured MiniCPM precompute recipe without importing its weights.
    """

    def __init__(self, model_id: str = DEFAULT_WHISPER,
                 device: str = "auto") -> None:
        import torch
        from transformers import WhisperFeatureExtractor, WhisperModel

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "mps" else torch.float32
        model = WhisperModel.from_pretrained(model_id, torch_dtype=self.dtype)
        self.encoder = model.encoder.to(self.device).eval().requires_grad_(False)
        self.extractor = WhisperFeatureExtractor.from_pretrained(model_id)
        self.model_id = model_id

    def _encode_variable(self, mel):
        import torch.nn.functional as functional

        enc = self.encoder
        hidden = functional.gelu(enc.conv1(mel))
        hidden = functional.gelu(enc.conv2(hidden)).permute(0, 2, 1)
        if hidden.shape[1] > enc.embed_positions.weight.shape[0]:
            raise ValueError("audio exceeds Whisper's 30-second context")
        hidden = hidden + enc.embed_positions.weight[:hidden.shape[1]]
        for layer in enc.layers:
            hidden = layer(hidden, None)
        return enc.layer_norm(hidden)

    def _encode_short(self, waveform: np.ndarray) -> np.ndarray:
        """Encode one bounded window as unpooled 50 Hz states."""
        batch = self.extractor(
            waveform, sampling_rate=SAMPLE_RATE, padding="max_length",
            return_attention_mask=True, return_tensors="pt")
        frame_count = int(batch.attention_mask.sum().item())
        mel = batch.input_features[:, :, :frame_count].to(
            self.device, dtype=self.dtype)
        with self.torch.inference_mode():
            states = self._encode_variable(mel)
        encoder_frames = (frame_count + 1) // 2
        return states[0, :encoder_frames].float().cpu().numpy()

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Encode arbitrary audio, joining overlapping windows in state space.

        Whisper remains bidirectional inside each 28-second window. A one
        second overlap protects boundary words; duplicate overlap states are
        discarded from every later window. No decoder or transcript exists.
        """
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not waveform.size:
            raise ValueError("cannot encode empty audio")
        window = WHISPER_WINDOW_S * SAMPLE_RATE
        overlap = WHISPER_OVERLAP_S * SAMPLE_RATE
        if len(waveform) <= window:
            return self._encode_short(waveform)
        hop = window - overlap
        drop_frames = WHISPER_OVERLAP_S * 50
        chunks = []
        start = 0
        while start < len(waveform):
            end = min(start + window, len(waveform))
            encoded = self._encode_short(waveform[start:end])
            if start:
                encoded = encoded[drop_frames:]
            if len(encoded):
                chunks.append(encoded)
            if end == len(waveform):
                break
            start += hop
        return np.concatenate(chunks, axis=0)


def _pool_mlx(mx, values, pool_step: int = POOL_STEP):
    """Average time after the nonlinear adapter, matching MiniCPM's order."""
    usable = (values.shape[0] // pool_step) * pool_step
    if usable:
        return mx.mean(
            values[:usable].reshape(-1, pool_step, values.shape[-1]), axis=1)
    return mx.mean(values, axis=0, keepdims=True)


def validate_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
    """Load and strictly identify a Qwen-free stock-Whisper checkpoint."""
    with np.load(path, allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    architecture = str(values.get("adapter_arch", ARCH_PAIRED))
    encoder = str(values.get("encoder", ""))
    if encoder not in SUPPORTED_WHISPER:
        raise ValueError(f"{path}: unsupported encoder {encoder!r}")
    input_dim = int(values.get("input_dim", ENCODER_DIMS[encoder]))
    if input_dim != ENCODER_DIMS[encoder]:
        raise ValueError(
            f"{path}: encoder {encoder!r} requires input_dim "
            f"{ENCODER_DIMS[encoder]}, got {input_dim}")
    if architecture in (ARCH_GEMMA_NATIVE_STACK,
                         ARCH_GEMMA_NATIVE_RESIDUAL,
                         ARCH_GEMMA_NATIVE_TEMPORAL,
                         ARCH_GEMMA_NATIVE_TEMPORAL2,
                         ARCH_GEMMA_NATIVE_CONTEXT):
        stacked = input_dim * int(values.get("pool_step", 0))
        expected = {
            "rms_pre_gamma": (stacked,),
            "stack_w1": (OUTPUT_DIM * 2, stacked),
            "rms_mid_gamma": (OUTPUT_DIM,),
            "stack_w2": (OUTPUT_DIM, OUTPUT_DIM),
        }
        if architecture in (ARCH_GEMMA_NATIVE_RESIDUAL,
                            ARCH_GEMMA_NATIVE_TEMPORAL,
                            ARCH_GEMMA_NATIVE_TEMPORAL2,
                            ARCH_GEMMA_NATIVE_CONTEXT):
            residual_hidden = int(values.get("residual_hidden", 0))
            if residual_hidden <= 0:
                raise ValueError(f"{path}: invalid residual_hidden")
            expected.update({
                "res_rms_pre_gamma": (OUTPUT_DIM,),
                "res_w1": (residual_hidden * 2, OUTPUT_DIM),
                "res_rms_mid_gamma": (residual_hidden,),
                "res_w2": (OUTPUT_DIM, residual_hidden),
            })
        if architecture in (ARCH_GEMMA_NATIVE_TEMPORAL,
                            ARCH_GEMMA_NATIVE_TEMPORAL2,
                            ARCH_GEMMA_NATIVE_CONTEXT):
            temporal_hidden = int(values.get("temporal_hidden", 0))
            if temporal_hidden <= 0:
                raise ValueError(f"{path}: invalid temporal_hidden")
            expected.update({
                "temp_rms_pre_gamma": (OUTPUT_DIM,),
                "temp_w1": (temporal_hidden * 2, OUTPUT_DIM),
                "temp_dw": (temporal_hidden, 3, 1),
                "temp_rms_mid_gamma": (temporal_hidden,),
                "temp_w2": (OUTPUT_DIM, temporal_hidden),
            })
        if architecture in (ARCH_GEMMA_NATIVE_TEMPORAL2,
                            ARCH_GEMMA_NATIVE_CONTEXT):
            temporal2_hidden = int(values.get("temporal2_hidden", 0))
            temporal2_kernel = int(values.get("temporal2_kernel", 0))
            if temporal2_hidden <= 0:
                raise ValueError(f"{path}: invalid temporal2_hidden")
            if temporal2_kernel <= 0 or temporal2_kernel % 2 == 0:
                raise ValueError(f"{path}: invalid temporal2_kernel")
            expected.update({
                "temp2_rms_pre_gamma": (OUTPUT_DIM,),
                "temp2_w1": (temporal2_hidden * 2, OUTPUT_DIM),
                "temp2_dw": (temporal2_hidden, temporal2_kernel, 1),
                "temp2_rms_mid_gamma": (temporal2_hidden,),
                "temp2_w2": (OUTPUT_DIM, temporal2_hidden),
            })
        if architecture == ARCH_GEMMA_NATIVE_CONTEXT:
            context_hidden = int(values.get("context_hidden", 0))
            context_heads = int(values.get("context_heads", 0))
            context_kernel = int(values.get("context_kernel", 0))
            if context_hidden <= 0:
                raise ValueError(f"{path}: invalid context_hidden")
            if context_heads <= 0 or context_hidden % context_heads:
                raise ValueError(f"{path}: invalid context_heads")
            if context_kernel <= 0 or context_kernel % 2 == 0:
                raise ValueError(f"{path}: invalid context_kernel")
            expected.update({
                "context_hidden": (), "context_heads": (),
                "context_kernel": (),
                "ctx_rms_pre_gamma": (OUTPUT_DIM,),
                "ctx_w1": (context_hidden * 2, OUTPUT_DIM),
                "ctx_rms_attn_gamma": (context_hidden,),
                "ctx_qkv_w": (context_hidden * 3, context_hidden),
                "ctx_attn_w": (context_hidden, context_hidden),
                "ctx_rms_conv_gamma": (context_hidden,),
                "ctx_dw": (context_hidden, context_kernel, 1),
                "ctx_rms_mid_gamma": (context_hidden,),
                "ctx_w2": (OUTPUT_DIM, context_hidden),
            })
        if stacked != input_dim * 2:
            raise ValueError(f"{path}: stack adapter requires pool_step 2")
    elif architecture == ARCH_GEMMA_NATIVE:
        hidden = int(values.get("hidden_dim", 0))
        expected = {
            "ln_gamma": (input_dim,), "ln_beta": (input_dim,),
            "w1": (hidden, input_dim), "b1": (hidden,),
            "w2": (OUTPUT_DIM, hidden), "b2": (OUTPUT_DIM,),
        }
        if hidden <= 0:
            raise ValueError(f"{path}: invalid hidden_dim {hidden}")
    elif architecture == ARCH_PAIRED:
        expected = {
            "linear1_weight": (HIDDEN_DIM, input_dim),
            "linear1_bias": (HIDDEN_DIM,),
            "linear2_weight": (OUTPUT_DIM, HIDDEN_DIM),
            "linear2_bias": (OUTPUT_DIM,),
        }
    else:
        raise ValueError(f"{path}: unsupported adapter_arch {architecture!r}")
    for key, shape in expected.items():
        if key not in values or values[key].shape != shape:
            raise ValueError(f"{path}: {key} must be {shape}")
    boundary = str(values.get("input_boundary", ""))
    lineage = str(values.get("lineage", ""))
    if boundary != INPUT_BOUNDARY:
        raise ValueError(f"{path}: wrong input boundary {boundary!r}")
    if lineage != "qwen_free":
        raise ValueError(f"{path}: checkpoint is not marked qwen_free")
    pool_step = int(values.get("pool_step", POOL_STEP))
    if pool_step not in (2, POOL_STEP):
        raise ValueError(f"{path}: unsupported pool step {pool_step}")
    if not all(np.isfinite(values[key]).all() for key in expected):
        raise ValueError(f"{path}: non-finite adapter value")
    return values


class StockWhisperGemmaSession:
    """Minimal session contract used by :mod:`jaeger_agent_omni.omni`."""

    def __init__(self, adapter_path: str | Path,
                 model_path: str | Path = DEFAULT_GEMMA,
                 whisper_id: str | None = None,
                 system: str | None = None,
                 device: str = "auto") -> None:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load

        from jaeger_agent_omni.model_nodes import (
            resolve_audio_node,
            validate_node_checkpoint,
        )

        adapter_path, node = resolve_audio_node(adapter_path)
        values = validate_checkpoint(adapter_path)
        if node is not None:
            validate_node_checkpoint(node, values)
        self.adapter_path = Path(adapter_path).resolve()
        self.node = node
        architecture = str(values.get("adapter_arch", ARCH_PAIRED))
        checkpoint_encoder = str(values["encoder"])
        self.input_dim = int(values.get(
            "input_dim", ENCODER_DIMS[checkpoint_encoder]))
        whisper_id = whisper_id or checkpoint_encoder
        output_mode = str(values.get("output_mode", OUTPUT_MODE_NORMALIZED))
        if output_mode not in (
                OUTPUT_MODE_NORMALIZED, OUTPUT_MODE_RAW, OUTPUT_MODE_DIRECT):
            raise ValueError(
                f"{adapter_path}: unsupported output mode {output_mode!r}")
        if checkpoint_encoder != whisper_id:
            raise ValueError(
                f"{adapter_path}: encoder is {checkpoint_encoder!r}, but "
                f"runtime requested {whisper_id!r}")
        self.mx = mx
        self.nn = nn
        # A packaged node owns a complete pinned encoder snapshot. Legacy raw
        # checkpoint callers retain the Hugging Face ID behavior.
        encoder_source = str(node.encoder_path) if node is not None else whisper_id
        self.front_end = StockWhisperFrontEnd(encoder_source, device)
        self.model, self.tokenizer = load(str(model_path), lazy=False)
        self.model.freeze()
        self.model.eval()
        self.processor = type("Processor", (), {"tokenizer": self.tokenizer})()
        self.system = system
        self.pool_step = int(values.get("pool_step", POOL_STEP))
        self.output_mode = output_mode
        self._audio: list[np.ndarray] = []
        self._features: np.ndarray | None = None
        self._metadata: list[str] = []
        self._cancelled = False
        self.n_prefilled = 0
        self.hidden: list = []
        self.output_ids: list[int] = []

        class Adapter(nn.Module):
            def __init__(inner):
                super().__init__()
                if architecture in (ARCH_GEMMA_NATIVE_STACK,
                                     ARCH_GEMMA_NATIVE_RESIDUAL,
                                     ARCH_GEMMA_NATIVE_TEMPORAL,
                                     ARCH_GEMMA_NATIVE_TEMPORAL2,
                                     ARCH_GEMMA_NATIVE_CONTEXT):
                    inner.pre = nn.RMSNorm(self.input_dim * self.pool_step,
                                           eps=1e-6)
                    inner.linear1 = nn.Linear(
                        self.input_dim * self.pool_step, OUTPUT_DIM * 2,
                        bias=False)
                    inner.mid = nn.RMSNorm(OUTPUT_DIM, eps=1e-6)
                    inner.linear2 = nn.Linear(OUTPUT_DIM, OUTPUT_DIM,
                                              bias=False)
                    inner.pre.weight = mx.array(values["rms_pre_gamma"])
                    inner.linear1.weight = mx.array(values["stack_w1"])
                    inner.mid.weight = mx.array(values["rms_mid_gamma"])
                    inner.linear2.weight = mx.array(values["stack_w2"])
                    if architecture in (ARCH_GEMMA_NATIVE_RESIDUAL,
                                        ARCH_GEMMA_NATIVE_TEMPORAL,
                                        ARCH_GEMMA_NATIVE_TEMPORAL2,
                                        ARCH_GEMMA_NATIVE_CONTEXT):
                        residual_hidden = int(values["residual_hidden"])
                        inner.res_pre = nn.RMSNorm(OUTPUT_DIM, eps=1e-6)
                        inner.res_up = nn.Linear(
                            OUTPUT_DIM, residual_hidden * 2, bias=False)
                        inner.res_mid = nn.RMSNorm(
                            residual_hidden, eps=1e-6)
                        inner.res_out = nn.Linear(
                            residual_hidden, OUTPUT_DIM, bias=False)
                        inner.res_pre.weight = mx.array(
                            values["res_rms_pre_gamma"])
                        inner.res_up.weight = mx.array(values["res_w1"])
                        inner.res_mid.weight = mx.array(
                            values["res_rms_mid_gamma"])
                        inner.res_out.weight = mx.array(values["res_w2"])
                    if architecture in (ARCH_GEMMA_NATIVE_TEMPORAL,
                                        ARCH_GEMMA_NATIVE_TEMPORAL2,
                                        ARCH_GEMMA_NATIVE_CONTEXT):
                        temporal_hidden = int(values["temporal_hidden"])
                        inner.temp_pre = nn.RMSNorm(OUTPUT_DIM, eps=1e-6)
                        inner.temp_up = nn.Linear(
                            OUTPUT_DIM, temporal_hidden * 2, bias=False)
                        inner.temp_conv = nn.Conv1d(
                            temporal_hidden, temporal_hidden, 3, padding=1,
                            groups=temporal_hidden, bias=False)
                        inner.temp_mid = nn.RMSNorm(
                            temporal_hidden, eps=1e-6)
                        inner.temp_out = nn.Linear(
                            temporal_hidden, OUTPUT_DIM, bias=False)
                        inner.temp_pre.weight = mx.array(
                            values["temp_rms_pre_gamma"])
                        inner.temp_up.weight = mx.array(values["temp_w1"])
                        inner.temp_conv.weight = mx.array(values["temp_dw"])
                        inner.temp_mid.weight = mx.array(
                            values["temp_rms_mid_gamma"])
                        inner.temp_out.weight = mx.array(values["temp_w2"])
                    if architecture in (ARCH_GEMMA_NATIVE_TEMPORAL2,
                                        ARCH_GEMMA_NATIVE_CONTEXT):
                        temporal2_hidden = int(values["temporal2_hidden"])
                        temporal2_kernel = int(values["temporal2_kernel"])
                        inner.temp2_pre = nn.RMSNorm(OUTPUT_DIM, eps=1e-6)
                        inner.temp2_up = nn.Linear(
                            OUTPUT_DIM, temporal2_hidden * 2, bias=False)
                        inner.temp2_conv = nn.Conv1d(
                            temporal2_hidden, temporal2_hidden,
                            temporal2_kernel, padding=temporal2_kernel // 2,
                            groups=temporal2_hidden, bias=False)
                        inner.temp2_mid = nn.RMSNorm(
                            temporal2_hidden, eps=1e-6)
                        inner.temp2_out = nn.Linear(
                            temporal2_hidden, OUTPUT_DIM, bias=False)
                        inner.temp2_pre.weight = mx.array(
                            values["temp2_rms_pre_gamma"])
                        inner.temp2_up.weight = mx.array(values["temp2_w1"])
                        inner.temp2_conv.weight = mx.array(values["temp2_dw"])
                        inner.temp2_mid.weight = mx.array(
                            values["temp2_rms_mid_gamma"])
                        inner.temp2_out.weight = mx.array(values["temp2_w2"])
                    if architecture == ARCH_GEMMA_NATIVE_CONTEXT:
                        context_hidden = int(values["context_hidden"])
                        context_kernel = int(values["context_kernel"])
                        inner.context_heads = int(values["context_heads"])
                        inner.ctx_pre = nn.RMSNorm(OUTPUT_DIM, eps=1e-6)
                        inner.ctx_up = nn.Linear(
                            OUTPUT_DIM, context_hidden * 2, bias=False)
                        inner.ctx_attn_norm = nn.RMSNorm(
                            context_hidden, eps=1e-6)
                        inner.ctx_qkv = nn.Linear(
                            context_hidden, context_hidden * 3, bias=False)
                        inner.ctx_attn_out = nn.Linear(
                            context_hidden, context_hidden, bias=False)
                        inner.ctx_conv_norm = nn.RMSNorm(
                            context_hidden, eps=1e-6)
                        inner.ctx_conv = nn.Conv1d(
                            context_hidden, context_hidden, context_kernel,
                            padding=context_kernel // 2,
                            groups=context_hidden, bias=False)
                        inner.ctx_mid = nn.RMSNorm(context_hidden, eps=1e-6)
                        inner.ctx_out = nn.Linear(
                            context_hidden, OUTPUT_DIM, bias=False)
                        inner.ctx_pre.weight = mx.array(
                            values["ctx_rms_pre_gamma"])
                        inner.ctx_up.weight = mx.array(values["ctx_w1"])
                        inner.ctx_attn_norm.weight = mx.array(
                            values["ctx_rms_attn_gamma"])
                        inner.ctx_qkv.weight = mx.array(values["ctx_qkv_w"])
                        inner.ctx_attn_out.weight = mx.array(
                            values["ctx_attn_w"])
                        inner.ctx_conv_norm.weight = mx.array(
                            values["ctx_rms_conv_gamma"])
                        inner.ctx_conv.weight = mx.array(values["ctx_dw"])
                        inner.ctx_mid.weight = mx.array(
                            values["ctx_rms_mid_gamma"])
                        inner.ctx_out.weight = mx.array(values["ctx_w2"])
                elif architecture == ARCH_GEMMA_NATIVE:
                    hidden = int(values["hidden_dim"])
                    inner.layer_norm = nn.LayerNorm(self.input_dim)
                    inner.linear1 = nn.Linear(
                        self.input_dim, hidden, bias=True)
                    inner.linear2 = nn.Linear(hidden, OUTPUT_DIM, bias=True)
                    inner.layer_norm.weight = mx.array(values["ln_gamma"])
                    inner.layer_norm.bias = mx.array(values["ln_beta"])
                    inner.linear1.weight = mx.array(values["w1"])
                    inner.linear1.bias = mx.array(values["b1"])
                    inner.linear2.weight = mx.array(values["w2"])
                    inner.linear2.bias = mx.array(values["b2"])
                else:
                    inner.linear1 = nn.Linear(
                        self.input_dim, HIDDEN_DIM, bias=True)
                    inner.linear2 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM, bias=True)
                    inner.linear1.weight = mx.array(values["linear1_weight"])
                    inner.linear1.bias = mx.array(values["linear1_bias"])
                    inner.linear2.weight = mx.array(values["linear2_weight"])
                    inner.linear2.bias = mx.array(values["linear2_bias"])

            def __call__(inner, x):
                if architecture in (ARCH_GEMMA_NATIVE_STACK,
                                     ARCH_GEMMA_NATIVE_RESIDUAL,
                                     ARCH_GEMMA_NATIVE_TEMPORAL,
                                     ARCH_GEMMA_NATIVE_TEMPORAL2,
                                     ARCH_GEMMA_NATIVE_CONTEXT):
                    usable = (x.shape[0] // self.pool_step) * self.pool_step
                    if usable == 0:
                        x = mx.concatenate([x, x], axis=0)
                        usable = self.pool_step
                    stacked = x[:usable].reshape(
                        -1, self.input_dim * self.pool_step)
                    gate, value = mx.split(
                        inner.linear1(inner.pre(stacked)), 2, axis=-1)
                    y = inner.linear2(inner.mid(nn.silu(gate) * value))
                    if architecture in (ARCH_GEMMA_NATIVE_RESIDUAL,
                                        ARCH_GEMMA_NATIVE_TEMPORAL,
                                        ARCH_GEMMA_NATIVE_TEMPORAL2,
                                        ARCH_GEMMA_NATIVE_CONTEXT):
                        res_gate, res_value = mx.split(
                            inner.res_up(inner.res_pre(y)), 2, axis=-1)
                        y = y + inner.res_out(inner.res_mid(
                            nn.silu(res_gate) * res_value))
                    if architecture in (ARCH_GEMMA_NATIVE_TEMPORAL,
                                        ARCH_GEMMA_NATIVE_TEMPORAL2,
                                        ARCH_GEMMA_NATIVE_CONTEXT):
                        temp_gate, temp_value = mx.split(
                            inner.temp_up(inner.temp_pre(y)), 2, axis=-1)
                        temporal = nn.silu(temp_gate) * temp_value
                        temporal = inner.temp_conv(temporal[None, :, :])[0]
                        y = y + inner.temp_out(inner.temp_mid(temporal))
                    if architecture in (ARCH_GEMMA_NATIVE_TEMPORAL2,
                                        ARCH_GEMMA_NATIVE_CONTEXT):
                        temp2_gate, temp2_value = mx.split(
                            inner.temp2_up(inner.temp2_pre(y)), 2, axis=-1)
                        temporal2 = nn.silu(temp2_gate) * temp2_value
                        temporal2 = inner.temp2_conv(
                            temporal2[None, :, :])[0]
                        y = y + inner.temp2_out(inner.temp2_mid(temporal2))
                    if architecture == ARCH_GEMMA_NATIVE_CONTEXT:
                        context_gate, context_value = mx.split(
                            inner.ctx_up(inner.ctx_pre(y)), 2, axis=-1)
                        context = nn.silu(context_gate) * context_value
                        normalized = inner.ctx_attn_norm(context)
                        query, key, value = mx.split(
                            inner.ctx_qkv(normalized), 3, axis=-1)
                        heads = inner.context_heads
                        head_dim = query.shape[-1] // heads
                        query = query.reshape(-1, heads, head_dim)
                        key = key.reshape(-1, heads, head_dim)
                        value = value.reshape(-1, heads, head_dim)
                        scores = mx.einsum(
                            "thd,shd->hts", query, key) / math.sqrt(head_dim)
                        weights = mx.softmax(scores, axis=-1)
                        attended = mx.einsum(
                            "hts,shd->thd", weights, value).reshape(
                                context.shape)
                        context = context + inner.ctx_attn_out(attended)
                        convolved = inner.ctx_conv(
                            inner.ctx_conv_norm(context)[None, :, :])[0]
                        context = context + nn.silu(convolved)
                        y = y + inner.ctx_out(inner.ctx_mid(context))
                elif architecture == ARCH_GEMMA_NATIVE:
                    y = nn.gelu(inner.linear1(inner.layer_norm(x)))
                    y = _pool_mlx(mx, y, self.pool_step)
                    y = inner.linear2(y)
                else:
                    y = inner.linear2(nn.relu(inner.linear1(x)))
                    y = _pool_mlx(mx, y, self.pool_step)
                if output_mode in (OUTPUT_MODE_RAW, OUTPUT_MODE_DIRECT):
                    return y
                norms = mx.sqrt(mx.sum(y * y, axis=-1, keepdims=True) + 1e-8)
                return y / norms * TARGET_NORM

        self.adapter = Adapter()
        mx.eval(self.adapter.parameters())

    def warmup(self) -> float:
        return 0.0

    def reset(self) -> None:
        self._audio.clear()
        self._features = None
        self._metadata.clear()
        self._cancelled = False
        self.n_prefilled = 0
        self.hidden.clear()
        self.output_ids.clear()

    def open_turn(self) -> None:
        self._cancelled = False

    def feed_audio(self, samples) -> int:
        block = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
        if block.size:
            self._audio.append(block)
        return 0

    def feed_text(self, text: str) -> int:
        """Metadata only: speaker tags are permitted between audio turns."""
        self._metadata.append(str(text))
        return 0

    def end_of_turn(self, close: bool = True) -> None:
        if not self._audio:
            raise ValueError("no audio in turn")
        self._features = self.front_end(np.concatenate(self._audio))
        self.n_prefilled = max(1, len(self._features) // self.pool_step)

    def _prompt_parts(self) -> tuple[list[int], list[int]]:
        content = "".join(self._metadata) + AUDIO_MARKER
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": content})
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        if AUDIO_MARKER not in rendered:
            raise RuntimeError("audio marker disappeared from chat template")
        prefix, suffix = rendered.split(AUDIO_MARKER, 1)
        return (self.tokenizer.encode(prefix, add_special_tokens=False),
                self.tokenizer.encode(suffix, add_special_tokens=False))

    def respond(self, max_tokens: int = 96, eos_ids=(), on_token=None,
                collect_hidden: bool = False) -> str:
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_logits_processors

        if self._features is None:
            raise ValueError("end the audio turn before responding")
        if collect_hidden:
            raise NotImplementedError(
                "stock-Whisper MLX generation does not expose Talker hidden states")
        mx = self.mx
        prefix, suffix = self._prompt_parts()
        n_audio = max(1, len(self._features) // self.pool_step)
        ids_list = prefix + [0] * n_audio + suffix
        ids = mx.array(ids_list, dtype=mx.int32)
        tokens = self.model.language_model.model.embed_tokens(ids[None, :])[0]
        raw_audio = self.adapter(mx.array(self._features))
        audio = (raw_audio if self.output_mode == OUTPUT_MODE_DIRECT else
                 raw_audio / math.sqrt(OUTPUT_DIM))
        start = len(prefix)
        embeds = mx.concatenate(
            [tokens[:start], audio, tokens[start + n_audio:]], axis=0)
        generated = []
        self._cancelled = False
        blocked = {
            int(self.tokenizer.convert_tokens_to_ids("<|think|>")),
            int(self.tokenizer.convert_tokens_to_ids("<|channel>")),
        }
        processors = make_logits_processors(
            logit_bias={token: -1e9 for token in blocked},
            repetition_penalty=1.05,
            repetition_context_size=20,
        )
        for token, _ in generate_step(
                ids, self.model, input_embeddings=embeds,
                max_tokens=max_tokens, logits_processors=processors):
            token = int(token)
            if self._cancelled or token in set(eos_ids) | {1, 106}:
                break
            generated.append(token)
            if on_token:
                on_token(token, len(generated))
        self.output_ids = generated
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def cancel(self) -> None:
        self._cancelled = True

    def interrupt(self) -> None:
        self._cancelled = False

    def resume_listening(self) -> None:
        pass

    def close_turn(self) -> None:
        pass


def selfcheck() -> None:
    assert INPUT_DIM == 1024 and OUTPUT_DIM == 2560
    assert "qwen" not in DEFAULT_WHISPER.lower()
    five_minute_chunks = (
        math.ceil((5 * 60 - WHISPER_WINDOW_S) /
                  (WHISPER_WINDOW_S - WHISPER_OVERLAP_S)) + 1)
    assert five_minute_chunks == 12
    fake = object.__new__(StockWhisperFrontEnd)
    calls = []

    def fake_encode(waveform):
        calls.append(len(waveform))
        frames = math.ceil(len(waveform) / (SAMPLE_RATE / 50))
        return np.zeros((frames, 1), dtype=np.float32)

    fake._encode_short = fake_encode
    joined = fake(np.zeros(5 * 60 * SAMPLE_RATE, dtype=np.float32))
    assert len(calls) == 12 and joined.shape == (15_000, 1)
    print("runtime.stock_whisper: stock encoder -> owned Jaeger connector -> Gemma; "
          "ASR decoder unused; five-minute window chunking contract OK")


if __name__ == "__main__":
    selfcheck()
