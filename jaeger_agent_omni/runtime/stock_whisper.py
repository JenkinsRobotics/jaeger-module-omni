"""Qwen-free stock-Whisper audio front end for frozen Gemma.

The runtime boundary is deliberately small and explicit::

    PCM -> stock OpenAI Whisper encoder
        -> fresh Jaeger MLP -> temporal pool -> Gemma 4 E4B -> answer tokens

Whisper's decoder is neither loaded nor run for a packaged node.  The loader
constructs ``WhisperEncoder`` directly and selects only ``model.encoder.*``
weights from the pinned safetensors snapshot. Legacy Hugging Face-ID callers
retain the full-checkpoint compatibility loader. The adapter checkpoint cannot
silently carry MiniCPM or Qwen weights.

Whisper is bidirectional, so each completed microphone turn is encoded once.
The resulting audio embeddings and every later turn share one persistent
Gemma KV cache.  The microphone queue can continue collecting while that
worker runs; causal incremental Whisper remains a separate measured milestone.
"""

from __future__ import annotations

import math
import os
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
ARCH_CLEANROOM_CONTEXT = "cleanroom_context_v2"
ARCH_STREAMING_CAUSAL = "streaming_causal_v1"
WHISPER_WINDOW_S = 28
WHISPER_OVERLAP_S = 1
LIVE_WINDOW_S = 12
LIVE_OVERLAP_S = 1


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
        from transformers import WhisperConfig, WhisperFeatureExtractor
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "mps" else torch.float32
        source = Path(model_id)
        weights_path = source / "model.safetensors"
        if weights_path.is_file():
            # The registered node pins a normal Whisper full checkpoint for
            # provenance, but production only materialises its encoder half.
            # This avoids constructing/allocating 24 decoder layers merely to
            # discard them before the first microphone block.
            from safetensors import safe_open

            config = WhisperConfig.from_pretrained(str(source))
            encoder = WhisperEncoder(config)
            with safe_open(weights_path, framework="pt", device="cpu") as saved:
                prefix = "model.encoder."
                state = {
                    key[len(prefix):]: saved.get_tensor(key)
                    for key in saved.keys() if key.startswith(prefix)
                }
            missing, unexpected = encoder.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise ValueError(
                    f"encoder-only load mismatch: missing={missing[:4]} "
                    f"unexpected={unexpected[:4]}")
            del state
            self.encoder_only_load = True
            self.encoder = encoder.to(
                self.device, dtype=self.dtype).eval().requires_grad_(False)
        else:
            from transformers import WhisperModel

            model = WhisperModel.from_pretrained(
                model_id, torch_dtype=self.dtype)
            self.encoder = model.encoder.to(
                self.device).eval().requires_grad_(False)
            self.encoder_only_load = False
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
    cleanroom_arch = values.get("arch")
    arch_tag = (bytes(np.asarray(cleanroom_arch, dtype=np.uint8))
                if cleanroom_arch is not None else b"")
    if ("adapter_arch" not in values and
            arch_tag in (b"context_v2", b"streaming_causal_v1")):
        # The clean-room trainer intentionally emits only its owned weights
        # plus a byte architecture tag. Supply deployment metadata in memory;
        # the packaged node manifest independently pins the same boundary.
        values.update({
            "adapter_arch": np.array(
                ARCH_STREAMING_CAUSAL
                if arch_tag == b"streaming_causal_v1"
                else ARCH_CLEANROOM_CONTEXT),
            "encoder": np.array(DEFAULT_WHISPER),
            "input_dim": np.array(INPUT_DIM, dtype=np.int32),
            "input_boundary": np.array(INPUT_BOUNDARY),
            "lineage": np.array("qwen_free"),
            "pool_step": np.array(2, dtype=np.int32),
            "output_mode": np.array(OUTPUT_MODE_RAW),
        })
    architecture = str(values.get("adapter_arch", ARCH_PAIRED))
    encoder = str(values.get("encoder", ""))
    if encoder not in SUPPORTED_WHISPER:
        raise ValueError(f"{path}: unsupported encoder {encoder!r}")
    input_dim = int(values.get("input_dim", ENCODER_DIMS[encoder]))
    if input_dim != ENCODER_DIMS[encoder]:
        raise ValueError(
            f"{path}: encoder {encoder!r} requires input_dim "
            f"{ENCODER_DIMS[encoder]}, got {input_dim}")
    if architecture in (ARCH_CLEANROOM_CONTEXT, ARCH_STREAMING_CAUSAL):
        hidden_values = np.asarray(values.get("hidden", []), dtype=np.int64)
        if hidden_values.size != 1 or int(hidden_values[0]) <= 0:
            raise ValueError(f"{path}: invalid clean-room hidden width")
        hidden = int(hidden_values[0])
        expected = {
            "ln.weight": (input_dim,), "ln.bias": (input_dim,),
            "up.weight": (hidden, input_dim), "up.bias": (hidden,),
            "attn.query_proj.weight": (hidden, hidden),
            "attn.key_proj.weight": (hidden, hidden),
            "attn.value_proj.weight": (hidden, hidden),
            "attn.out_proj.weight": (hidden, hidden),
            "ln_a.weight": (hidden,), "ln_a.bias": (hidden,),
            "ff1.weight": (hidden * 2, hidden),
            "ff1.bias": (hidden * 2,),
            "ff2.weight": (hidden, hidden * 2), "ff2.bias": (hidden,),
            "ln_f.weight": (hidden,), "ln_f.bias": (hidden,),
            "out.weight": (OUTPUT_DIM, hidden), "out.bias": (OUTPUT_DIM,),
        }
        if architecture == ARCH_STREAMING_CAUSAL:
            expected.update({
                "head_speaker_change.weight": (1, hidden),
                "head_speaker_change.bias": (1,),
                "head_overlap.weight": (1, hidden),
                "head_overlap.bias": (1,),
                "head_turn_end.weight": (1, hidden),
                "head_turn_end.bias": (1,),
                "head_speaker_id.weight": (4, hidden),
                "head_speaker_id.bias": (4,),
                "self_proj.weight": (hidden, 1),
                "self_proj.bias": (hidden,),
            })
    elif architecture in (ARCH_GEMMA_NATIVE_STACK,
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
        from mlx_lm.models.cache import make_prompt_cache

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
        self._make_prompt_cache = make_prompt_cache
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
        # Inference-only, mathematically equivalent QKV packing for the 41M
        # clean-room connector. Keep it opt-in until the checkpoint-pinned
        # latency harness verifies identical replies and a real speed gain.
        self.fused_cleanroom_attention = (
            architecture == ARCH_CLEANROOM_CONTEXT and
            os.environ.get("JAEGER_FUSED_CLEANROOM_ATTN") == "1")
        self._audio: list[np.ndarray] = []
        self._features: np.ndarray | None = None
        self._metadata: list[str] = []
        self._cancelled = False
        self._cancel_at = 0.0
        self.cancel_latency_ms: float | None = None
        self.n_prefilled = 0
        # Lifetime counters deliberately survive reset().  live.py samples
        # their deltas across an explicit context roll; resetting them was the
        # source of the historical negative ms/token soak rows.
        self.total_prefill_tokens = 0
        self.total_prefill_ms = 0.0
        self.cache = None
        self._has_history = False
        self._assistant_closed = False
        self._generation_start = 0
        self._turn_has_audio = False
        self._had_stream_chunk = False
        self._turn_prefix_open = False
        self._turn_closed = False
        self._response_seed: int | None = None
        self.hidden: list = []
        self.output_ids: list[int] = []

        class Adapter(nn.Module):
            def __init__(inner):
                super().__init__()
                if architecture in (ARCH_CLEANROOM_CONTEXT,
                                     ARCH_STREAMING_CAUSAL):
                    hidden = int(values["hidden"][0])
                    inner.ln = nn.LayerNorm(self.input_dim)
                    inner.up = nn.Linear(self.input_dim, hidden)
                    if self.fused_cleanroom_attention:
                        inner.qkv = nn.Linear(hidden, hidden * 3, bias=False)
                        inner.attn_out = nn.Linear(hidden, hidden, bias=False)
                    else:
                        inner.attn = nn.MultiHeadAttention(hidden, 8)
                    inner.ln_a = nn.LayerNorm(hidden)
                    inner.ff1 = nn.Linear(hidden, hidden * 2)
                    inner.ff2 = nn.Linear(hidden * 2, hidden)
                    inner.ln_f = nn.LayerNorm(hidden)
                    inner.out = nn.Linear(hidden, OUTPUT_DIM)
                    inner.ln.weight = mx.array(values["ln.weight"])
                    inner.ln.bias = mx.array(values["ln.bias"])
                    inner.up.weight = mx.array(values["up.weight"])
                    inner.up.bias = mx.array(values["up.bias"])
                    if self.fused_cleanroom_attention:
                        inner.qkv.weight = mx.array(np.concatenate((
                            values["attn.query_proj.weight"],
                            values["attn.key_proj.weight"],
                            values["attn.value_proj.weight"],
                        ), axis=0))
                        inner.attn_out.weight = mx.array(
                            values["attn.out_proj.weight"])
                    else:
                        inner.attn.query_proj.weight = mx.array(
                            values["attn.query_proj.weight"])
                        inner.attn.key_proj.weight = mx.array(
                            values["attn.key_proj.weight"])
                        inner.attn.value_proj.weight = mx.array(
                            values["attn.value_proj.weight"])
                        inner.attn.out_proj.weight = mx.array(
                            values["attn.out_proj.weight"])
                    inner.ln_a.weight = mx.array(values["ln_a.weight"])
                    inner.ln_a.bias = mx.array(values["ln_a.bias"])
                    inner.ff1.weight = mx.array(values["ff1.weight"])
                    inner.ff1.bias = mx.array(values["ff1.bias"])
                    inner.ff2.weight = mx.array(values["ff2.weight"])
                    inner.ff2.bias = mx.array(values["ff2.bias"])
                    inner.ln_f.weight = mx.array(values["ln_f.weight"])
                    inner.ln_f.bias = mx.array(values["ln_f.bias"])
                    inner.out.weight = mx.array(values["out.weight"])
                    inner.out.bias = mx.array(values["out.bias"])
                    if architecture == ARCH_STREAMING_CAUSAL:
                        inner.head_speaker_change = nn.Linear(hidden, 1)
                        inner.head_overlap = nn.Linear(hidden, 1)
                        inner.head_turn_end = nn.Linear(hidden, 1)
                        inner.head_speaker_id = nn.Linear(hidden, 4)
                        inner.self_proj = nn.Linear(1, hidden)
                        for name in (
                                "speaker_change", "overlap", "turn_end",
                                "speaker_id"):
                            layer = getattr(inner, f"head_{name}")
                            layer.weight = mx.array(
                                values[f"head_{name}.weight"])
                            layer.bias = mx.array(values[f"head_{name}.bias"])
                        inner.self_proj.weight = mx.array(
                            values["self_proj.weight"])
                        inner.self_proj.bias = mx.array(
                            values["self_proj.bias"])
                elif architecture in (ARCH_GEMMA_NATIVE_STACK,
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

            def _context_trunk(inner, x, self_speech=None):
                """Run the shared 41M trunk and retain its 2048-d frames."""
                hidden = int(values["hidden"][0])
                h = nn.gelu(inner.up(inner.ln(x)))
                usable = (h.shape[0] // self.pool_step) * self.pool_step
                if usable == 0:
                    h = mx.mean(h, axis=0, keepdims=True)
                else:
                    h = mx.mean(
                        h[:usable].reshape(-1, self.pool_step, hidden),
                        axis=1)
                if (architecture == ARCH_STREAMING_CAUSAL and
                        self_speech is not None):
                    flag = np.asarray(self_speech, dtype=np.float32).reshape(-1)
                    flag = flag[:h.shape[0]]
                    if len(flag) < h.shape[0]:
                        flag = np.pad(flag, (0, h.shape[0] - len(flag)))
                    h = h + inner.self_proj(mx.array(flag[:, None]))
                context = h[None, :]
                mask = None
                if architecture == ARCH_STREAMING_CAUSAL and context.shape[1] > 1:
                    n = context.shape[1]
                    mask = mx.where(
                        mx.tril(mx.ones((n, n))) > 0, 0.0, -1e9).astype(
                            context.dtype)
                if self.fused_cleanroom_attention:
                    query, key, value = mx.split(
                        inner.qkv(context), 3, axis=-1)
                    heads = 8
                    head_dim = hidden // heads
                    query = query.reshape(
                        1, -1, heads, head_dim).transpose(0, 2, 1, 3)
                    key = key.reshape(
                        1, -1, heads, head_dim).transpose(0, 2, 1, 3)
                    value = value.reshape(
                        1, -1, heads, head_dim).transpose(0, 2, 1, 3)
                    attended = mx.fast.scaled_dot_product_attention(
                        query, key, value,
                        scale=math.sqrt(1.0 / head_dim), mask=mask)
                    attended = attended.transpose(
                        0, 2, 1, 3).reshape(context.shape)
                    attended = inner.attn_out(attended)
                else:
                    attended = inner.attn(
                        context, context, context, mask=mask)
                context = inner.ln_a(context + attended)
                return inner.ln_f(
                    context + inner.ff2(nn.gelu(inner.ff1(context))))

            def perceive(inner, x, self_speech=None):
                """Return audio embeddings and honest 25 Hz perception logits."""
                if architecture != ARCH_STREAMING_CAUSAL:
                    raise ValueError(
                        "perception heads require streaming_causal_v1")
                context = inner._context_trunk(x, self_speech)
                frames = context[0]
                return inner.out(frames), {
                    "speaker_change": inner.head_speaker_change(context)[0, :, 0],
                    "overlap": inner.head_overlap(context)[0, :, 0],
                    "turn_end": inner.head_turn_end(context)[0, :, 0],
                    "speaker_id": inner.head_speaker_id(context)[0],
                }

            def __call__(inner, x):
                if architecture in (ARCH_CLEANROOM_CONTEXT,
                                     ARCH_STREAMING_CAUSAL):
                    y = inner.out(inner._context_trunk(x)[0])
                elif architecture in (ARCH_GEMMA_NATIVE_STACK,
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
        """Compile the actual encoder, adapter, prefill, and decode path."""
        import time

        started = time.monotonic()
        self.reset()
        self.open_turn()
        self.feed_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
        self.end_of_turn()
        self.respond(max_tokens=1, eos_ids=())
        self.reset()
        return (time.monotonic() - started) * 1000

    def reset(self) -> None:
        self._audio.clear()
        self._features = None
        self._metadata.clear()
        self._cancelled = False
        self._cancel_at = 0.0
        self.cancel_latency_ms = None
        self.rewind_failures = getattr(self, "rewind_failures", 0)
        self.n_prefilled = 0
        self.cache = None
        self._has_history = False
        self._assistant_closed = False
        self._generation_start = 0
        self._turn_has_audio = False
        self._had_stream_chunk = False
        self._turn_prefix_open = False
        self._turn_closed = False
        self._response_seed = None
        self.hidden.clear()
        self.output_ids.clear()

    def open_turn(self) -> None:
        self._cancelled = False
        self._audio.clear()
        self._features = None
        self._metadata.clear()
        self._turn_has_audio = False
        self._had_stream_chunk = False
        self._turn_prefix_open = False
        self._turn_closed = False
        self._response_seed = None

    def feed_audio(self, samples) -> int:
        block = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
        if block.size:
            self._audio.append(block)
            self._turn_has_audio = True
        return self._encode_available(final=False)

    def feed_text(self, text: str) -> int:
        """Metadata only: speaker tags are permitted between audio turns."""
        self._metadata.append(str(text))
        return 0

    def end_of_turn(self, close: bool = True) -> None:
        if not self._turn_has_audio:
            raise ValueError("no audio in turn")
        self._encode_available(final=True)
        if close:
            self.close_turn()

    def _prompt_parts(self) -> tuple[list[int], list[int]]:
        # Metadata is appended after the audio, matching StreamingSession's
        # live inline annotation order.  This also lets diarization arrive in
        # before_respond after the last audio window is already encoded.
        content = AUDIO_MARKER + "".join(self._metadata)
        messages = []
        if self.system and not self._has_history:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": content})
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        if AUDIO_MARKER not in rendered:
            raise RuntimeError("audio marker disappeared from chat template")
        prefix, suffix = rendered.split(AUDIO_MARKER, 1)
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        if self._has_history:
            # A standalone chat template starts with BOS.  The persistent
            # cache already owns BOS and the preceding assistant turn.  Gemma
            # emits <turn|> (106) but not the following newline (107), while a
            # max-token or interrupted reply needs both closure and newline.
            bos = getattr(self.tokenizer, "bos_token_id", None)
            if prefix_ids and bos is not None and prefix_ids[0] == bos:
                prefix_ids = prefix_ids[1:]
            newline = self.tokenizer.encode("\n", add_special_tokens=False)
            if self._assistant_closed:
                prefix_ids = newline + prefix_ids
            else:
                close = self.tokenizer.encode(
                    "<turn|>\n", add_special_tokens=False)
                prefix_ids = close + prefix_ids
        return prefix_ids, suffix_ids

    def _cache_size(self) -> int:
        if not self.cache:
            return 0
        # Gemma 3n mixes full-attention KVCache layers with 512-token
        # RotatingKVCache layers.  ``size()`` on a rotating layer is its
        # physical ring-buffer occupancy, not the logical conversation
        # position, so it plateaus at 512 and used to prevent max-context
        # rolls entirely.  Every layer advances ``offset`` for every token;
        # use that authoritative logical position when it is available.
        offset = getattr(self.cache[0], "offset", None)
        if offset is not None:
            return int(offset)
        size = getattr(self.cache[0], "size", None)
        return int(size()) if callable(size) else self.n_prefilled

    def _prefill(self, ids: list[int], embeddings=None) -> float:
        """Append one text/audio segment to the persistent Gemma cache."""
        import time

        if not ids:
            return 0.0
        mx = self.mx
        if self.cache is None:
            self.cache = self._make_prompt_cache(self.model)
        token_ids = mx.array(ids, dtype=mx.int32)
        if embeddings is None:
            embeddings = self.model.language_model.model.embed_tokens(
                token_ids[None, :])[0]
        started = time.monotonic()
        out = self.model(
            token_ids[None, :], cache=self.cache,
            input_embeddings=embeddings[None, :])
        logits = out.logits if hasattr(out, "logits") else out
        mx.eval(logits)
        elapsed = (time.monotonic() - started) * 1000
        self.n_prefilled = self._cache_size()
        self.total_prefill_tokens += len(ids)
        self.total_prefill_ms += elapsed
        return elapsed

    def _ensure_turn_prefix(self) -> None:
        if self._turn_prefix_open:
            return
        prefix, _ = self._prompt_parts()
        self._prefill(prefix)
        self._turn_prefix_open = True

    def _prefill_audio(self, features: np.ndarray) -> int:
        mx = self.mx
        self._ensure_turn_prefix()
        raw = self.adapter(mx.array(features))
        audio = (raw if self.output_mode == OUTPUT_MODE_DIRECT else
                 raw / math.sqrt(OUTPUT_DIM))
        count = int(audio.shape[0])
        self._prefill([0] * count, audio)
        self._features = features
        return count

    def _encode_available(self, final: bool) -> int:
        """Encode bounded windows, retaining one second across boundaries.

        Short turns remain a single bidirectional Whisper pass. Continuous
        speech is committed every 11 new seconds, so a five-minute meeting
        cannot become one unbounded encoder/adapter operation at turn end.
        """
        if not self._audio:
            return 0
        waveform = np.concatenate(self._audio)
        window = LIVE_WINDOW_S * SAMPLE_RATE
        overlap = LIVE_OVERLAP_S * SAMPLE_RATE
        encoded = 0
        while len(waveform) >= window and (not final or len(waveform) > window):
            features = self.front_end(waveform[:window])
            if self._had_stream_chunk:
                features = features[LIVE_OVERLAP_S * 50:]
            self._prefill_audio(features)
            encoded += 1
            self._had_stream_chunk = True
            waveform = waveform[window - overlap:]
        if final:
            # When a prior full window ended exactly at the turn boundary,
            # only its retained overlap remains and must not be duplicated.
            if len(waveform) > (overlap if self._had_stream_chunk else 0):
                features = self.front_end(waveform)
                if self._had_stream_chunk:
                    features = features[LIVE_OVERLAP_S * 50:]
                if len(features):
                    self._prefill_audio(features)
                    encoded += 1
            waveform = np.empty(0, dtype=np.float32)
        self._audio = [waveform] if len(waveform) else []
        return encoded

    def respond(self, max_tokens: int = 96, eos_ids=(), on_token=None,
                collect_hidden: bool = False) -> str:
        import time

        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_logits_processors

        if not self._turn_closed or self._response_seed is None:
            raise ValueError("close the audio turn before responding")
        if collect_hidden:
            raise NotImplementedError(
                "stock-Whisper MLX generation does not expose Talker hidden states")
        mx = self.mx
        ids = mx.array([self._response_seed], dtype=mx.int32)
        tokens = self.model.language_model.model.embed_tokens(ids[None, :])[0]
        embeds = tokens
        generated = []
        self._cancelled = False
        self.cancel_latency_ms = None
        if self.cache is None:
            self.cache = self._make_prompt_cache(self.model)
        prompt_tokens = 1
        self._generation_start = self.n_prefilled + prompt_tokens
        prefill_started = time.monotonic()
        first_token_at = None
        assistant_closed = False
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
                max_tokens=max_tokens, logits_processors=processors,
                prompt_cache=self.cache):
            if first_token_at is None:
                first_token_at = time.monotonic()
                self.total_prefill_tokens += prompt_tokens
                self.total_prefill_ms += (
                    first_token_at - prefill_started) * 1000
            token = int(token)
            if self._cancelled:
                self.cancel_latency_ms = (
                    time.monotonic() - self._cancel_at) * 1000
                break
            if token in set(eos_ids) | {1, 106}:
                assistant_closed = token == 106
                break
            generated.append(token)
            if on_token:
                on_token(token, len(generated))
        # generate_step updates the supplied cache in place.  Its one-token
        # lookahead has cached exactly every token yielded so far, including a
        # closing token.  Keep the authoritative size for rolling-context and
        # precise interrupt rewind accounting.
        self.n_prefilled = self._cache_size()
        if first_token_at is None:
            # Defensive for max_tokens=0; ordinarily generation always yields.
            self.total_prefill_tokens += prompt_tokens
            self.total_prefill_ms += (time.monotonic() - prefill_started) * 1000
        self._assistant_closed = assistant_closed
        self._has_history = True
        self.output_ids = generated
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def cancel(self) -> None:
        import time

        self._cancel_at = time.monotonic()
        self._cancelled = True

    def interrupt(self) -> int:
        """Remove the abandoned assistant suffix while preserving its prompt."""
        from mlx_lm.models.cache import trim_prompt_cache

        drop = self.n_prefilled - self._generation_start
        if drop <= 0 or not self.cache:
            self._cancelled = False
            return 0
        trimmed = trim_prompt_cache(self.cache, drop)
        self._cancelled = False
        if trimmed != drop:
            # A wrapped RotatingKVCache layer (>512 tokens of history)
            # refuses to rewind, so trim returns 0.  Raising here turned a
            # live barge-in into a crash four minutes into a conversation
            # (measured 2026-08-29).  Keeping the few abandoned tokens is
            # the lesser harm: the next prompt prefix already closes an
            # unfinished assistant turn, so the dangling suffix is
            # terminated rather than continued.
            self.rewind_failures += 1
            self._assistant_closed = False
            return 0
        self.n_prefilled -= drop
        self._assistant_closed = False
        return drop

    def resume_listening(self) -> None:
        self.open_turn()

    def close_turn(self) -> None:
        if self._turn_closed:
            return
        if not self._turn_has_audio:
            raise ValueError("no audio in turn")
        self._ensure_turn_prefix()
        _, suffix = self._prompt_parts()
        if not suffix:
            raise RuntimeError("chat template produced no assistant seed")
        self._prefill(suffix[:-1])
        self._response_seed = int(suffix[-1])
        self._turn_closed = True


def selfcheck() -> None:
    assert INPUT_DIM == 1024 and OUTPUT_DIM == 2560
    assert "qwen" not in DEFAULT_WHISPER.lower()
    # Packing three bias-free projections into one larger matrix is exact;
    # the exclusive-GPU A/B separately proves that optimized inference keeps
    # the real checkpoint's generated replies unchanged.
    rng = np.random.default_rng(20260828)
    probe_values = rng.normal(size=(5, 7)).astype(np.float32)
    projection_weights = [
        rng.normal(size=(11, 7)).astype(np.float32) for _ in range(3)]
    separate = np.concatenate(
        [probe_values @ weight.T for weight in projection_weights], axis=-1)
    fused = probe_values @ np.concatenate(projection_weights, axis=0).T
    assert np.allclose(separate, fused, rtol=1e-6, atol=1e-6)
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

    class FakeTokenizer:
        bos_token_id = 2

        @staticmethod
        def apply_chat_template(messages, tokenize=False,
                                add_generation_prompt=True):
            body = "".join(
                f"<|turn>{m['role']}\n{m['content']}<turn|>\n"
                for m in messages)
            return "<bos>" + body + "<|turn>model\n"

        @staticmethod
        def encode(value, add_special_tokens=False):
            if value == "\n":
                return [107]
            if value == "<turn|>\n":
                return [106, 107]
            result = []
            if value.startswith("<bos>"):
                result.append(2)
            result.append(105)
            return result

    # A second turn must not inject another BOS. A naturally closed assistant
    # needs only a newline; an interrupted/max-token reply needs an explicit
    # model-turn close before the new user turn.
    probe = object.__new__(StockWhisperGemmaSession)
    probe._metadata = []
    probe.system = "system"
    probe.tokenizer = FakeTokenizer()
    probe._has_history = True
    probe._assistant_closed = True
    closed_prefix, _ = probe._prompt_parts()
    assert closed_prefix[:2] == [107, 105]
    probe._assistant_closed = False
    open_prefix, _ = probe._prompt_parts()
    assert open_prefix[:3] == [106, 107, 105]

    class FakeCache:
        def __init__(self, size):
            self.offset = size

        def is_trimmable(self):
            return True

        def trim(self, amount):
            amount = min(amount, self.offset)
            self.offset -= amount
            return amount

        def size(self):
            # Model Gemma's bounded sliding-cache storage.  Logical context
            # must continue beyond this physical capacity.
            return min(self.offset, 512)

    probe.cache = [FakeCache(13)]
    probe.n_prefilled = 13
    probe._generation_start = 9
    probe._cancelled = True
    assert probe.interrupt() == 4
    assert probe.n_prefilled == 9 and probe.cache[0].offset == 9
    probe.cache = [FakeCache(6001)]
    assert probe._cache_size() == 6001

    # The live session uses the adapter's training-length window instead of
    # handing a five-minute uninterrupted meeting to one quadratic context
    # block. Overlap is present at the encoder boundary but never duplicated
    # in what reaches Gemma.
    live = object.__new__(StockWhisperGemmaSession)
    live._audio = [np.zeros(5 * 60 * SAMPLE_RATE, dtype=np.float32)]
    live._had_stream_chunk = False
    live.front_end = lambda wav: np.zeros(
        (math.ceil(len(wav) / (SAMPLE_RATE / 50)), 1), dtype=np.float32)
    live_frames = []
    live._prefill_audio = lambda features: live_frames.append(len(features))
    windows = live._encode_available(final=False)
    windows += live._encode_available(final=True)
    assert windows == 28 and sum(live_frames) == 15_000, (
        windows, sum(live_frames))
    print("runtime.stock_whisper: stock encoder -> owned Jaeger connector -> Gemma; "
          "ASR decoder unused; persistent turns + rewind + bounded five-minute "
          "streaming contracts OK")


if __name__ == "__main__":
    selfcheck()
