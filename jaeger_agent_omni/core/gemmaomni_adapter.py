#!/usr/bin/env python3
"""GEMMAOMNI owned audio input — zero Qwen/MiniCPM lineage.

    PCM -> stock OpenAI whisper-medium encoder (frozen, MIT weights)
        -> fresh Jaeger adapter, PROJECT-THEN-POOL (OURS, trainable):
             LN(1024) -> Linear 2048 -> GELU -> avgpool/2 -> Linear 2560
        -> Gemma-4 E4B MLX (frozen)
        -> text

Project-then-pool, not pool-then-project: averaging raw encoder states over an
80 ms window blurs phonemes (a phoneme is 50-100 ms), and v1 measurably
collapsed — content recall@10 fell 2.4% -> 0.8% and frame diversity 86% -> 45%
while CE still dropped, i.e. the adapter learned to emit uninformative vectors
and let Gemma's prior do the talking. Projecting first lets the network choose
what survives averaging. Same parameter count.

MiniCPM is a conceptual reference only: no MiniCPM encoder weights, no linear1,
no Qwen-aligned init, no ridge maps, no Qwen templates. The BRIDGE track
(mlx_audio_streaming.py, MiniCPM-derived) stays untouched as the accuracy
bridge; this file is the challenger that carries the real Jaeger lineage.

Decode, endpointing, speaker tagging and the ledger are shared with the bridge
(CleanroomTranscriber subclasses the bridge transcriber and swaps ONLY the
front end + adapter), so comparisons measure the seam, nothing else.

    python gemmaomni_adapter.py --smoke            # untrained plumbing check
    python gemmaomni_adapter.py --smoke --adapter checkpoints/x.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
import mlx_audio_streaming as mas  # noqa: E402  (bridge module — reused, not modified)

STOCK_WHISPER = "openai/whisper-medium"
POOL = 2                       # 50 Hz encoder -> 25 Hz, MEASURED to match
                               # Gemma's own audio tower (147 frames / 5.86 s)
ENC_DIM = 1024
HID_DIM = 2048
OUT_DIM = 2560
DEFAULT_ADAPTER = Path(__file__).resolve().parent / "checkpoints/adapter_cleanroom.npz"
ARCH_MEAN_GELU = "mean_pool_gelu_v1"
ARCH_STACK_SWIGLU = "frame_stack_swiglu_v1"
ARCH_STACK_RESIDUAL = "frame_stack_swiglu_residual_v1"
ARCH_STACK_RESIDUAL2 = "frame_stack_swiglu_residual2_v1"
ARCH_STACK_CALIBRATION = "frame_stack_swiglu_residual_calibration_v1"
ARCH_STACK_TEMPORAL = "frame_stack_swiglu_residual_temporal_v1"
ARCH_STACK_TEMPORAL2 = "frame_stack_swiglu_residual_temporal2_v1"
ARCH_STACK_CONTEXT = "frame_stack_swiglu_residual_temporal2_context_v1"


class StockWhisperFrontEnd:
    """Stock whisper-medium ENCODER only — no MiniCPM tuning, no linear1,
    no Whisper text decoder. Variable-length encode, then stride-4 mean pool.
    Returns (T', 1024) float32."""

    def __init__(self, device: str = "mps", source: str = STOCK_WHISPER,
                 output_dim: int = ENC_DIM):
        """`source` is a hub id OR a local materialised encoder directory.

        Both come from a node manifest's encoder spec, which is what makes the
        encoder swappable: whisper-medium (1024-d) and large-v3 (1280-d) are
        different nodes, and the width is asserted below rather than assumed,
        because a silent width mismatch would train an adapter against the
        wrong contract and only show up as bad WER much later.
        """
        import torch
        from transformers import WhisperFeatureExtractor, WhisperModel

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float16 if device == "mps" else torch.float32
        self.source = source
        self.output_dim = int(output_dim)
        model = WhisperModel.from_pretrained(source)
        self.encoder = model.encoder.to(self.device, dtype=self.dtype).eval()
        self.encoder.requires_grad_(False)
        del model
        self.extractor = WhisperFeatureExtractor.from_pretrained(source)
        actual = int(self.encoder.config.d_model)
        if actual != self.output_dim:
            raise ValueError(
                f"encoder width mismatch: {source} emits {actual}-d but the "
                f"node declares {self.output_dim}-d. An adapter trained "
                f"against one width cannot consume the other.")

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        import torch.nn.functional as functional

        pcm = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not pcm.size:
            raise ValueError("cannot encode an empty waveform")
        if pcm.size > mas.SAMPLE_RATE * 29:
            raise ValueError("turn exceeds the 29s single-window cap")
        batch = self.extractor(pcm, sampling_rate=mas.SAMPLE_RATE,
                               padding="max_length",
                               return_attention_mask=True,
                               return_tensors="pt")
        frames = int(batch.attention_mask.sum().item())
        mel = batch.input_features[:, :, :frames].to(self.device,
                                                     dtype=self.dtype)
        enc = self.encoder
        with self.torch.inference_mode():
            hidden = functional.gelu(enc.conv1(mel))
            hidden = functional.gelu(enc.conv2(hidden)).permute(0, 2, 1)
            if hidden.shape[1] > enc.embed_positions.weight.shape[0]:
                raise ValueError("turn exceeds Whisper positional context")
            hidden = hidden + enc.embed_positions.weight[:hidden.shape[1]]
            for layer in enc.layers:
                hidden = layer(hidden, None)
            hidden = enc.layer_norm(hidden)
            states = hidden[0]                       # (T, 1024) fp16 @ 50Hz
        # NO pooling here — the adapter pools after its own projection.
        return states.float().cpu().numpy()


def fresh_adapter_arrays(seed: int = 0, hidden: int | None = None,
                         architecture: str = ARCH_MEAN_GELU
                         ) -> dict[str, np.ndarray]:
    """He-init fresh adapter — the only trainable module in the pipeline.

    `hidden` overrides the default width for capacity A/Bs. 2048 is ~7.3M
    params; 4096 is ~14.7M, matching the projector width a parallel effort
    used to reach 16.34% WER. Checkpoints carry their own shapes, so wide and
    narrow adapters load interchangeably.
    """
    if architecture == ARCH_STACK_RESIDUAL2:
        first = fresh_adapter_arrays(seed, hidden, ARCH_STACK_RESIDUAL)
        return add_second_residual_arrays(
            first, seed=seed, hidden=hidden or 1024)
    if architecture == ARCH_STACK_RESIDUAL:
        base = fresh_adapter_arrays(seed, None, ARCH_STACK_SWIGLU)
        return add_residual_arrays(base, seed=seed, hidden=hidden or 1024)
    rng = np.random.default_rng(seed)
    if architecture == ARCH_STACK_SWIGLU:
        # Ultravox's proven connector shape, adjusted only to the measured
        # 50 -> 25 Hz boundary here: stack frames (preserve order), RMSNorm,
        # Linear -> SwiGLU, RMSNorm, Linear. Its mid width equals the target
        # language-model width and all affine biases are intentionally absent.
        stacked = ENC_DIM * POOL
        return {
            "adapter_arch": np.array(ARCH_STACK_SWIGLU),
            "rms_pre_gamma": np.ones(stacked, np.float32),
            "stack_w1": (rng.standard_normal((OUT_DIM * 2, stacked))
                         * np.sqrt(2.0 / stacked)).astype(np.float32),
            "rms_mid_gamma": np.ones(OUT_DIM, np.float32),
            "stack_w2": (rng.standard_normal((OUT_DIM, OUT_DIM))
                         * np.sqrt(2.0 / OUT_DIM)).astype(np.float32),
        }
    if architecture != ARCH_MEAN_GELU:
        raise ValueError(f"unsupported architecture {architecture!r}")
    hid = hidden or HID_DIM
    return {
        "adapter_arch": np.array(ARCH_MEAN_GELU),
        "ln_gamma": np.ones(ENC_DIM, np.float32),
        "ln_beta": np.zeros(ENC_DIM, np.float32),
        "w1": (rng.standard_normal((hid, ENC_DIM))
               * np.sqrt(2.0 / ENC_DIM)).astype(np.float32),
        "b1": np.zeros(hid, np.float32),
        "w2": (rng.standard_normal((OUT_DIM, hid))
               * np.sqrt(2.0 / hid)).astype(np.float32),
        "b2": np.zeros(OUT_DIM, np.float32),
    }


def add_residual_arrays(arrays: dict[str, np.ndarray], seed: int = 0,
                        hidden: int = 1024) -> dict[str, np.ndarray]:
    """Upgrade a trained stack connector without changing its output.

    The residual down projection starts at zero, so the upgraded module is
    bit-for-bit equivalent before training. This isolates added capacity from
    destructive reinitialisation of the working base connector.
    """
    architecture = str(arrays.get("adapter_arch", ARCH_MEAN_GELU))
    if architecture != ARCH_STACK_SWIGLU:
        raise ValueError("residual upgrade requires frame_stack_swiglu_v1")
    if hidden <= 0:
        raise ValueError("residual hidden width must be positive")
    rng = np.random.default_rng(seed)
    upgraded = dict(arrays)
    upgraded.update({
        "adapter_arch": np.array(ARCH_STACK_RESIDUAL),
        "residual_hidden": np.array(hidden, dtype=np.int32),
        "res_rms_pre_gamma": np.ones(OUT_DIM, np.float32),
        "res_w1": (rng.standard_normal((hidden * 2, OUT_DIM))
                   * np.sqrt(2.0 / OUT_DIM)).astype(np.float32),
        "res_rms_mid_gamma": np.ones(hidden, np.float32),
        "res_w2": np.zeros((OUT_DIM, hidden), np.float32),
    })
    return upgraded


def add_second_residual_arrays(arrays: dict[str, np.ndarray], seed: int = 0,
                               hidden: int = 1024) -> dict[str, np.ndarray]:
    """Add a second exact-zero residual stage to a trained first stage."""
    architecture = str(arrays.get("adapter_arch", ARCH_MEAN_GELU))
    if architecture != ARCH_STACK_RESIDUAL:
        raise ValueError("second residual requires the trained first residual")
    if hidden <= 0:
        raise ValueError("second residual hidden width must be positive")
    rng = np.random.default_rng(seed)
    upgraded = dict(arrays)
    upgraded.update({
        "adapter_arch": np.array(ARCH_STACK_RESIDUAL2),
        "res2_hidden": np.array(hidden, dtype=np.int32),
        "res2_rms_pre_gamma": np.ones(OUT_DIM, np.float32),
        "res2_w1": (rng.standard_normal((hidden * 2, OUT_DIM))
                    * np.sqrt(2.0 / OUT_DIM)).astype(np.float32),
        "res2_rms_mid_gamma": np.ones(hidden, np.float32),
        "res2_w2": np.zeros((OUT_DIM, hidden), np.float32),
    })
    return upgraded


def add_calibration_arrays(arrays: dict[str, np.ndarray]
                           ) -> dict[str, np.ndarray]:
    """Add an exact-identity feature-wise affine output calibration."""
    architecture = str(arrays.get("adapter_arch", ARCH_MEAN_GELU))
    if architecture != ARCH_STACK_RESIDUAL:
        raise ValueError("calibration requires the trained first residual")
    upgraded = dict(arrays)
    upgraded.update({
        "adapter_arch": np.array(ARCH_STACK_CALIBRATION),
        "cal_scale": np.zeros(OUT_DIM, np.float32),
        "cal_bias": np.zeros(OUT_DIM, np.float32),
    })
    return upgraded


def add_temporal_arrays(arrays: dict[str, np.ndarray], seed: int = 0,
                        hidden: int = 256) -> dict[str, np.ndarray]:
    """Add an exact-zero Conformer-style local temporal residual.

    The depthwise three-token convolution sees 120 ms at the 25 Hz adapter
    boundary.  The final projection starts at zero, so upgrading a measured
    checkpoint preserves every output exactly until training begins.
    """
    architecture = str(arrays.get("adapter_arch", ARCH_MEAN_GELU))
    if architecture != ARCH_STACK_RESIDUAL:
        raise ValueError("temporal upgrade requires the trained first residual")
    if hidden <= 0:
        raise ValueError("temporal hidden width must be positive")
    rng = np.random.default_rng(seed)
    upgraded = dict(arrays)
    upgraded.update({
        "adapter_arch": np.array(ARCH_STACK_TEMPORAL),
        "temporal_hidden": np.array(hidden, dtype=np.int32),
        "temp_rms_pre_gamma": np.ones(OUT_DIM, np.float32),
        "temp_w1": (rng.standard_normal((hidden * 2, OUT_DIM))
                    * np.sqrt(2.0 / OUT_DIM)).astype(np.float32),
        "temp_dw": (rng.standard_normal((hidden, 3, 1))
                    * np.sqrt(2.0 / 3.0)).astype(np.float32),
        "temp_rms_mid_gamma": np.ones(hidden, np.float32),
        "temp_w2": np.zeros((OUT_DIM, hidden), np.float32),
    })
    return upgraded


def add_second_temporal_arrays(arrays: dict[str, np.ndarray], seed: int = 0,
                               hidden: int = 256, kernel: int = 31
                               ) -> dict[str, np.ndarray]:
    """Add an exact-zero long-context Conformer convolution residual.

    The selected first temporal branch has a three-token (120 ms) receptive
    field. Conformer commonly uses a 31-frame depthwise kernel; at this 25 Hz
    boundary that spans 1.24 seconds. The final projection starts at zero, so
    this upgrade preserves the measured first branch exactly at step zero.
    """
    architecture = str(arrays.get("adapter_arch", ARCH_MEAN_GELU))
    if architecture != ARCH_STACK_TEMPORAL:
        raise ValueError("second temporal upgrade requires temporal connector")
    if hidden <= 0:
        raise ValueError("second temporal hidden width must be positive")
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("second temporal kernel must be a positive odd value")
    rng = np.random.default_rng(seed)
    upgraded = dict(arrays)
    upgraded.update({
        "adapter_arch": np.array(ARCH_STACK_TEMPORAL2),
        "temporal2_hidden": np.array(hidden, dtype=np.int32),
        "temporal2_kernel": np.array(kernel, dtype=np.int32),
        "temp2_rms_pre_gamma": np.ones(OUT_DIM, np.float32),
        "temp2_w1": (rng.standard_normal((hidden * 2, OUT_DIM))
                     * np.sqrt(2.0 / OUT_DIM)).astype(np.float32),
        "temp2_dw": (rng.standard_normal((hidden, kernel, 1))
                     * np.sqrt(2.0 / kernel)).astype(np.float32),
        "temp2_rms_mid_gamma": np.ones(hidden, np.float32),
        "temp2_w2": np.zeros((OUT_DIM, hidden), np.float32),
    })
    return upgraded


def add_context_arrays(arrays: dict[str, np.ndarray], seed: int = 0,
                       hidden: int = 1536, heads: int = 12,
                       kernel: int = 31) -> dict[str, np.ndarray]:
    """Add an exact-zero attention+convolution contextual residual.

    At the default width this grows the selected 28.9M connector to 50.2M
    parameters.  The zero final projection makes step zero exactly equivalent
    to the measured temporal2 checkpoint; only the new branch is trained.
    Its bottleneck is also a stable seam for later speaker/environment/turn
    heads, without feeding a transcript back into the audio path.
    """
    architecture = str(arrays.get("adapter_arch", ARCH_MEAN_GELU))
    if architecture != ARCH_STACK_TEMPORAL2:
        raise ValueError("context upgrade requires temporal2 connector")
    if hidden <= 0 or heads <= 0 or hidden % heads:
        raise ValueError("context hidden must divide evenly across heads")
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("context kernel must be a positive odd value")
    rng = np.random.default_rng(seed)

    def he(shape, fan_in):
        return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(
            np.float32)

    upgraded = dict(arrays)
    upgraded.update({
        "adapter_arch": np.array(ARCH_STACK_CONTEXT),
        "context_hidden": np.array(hidden, dtype=np.int32),
        "context_heads": np.array(heads, dtype=np.int32),
        "context_kernel": np.array(kernel, dtype=np.int32),
        "ctx_rms_pre_gamma": np.ones(OUT_DIM, np.float32),
        "ctx_w1": he((hidden * 2, OUT_DIM), OUT_DIM),
        "ctx_rms_attn_gamma": np.ones(hidden, np.float32),
        "ctx_qkv_w": he((hidden * 3, hidden), hidden),
        "ctx_attn_w": he((hidden, hidden), hidden),
        "ctx_rms_conv_gamma": np.ones(hidden, np.float32),
        "ctx_dw": he((hidden, kernel, 1), kernel),
        "ctx_rms_mid_gamma": np.ones(hidden, np.float32),
        "ctx_w2": np.zeros((OUT_DIM, hidden), np.float32),
    })
    return upgraded


def load_adapter_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        architecture = str(archive["adapter_arch"]) \
            if "adapter_arch" in archive.files else ARCH_MEAN_GELU
        if architecture == ARCH_STACK_CONTEXT:
            keys = ("rms_pre_gamma", "stack_w1", "rms_mid_gamma", "stack_w2",
                    "residual_hidden", "res_rms_pre_gamma", "res_w1",
                    "res_rms_mid_gamma", "res_w2", "temporal_hidden",
                    "temp_rms_pre_gamma", "temp_w1", "temp_dw",
                    "temp_rms_mid_gamma", "temp_w2", "temporal2_hidden",
                    "temporal2_kernel", "temp2_rms_pre_gamma", "temp2_w1",
                    "temp2_dw", "temp2_rms_mid_gamma", "temp2_w2",
                    "context_hidden", "context_heads", "context_kernel",
                    "ctx_rms_pre_gamma", "ctx_w1", "ctx_rms_attn_gamma",
                    "ctx_qkv_w", "ctx_attn_w", "ctx_rms_conv_gamma",
                    "ctx_dw", "ctx_rms_mid_gamma", "ctx_w2")
        elif architecture == ARCH_STACK_TEMPORAL2:
            keys = ("rms_pre_gamma", "stack_w1", "rms_mid_gamma", "stack_w2",
                    "residual_hidden", "res_rms_pre_gamma", "res_w1",
                    "res_rms_mid_gamma", "res_w2", "temporal_hidden",
                    "temp_rms_pre_gamma", "temp_w1", "temp_dw",
                    "temp_rms_mid_gamma", "temp_w2", "temporal2_hidden",
                    "temporal2_kernel", "temp2_rms_pre_gamma", "temp2_w1",
                    "temp2_dw", "temp2_rms_mid_gamma", "temp2_w2")
        elif architecture == ARCH_STACK_TEMPORAL:
            keys = ("rms_pre_gamma", "stack_w1", "rms_mid_gamma", "stack_w2",
                    "residual_hidden", "res_rms_pre_gamma", "res_w1",
                    "res_rms_mid_gamma", "res_w2", "temporal_hidden",
                    "temp_rms_pre_gamma", "temp_w1", "temp_dw",
                    "temp_rms_mid_gamma", "temp_w2")
        elif architecture == ARCH_STACK_CALIBRATION:
            keys = ("rms_pre_gamma", "stack_w1", "rms_mid_gamma", "stack_w2",
                    "residual_hidden", "res_rms_pre_gamma", "res_w1",
                    "res_rms_mid_gamma", "res_w2", "cal_scale", "cal_bias")
        elif architecture == ARCH_STACK_RESIDUAL2:
            keys = ("rms_pre_gamma", "stack_w1", "rms_mid_gamma", "stack_w2",
                    "residual_hidden", "res_rms_pre_gamma", "res_w1",
                    "res_rms_mid_gamma", "res_w2", "res2_hidden",
                    "res2_rms_pre_gamma", "res2_w1", "res2_rms_mid_gamma",
                    "res2_w2")
        elif architecture == ARCH_STACK_RESIDUAL:
            keys = ("rms_pre_gamma", "stack_w1", "rms_mid_gamma", "stack_w2",
                    "residual_hidden", "res_rms_pre_gamma", "res_w1",
                    "res_rms_mid_gamma", "res_w2")
        elif architecture == ARCH_STACK_SWIGLU:
            keys = ("rms_pre_gamma", "stack_w1", "rms_mid_gamma", "stack_w2")
        else:
            keys = ("ln_gamma", "ln_beta", "w1", "b1", "w2", "b2")
        arrays = {k: archive[k] for k in keys}
        arrays["adapter_arch"] = np.array(architecture)
    for key, value in arrays.items():
        if key == "adapter_arch":
            continue
        if not np.isfinite(value).all():
            raise ValueError("cleanroom adapter contains non-finite values")
    return arrays


def build_context_adapter(hidden: int = 2048, seed: int = 0,
                          arrays: dict[str, np.ndarray] | None = None):
    """v2 adapter: CONTEXT-MIXING, ~41M params at hidden=2048.

    The v1 MLP translates every pooled frame in isolation, but phonetic
    disambiguation is inherently contextual — "Quilter" vs "Keuler" cannot be
    resolved from 80 ms alone, and the measured error mode is exactly such
    near-phone substitutions. One bidirectional attention block + FFN over the
    pooled sequence lets each frame see its neighbours before projection.
    Whole turns are <=29 s (<=725 pooled frames), so full attention is cheap.

        LN -> Linear(1024,h) -> GELU -> pool/2
           -> [MHA + residual + LN] -> [FFN + residual + LN]
           -> Linear(h, 2560)

    Saved as a flattened parameter tree with an `arch` marker; v1 checkpoints
    (ln/up/out keys) load through build_mlx_adapter as before.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten

    class ContextAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(ENC_DIM)
            self.up = nn.Linear(ENC_DIM, hidden)
            self.attn = nn.MultiHeadAttention(hidden, 8)
            self.ln_a = nn.LayerNorm(hidden)
            self.ff1 = nn.Linear(hidden, hidden * 2)
            self.ff2 = nn.Linear(hidden * 2, hidden)
            self.ln_f = nn.LayerNorm(hidden)
            self.out = nn.Linear(hidden, OUT_DIM)

        def __call__(self, values):
            h = nn.gelu(self.up(self.ln(values)))          # (T, h) @ 50Hz
            usable = (h.shape[0] // POOL) * POOL
            if usable == 0:
                h = mx.mean(h, axis=0, keepdims=True)
            else:
                h = mx.mean(h[:usable].reshape(-1, POOL, hidden), axis=1)
            x = h[None, :]                                  # (1, T', h)
            x = self.ln_a(x + self.attn(x, x, x))
            x = self.ln_f(x + self.ff2(nn.gelu(self.ff1(x))))
            return self.out(x[0])

    adapter = ContextAdapter()
    if arrays is not None:
        flat = [(k, mx.array(v)) for k, v in arrays.items()
                if k not in ("arch", "hidden")]
        adapter.update(tree_unflatten(flat))
    else:
        # deterministic init for reproducibility
        mx.random.seed(seed)
        adapter.update(tree_unflatten(
            [(k, v) for k, v in tree_flatten(adapter.parameters())]))
    mx.eval(adapter.parameters())
    return adapter


def context_arrays_from_module(adapter, hidden: int) -> dict[str, np.ndarray]:
    from mlx.utils import tree_flatten
    out = {k: np.array(v) for k, v in tree_flatten(adapter.parameters())}
    out["arch"] = np.frombuffer(b"context_v2", dtype=np.uint8)
    out["hidden"] = np.array([hidden], dtype=np.int64)
    return out


def make_adapter(arch: str = "mlp", hidden: int | None = None, seed: int = 0,
                 resume=None):
    """One entry point for both trainers. Resume dispatches on the checkpoint's
    own format; fresh builds follow the requested arch."""
    if resume:
        return load_any_adapter(Path(resume))
    if arch == "context":
        return build_context_adapter(hidden or 2048, seed), "context_v2"
    return build_mlx_adapter(fresh_adapter_arrays(seed, hidden)), "mlp_v1"


def save_any_adapter(path, adapter, arch: str) -> None:
    if arch == "context_v2":
        hidden = int(adapter.up.weight.shape[0])
        np.savez(path, **context_arrays_from_module(adapter, hidden))
    else:
        np.savez(path, **adapter_arrays_from_module(adapter))


def load_any_adapter(path: Path):
    """Dispatch on checkpoint format: streaming_causal_v1, context v2, or v1."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arch_tag = (bytes(arrays["arch"]).decode(errors="replace")
                if "arch" in arrays else "")
    if "streaming_causal" in arch_tag:
        # Causal trunk + perception heads. Built by streaming_adapter so the
        # head parameters are recognised; the plain context builder rejects
        # them ("Module does not have parameter named head_*").
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from streaming_adapter import build_streaming_adapter
        hidden = int(arrays["hidden"][0])
        return build_streaming_adapter(hidden, arrays=arrays,
                                       causal=True), "streaming_causal_v1"
    if "arch" in arrays:
        hidden = int(arrays["hidden"][0])
        # Packaged checkpoints also carry provenance/architecture metadata.
        # Only named parameters belong in the module tree; keep the NPZ intact.
        params = {k: v for k, v in arrays.items() if "." in k}
        return build_context_adapter(hidden, arrays=params), "context_v2"
    return build_mlx_adapter(
        {k: arrays[k] for k in
         ("ln_gamma", "ln_beta", "w1", "b1", "w2", "b2")}), "mlp_v1"


def build_mlx_adapter(arrays: dict[str, np.ndarray]):
    """Fresh adapter as an MLX module — shared by runtime and trainer."""
    import mlx.core as mx
    import mlx.nn as nn

    architecture = str(arrays.get("adapter_arch", ARCH_MEAN_GELU))

    if architecture in (ARCH_STACK_SWIGLU, ARCH_STACK_RESIDUAL,
                         ARCH_STACK_RESIDUAL2, ARCH_STACK_CALIBRATION,
                         ARCH_STACK_TEMPORAL, ARCH_STACK_TEMPORAL2,
                         ARCH_STACK_CONTEXT):
        class StackSwiGLUAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.pre = nn.RMSNorm(ENC_DIM * POOL, eps=1e-6)
                self.up = nn.Linear(ENC_DIM * POOL, OUT_DIM * 2, bias=False)
                self.mid = nn.RMSNorm(OUT_DIM, eps=1e-6)
                self.out = nn.Linear(OUT_DIM, OUT_DIM, bias=False)
                self.pre.weight = mx.array(arrays["rms_pre_gamma"])
                self.up.weight = mx.array(arrays["stack_w1"])
                self.mid.weight = mx.array(arrays["rms_mid_gamma"])
                self.out.weight = mx.array(arrays["stack_w2"])
                if architecture in (ARCH_STACK_RESIDUAL,
                                     ARCH_STACK_RESIDUAL2,
                                     ARCH_STACK_CALIBRATION,
                                     ARCH_STACK_TEMPORAL,
                                     ARCH_STACK_TEMPORAL2,
                                     ARCH_STACK_CONTEXT):
                    residual_hidden = int(arrays["residual_hidden"])
                    self.res_pre = nn.RMSNorm(OUT_DIM, eps=1e-6)
                    self.res_up = nn.Linear(
                        OUT_DIM, residual_hidden * 2, bias=False)
                    self.res_mid = nn.RMSNorm(residual_hidden, eps=1e-6)
                    self.res_out = nn.Linear(
                        residual_hidden, OUT_DIM, bias=False)
                    self.res_pre.weight = mx.array(
                        arrays["res_rms_pre_gamma"])
                    self.res_up.weight = mx.array(arrays["res_w1"])
                    self.res_mid.weight = mx.array(
                        arrays["res_rms_mid_gamma"])
                    self.res_out.weight = mx.array(arrays["res_w2"])
                if architecture == ARCH_STACK_RESIDUAL2:
                    residual2_hidden = int(arrays["res2_hidden"])
                    self.res2_pre = nn.RMSNorm(OUT_DIM, eps=1e-6)
                    self.res2_up = nn.Linear(
                        OUT_DIM, residual2_hidden * 2, bias=False)
                    self.res2_mid = nn.RMSNorm(residual2_hidden, eps=1e-6)
                    self.res2_out = nn.Linear(
                        residual2_hidden, OUT_DIM, bias=False)
                    self.res2_pre.weight = mx.array(
                        arrays["res2_rms_pre_gamma"])
                    self.res2_up.weight = mx.array(arrays["res2_w1"])
                    self.res2_mid.weight = mx.array(
                        arrays["res2_rms_mid_gamma"])
                    self.res2_out.weight = mx.array(arrays["res2_w2"])
                if architecture == ARCH_STACK_CALIBRATION:
                    self.cal_scale = mx.array(arrays["cal_scale"])
                    self.cal_bias = mx.array(arrays["cal_bias"])
                if architecture in (ARCH_STACK_TEMPORAL,
                                     ARCH_STACK_TEMPORAL2,
                                     ARCH_STACK_CONTEXT):
                    temporal_hidden = int(arrays["temporal_hidden"])
                    self.temp_pre = nn.RMSNorm(OUT_DIM, eps=1e-6)
                    self.temp_up = nn.Linear(
                        OUT_DIM, temporal_hidden * 2, bias=False)
                    self.temp_conv = nn.Conv1d(
                        temporal_hidden, temporal_hidden, 3, padding=1,
                        groups=temporal_hidden, bias=False)
                    self.temp_mid = nn.RMSNorm(temporal_hidden, eps=1e-6)
                    self.temp_out = nn.Linear(
                        temporal_hidden, OUT_DIM, bias=False)
                    self.temp_pre.weight = mx.array(
                        arrays["temp_rms_pre_gamma"])
                    self.temp_up.weight = mx.array(arrays["temp_w1"])
                    self.temp_conv.weight = mx.array(arrays["temp_dw"])
                    self.temp_mid.weight = mx.array(
                        arrays["temp_rms_mid_gamma"])
                    self.temp_out.weight = mx.array(arrays["temp_w2"])
                if architecture in (ARCH_STACK_TEMPORAL2,
                                    ARCH_STACK_CONTEXT):
                    temporal2_hidden = int(arrays["temporal2_hidden"])
                    temporal2_kernel = int(arrays["temporal2_kernel"])
                    self.temp2_pre = nn.RMSNorm(OUT_DIM, eps=1e-6)
                    self.temp2_up = nn.Linear(
                        OUT_DIM, temporal2_hidden * 2, bias=False)
                    self.temp2_conv = nn.Conv1d(
                        temporal2_hidden, temporal2_hidden, temporal2_kernel,
                        padding=temporal2_kernel // 2,
                        groups=temporal2_hidden, bias=False)
                    self.temp2_mid = nn.RMSNorm(temporal2_hidden, eps=1e-6)
                    self.temp2_out = nn.Linear(
                        temporal2_hidden, OUT_DIM, bias=False)
                    self.temp2_pre.weight = mx.array(
                        arrays["temp2_rms_pre_gamma"])
                    self.temp2_up.weight = mx.array(arrays["temp2_w1"])
                    self.temp2_conv.weight = mx.array(arrays["temp2_dw"])
                    self.temp2_mid.weight = mx.array(
                        arrays["temp2_rms_mid_gamma"])
                    self.temp2_out.weight = mx.array(arrays["temp2_w2"])
                if architecture == ARCH_STACK_CONTEXT:
                    context_hidden = int(arrays["context_hidden"])
                    context_kernel = int(arrays["context_kernel"])
                    self.context_heads = int(arrays["context_heads"])
                    self.ctx_pre = nn.RMSNorm(OUT_DIM, eps=1e-6)
                    self.ctx_up = nn.Linear(
                        OUT_DIM, context_hidden * 2, bias=False)
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
                    self.ctx_out = nn.Linear(
                        context_hidden, OUT_DIM, bias=False)
                    self.ctx_pre.weight = mx.array(arrays["ctx_rms_pre_gamma"])
                    self.ctx_up.weight = mx.array(arrays["ctx_w1"])
                    self.ctx_attn_norm.weight = mx.array(
                        arrays["ctx_rms_attn_gamma"])
                    self.ctx_qkv.weight = mx.array(arrays["ctx_qkv_w"])
                    self.ctx_attn_out.weight = mx.array(arrays["ctx_attn_w"])
                    self.ctx_conv_norm.weight = mx.array(
                        arrays["ctx_rms_conv_gamma"])
                    self.ctx_conv.weight = mx.array(arrays["ctx_dw"])
                    self.ctx_mid.weight = mx.array(
                        arrays["ctx_rms_mid_gamma"])
                    self.ctx_out.weight = mx.array(arrays["ctx_w2"])

            def __call__(self, values, return_context=False):
                usable = (values.shape[0] // POOL) * POOL
                if usable == 0:
                    values = mx.concatenate([values, values], axis=0)
                    usable = POOL
                stacked = values[:usable].reshape(-1, ENC_DIM * POOL)
                gate, value = mx.split(self.up(self.pre(stacked)), 2, axis=-1)
                base = self.out(self.mid(nn.silu(gate) * value))
                if architecture in (ARCH_STACK_RESIDUAL,
                                     ARCH_STACK_RESIDUAL2,
                                     ARCH_STACK_CALIBRATION,
                                     ARCH_STACK_TEMPORAL,
                                     ARCH_STACK_TEMPORAL2,
                                     ARCH_STACK_CONTEXT):
                    r_gate, r_value = mx.split(
                        self.res_up(self.res_pre(base)), 2, axis=-1)
                    residual = self.res_out(
                        self.res_mid(nn.silu(r_gate) * r_value))
                    base = base + residual
                if architecture == ARCH_STACK_RESIDUAL2:
                    r2_gate, r2_value = mx.split(
                        self.res2_up(self.res2_pre(base)), 2, axis=-1)
                    base = base + self.res2_out(
                        self.res2_mid(nn.silu(r2_gate) * r2_value))
                if architecture == ARCH_STACK_CALIBRATION:
                    base = base * (1.0 + self.cal_scale) + self.cal_bias
                if architecture in (ARCH_STACK_TEMPORAL,
                                     ARCH_STACK_TEMPORAL2,
                                     ARCH_STACK_CONTEXT):
                    t_gate, t_value = mx.split(
                        self.temp_up(self.temp_pre(base)), 2, axis=-1)
                    temporal = nn.silu(t_gate) * t_value
                    temporal = self.temp_conv(temporal[None, :, :])[0]
                    base = base + self.temp_out(self.temp_mid(temporal))
                if architecture in (ARCH_STACK_TEMPORAL2,
                                    ARCH_STACK_CONTEXT):
                    t2_gate, t2_value = mx.split(
                        self.temp2_up(self.temp2_pre(base)), 2, axis=-1)
                    temporal2 = nn.silu(t2_gate) * t2_value
                    temporal2 = self.temp2_conv(temporal2[None, :, :])[0]
                    base = base + self.temp2_out(self.temp2_mid(temporal2))
                if architecture == ARCH_STACK_CONTEXT:
                    c_gate, c_value = mx.split(
                        self.ctx_up(self.ctx_pre(base)), 2, axis=-1)
                    context = nn.silu(c_gate) * c_value
                    normalized = self.ctx_attn_norm(context)
                    query, key, value = mx.split(
                        self.ctx_qkv(normalized), 3, axis=-1)
                    heads = self.context_heads
                    head_dim = query.shape[-1] // heads
                    query = query.reshape(-1, heads, head_dim)
                    key = key.reshape(-1, heads, head_dim)
                    value = value.reshape(-1, heads, head_dim)
                    scores = mx.einsum(
                        "thd,shd->hts", query, key) / np.sqrt(head_dim)
                    weights = mx.softmax(scores, axis=-1)
                    attended = mx.einsum(
                        "hts,shd->thd", weights, value).reshape(context.shape)
                    context = context + self.ctx_attn_out(attended)
                    convolved = self.ctx_conv(
                        self.ctx_conv_norm(context)[None, :, :])[0]
                    context = context + nn.silu(convolved)
                    base = base + self.ctx_out(self.ctx_mid(context))
                    if return_context:
                        return base, context
                if return_context:
                    raise ValueError(
                        "context features require the contextual architecture")
                return base

        adapter = StackSwiGLUAdapter()
        mx.eval(adapter.parameters())
        return adapter

    if architecture != ARCH_MEAN_GELU:
        raise ValueError(f"unsupported architecture {architecture!r}")
    hid = int(arrays["w1"].shape[0])          # shape is the source of truth

    class FreshAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(ENC_DIM)
            self.up = nn.Linear(ENC_DIM, hid)
            self.out = nn.Linear(hid, OUT_DIM)
            self.ln.weight = mx.array(arrays["ln_gamma"])
            self.ln.bias = mx.array(arrays["ln_beta"])
            self.up.weight = mx.array(arrays["w1"])
            self.up.bias = mx.array(arrays["b1"])
            self.out.weight = mx.array(arrays["w2"])
            self.out.bias = mx.array(arrays["b2"])

        def __call__(self, values):
            hidden = nn.gelu(self.up(self.ln(values)))     # (T, hid) @ 50Hz
            usable = (hidden.shape[0] // POOL) * POOL
            if usable == 0:                                # sub-pool-window clip
                hidden = mx.mean(hidden, axis=0, keepdims=True)
            else:
                hidden = mx.mean(
                    hidden[:usable].reshape(-1, POOL, hid), axis=1)
            return self.out(hidden)                        # (T/4, 2560)

    adapter = FreshAdapter()
    mx.eval(adapter.parameters())
    return adapter


def adapter_arrays_from_module(adapter) -> dict[str, np.ndarray]:
    import numpy as _np
    if hasattr(adapter, "pre"):
        result = {
            "adapter_arch": _np.array(ARCH_STACK_SWIGLU),
            "rms_pre_gamma": _np.array(adapter.pre.weight),
            "stack_w1": _np.array(adapter.up.weight),
            "rms_mid_gamma": _np.array(adapter.mid.weight),
            "stack_w2": _np.array(adapter.out.weight),
        }
        if hasattr(adapter, "res_pre"):
            result.update({
                "adapter_arch": _np.array(ARCH_STACK_RESIDUAL),
                "residual_hidden": _np.array(
                    adapter.res_up.weight.shape[0] // 2, dtype=_np.int32),
                "res_rms_pre_gamma": _np.array(adapter.res_pre.weight),
                "res_w1": _np.array(adapter.res_up.weight),
                "res_rms_mid_gamma": _np.array(adapter.res_mid.weight),
                "res_w2": _np.array(adapter.res_out.weight),
            })
        if hasattr(adapter, "res2_pre"):
            result.update({
                "adapter_arch": _np.array(ARCH_STACK_RESIDUAL2),
                "res2_hidden": _np.array(
                    adapter.res2_up.weight.shape[0] // 2, dtype=_np.int32),
                "res2_rms_pre_gamma": _np.array(adapter.res2_pre.weight),
                "res2_w1": _np.array(adapter.res2_up.weight),
                "res2_rms_mid_gamma": _np.array(adapter.res2_mid.weight),
                "res2_w2": _np.array(adapter.res2_out.weight),
            })
        if hasattr(adapter, "cal_scale"):
            result.update({
                "adapter_arch": _np.array(ARCH_STACK_CALIBRATION),
                "cal_scale": _np.array(adapter.cal_scale),
                "cal_bias": _np.array(adapter.cal_bias),
            })
        if hasattr(adapter, "temp_pre"):
            result.update({
                "adapter_arch": _np.array(ARCH_STACK_TEMPORAL),
                "temporal_hidden": _np.array(
                    adapter.temp_up.weight.shape[0] // 2, dtype=_np.int32),
                "temp_rms_pre_gamma": _np.array(adapter.temp_pre.weight),
                "temp_w1": _np.array(adapter.temp_up.weight),
                "temp_dw": _np.array(adapter.temp_conv.weight),
                "temp_rms_mid_gamma": _np.array(adapter.temp_mid.weight),
                "temp_w2": _np.array(adapter.temp_out.weight),
            })
        if hasattr(adapter, "temp2_pre"):
            result.update({
                "adapter_arch": _np.array(ARCH_STACK_TEMPORAL2),
                "temporal2_hidden": _np.array(
                    adapter.temp2_up.weight.shape[0] // 2, dtype=_np.int32),
                "temporal2_kernel": _np.array(
                    adapter.temp2_conv.weight.shape[1], dtype=_np.int32),
                "temp2_rms_pre_gamma": _np.array(adapter.temp2_pre.weight),
                "temp2_w1": _np.array(adapter.temp2_up.weight),
                "temp2_dw": _np.array(adapter.temp2_conv.weight),
                "temp2_rms_mid_gamma": _np.array(adapter.temp2_mid.weight),
                "temp2_w2": _np.array(adapter.temp2_out.weight),
            })
        if hasattr(adapter, "ctx_pre"):
            result.update({
                "adapter_arch": _np.array(ARCH_STACK_CONTEXT),
                "context_hidden": _np.array(
                    adapter.ctx_up.weight.shape[0] // 2, dtype=_np.int32),
                "context_heads": _np.array(
                    adapter.context_heads, dtype=_np.int32),
                "context_kernel": _np.array(
                    adapter.ctx_conv.weight.shape[1], dtype=_np.int32),
                "ctx_rms_pre_gamma": _np.array(adapter.ctx_pre.weight),
                "ctx_w1": _np.array(adapter.ctx_up.weight),
                "ctx_rms_attn_gamma": _np.array(
                    adapter.ctx_attn_norm.weight),
                "ctx_qkv_w": _np.array(adapter.ctx_qkv.weight),
                "ctx_attn_w": _np.array(adapter.ctx_attn_out.weight),
                "ctx_rms_conv_gamma": _np.array(
                    adapter.ctx_conv_norm.weight),
                "ctx_dw": _np.array(adapter.ctx_conv.weight),
                "ctx_rms_mid_gamma": _np.array(adapter.ctx_mid.weight),
                "ctx_w2": _np.array(adapter.ctx_out.weight),
            })
        return result
    return {
        "adapter_arch": _np.array(ARCH_MEAN_GELU),
        "ln_gamma": _np.array(adapter.ln.weight), "ln_beta": _np.array(adapter.ln.bias),
        "w1": _np.array(adapter.up.weight), "b1": _np.array(adapter.up.bias),
        "w2": _np.array(adapter.out.weight), "b2": _np.array(adapter.out.bias),
    }


class CleanroomTranscriber(mas.MlxGemmaAudioTranscriber):
    """The bridge transcriber with ONLY the seam swapped: stock encoder in,
    fresh MLP adapter on. Decode path, template, suppression, loop blockers
    are inherited unchanged so evaluations compare seams, not decoders."""

    def __init__(self, adapter_path: Path | None = None,
                 model_path: Path | None = None,
                 encoder: dict | None = None, **kwargs):
        # model_path lets evaluation run on the SAME export the distillation
        # teacher came from. The 4-bit text export and the full multimodal
        # export are the same Gemma weights at different precision, but a
        # distilled adapter targets one embedding space and that should be
        # measured, not assumed.
        super().__init__(model_path or mas.DEFAULT_MODEL, mas.DEFAULT_ADAPTER,
                         mas.DEFAULT_AUDIO_GGUF, **kwargs)
        # replace the MiniCPM front end + affine with the GemmaOmni pair
        enc = encoder or {}
        self.front_end = StockWhisperFrontEnd(
            device="mps" if self.torch_available() else "cpu",
            source=enc.get("source", STOCK_WHISPER),
            output_dim=int(enc.get("output_dim", ENC_DIM)))
        if adapter_path:
            self.adapter, self.adapter_arch = load_any_adapter(adapter_path)
        else:
            self.adapter = build_mlx_adapter(fresh_adapter_arrays())
            self.adapter_arch = "mlp_v1"
        self.adapter_path = adapter_path or Path("<fresh-untrained>")

    @staticmethod
    def torch_available() -> bool:
        import torch
        return torch.backends.mps.is_available()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true",
                    help="run question.wav through the full endpointed "
                         "pipeline; untrained adapter -> gibberish is the "
                         "EXPECTED text, the check is plumbing + shapes")
    ap.add_argument("--adapter", type=Path, default=None)
    args = ap.parse_args()
    if not args.smoke:
        ap.print_help()
        return 0

    import soundfile as sf
    wav = sorted(ROOT.parent.glob("*/assets/question.wav"))[0]
    pcm, _ = sf.read(str(wav), dtype="float32")
    pcm = np.concatenate([pcm, np.zeros(mas.SAMPLE_RATE * 2, np.float32)])
    commits = []
    session = mas.EndpointedSession(
        lambda: CleanroomTranscriber(args.adapter),
        frame_ms=30, rms_threshold=0.015, vad_backend="silero",
        on_commit=lambda i, t: commits.append((i, t)))
    print("loading GemmaOmni pipeline (stock whisper + fresh adapter + "
          "Gemma)…", flush=True)
    session.start()
    for i in range(0, len(pcm) - 480, 480):
        session.push(pcm[i:i + 480])
    result = session.finish(timeout=180)
    trained = args.adapter is not None
    print(f"turns {len(result['turns'])} · dropped "
          f"{result['samples_dropped']} · adapter "
          f"{'TRAINED' if trained else 'FRESH/untrained'}")
    print(f"text: {result['transcript'][:200]!r}")
    print("CLEANROOM PLUMBING OK — commits flowed end to end"
          if result["turns"] else "NO TURNS — check VAD/levels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
