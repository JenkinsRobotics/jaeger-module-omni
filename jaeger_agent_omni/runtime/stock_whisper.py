"""Qwen-free stock-Whisper audio front end for frozen Gemma.

The runtime boundary is deliberately small and explicit::

    PCM -> stock OpenAI Whisper encoder
        -> fresh Jaeger MLP -> AvgPool5 -> Gemma 4 E4B -> answer tokens

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
AUDIO_MARKER = "__JAEGER_STOCK_WHISPER_AUDIO__"
DEFAULT_WHISPER = "openai/whisper-medium"
SUPPORTED_WHISPER = frozenset((
    "openai/whisper-medium",
    "openai/whisper-medium.en",
))
DEFAULT_GEMMA = Path.home() / "models/gemma-4-E4B-it-4bit-mlx"
INPUT_BOUNDARY = "stock_whisper_encoder_50hz"
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


def _pool_mlx(mx, values):
    """Average time after the nonlinear adapter, matching MiniCPM's order."""
    usable = (values.shape[0] // POOL_STEP) * POOL_STEP
    if usable:
        return mx.mean(
            values[:usable].reshape(-1, POOL_STEP, values.shape[-1]), axis=1)
    return mx.mean(values, axis=0, keepdims=True)


def validate_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
    """Load and strictly identify a Qwen-free stock-Whisper checkpoint."""
    with np.load(path, allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    expected = {
        "linear1_weight": (HIDDEN_DIM, INPUT_DIM),
        "linear1_bias": (HIDDEN_DIM,),
        "linear2_weight": (OUTPUT_DIM, HIDDEN_DIM),
        "linear2_bias": (OUTPUT_DIM,),
    }
    for key, shape in expected.items():
        if key not in values or values[key].shape != shape:
            raise ValueError(f"{path}: {key} must be {shape}")
    boundary = str(values.get("input_boundary", ""))
    encoder = str(values.get("encoder", ""))
    lineage = str(values.get("lineage", ""))
    if boundary != INPUT_BOUNDARY:
        raise ValueError(f"{path}: wrong input boundary {boundary!r}")
    if encoder not in SUPPORTED_WHISPER:
        raise ValueError(f"{path}: unsupported encoder {encoder!r}")
    if lineage != "qwen_free":
        raise ValueError(f"{path}: checkpoint is not marked qwen_free")
    if not all(np.isfinite(values[key]).all() for key in expected):
        raise ValueError(f"{path}: non-finite adapter value")
    return values


class StockWhisperGemmaSession:
    """Minimal session contract used by :mod:`jaeger_agent_omni.omni`."""

    def __init__(self, adapter_path: str | Path,
                 model_path: str | Path = DEFAULT_GEMMA,
                 whisper_id: str = DEFAULT_WHISPER,
                 system: str | None = None,
                 device: str = "auto") -> None:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load

        values = validate_checkpoint(adapter_path)
        checkpoint_encoder = str(values["encoder"])
        output_mode = str(values.get("output_mode", OUTPUT_MODE_NORMALIZED))
        if output_mode not in (OUTPUT_MODE_NORMALIZED, OUTPUT_MODE_RAW):
            raise ValueError(
                f"{adapter_path}: unsupported output mode {output_mode!r}")
        if checkpoint_encoder != whisper_id:
            raise ValueError(
                f"{adapter_path}: encoder is {checkpoint_encoder!r}, but "
                f"runtime requested {whisper_id!r}")
        self.mx = mx
        self.nn = nn
        self.front_end = StockWhisperFrontEnd(whisper_id, device)
        self.model, self.tokenizer = load(str(model_path), lazy=False)
        self.model.freeze()
        self.model.eval()
        self.processor = type("Processor", (), {"tokenizer": self.tokenizer})()
        self.system = system
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
                inner.linear1 = nn.Linear(INPUT_DIM, HIDDEN_DIM, bias=True)
                inner.linear2 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM, bias=True)
                inner.linear1.weight = mx.array(values["linear1_weight"])
                inner.linear1.bias = mx.array(values["linear1_bias"])
                inner.linear2.weight = mx.array(values["linear2_weight"])
                inner.linear2.bias = mx.array(values["linear2_bias"])

            def __call__(inner, x):
                y = inner.linear2(nn.relu(inner.linear1(x)))
                y = _pool_mlx(mx, y)
                if output_mode == OUTPUT_MODE_RAW:
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
        self.n_prefilled = max(1, len(self._features) // POOL_STEP)

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
        n_audio = max(1, len(self._features) // POOL_STEP)
        ids_list = prefix + [0] * n_audio + suffix
        ids = mx.array(ids_list, dtype=mx.int32)
        tokens = self.model.language_model.model.embed_tokens(ids[None, :])[0]
        raw_audio = self.adapter(mx.array(self._features))
        audio = raw_audio / math.sqrt(OUTPUT_DIM)
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
    print("runtime.stock_whisper: stock encoder -> fresh Jaeger MLP -> Gemma; "
          "ASR decoder unused; five-minute window chunking contract OK")


if __name__ == "__main__":
    selfcheck()
