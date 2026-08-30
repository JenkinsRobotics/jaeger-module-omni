"""Qwen-free tuned-Whisper audio input for Gemma 4.

This is the production form of the adapter recipe proven in the VoiceLLM
playground and recorded in ``dev/lessons from gemma-backbone.md``::

    PCM -> MiniCPM-tuned Whisper encoder
        -> ReLU(MiniCPM audio_projector.linear1) -> AvgPool5
        -> trained Jaeger 4096x2560 affine -> Gemma 4 E4B -> tokens

The audio GGUF is only a container for the tuned Whisper encoder and the first
audio-projector affine.  The Qwen language model, Whisper decoder, and the
original Qwen-facing second projector are never loaded.  No transcript exists
between the audio encoder and Gemma.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
WHISPER_DIM = 1024
BRIDGE_DIM = 4096
GEMMA_DIM = 2560
POOL_STEP = 5
WHISPER_WINDOW_S = 28
WHISPER_OVERLAP_S = 1
AUDIO_MARKER = "__JAEGER_TUNED_WHISPER_AUDIO__"
INPUT_BOUNDARY = "avgpool5(relu(audio_projector.linear1(APM)))"
DEFAULT_MODEL = Path.home() / "models/gemma-4-E4B-it-4bit-mlx"
DEFAULT_ADAPTER = (
    Path.home() / "models/gemma-audio-adapter"
    / "adapter_sparse_vox1375_k022_b000.npz"
)
DEFAULT_AUDIO_GGUF = (
    Path.home() / "models/MiniCPM-o-4_5-gguf/audio"
    / "MiniCPM-o-4_5-audio-F16.gguf"
)


def _tensor(reader, name: str) -> np.ndarray:
    """Read one named GGUF tensor without materialising unrelated weights."""
    from gguf.quants import dequantize

    try:
        tensor = next(item for item in reader.tensors if item.name == name)
    except StopIteration as exc:
        raise KeyError(f"audio GGUF is missing {name!r}") from exc
    return dequantize(tensor.data, tensor.tensor_type).astype(np.float32)


def _whisper_state(reader) -> dict[str, object]:
    """Map only the audio-tuned Whisper encoder tensors into HF names."""
    import torch

    mapping = {
        "encoder.conv1.weight": "conv1.weight",
        "encoder.conv1.bias": "conv1.bias",
        "encoder.conv2.weight": "conv2.weight",
        "encoder.conv2.bias": "conv2.bias",
        "encoder.positional_embedding": "embed_positions.weight",
        "encoder.ln_post.weight": "layer_norm.weight",
        "encoder.ln_post.bias": "layer_norm.bias",
    }
    for layer in range(24):
        source = f"encoder.blocks.{layer}"
        target = f"layers.{layer}"
        mapping.update({
            f"{source}.attn.key.weight": f"{target}.self_attn.k_proj.weight",
            f"{source}.attn.query.weight": f"{target}.self_attn.q_proj.weight",
            f"{source}.attn.query.bias": f"{target}.self_attn.q_proj.bias",
            f"{source}.attn.value.weight": f"{target}.self_attn.v_proj.weight",
            f"{source}.attn.value.bias": f"{target}.self_attn.v_proj.bias",
            f"{source}.attn.out.weight": f"{target}.self_attn.out_proj.weight",
            f"{source}.attn.out.bias": f"{target}.self_attn.out_proj.bias",
            f"{source}.attn_ln.weight":
                f"{target}.self_attn_layer_norm.weight",
            f"{source}.attn_ln.bias": f"{target}.self_attn_layer_norm.bias",
            f"{source}.mlp.0.weight": f"{target}.fc1.weight",
            f"{source}.mlp.0.bias": f"{target}.fc1.bias",
            f"{source}.mlp.2.weight": f"{target}.fc2.weight",
            f"{source}.mlp.2.bias": f"{target}.fc2.bias",
            f"{source}.mlp_ln.weight": f"{target}.final_layer_norm.weight",
            f"{source}.mlp_ln.bias": f"{target}.final_layer_norm.bias",
        })
    state = {}
    for source, target in mapping.items():
        value = _tensor(reader, source)
        if source in ("encoder.conv1.bias", "encoder.conv2.bias"):
            value = value.reshape(-1)
        state[target] = torch.from_numpy(value)
    return state


class TunedWhisperFrontEnd:
    """Tuned Whisper plus the audio-semantic first projector, no decoder."""

    def __init__(self, audio_gguf: str | Path = DEFAULT_AUDIO_GGUF,
                 device: str = "auto") -> None:
        import torch
        from gguf import GGUFReader
        from transformers import WhisperConfig, WhisperFeatureExtractor
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        audio_gguf = Path(audio_gguf)
        if not audio_gguf.exists():
            raise FileNotFoundError(audio_gguf)
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        reader = GGUFReader(str(audio_gguf))
        config = WhisperConfig(
            d_model=WHISPER_DIM,
            encoder_layers=24,
            encoder_attention_heads=16,
            encoder_ffn_dim=BRIDGE_DIM,
            num_mel_bins=80,
            max_source_positions=1500,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            activation_function="gelu",
        )
        self.encoder = WhisperEncoder(config)
        self.encoder.load_state_dict(_whisper_state(reader), strict=True)
        self.linear1_weight = torch.from_numpy(
            _tensor(reader, "audio_projector.linear1.weight"))
        self.linear1_bias = torch.from_numpy(
            _tensor(reader, "audio_projector.linear1.bias"))
        del reader

        self.dtype = (
            torch.float16 if self.device.type == "mps" else torch.float32)
        self.encoder.to(device=self.device, dtype=self.dtype).eval()
        self.encoder.requires_grad_(False)
        self.linear1_weight = self.linear1_weight.to(
            self.device, dtype=self.dtype)
        self.linear1_bias = self.linear1_bias.to(
            self.device, dtype=self.dtype)
        self.extractor = WhisperFeatureExtractor(
            feature_size=80,
            sampling_rate=SAMPLE_RATE,
            hop_length=160,
            chunk_length=30,
            n_fft=400,
            padding_value=0.0,
            return_attention_mask=True,
        )
        self.audio_gguf = audio_gguf.resolve()

    def _encode_variable(self, mel):
        import torch.nn.functional as functional

        hidden = functional.gelu(self.encoder.conv1(mel))
        hidden = functional.gelu(self.encoder.conv2(hidden)).permute(0, 2, 1)
        if hidden.shape[1] > self.encoder.embed_positions.weight.shape[0]:
            raise ValueError("audio exceeds Whisper's positional context")
        hidden = hidden + self.encoder.embed_positions.weight[:hidden.shape[1]]
        for layer in self.encoder.layers:
            hidden = layer(hidden, None)
        return self.encoder.layer_norm(hidden)

    def _encode_short(self, waveform: np.ndarray) -> np.ndarray:
        """Encode one bounded window into the adapter's exact 10 Hz boundary."""
        import torch.nn.functional as functional

        batch = self.extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        frame_count = int(batch.attention_mask.sum().item())
        mel = batch.input_features[:, :, :frame_count].to(
            self.device, dtype=self.dtype)
        with self.torch.inference_mode():
            states = self._encode_variable(mel)
            hidden = functional.relu(functional.linear(
                states, self.linear1_weight, self.linear1_bias))
            if hidden.shape[1] < POOL_STEP:
                hidden = hidden.mean(dim=1, keepdim=True)
            else:
                hidden = functional.avg_pool1d(
                    hidden.transpose(1, 2), POOL_STEP, POOL_STEP
                ).transpose(1, 2)
        encoder_frames = (frame_count + 1) // 2
        expected = max(1, encoder_frames // POOL_STEP)
        return hidden[0, :expected].float().cpu().numpy()

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Encode arbitrary audio using overlapping 28-second windows.

        The one-second overlap protects words at a bidirectional Whisper
        boundary. Duplicate 10 Hz overlap states are removed from each later
        window. A five-minute meeting therefore occupies about 3,000 Gemma
        audio tokens and remains inside the model's context window.
        """
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not waveform.size:
            raise ValueError("cannot encode empty audio")
        window = WHISPER_WINDOW_S * SAMPLE_RATE
        overlap = WHISPER_OVERLAP_S * SAMPLE_RATE
        if len(waveform) <= window:
            return self._encode_short(waveform)
        hop = window - overlap
        drop_tokens = WHISPER_OVERLAP_S * (50 // POOL_STEP)
        chunks = []
        start = 0
        while start < len(waveform):
            end = min(start + window, len(waveform))
            encoded = self._encode_short(waveform[start:end])
            if start:
                encoded = encoded[drop_tokens:]
            if len(encoded):
                chunks.append(encoded)
            if end == len(waveform):
                break
            start += hop
        return np.concatenate(chunks, axis=0)


def validate_adapter(path: str | Path) -> dict[str, np.ndarray]:
    """Validate the exact tuned-audio-to-Gemma checkpoint boundary."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    if values.get("weight", np.empty(0)).shape != (GEMMA_DIM, BRIDGE_DIM):
        raise ValueError(f"{path}: weight must be {(GEMMA_DIM, BRIDGE_DIM)}")
    if values.get("bias", np.empty(0)).shape != (GEMMA_DIM,):
        raise ValueError(f"{path}: bias must be {(GEMMA_DIM,)}")
    boundary = str(values.get("input_boundary", ""))
    if boundary and boundary != INPUT_BOUNDARY:
        raise ValueError(f"{path}: wrong input boundary {boundary!r}")
    for value in values.values():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"{path}: contains a non-finite value")
    return values


def _token_loop_blocker(size: int = 4, max_occurrences: int = 1):
    """Prevent a token candidate from extending an already repeated n-gram."""
    def processor(tokens, logits):
        import mlx.core as mx

        history = [int(token) for token in tokens.tolist()]
        if len(history) < size - 1:
            return logits
        prefix = tuple(history[-(size - 1):])
        counts = Counter(
            tuple(history[index:index + size])
            for index in range(len(history) - size + 1))
        blocked = [
            gram[-1] for gram, count in counts.items()
            if count >= max_occurrences and gram[:-1] == prefix]
        if not blocked:
            return logits
        return logits.at[:, mx.array(sorted(set(blocked)))].add(-1e9)

    return processor


def _word_loop_blocker(tokenizer, size: int = 4,
                       max_occurrences: int = 1, candidate_count: int = 8):
    """Veto top candidates that complete a repeated normalized word n-gram."""
    def normalized(token_ids):
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return re.findall(r"[a-z0-9]+", text.lower())

    def processor(tokens, logits):
        import mlx.core as mx

        history = [int(token) for token in tokens.tolist()][1:]
        if len(normalized(history)) < size - 1:
            return logits
        count = min(candidate_count, int(logits.shape[-1]))
        indices = mx.argpartition(logits, kth=-count, axis=-1)[:, -count:]
        blocked = []
        for candidate in [int(value) for value in indices.reshape(-1).tolist()]:
            words = normalized(history + [candidate])
            if len(words) < size:
                continue
            latest = tuple(words[-size:])
            occurrences = sum(
                tuple(words[index:index + size]) == latest
                for index in range(len(words) - size))
            if occurrences >= max_occurrences:
                blocked.append(candidate)
        if not blocked:
            return logits
        return logits.at[:, mx.array(sorted(set(blocked)))].add(-1e9)

    return processor


class TunedWhisperGemmaSession:
    """Session-compatible direct neural audio input using frozen Gemma MLX."""

    def __init__(self, adapter_path: str | Path = DEFAULT_ADAPTER,
                 model_path: str | Path = DEFAULT_MODEL,
                 audio_gguf: str | Path = DEFAULT_AUDIO_GGUF,
                 system: str | None = None, device: str = "auto") -> None:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load

        for path in (Path(model_path), Path(adapter_path), Path(audio_gguf)):
            if not path.exists():
                raise FileNotFoundError(path)
        values = validate_adapter(adapter_path)
        self.mx = mx
        self.model, self.tokenizer = load(str(model_path), lazy=False)
        self.model.freeze()
        self.model.eval()
        self.processor = type("Processor", (), {"tokenizer": self.tokenizer})()
        self.front_end = TunedWhisperFrontEnd(audio_gguf, device=device)
        self.system = system
        self._audio: list[np.ndarray] = []
        self._features: np.ndarray | None = None
        self._metadata: list[str] = []
        self._cancelled = False
        self.n_prefilled = 0
        self.hidden: list = []
        self.output_ids: list[int] = []
        self.adapter_path = Path(adapter_path).resolve()

        class Adapter(nn.Module):
            def __init__(inner):
                super().__init__()
                inner.proj = nn.Linear(BRIDGE_DIM, GEMMA_DIM, bias=True)
                inner.proj.weight = mx.array(values["weight"].astype(np.float32))
                inner.proj.bias = mx.array(values["bias"].astype(np.float32))
                inner.residual_down = None
                inner.residual_up = None
                if "residual_down_weight" in values:
                    residual_dim = int(values["residual_down_weight"].shape[0])
                    inner.residual_down = nn.Linear(
                        BRIDGE_DIM, residual_dim, bias=True)
                    inner.residual_down.weight = mx.array(
                        values["residual_down_weight"].astype(np.float32))
                    inner.residual_down.bias = mx.array(
                        values["residual_down_bias"].astype(np.float32))
                    inner.residual_up = nn.Linear(
                        residual_dim, GEMMA_DIM, bias=False)
                    inner.residual_up.weight = mx.array(
                        values["residual_up_weight"].astype(np.float32))

            def __call__(inner, values_in):
                output = inner.proj(values_in)
                if inner.residual_down is not None:
                    output = output + inner.residual_up(
                        nn.gelu(inner.residual_down(values_in)))
                return output

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
        """Accept only explicit between-turn metadata such as speaker tags."""
        self._metadata.append(str(text))
        return 0

    def end_of_turn(self, close: bool = True) -> None:
        if not self._audio:
            raise ValueError("no audio in turn")
        self._features = self.front_end(np.concatenate(self._audio))
        self.n_prefilled = len(self._features)

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

    def _embed_ids(self, ids: list[int]):
        token_ids = self.mx.array(ids, dtype=self.mx.int32)
        return self.model.language_model.model.embed_tokens(token_ids[None, :])[0], token_ids

    def respond(self, max_tokens: int = 96, eos_ids=(), on_token=None,
                collect_hidden: bool = False) -> str:
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_logits_processors

        if self._features is None:
            raise ValueError("end the audio turn before responding")
        if collect_hidden:
            raise NotImplementedError(
                "tuned MLX generation does not expose Talker hidden states")
        mx = self.mx
        prefix, suffix = self._prompt_parts()
        audio = self.adapter(mx.array(self._features)) / math.sqrt(GEMMA_DIM)
        n_audio = int(audio.shape[0])
        ids_list = prefix + [0] * n_audio + suffix
        ids = mx.array(ids_list, dtype=mx.int32)
        tokens = self.model.language_model.model.embed_tokens(ids[None, :])[0]
        start = len(prefix)
        embeddings = mx.concatenate(
            [tokens[:start], audio.astype(tokens.dtype),
             tokens[start + n_audio:]], axis=0)
        mx.eval(ids, embeddings)
        blocked = {
            int(self.tokenizer.convert_tokens_to_ids("<|think|>")),
            int(self.tokenizer.convert_tokens_to_ids("<|channel>")),
        }
        processors = make_logits_processors(
            logit_bias={token: -1e9 for token in blocked},
            repetition_penalty=1.05,
            repetition_context_size=20,
        )
        processors.extend((
            _token_loop_blocker(4, 1),
            _word_loop_blocker(self.tokenizer, 4, 1),
        ))
        generated = []
        stop_ids = {int(token) for token in eos_ids} | {1, 106}
        self._cancelled = False
        for token, _ in generate_step(
                ids, self.model, input_embeddings=embeddings,
                max_tokens=max_tokens, logits_processors=processors):
            token = int(token)
            if self._cancelled or token in stop_ids:
                break
            generated.append(token)
            if on_token:
                on_token(token, len(generated))
        self.output_ids = generated
        return self.tokenizer.decode(
            generated, skip_special_tokens=True).strip()

    def cancel(self) -> None:
        self._cancelled = True

    def interrupt(self) -> None:
        self._cancelled = False

    def resume_listening(self) -> None:
        pass

    def close_turn(self) -> None:
        pass


def selfcheck() -> None:
    """Run structural, adapter-boundary, and five-minute window checks."""
    assert WHISPER_DIM == 1024 and BRIDGE_DIM == 4096 and GEMMA_DIM == 2560
    assert "qwen" not in str(DEFAULT_MODEL).lower()
    assert "qwen" not in str(DEFAULT_ADAPTER).lower()
    fake = object.__new__(TunedWhisperFrontEnd)
    calls = []

    def fake_encode(waveform):
        calls.append(len(waveform))
        tokens = max(1, math.ceil(len(waveform) / (SAMPLE_RATE / 10)))
        return np.zeros((tokens, 1), dtype=np.float32)

    fake._encode_short = fake_encode
    joined = fake(np.zeros(5 * 60 * SAMPLE_RATE, dtype=np.float32))
    assert len(calls) == 12 and joined.shape == (3_000, 1)
    print("runtime.tuned_whisper: tuned Whisper + linear1 -> Jaeger adapter "
          "-> Gemma; no Qwen/decoder; five-minute window contract OK")


if __name__ == "__main__":
    selfcheck()
