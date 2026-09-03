#!/usr/bin/env python3
"""Direct streaming audio -> Gemma transcription on Apple Silicon.

This is the production-shaped runner for the trained Gemma audio adapter.  It
contains no text handoff between models:

    PCM -> MiniCPM-tuned Whisper encoder -> linear1/ReLU/pool
        -> trained 4096x2560 adapter -> Gemma 4 E4B MLX -> displayed text

Rolling text exists only after Gemma, where it is used to provide a live
preview and an immutable, duplicate-free terminal commit ledger.  The audio
callback never runs a model: it copies numbered PCM packets into an unbounded
queue, and one worker serializes PyTorch/MPS Whisper with MLX Gemma.  This
avoids both callback stalls and concurrent Metal command streams.

Examples:

    python mlx_audio_streaming.py --audio speech.wav --paced
    python mlx_audio_streaming.py --audio speech.wav --reference speech.txt
    python mlx_audio_streaming.py --self-test

The present adapter was trained for utterance transcription, not an
incremental Whisper KV cache.  Consequently rolling previews re-encode the
growing turn.  Turns are capped below Whisper's 30 second positional context
and fail visibly rather than silently truncating audio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import re
import threading
import time
from dataclasses import dataclass, field
import sys
from pathlib import Path
from typing import Callable

import numpy as np


SAMPLE_RATE = 16_000
IN_DIM = 4096
OUT_DIM = 2560
POOL_STEP = 5
AUDIO_MARKER = "__MINICPM_AUDIO_EMBEDDINGS__"
DEFAULT_MODEL = Path.home() / "models/gemma-4-E4B-it-4bit-mlx"
DEFAULT_ADAPTER = (
    Path.home()
    / "models/gemma-audio-adapter/adapter_sparse_vox1375_k022_b000.npz"
)
DEFAULT_SILERO_VAD = (
    Path.home() / "models/gemma-audio-adapter/silero_vad_16k_op15.onnx"
)
DEFAULT_AUDIO_GGUF = (
    Path.home()
    / "models/MiniCPM-o-4_5-gguf/audio/MiniCPM-o-4_5-audio-F16.gguf"
)
EOS_IDS = frozenset((1, 106))
PACKET_QUEUE_CAPACITY = 4096  # 122.88 seconds at the 30 ms production packet
TURN_QUEUE_CAPACITY = 64


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact(text: str) -> str:
    return " ".join(text.split())


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _word_distance(reference: list[str], hypothesis: list[str]) -> int:
    row = list(range(len(hypothesis) + 1))
    for i, source in enumerate(reference, 1):
        nxt = [i]
        for j, target in enumerate(hypothesis, 1):
            nxt.append(min(
                nxt[-1] + 1,
                row[j] + 1,
                row[j - 1] + (source != target),
            ))
        row = nxt
    return row[-1]


_ABBREVIATIONS = (
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "vs", "etc",
    "e.g", "i.e", "U.S", "U.K",
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+")


def split_sentences(text: str) -> list[str]:
    """Split completed punctuation while protecting common abbreviations."""
    clean = _compact(text)
    if not clean:
        return []
    protected = clean
    token = "\u2024"
    for abbreviation in _ABBREVIATIONS:
        protected = re.sub(
            rf"\b{re.escape(abbreviation)}\.",
            lambda match: match.group(0).replace(".", token),
            protected,
            flags=re.IGNORECASE,
        )
    protected = re.sub(r"(?<=\d)\.(?=\d)", token, protected)
    return [
        part.replace(token, ".").strip()
        for part in _SENTENCE_BOUNDARY.split(protected)
        if part.strip()
    ]


def _same_sentence(left: str, right: str) -> bool:
    """Ignore only presentation drift; lexical revisions remain revisions."""
    return _words(left) == _words(right)


def _has_terminal_punctuation(sentence: str) -> bool:
    return bool(re.search(r"[.!?][\"')\]]*$", sentence.rstrip()))


@dataclass
class StableCommitter:
    """Commit rolling Gemma sentences once and only in source order."""

    stable_passes: int = 2
    committed: list[str] = field(default_factory=list)
    _seen: dict[int, tuple[str, int]] = field(default_factory=dict)
    alignment_failures: int = 0

    def _aligned_prefix(self, sentences: list[str]) -> bool:
        if len(sentences) < len(self.committed):
            return False
        return all(
            _same_sentence(sentences[index], committed)
            for index, committed in enumerate(self.committed)
        )

    def update(self, transcript: str, *, allow_tail: bool = True) -> list[str]:
        sentences = split_sentences(transcript)
        if not self._aligned_prefix(sentences):
            # Never reinterpret already-committed audio as new text.  A later
            # clean decode can realign; the final decode preserves the ledger.
            self.alignment_failures += 1
            return []

        ready: list[str] = []
        first_pending = len(self.committed)
        for index in range(first_pending, len(sentences)):
            sentence = sentences[index]
            previous, passes = self._seen.get(index, ("", 0))
            passes = passes + 1 if _same_sentence(sentence, previous) else 1
            self._seen[index] = (sentence, passes)
            superseded = index < len(sentences) - 1
            # A long, unpunctuated Gemma tail can stay byte-identical for two
            # passes and then revise substantially as more audio arrives (the
            # 27 s JFK replay demonstrated this seven times).  Punctuation is
            # the model's only sentence-boundary signal, so stability may
            # finalize a tail only when that boundary is present.  End-of-turn
            # finish() still releases an unpunctuated final transcript.
            stable = (
                passes >= self.stable_passes
                and (superseded or allow_tail)
                and _has_terminal_punctuation(sentence)
            )
            if index == len(self.committed) and (superseded or stable):
                self.committed.append(sentence)
                ready.append(sentence)
        return ready

    def provisional(self, transcript: str) -> str:
        sentences = split_sentences(transcript)
        if not self._aligned_prefix(sentences):
            return transcript
        return " ".join(sentences[len(self.committed):])

    def finish(self, transcript: str) -> list[str]:
        """Use the final decode, but never mutate earlier immutable commits."""
        sentences = split_sentences(transcript)
        if not self._aligned_prefix(sentences):
            # A final lexical revision of committed material must not create a
            # duplicate.  Keep the immutable prefix and release no ambiguous
            # suffix; this condition is surfaced in the report.
            self.alignment_failures += 1
            return []
        ready = []
        for sentence in sentences[len(self.committed):]:
            self.committed.append(sentence)
            ready.append(sentence)
        return ready

    @property
    def text(self) -> str:
        return " ".join(self.committed)


def _tensor(reader, name: str) -> np.ndarray:
    from gguf.quants import dequantize

    try:
        tensor = next(item for item in reader.tensors if item.name == name)
    except StopIteration as exc:
        raise KeyError(f"audio GGUF is missing {name!r}") from exc
    return dequantize(tensor.data, tensor.tensor_type).astype(np.float32)


def _whisper_state(reader) -> dict[str, object]:
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
            f"{source}.attn_ln.weight": f"{target}.self_attn_layer_norm.weight",
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


class AudioFrontEnd:
    """MiniCPM-tuned Whisper through AvgPool5(ReLU(linear1(...)))."""

    def __init__(self, audio_gguf: Path, device: str = "mps"):
        import torch
        from gguf import GGUFReader
        from transformers import WhisperConfig, WhisperFeatureExtractor
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        self.torch = torch
        self.device = torch.device(device)
        reader = GGUFReader(str(audio_gguf))
        config = WhisperConfig(
            d_model=1024,
            encoder_layers=24,
            encoder_attention_heads=16,
            encoder_ffn_dim=4096,
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
            _tensor(reader, "audio_projector.linear1.weight")
        )
        self.linear1_bias = torch.from_numpy(
            _tensor(reader, "audio_projector.linear1.bias")
        )
        del reader

        self.dtype = torch.float16 if self.device.type == "mps" else torch.float32
        self.encoder.to(device=self.device, dtype=self.dtype).eval()
        self.encoder.requires_grad_(False)
        self.linear1_weight = self.linear1_weight.to(self.device, dtype=self.dtype)
        self.linear1_bias = self.linear1_bias.to(self.device, dtype=self.dtype)
        self.extractor = WhisperFeatureExtractor(
            feature_size=80,
            sampling_rate=SAMPLE_RATE,
            hop_length=160,
            chunk_length=30,
            n_fft=400,
            padding_value=0.0,
            return_attention_mask=True,
        )

    def _encode_variable(self, mel):
        import torch.nn.functional as functional

        hidden = functional.gelu(self.encoder.conv1(mel))
        hidden = functional.gelu(self.encoder.conv2(hidden)).permute(0, 2, 1)
        if hidden.shape[1] > self.encoder.embed_positions.weight.shape[0]:
            raise ValueError(
                "audio turn exceeds Whisper's 30-second positional context"
            )
        hidden = hidden + self.encoder.embed_positions.weight[:hidden.shape[1]]
        for layer in self.encoder.layers:
            hidden = layer(hidden, None)
        return self.encoder.layer_norm(hidden)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        import torch.nn.functional as functional

        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not waveform.size:
            raise ValueError("cannot encode an empty waveform")
        if waveform.size > SAMPLE_RATE * 29:
            raise ValueError(
                f"turn is {waveform.size / SAMPLE_RATE:.2f}s; direct streaming "
                "currently caps turns at 29s to avoid silent truncation"
            )
        batch = self.extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        frame_count = int(batch.attention_mask.sum().item())
        mel = batch.input_features[:, :, :frame_count].to(
            self.device, dtype=self.dtype
        )
        with self.torch.inference_mode():
            states = self._encode_variable(mel)
            hidden = functional.relu(functional.linear(
                states, self.linear1_weight, self.linear1_bias
            ))
            hidden = functional.avg_pool1d(
                hidden.transpose(1, 2), POOL_STEP, POOL_STEP
            ).transpose(1, 2)
        expected = ((frame_count - 1) // 2 + 1 - POOL_STEP) // POOL_STEP + 1
        return hidden[0, :expected].float().cpu().numpy()


class MlxGemmaAudioTranscriber:
    """Frozen MLX Gemma plus the one trained audio adapter."""

    def __init__(self, model_path: Path, adapter_path: Path,
                 audio_gguf: Path, device: str = "mps",
                 max_tokens: int = 128, repetition_penalty: float = 1.05,
                 repetition_context: int = 20,
                 suppress_thinking: bool = True,
                 loop_ngram_size: int = 4,
                 max_ngram_occurrences: int = 1,
                 word_loop_ngram_size: int = 4,
                 word_max_ngram_occurrences: int = 1):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load

        self._owner_thread = threading.get_ident()
        self.prepare_worker()
        for path in (model_path, adapter_path, audio_gguf):
            if not path.exists():
                raise FileNotFoundError(path)
        started = time.perf_counter()
        self.model, self.tokenizer = load(str(model_path), lazy=False)
        self.model.freeze()
        self.model.eval()
        self.front_end = AudioFrontEnd(audio_gguf, device=device)
        with np.load(adapter_path, allow_pickle=False) as archive:
            saved = {name: archive[name] for name in archive.files}
        if saved.get("weight", np.empty(0)).shape != (OUT_DIM, IN_DIM):
            raise ValueError(
                f"adapter weight must be {(OUT_DIM, IN_DIM)}, got "
                f"{saved.get('weight', np.empty(0)).shape}"
            )
        if saved.get("bias", np.empty(0)).shape != (OUT_DIM,):
            raise ValueError(
                f"adapter bias must be {(OUT_DIM,)}, got "
                f"{saved.get('bias', np.empty(0)).shape}"
            )
        numeric_values = (
            value for value in saved.values()
            if np.issubdtype(value.dtype, np.number)
        )
        if not all(np.isfinite(value).all() for value in numeric_values):
            raise ValueError("adapter contains non-finite values")
        self.model_path = model_path.resolve()
        self.adapter_path = adapter_path.resolve()
        self.adapter_sha256 = _file_sha256(adapter_path)
        self.audio_gguf_path = audio_gguf.resolve()

        class Adapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(IN_DIM, OUT_DIM, bias=True)
                self.proj.weight = mx.array(saved["weight"].astype(np.float32))
                self.proj.bias = mx.array(saved["bias"].astype(np.float32))
                self.residual_down = None
                self.residual_up = None
                if "residual_down_weight" in saved:
                    residual_dim = int(saved["residual_down_weight"].shape[0])
                    self.residual_down = nn.Linear(
                        IN_DIM, residual_dim, bias=True
                    )
                    self.residual_down.weight = mx.array(
                        saved["residual_down_weight"].astype(np.float32)
                    )
                    self.residual_down.bias = mx.array(
                        saved["residual_down_bias"].astype(np.float32)
                    )
                    self.residual_up = nn.Linear(
                        residual_dim, OUT_DIM, bias=False
                    )
                    self.residual_up.weight = mx.array(
                        saved["residual_up_weight"].astype(np.float32)
                    )

            def __call__(self, values):
                output = self.proj(values)
                if self.residual_down is not None:
                    output = output + self.residual_up(
                        nn.gelu(self.residual_down(values))
                    )
                return output

        self.adapter = Adapter()
        mx.eval(self.adapter.parameters())
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": AUDIO_MARKER}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if AUDIO_MARKER not in rendered:
            raise RuntimeError("Gemma chat template discarded the audio marker")
        prefix, suffix = rendered.split(AUDIO_MARKER, 1)
        self.prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.repetition_context = repetition_context
        self.suppress_thinking = suppress_thinking
        self.loop_ngram_size = loop_ngram_size
        self.max_ngram_occurrences = max_ngram_occurrences
        self.word_loop_ngram_size = word_loop_ngram_size
        self.word_max_ngram_occurrences = word_max_ngram_occurrences
        self.thinking_control_ids = (
            int(self.tokenizer.convert_tokens_to_ids("<|think|>")),
            int(self.tokenizer.convert_tokens_to_ids("<|channel>")),
        )
        self.load_s = time.perf_counter() - started

    def prepare_worker(self) -> None:
        """Create MLX's thread-local streams on the inference owner thread."""
        import mlx.core as mx

        # Arrays and model weights are device objects, not stream-bound.  The
        # execution defaults are thread-local, however, and generate_step's KV
        # cache evaluates on them.  Create both defaults here before the first
        # worker-side array operation.
        for stream in (mx.new_stream(mx.cpu), mx.new_stream(mx.gpu)):
            mx.set_default_stream(stream)

    def respond(self, waveform: np.ndarray, system: str | None = None,
                history: tuple | list = (),
                max_tokens: int | None = None
                ) -> tuple[str, dict[str, float | int]]:
        """DIRECT REPLY from audio embeddings — the same frozen Gemma that
        receives the adapter output ANSWERS it; understanding never passes
        through text. The template around the audio marker becomes
        system + history + audio-user-turn instead of the bare transcription
        wrapper. Experimental and MEASURED, never assumed: the adapter was
        distilled toward transcription. Reuses transcribe()'s generation
        path verbatim by swapping the cached prompt ids (single-owner-thread
        contract already enforced there)."""
        msgs = ([{"role": "system", "content": system}] if system else [])
        msgs += list(history)
        msgs += [{"role": "user", "content": AUDIO_MARKER}]
        rendered = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        if AUDIO_MARKER not in rendered:
            raise RuntimeError("chat template discarded the audio marker")
        prefix, suffix = rendered.split(AUDIO_MARKER, 1)
        saved = (self.prefix_ids, self.suffix_ids, self.max_tokens)
        try:
            self.prefix_ids = self.tokenizer.encode(
                prefix, add_special_tokens=False)
            # PREFILL the assistant turn with "HEARD:" — with history in
            # context the model imitates the history's plain-reply shape and
            # silently drops the two-line format (measured 2026-09-02: every
            # follow-up turn parsed as heard=""). Forcing the first tokens
            # pins the format at zero added latency.
            self.suffix_ids = (self.tokenizer.encode(
                suffix, add_special_tokens=False)
                + self.tokenizer.encode("HEARD:", add_special_tokens=False))
            if max_tokens:
                self.max_tokens = max_tokens
            raw, met = self.transcribe(waveform)
            if not raw.lstrip().upper().startswith(("HEARD", "REPLY")):
                raw = "HEARD:" + raw          # generation excludes the prefill
            return raw, met
        finally:
            (self.prefix_ids, self.suffix_ids, self.max_tokens) = saved

    def respond_text(self, text: str, system: str | None = None,
                     history: tuple | list = (),
                     max_tokens: int | None = None) -> str:
        """Text-only generation on the SAME model and history as the audio
        path — one brain, one memory across modalities. Closes the
        split-memory gap: typed turns used to run on a second Gemma with
        its own history."""
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_logits_processors

        if threading.get_ident() != self._owner_thread:
            raise RuntimeError(
                "MLX transcriber must execute on the thread that loaded it")
        msgs = ([{"role": "system", "content": system}] if system else [])
        msgs += list(history)
        msgs += [{"role": "user", "content": text}]
        rendered = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids_list = self.tokenizer.encode(rendered, add_special_tokens=False)
        ids = mx.array(ids_list, dtype=mx.int32)
        embeddings = self.model.language_model.model.embed_tokens(
            ids[None, :])[0]
        mx.eval(ids, embeddings)
        processors = make_logits_processors(
            logit_bias=({t: -1e9 for t in self.thinking_control_ids}
                        if self.suppress_thinking else None),
            repetition_penalty=(self.repetition_penalty
                                if self.repetition_penalty != 1.0 else None),
            repetition_context_size=self.repetition_context)
        generated = []
        for token, _ in generate_step(
                ids, self.model, input_embeddings=embeddings,
                max_tokens=max_tokens or self.max_tokens,
                logits_processors=processors):
            if token in EOS_IDS:
                break
            generated.append(token)
        return self.tokenizer.decode(
            generated, skip_special_tokens=True).strip()

    def transcribe(self, waveform: np.ndarray) -> tuple[str, dict[str, float | int]]:
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_logits_processors

        if threading.get_ident() != self._owner_thread:
            raise RuntimeError(
                "MLX transcriber must execute on the thread that loaded it"
            )
        started = time.perf_counter()
        cached = self.front_end(waveform)
        front_end_s = time.perf_counter() - started
        # Adapter output length may differ from feature length (the GemmaOmni
        # adapter pools internally); derive the placeholder count from what the
        # adapter actually produced. Identical for the bridge's 1:1 affine.
        audio = self.adapter(mx.array(cached)) / math.sqrt(OUT_DIM)
        n_audio = audio.shape[0]
        ids_list = self.prefix_ids + [0] * n_audio + self.suffix_ids
        ids = mx.array(ids_list, dtype=mx.int32)
        token_embeddings = self.model.language_model.model.embed_tokens(ids[None, :])[0]
        start = len(self.prefix_ids)
        embeddings = mx.concatenate([
            token_embeddings[:start],
            audio,
            token_embeddings[start + n_audio:],
        ], axis=0)
        mx.eval(ids, embeddings)
        prepared = time.perf_counter()

        generated = []
        logits_processors = make_logits_processors(
            logit_bias=(
                {token_id: -1e9 for token_id in self.thinking_control_ids}
                if self.suppress_thinking
                else None
            ),
            repetition_penalty=(
                self.repetition_penalty
                if self.repetition_penalty != 1.0 else None
            ),
            repetition_context_size=self.repetition_context,
        )
        if self.max_ngram_occurrences:
            from mlx_transcript_decoding import make_ngram_loop_blocker

            logits_processors.append(
                make_ngram_loop_blocker(
                    self.loop_ngram_size, self.max_ngram_occurrences
                )
            )
        if self.word_loop_ngram_size:
            from mlx_transcript_decoding import make_word_ngram_loop_blocker

            logits_processors.append(
                make_word_ngram_loop_blocker(
                    self.tokenizer,
                    self.word_loop_ngram_size,
                    self.word_max_ngram_occurrences,
                )
            )
        first_token_s = None
        generation_started = time.perf_counter()
        for token, _ in generate_step(
            ids,
            self.model,
            input_embeddings=embeddings,
            max_tokens=self.max_tokens,
            logits_processors=logits_processors,
        ):
            if first_token_s is None:
                first_token_s = time.perf_counter() - generation_started
            if token in EOS_IDS:
                break
            generated.append(token)
        finished = time.perf_counter()
        text = self.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()
        return text, {
            "audio_s": len(waveform) / SAMPLE_RATE,
            "audio_tokens": len(cached),
            "generated_tokens": len(generated),
            "front_end_s": front_end_s,
            "prepare_s": prepared - started,
            "ttft_s": first_token_s or 0.0,
            "generation_s": finished - generation_started,
            "total_s": finished - started,
        }


class SpeakerRegistry:
    """Online cosine clustering of speaker embeddings into conversational ids.

    Nearest centroid within `threshold` joins that speaker (centroid updated);
    otherwise a new id opens. Default threshold calibrated on ECAPA embeddings
    of real two-speaker room audio: within-cluster mean sim 0.51,
    between-centroid 0.17 -> 0.30 splits them with margin. Honest scope: no
    overlap handling, no cross-session identity.
    """

    def __init__(self, threshold: float = 0.30, merge_threshold: float = 0.45,
                 max_speakers: int = 8):
        self.threshold = threshold
        self.merge_threshold = merge_threshold
        self.max_speakers = max_speakers
        self.centroids: list[np.ndarray | None] = []   # None = merged away
        self.counts: list[int] = []
        self._alias: dict[int, int] = {}
        self.similarities: list[float] = []   # per-assignment, for calibration

    def canonical(self, sid: int) -> int:
        while sid in self._alias:
            sid = self._alias[sid]
        return sid

    def _live(self):
        return [(i, c) for i, c in enumerate(self.centroids) if c is not None]

    def assign(self, signature: np.ndarray) -> int:
        v = np.asarray(signature, dtype=np.float32).reshape(-1)
        v = v / (np.linalg.norm(v) + 1e-9)
        live = self._live()
        if live:
            sims = [float(v @ c) for _, c in live]
            k = int(np.argmax(sims))
            self.similarities.append(sims[k])
            best = live[k][0]
            if sims[k] >= self.threshold or (
                    len(live) >= self.max_speakers):
                n = self.counts[best]
                merged = (self.centroids[best] * n + v) / (n + 1)
                self.centroids[best] = merged / (np.linalg.norm(merged) + 1e-9)
                self.counts[best] += 1
                return best + 1
        self.centroids.append(v)
        self.counts.append(1)
        return len(self.centroids)

    def consolidate(self) -> None:
        """Fold fragments of the same voice together. Greedy online assignment
        over-splits early (single windows make noisy centroids — a real
        two-speaker room opened 8 ids); once centroids carry more evidence,
        pairs above merge_threshold are the same voice. Old ids stay valid via
        the alias table."""
        changed = True
        while changed:
            changed = False
            live = self._live()
            for a in range(len(live)):
                for b in range(a + 1, len(live)):
                    ia, ca = live[a]
                    ib, cb = live[b]
                    if float(ca @ cb) >= self.merge_threshold:
                        big, small = (ia, ib) if self.counts[ia] >= self.counts[ib] else (ib, ia)
                        n1, n2 = self.counts[big], self.counts[small]
                        merged = (self.centroids[big] * n1 +
                                  self.centroids[small] * n2) / (n1 + n2)
                        self.centroids[big] = merged / (np.linalg.norm(merged) + 1e-9)
                        self.counts[big] = n1 + n2
                        self.centroids[small] = None
                        self._alias[small + 1] = big + 1
                        changed = True
                        break
                if changed:
                    break


class SpeakerEmbedder:
    """The dedicated SPEAKER pipeline (Whisper answers WHAT; this answers WHO).

    ECAPA-TDNN (speechbrain, VoxCeleb) on CPU — deliberately off Metal so it
    never competes with Whisper/MPS or Gemma/MLX. Each sealed turn is embedded
    in 1.5 s sub-windows and every window is assigned a speaker id; the turn
    reports the majority id plus purity, so hard-split turns that mix two
    voices are attributed honestly instead of pretending to be one voice.
    """

    def __init__(self, win_s: float = 1.5, hop_s: float = 0.75,
                 min_rms: float = 0.005):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch = torch
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"})
        self.win = int(SAMPLE_RATE * win_s)
        self.hop = int(SAMPLE_RATE * hop_s)
        self.min_rms = min_rms

    def turn_speaker(self, waveform: np.ndarray,
                     registry: "SpeakerRegistry") -> tuple[int, float]:
        """Majority speaker id for a sealed turn + purity (agreeing fraction).
        Returns (0, 0.0) when no voiced window exists."""
        pcm = np.asarray(waveform, dtype=np.float32).reshape(-1)
        votes: list[int] = []
        for i in range(0, max(1, pcm.size - self.win), self.hop):
            seg = pcm[i:i + self.win]
            if seg.size < self.win // 2:
                continue
            if float(np.sqrt(np.mean(np.square(seg)))) < self.min_rms:
                continue
            with self.torch.inference_mode():
                emb = self.encoder.encode_batch(
                    self.torch.from_numpy(seg.copy()).unsqueeze(0)
                ).squeeze().numpy()
            votes.append(registry.assign(emb))
        if not votes:
            return 0, 0.0
        registry.consolidate()
        votes = [registry.canonical(v) for v in votes]
        ids, counts = np.unique(votes, return_counts=True)
        best = int(ids[np.argmax(counts)])
        return best, float(np.max(counts) / len(votes))


@dataclass(frozen=True)
class AudioPacket:
    sequence: int
    captured_at: float
    samples: np.ndarray


class RealtimeTurn:
    """No-loss packet queue plus rolling direct-neural transcription."""

    _END = object()

    def __init__(self, transcriber=None, *,
                 transcriber_factory: Callable[[], object] | None = None,
                 decode_every_s: float = 1.5, first_decode_s: float = 1.0,
                 stable_passes: int = 2,
                 on_commit: Callable[[int, str], None] | None = None,
                 on_provisional: Callable[[str], None] | None = None):
        if decode_every_s <= 0 or first_decode_s <= 0:
            raise ValueError("decode intervals must be positive")
        if (transcriber is None) == (transcriber_factory is None):
            raise ValueError(
                "provide exactly one of transcriber or transcriber_factory"
            )
        self.transcriber = transcriber
        self.transcriber_factory = transcriber_factory
        self.decode_every_s = decode_every_s
        self.first_decode_s = min(first_decode_s, decode_every_s)
        self.committer = StableCommitter(stable_passes=stable_passes)
        self.on_commit = on_commit or (lambda index, text: None)
        self.on_provisional = on_provisional or (lambda text: None)
        self._queue: queue.Queue = queue.Queue(maxsize=PACKET_QUEUE_CAPACITY)
        self._next_sequence = 0
        self._depth = 0
        self._high_water = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._audio = np.empty(SAMPLE_RATE * 2, dtype=np.float32)
        self._audio_size = 0
        self._last_decode_size = 0
        self.decode_timings: list[dict] = []
        self.final_raw = ""
        self.samples_published = 0
        self.samples_processed = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("turn already started")
        self._thread = threading.Thread(
            target=self._work, name="gemma-mlx-audio-input", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(120.0):
            raise TimeoutError("audio inference worker did not become ready")
        if self._error is not None:
            raise RuntimeError(
                f"audio inference worker failed during startup: "
                f"{type(self._error).__name__}: {self._error}"
            ) from self._error

    def push(self, samples, *, sequence: int | None = None,
             captured_at: float | None = None) -> int:
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not values.size:
            raise ValueError("empty PCM packet")
        values = np.ascontiguousarray(values).copy()
        values.setflags(write=False)
        with self._lock:
            actual = self._next_sequence if sequence is None else int(sequence)
            if actual != self._next_sequence:
                raise RuntimeError(
                    f"expected PCM packet {self._next_sequence}, got {actual}"
                )
            self._next_sequence += 1
            try:
                self._queue.put_nowait(AudioPacket(
                    actual,
                    time.monotonic() if captured_at is None else captured_at,
                    values,
                ))
            except queue.Full as exc:
                raise RuntimeError(
                    f"PCM queue reached its capacity of "
                    f"{self._queue.maxsize}; refusing to lose audio"
                ) from exc
            self._depth += 1
            self._high_water = max(self._high_water, self._depth)
            self.samples_published += len(values)
            return actual

    def _append(self, samples: np.ndarray) -> None:
        needed = self._audio_size + len(samples)
        if needed > SAMPLE_RATE * 29:
            raise RuntimeError(
                "speech turn exceeded 29 seconds; endpointing must split it "
                "before Whisper's context limit"
            )
        if needed > self._audio.size:
            grown = np.empty(max(needed, self._audio.size * 2), dtype=np.float32)
            grown[:self._audio_size] = self._audio[:self._audio_size]
            self._audio = grown
        self._audio[self._audio_size:needed] = samples
        self._audio_size = needed

    def _publish(self, raw: str, *, final: bool) -> None:
        newly = (self.committer.finish(raw) if final
                 else self.committer.update(raw, allow_tail=True))
        first = len(self.committer.committed) - len(newly)
        for offset, sentence in enumerate(newly):
            self.on_commit(first + offset, sentence)
        self.on_provisional("" if final else self.committer.provisional(raw))

    def _decode(self, *, final: bool) -> None:
        if not self._audio_size:
            return
        raw, timing = self.transcriber.transcribe(
            self._audio[:self._audio_size].copy()
        )
        timing = dict(timing)
        timing["final"] = final
        timing["queue_depth_at_decode"] = self._depth
        self.decode_timings.append(timing)
        self.final_raw = raw
        self._last_decode_size = self._audio_size
        self._publish(raw, final=final)

    def _work(self) -> None:
        try:
            if self.transcriber_factory is not None:
                # MLX streams and every array materialized by model loading now
                # belong to this inference thread.  Loading elsewhere and only
                # creating new worker streams leaves KV-cache state referring
                # to the original thread's stream id.
                self.transcriber = self.transcriber_factory()
            else:
                prepare_worker = getattr(self.transcriber, "prepare_worker", None)
                if prepare_worker is not None:
                    prepare_worker()
            self._ready.set()
            decoded_once = False
            while True:
                item = self._queue.get()
                with self._lock:
                    self._depth -= 1
                if item is self._END:
                    break
                self._append(item.samples)
                self.samples_processed += len(item.samples)
                threshold = self.decode_every_s if decoded_once else self.first_decode_s
                if self._audio_size - self._last_decode_size >= threshold * SAMPLE_RATE:
                    self._decode(final=False)
                    decoded_once = True
            # Always perform a final decode: a prior rolling result may have
            # ended before the last packet or may have reached max_tokens.
            self._decode(final=True)
        except BaseException as exc:
            self._error = exc
            self._ready.set()

    def finish(self, timeout: float = 120.0) -> dict:
        if self._thread is None:
            raise RuntimeError("turn was not started")
        with self._lock:
            try:
                self._queue.put_nowait(self._END)
            except queue.Full as exc:
                raise RuntimeError(
                    "cannot finish: PCM queue is at capacity"
                ) from exc
            self._depth += 1
            self._high_water = max(self._high_water, self._depth)
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("audio input worker did not finish")
        if self._error is not None:
            raise RuntimeError(
                f"audio input worker failed: {type(self._error).__name__}: "
                f"{self._error}"
            ) from self._error
        if self.samples_processed != self.samples_published:
            raise RuntimeError(
                f"lost PCM: processed {self.samples_processed}, "
                f"published {self.samples_published}"
            )
        return {
            "transcript": self.committer.text,
            "raw_final": self.final_raw,
            "commits": list(enumerate(self.committer.committed)),
            "alignment_failures": self.committer.alignment_failures,
            "decodes": len(self.decode_timings),
            "decode_timings": self.decode_timings,
            "packets": self._next_sequence,
            "samples_published": self.samples_published,
            "samples_processed": self.samples_processed,
            "samples_dropped": 0,
            "queue_high_watermark": self._high_water,
            "queue_capacity": self._queue.maxsize,
        }


@dataclass(frozen=True)
class AudioTurn:
    index: int
    samples: np.ndarray
    submitted_at: float
    hard_split: bool


class EndpointedSession:
    """Energy-endpoint continuous PCM into adapter-trained neural turns.

    Endpointing sees PCM only.  It never produces or transfers transcript text;
    each completed waveform still travels directly through Whisper embeddings,
    the adapter, and Gemma.  Model calls are serialized on the worker that owns
    every MLX array and stream while capture continues to enqueue packets.
    """

    _END = object()

    def __init__(self, transcriber_factory: Callable[[], object], *,
                 frame_ms: int = 30, rms_threshold: float,
                 silence_ms: int = 450, pre_roll_ms: int = 750,
                 min_turn_ms: int = 300, max_turn_s: float = 9.5,
                 speech_start_ms: int = 120,
                 speech_start_window_ms: int = 180,
                 start_max_spectral_flatness: float = 0.04,
                 vad_backend: str = "silero", webrtc_mode: int = 2,
                 silero_model: Path = DEFAULT_SILERO_VAD,
                 silero_threshold: float = 0.30,
                 silero_release_threshold: float = 0.125,
                 turn_wav_dir: Path | None = None,
                 on_commit: Callable[[int, str], None] | None = None,
                 speaker_attribution: bool = False,
                 speaker_threshold: float = 0.30,
                 speaker_merge_threshold: float = 0.45,
                 on_commit_tagged: Callable[[int, str, int], None] | None = None):
        if frame_ms <= 0 or SAMPLE_RATE * frame_ms % 1000:
            raise ValueError("frame_ms must map to an integer sample count")
        if vad_backend not in ("energy", "webrtc", "silero"):
            raise ValueError(f"unsupported VAD backend: {vad_backend}")
        if vad_backend == "energy" and rms_threshold <= 0:
            raise ValueError("rms_threshold must be positive for energy VAD")
        if vad_backend == "webrtc" and frame_ms not in (10, 20, 30):
            raise ValueError("WebRTC VAD requires 10, 20, or 30 ms packets")
        if not 0 <= webrtc_mode <= 3:
            raise ValueError("WebRTC VAD mode must be in 0..3")
        if not 0.0 < silero_threshold < 1.0:
            raise ValueError("Silero VAD threshold must be in (0, 1)")
        if not 0.0 < silero_release_threshold <= silero_threshold:
            raise ValueError(
                "Silero release threshold must be positive and no greater "
                "than its opening threshold"
            )
        self.transcriber_factory = transcriber_factory
        self.speaker_attribution = speaker_attribution
        self.speaker_error: str | None = None
        self.speakers = SpeakerRegistry(
            threshold=speaker_threshold,
            merge_threshold=speaker_merge_threshold,
        )
        self._compact_ids: dict[int, int] = {}
        self.on_commit_tagged = on_commit_tagged
        self.frame_ms = frame_ms
        self.frame_samples = SAMPLE_RATE * frame_ms // 1000
        self.rms_threshold = rms_threshold
        self.vad_backend = vad_backend
        self.webrtc_mode = webrtc_mode
        self._webrtc_vad = None
        self._silero_session = None
        self._silero_state = None
        self._silero_probability_sum = 0.0
        self._silero_probability_max = 0.0
        self._silero_probability_frames = 0
        if vad_backend == "webrtc":
            try:
                import webrtcvad
            except ImportError as exc:
                raise RuntimeError(
                    "--vad webrtc requires the webrtcvad package; use "
                    "--vad energy for calibrated RMS fallback"
                ) from exc
            self._webrtc_vad = webrtcvad.Vad(webrtc_mode)
        elif vad_backend == "silero":
            if not silero_model.exists():
                raise FileNotFoundError(
                    f"missing Silero VAD model {silero_model}; use --vad energy "
                    "for the dependency-free fallback"
                )
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self._silero_session = ort.InferenceSession(
                str(silero_model),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._silero_state = np.zeros((2, 1, 128), dtype=np.float32)
        self.silero_model = silero_model
        self.silero_threshold = silero_threshold
        self.silero_release_threshold = silero_release_threshold
        self.silence_frames = max(1, math.ceil(silence_ms / frame_ms))
        self.pre_roll_frames = max(1, math.ceil(pre_roll_ms / frame_ms))
        self.min_turn_ms = min_turn_ms
        self.max_turn_s = max_turn_s
        self.speech_start_frames = max(1, math.ceil(speech_start_ms / frame_ms))
        self.speech_start_window_frames = max(
            self.speech_start_frames,
            math.ceil(speech_start_window_ms / frame_ms),
        )
        if not 0.0 <= start_max_spectral_flatness <= 1.0:
            raise ValueError("start_max_spectral_flatness must be in [0, 1]")
        self.start_max_spectral_flatness = start_max_spectral_flatness
        self._spectral_window = np.hanning(self.frame_samples).astype(np.float32)
        self.min_turn_samples = math.ceil(SAMPLE_RATE * min_turn_ms / 1000)
        self.max_turn_samples = int(SAMPLE_RATE * max_turn_s)
        self.turn_wav_dir = turn_wav_dir
        if self.turn_wav_dir is not None:
            self.turn_wav_dir.mkdir(parents=True, exist_ok=True)
        self.on_commit = on_commit or (lambda index, text: None)

        from collections import deque
        # The back half of a confirmed silence belongs before the next turn.
        # Retain at least a full endpoint window so midpoint splitting cannot
        # discard that boundary before speech resumes.
        self._pre_roll = deque(
            maxlen=max(self.pre_roll_frames, self.silence_frames)
        )
        # Do not open a neural turn on one cable click, key tap, or noise spike.
        # Once confirmed, pre-roll restores the complete acoustic onset.
        self._speech_start_votes = deque(maxlen=self.speech_start_window_frames)
        self._turn_packets: list[np.ndarray] = []
        self._turn_samples = 0
        self._silent_frames = 0
        self._in_speech = False
        self._next_packet = 0
        self._next_turn = 0
        self._tasks: queue.Queue = queue.Queue(maxsize=TURN_QUEUE_CAPACITY)
        self._task_depth_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self.transcriber = None
        self.commits: list[tuple[int, str]] = []
        self.turns: list[dict] = []
        self.samples_published = 0
        self.samples_endpoint_processed = 0
        self.samples_modelled = 0
        self._task_depth = 0
        self._task_high_water = 0
        self._hard_splits = 0
        self._vad_speech_frames = 0
        self._vad_non_speech_frames = 0
        self._speech_starts_confirmed = 0
        self._start_quality_pass_frames = 0
        self._start_quality_rejected_frames = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("endpoint session already started")
        self._thread = threading.Thread(
            target=self._work,
            name="gemma-mlx-endpointed-input",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(120.0):
            raise TimeoutError("endpoint inference worker did not become ready")
        if self._error is not None:
            raise RuntimeError(
                f"endpoint worker failed during startup: "
                f"{type(self._error).__name__}: {self._error}"
            ) from self._error

    def _voiced(self, samples: np.ndarray) -> bool:
        if self._silero_session is not None:
            detector_samples = samples
            if len(detector_samples) < self.frame_samples:
                detector_samples = np.pad(
                    detector_samples,
                    (0, self.frame_samples - len(detector_samples)),
                )
            probability, self._silero_state = self._silero_session.run(
                None,
                {
                    "input": detector_samples[None].astype(
                        np.float32, copy=False
                    ),
                    "state": self._silero_state,
                    "sr": np.array(SAMPLE_RATE, dtype=np.int64),
                },
            )
            value = float(probability[0, 0])
            self._silero_probability_sum += value
            self._silero_probability_max = max(
                self._silero_probability_max, value
            )
            self._silero_probability_frames += 1
            if self._in_speech and self.rms_threshold > 0:
                # Silero is the high-precision admission gate. Once admitted,
                # measured energy retains the proven pause-midpoint behavior
                # and does not fragment weak consonants into tiny turns.
                voiced = (
                    float(np.sqrt(np.mean(np.square(samples))))
                    >= self.rms_threshold
                )
            else:
                threshold = (
                    self.silero_release_threshold
                    if self._in_speech else self.silero_threshold
                )
                voiced = value >= threshold
        elif self._webrtc_vad is None:
            voiced = (
                float(np.sqrt(np.mean(np.square(samples))))
                >= self.rms_threshold
            )
        else:
            # WebRTC accepts signed 16-bit mono PCM in exact 10/20/30 ms
            # frames. The final file packet may be shorter, so zero-pad only
            # the detector copy; the original sample sequence remains intact.
            detector_samples = samples
            if len(detector_samples) < self.frame_samples:
                detector_samples = np.pad(
                    detector_samples,
                    (0, self.frame_samples - len(detector_samples)),
                )
            pcm = np.round(
                np.clip(detector_samples, -1.0, 1.0) * 32767.0
            ).astype("<i2", copy=False)
            voiced = bool(
                self._webrtc_vad.is_speech(pcm.tobytes(), SAMPLE_RATE)
            )
        if voiced:
            self._vad_speech_frames += 1
        else:
            self._vad_non_speech_frames += 1
        return voiced

    def _speechlike_start(self, samples: np.ndarray, voiced: bool) -> bool:
        """Reject broadband impulses before opening a neural speech turn.

        This gate is deliberately start-only. Once speech is confirmed,
        endpointing continues to use the selected VAD so unvoiced consonants
        and natural intra-word gaps are retained.
        """
        if not voiced:
            return False
        if self.start_max_spectral_flatness == 0:
            self._start_quality_pass_frames += 1
            return True
        values = samples
        if len(values) < self.frame_samples:
            values = np.pad(values, (0, self.frame_samples - len(values)))
        centered = values - float(np.mean(values))
        power = np.square(np.abs(np.fft.rfft(centered * self._spectral_window)))
        # Exclude DC and the first two bins, which otherwise make hum/click
        # frames appear more speech-like than their usable band energy.
        usable = power[3:] + 1e-12
        flatness = float(np.exp(np.mean(np.log(usable))) / np.mean(usable))
        speechlike = flatness <= self.start_max_spectral_flatness
        if speechlike:
            self._start_quality_pass_frames += 1
        else:
            self._start_quality_rejected_frames += 1
        return speechlike

    def _seal(self, *, hard_split: bool) -> None:
        if not self._turn_packets:
            return
        carry = []
        kept = self._turn_packets
        if not hard_split and self._silent_frames:
            # Cut at the measured pause midpoint, not after the entire endpoint
            # guard.  JFK moved from 11.59% to 26.09% WER when 225 ms from the
            # next phrase was assigned to the preceding neural turn.
            carry_count = self._silent_frames // 2
            if carry_count:
                kept = self._turn_packets[:-carry_count]
                carry = self._turn_packets[-carry_count:]
        waveform = np.concatenate(kept)
        if waveform.size < self.min_turn_samples:
            # Preserve it as pre-roll: a short vocal onset may continue in the
            # next packet and must not be thrown away as a false trigger.
            for packet in self._turn_packets[-self.pre_roll_frames:]:
                self._pre_roll.append(packet)
        else:
            waveform = np.ascontiguousarray(waveform)
            waveform.setflags(write=False)
            task = AudioTurn(
                self._next_turn, waveform, time.monotonic(), hard_split
            )
            with self._task_depth_lock:
                try:
                    self._tasks.put_nowait(task)
                except queue.Full as exc:
                    raise RuntimeError(
                        f"neural turn queue reached its capacity of "
                        f"{self._tasks.maxsize}; refusing to lose audio"
                    ) from exc
                self._next_turn += 1
                self._task_depth += 1
                self._task_high_water = max(
                    self._task_high_water, self._task_depth
                )
            if hard_split:
                self._hard_splits += 1
        self._turn_packets = []
        self._turn_samples = 0
        self._silent_frames = 0
        self._in_speech = False
        self._speech_start_votes.clear()
        for packet in carry:
            self._pre_roll.append(packet)

    def push(self, samples, *, sequence: int | None = None,
             captured_at: float | None = None) -> int:
        del captured_at
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not values.size or values.size > self.frame_samples:
            raise ValueError(
                f"expected 1..{self.frame_samples} mono samples, got {values.size}"
            )
        values = np.ascontiguousarray(values).copy()
        values.setflags(write=False)
        actual = self._next_packet if sequence is None else int(sequence)
        if actual != self._next_packet:
            raise RuntimeError(
                f"expected PCM packet {self._next_packet}, got {actual}"
            )
        self._next_packet += 1
        self.samples_published += len(values)
        self.samples_endpoint_processed += len(values)
        voiced = self._voiced(values)

        if not self._in_speech:
            self._pre_roll.append(values)
            self._speech_start_votes.append(
                self._speechlike_start(values, voiced)
            )
            if sum(self._speech_start_votes) < self.speech_start_frames:
                return actual
            self._in_speech = True
            self._speech_starts_confirmed += 1
            self._turn_packets = list(self._pre_roll)
            self._turn_samples = sum(len(item) for item in self._turn_packets)
            self._pre_roll.clear()
            self._speech_start_votes.clear()
            self._silent_frames = 0
            return actual

        self._turn_packets.append(values)
        self._turn_samples += len(values)
        self._silent_frames = 0 if voiced else self._silent_frames + 1
        if self._silent_frames >= self.silence_frames:
            self._seal(hard_split=False)
        elif self._turn_samples >= self.max_turn_samples:
            self._seal(hard_split=True)
        return actual

    def _work(self) -> None:
        try:
            self.transcriber = self.transcriber_factory()
            self._embedder = None
            if self.speaker_attribution:
                try:
                    self._embedder = SpeakerEmbedder()
                except Exception as exc:              # missing dep -> untagged
                    self.speaker_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(f"speaker pipeline unavailable ({exc}); "
                          "turns will carry speaker=0", flush=True)
            self._ready.set()
            while True:
                task = self._tasks.get()
                with self._task_depth_lock:
                    self._task_depth -= 1
                if task is self._END:
                    break
                started = time.monotonic()
                turn_wav = None
                if self.turn_wav_dir is not None:
                    import soundfile as soundfile

                    turn_wav = self.turn_wav_dir / f"turn_{task.index:04d}.wav"
                    soundfile.write(
                        str(turn_wav), task.samples, SAMPLE_RATE, subtype="PCM_16"
                    )
                text, timing = self.transcriber.transcribe(task.samples)
                clean = _compact(text)
                self.samples_modelled += len(task.samples)
                # WHO said it: the dedicated speaker pipeline (ECAPA on CPU).
                # Whisper-encoder means were tried first and measured useless
                # here (one cluster at 0.97+ similarity) — ASR training erases
                # voice identity, and hard-split turns mix speakers anyway.
                speaker, purity = 0, 0.0
                if self._embedder is not None:
                    canon, purity = self._embedder.turn_speaker(
                        task.samples, self.speakers)
                    canon = self.speakers.canonical(canon)
                    if canon and canon not in self._compact_ids:
                        self._compact_ids[canon] = len(self._compact_ids) + 1
                    speaker = self._compact_ids.get(canon, 0)
                self.commits.append((task.index, clean))
                self.turns.append({
                    "index": task.index,
                    "audio_s": len(task.samples) / SAMPLE_RATE,
                    "hard_split": task.hard_split,
                    "turn_wav": str(turn_wav.resolve()) if turn_wav else None,
                    "queue_wait_s": started - task.submitted_at,
                    "completion_s": time.monotonic() - task.submitted_at,
                    "text": clean,
                    "speaker": speaker,
                    "speaker_purity": round(purity, 3),
                    "timing": timing,
                })
                self.on_commit(task.index, clean)
                if self.on_commit_tagged is not None:
                    self.on_commit_tagged(task.index, clean, speaker)
        except BaseException as exc:
            self._error = exc
            self._ready.set()

    def finish(self, timeout: float = 180.0) -> dict:
        if self._thread is None:
            raise RuntimeError("endpoint session was not started")
        if self._in_speech:
            self._seal(hard_split=False)
        with self._task_depth_lock:
            try:
                self._tasks.put_nowait(self._END)
            except queue.Full as exc:
                raise RuntimeError(
                    "cannot finish: neural turn queue is at capacity"
                ) from exc
            self._task_depth += 1
            self._task_high_water = max(
                self._task_high_water, self._task_depth
            )
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("endpoint inference worker did not finish")
        if self._error is not None:
            raise RuntimeError(
                f"endpoint inference worker failed: "
                f"{type(self._error).__name__}: {self._error}"
            ) from self._error
        if self.samples_endpoint_processed != self.samples_published:
            raise RuntimeError("endpointing did not inspect every captured sample")
        # Turns carry compact ids assigned at commit time (stable for live
        # consumers). Late merges may have folded two canonicals together —
        # re-map through the alias table and renumber once more for output.
        canon_of_compact = {v: self.speakers.canonical(k)
                            for k, v in self._compact_ids.items()}
        final_ids: dict[int, int] = {}
        for t in self.turns:
            canon = canon_of_compact.get(t.get("speaker", 0), 0)
            if canon and canon not in final_ids:
                final_ids[canon] = len(final_ids) + 1
            t["speaker"] = final_ids.get(canon, 0)
        compact_ids = final_ids
        return {
            "transcript": " ".join(text for _, text in self.commits),
            "tagged_transcript": (
                " ".join(
                    f"S{t['speaker']}: {t['text']}" for t in self.turns
                    if t.get("text")
                )
                if self._embedder is not None else None
            ),
            "speaker_count": len(compact_ids),
            "speaker_attribution_requested": self.speaker_attribution,
            "speaker_attribution_available": self._embedder is not None,
            "speaker_error": self.speaker_error,
            "speaker_similarities": [round(x, 4)
                                     for x in self.speakers.similarities],
            "commits": self.commits,
            "turns": self.turns,
            "packets": self._next_packet,
            "samples_published": self.samples_published,
            "samples_endpoint_processed": self.samples_endpoint_processed,
            "samples_modelled": self.samples_modelled,
            "samples_dropped": 0,
            "non_speech_samples_not_modelled": (
                self.samples_published - self.samples_modelled
            ),
            "turn_queue_high_watermark": self._task_high_water,
            "turn_queue_capacity": self._tasks.maxsize,
            "hard_splits": self._hard_splits,
            "vad_backend": self.vad_backend,
            "vad_mode": (
                self.webrtc_mode if self.vad_backend == "webrtc" else None
            ),
            "silero_model": (
                str(self.silero_model.resolve())
                if self.vad_backend == "silero" else None
            ),
            "silero_threshold": (
                self.silero_threshold if self.vad_backend == "silero" else None
            ),
            "silero_release_threshold": (
                self.silero_release_threshold
                if self.vad_backend == "silero" else None
            ),
            "silero_energy_release": (
                self.vad_backend == "silero" and self.rms_threshold > 0
            ),
            "silero_probability_mean": (
                self._silero_probability_sum / self._silero_probability_frames
                if self._silero_probability_frames else None
            ),
            "silero_probability_max": (
                self._silero_probability_max
                if self._silero_probability_frames else None
            ),
            "vad_speech_frames": self._vad_speech_frames,
            "vad_non_speech_frames": self._vad_non_speech_frames,
            "speech_start_frames": self.speech_start_frames,
            "speech_start_window_frames": self.speech_start_window_frames,
            "speech_starts_confirmed": self._speech_starts_confirmed,
            "start_max_spectral_flatness": self.start_max_spectral_flatness,
            "start_quality_pass_frames": self._start_quality_pass_frames,
            "start_quality_rejected_frames": self._start_quality_rejected_frames,
            "endpoint_rms": self.rms_threshold,
            "endpoint_silence_ms": self.silence_frames * self.frame_ms,
            "packet_ms": self.frame_ms,
            "pre_roll_ms": self.pre_roll_frames * self.frame_ms,
            "min_turn_ms": self.min_turn_ms,
            "max_turn_s": self.max_turn_s,
            "turn_wav_dir": (
                str(self.turn_wav_dir.resolve()) if self.turn_wav_dir else None
            ),
        }


class LivePcmCapture:
    """PortAudio callback that only copies sequenced PCM into a queue."""

    def __init__(self, *, frame_ms: int = 30, device=None):
        self.frame_ms = frame_ms
        self.frame_samples = SAMPLE_RATE * frame_ms // 1000
        self.device = device
        self._queue: queue.Queue = queue.Queue(maxsize=PACKET_QUEUE_CAPACITY)
        self._stream = None
        self._next_sequence = 0
        self.samples_captured = 0
        self.queue_high_watermark = 0
        self.status_events: list[str] = []
        self.callback_error: BaseException | None = None

    def _callback(self, indata, frames, time_info, status) -> None:
        del time_info
        try:
            if frames != self.frame_samples:
                raise RuntimeError(
                    f"device delivered {frames} samples, expected "
                    f"{self.frame_samples}"
                )
            if status:
                # Still retain the delivered samples, but make the run fail
                # visibly: a PortAudio overflow/underflow means continuity is
                # no longer proven even when this particular block exists.
                self.status_events.append(str(status))
            samples = np.ascontiguousarray(indata[:, 0], dtype=np.float32).copy()
            samples.setflags(write=False)
            packet = AudioPacket(
                self._next_sequence, time.monotonic(), samples
            )
            self._next_sequence += 1
            self.samples_captured += len(samples)
            self._queue.put_nowait(packet)
            self.queue_high_watermark = max(
                self.queue_high_watermark, self._queue.qsize()
            )
        except BaseException as exc:
            self.callback_error = exc

    def start(self) -> None:
        import sounddevice as sounddevice

        self._stream = sounddevice.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            latency="low",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def read(self, timeout: float = 0.5) -> AudioPacket:
        if self.callback_error is not None:
            raise RuntimeError(
                f"microphone callback failed: {type(self.callback_error).__name__}: "
                f"{self.callback_error}"
            ) from self.callback_error
        return self._queue.get(timeout=timeout)

    def drain(self) -> list[AudioPacket]:
        packets = []
        while True:
            try:
                packets.append(self._queue.get_nowait())
            except queue.Empty:
                return packets

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self.callback_error is not None:
            raise RuntimeError(
                f"microphone callback failed: {type(self.callback_error).__name__}: "
                f"{self.callback_error}"
            ) from self.callback_error
        if self.status_events:
            raise RuntimeError(
                "PortAudio reported input discontinuity: "
                + "; ".join(self.status_events)
            )

    def health(self) -> dict:
        return {
            "samples_captured": self.samples_captured,
            "packets_captured": self._next_sequence,
            "queue_high_watermark": self.queue_high_watermark,
            "queue_capacity": self._queue.maxsize,
            "status_events": self.status_events,
            "callback_error": (
                None if self.callback_error is None
                else f"{type(self.callback_error).__name__}: {self.callback_error}"
            ),
        }


def calibrate_microphone(*, frame_ms: int = 30, seconds: float = 1.5,
                         device=None) -> tuple[float, dict]:
    """Measure a quiet room and return a bounded speech-admission threshold."""
    capture = LivePcmCapture(frame_ms=frame_ms, device=device)
    target = max(1, math.ceil(seconds * 1000 / frame_ms))
    levels = []
    capture.start()
    try:
        while len(levels) < target:
            packet = capture.read(timeout=max(1.0, seconds))
            levels.append(float(np.sqrt(np.mean(np.square(packet.samples)))))
    finally:
        capture.close()
    floor = float(np.median(levels)) if levels else 0.0
    threshold = min(max(floor * 2.0, 0.004), 0.10)
    return threshold, {
        "seconds": len(levels) * frame_ms / 1000,
        "frames": len(levels),
        "median_rms": floor,
        "threshold_rms": threshold,
        "microphone": capture.health(),
    }


def _read_audio(path: Path) -> np.ndarray:
    import soundfile as soundfile

    waveform, rate = soundfile.read(path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly
        divisor = math.gcd(rate, SAMPLE_RATE)
        waveform = resample_poly(
            waveform, SAMPLE_RATE // divisor, rate // divisor
        ).astype(np.float32)
    return np.asarray(waveform, dtype=np.float32)


def _self_test() -> None:
    assert split_sentences("Ask Dr. Smith. Then leave.") == [
        "Ask Dr. Smith.", "Then leave."
    ]
    assert split_sentences("The value is 3.5. Correct?") == [
        "The value is 3.5.", "Correct?"
    ]
    committer = StableCommitter(stable_passes=2)
    assert committer.update("We choose a moon") == []
    assert committer.update("We choose the Moon.") == []
    assert committer.update("We choose the Moon. Not because") == [
        "We choose the Moon."
    ]
    assert committer.update("We choose the Moon. Not because it is") == []
    final = "We choose the Moon. Not because it is easy."
    assert committer.finish(final) == ["Not because it is easy."]
    assert committer.finish(final) == []
    assert committer.text == final
    # A revision of immutable words is withheld, never re-emitted.
    assert committer.update("We chose Mars. Not because it is easy.") == []
    assert committer.alignment_failures == 1

    class FakeTranscriber:
        calls = 0

        def transcribe(self, waveform):
            self.calls += 1
            text = "Alpha." if self.calls == 1 else "Alpha. Beta."
            return text, {
                "audio_s": len(waveform) / SAMPLE_RATE,
                "audio_tokens": 1,
                "generated_tokens": 2,
                "front_end_s": 0.0,
                "prepare_s": 0.0,
                "ttft_s": 0.0,
                "generation_s": 0.0,
                "total_s": 0.0,
            }

    emitted = []
    turn = RealtimeTurn(
        FakeTranscriber(),
        first_decode_s=0.03,
        decode_every_s=0.03,
        on_commit=lambda index, text: emitted.append((index, text)),
    )
    turn.start()
    turn.push(np.zeros(480, np.float32), sequence=0)
    turn.push(np.zeros(480, np.float32), sequence=1)
    result = turn.finish()
    assert emitted == [(0, "Alpha."), (1, "Beta.")], emitted
    assert result["transcript"] == "Alpha. Beta."
    assert result["samples_published"] == result["samples_processed"] == 960
    assert result["samples_dropped"] == 0

    endpoint_commits = []
    endpoint = EndpointedSession(
        lambda: FakeTranscriber(),
        rms_threshold=0.1,
        silence_ms=60,
        pre_roll_ms=30,
        min_turn_ms=30,
        speech_start_ms=30,
        speech_start_window_ms=30,
        start_max_spectral_flatness=0,
        vad_backend="energy",
        on_commit=lambda index, text: endpoint_commits.append((index, text)),
    )
    endpoint.start()
    packets = [
        np.full(480, 0.2, np.float32),
        np.full(480, 0.2, np.float32),
        np.full(480, 0.2, np.float32),
        np.zeros(480, np.float32),
        np.zeros(480, np.float32),
    ]
    for index, packet in enumerate(packets):
        endpoint.push(packet, sequence=index)
    endpoint_result = endpoint.finish()
    assert endpoint_commits == [(0, "Alpha.")], endpoint_commits
    assert endpoint_result["samples_published"] == 5 * 480
    assert endpoint_result["samples_endpoint_processed"] == 5 * 480
    assert endpoint_result["samples_dropped"] == 0
    assert endpoint_result["hard_splits"] == 0
    assert endpoint_result["vad_backend"] == "energy"
    assert endpoint_result["vad_speech_frames"] == 3
    assert endpoint_result["vad_non_speech_frames"] == 2
    assert endpoint_result["speech_starts_confirmed"] == 1

    # Isolated high-energy transients are inspected but never reach the neural
    # graph. Four voiced frames within the default six-frame window are needed
    # to open a turn, and pre-roll restores the onset after confirmation.
    noise_commits = []
    noise = EndpointedSession(
        lambda: FakeTranscriber(),
        rms_threshold=0.1,
        silence_ms=60,
        pre_roll_ms=180,
        min_turn_ms=30,
        start_max_spectral_flatness=0,
        vad_backend="energy",
        on_commit=lambda index, text: noise_commits.append((index, text)),
    )
    noise.start()
    for index in range(10):
        amplitude = 0.2 if index in (0, 3, 7) else 0.0
        noise.push(np.full(480, amplitude, np.float32), sequence=index)
    noise_result = noise.finish()
    assert noise_commits == []
    assert noise_result["speech_starts_confirmed"] == 0
    assert noise_result["samples_modelled"] == 0
    assert noise_result["samples_endpoint_processed"] == 10 * 480

    capture = LivePcmCapture(frame_ms=30)
    capture._callback(
        np.zeros((480, 1), np.float32), 480, None, False
    )
    packet = capture.read(timeout=0.1)
    assert packet.sequence == 0 and len(packet.samples) == 480
    assert capture.health()["samples_captured"] == 480
    print("mlx_audio_streaming: commits, packet order, and no-loss queue checks passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--audio", type=Path)
    source.add_argument("--microphone", action="store_true")
    reference_source = parser.add_mutually_exclusive_group()
    reference_source.add_argument("--reference", type=Path)
    reference_source.add_argument(
        "--reference-text",
        help="literal evaluation transcript; scoring only, never model input",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--audio-gguf", type=Path, default=DEFAULT_AUDIO_GGUF)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--packet-ms", type=int, default=30)
    parser.add_argument("--decode-every", type=float, default=1.5)
    parser.add_argument("--first-decode", type=float, default=1.0)
    parser.add_argument("--stable-passes", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--repetition-context", type=int, default=20)
    parser.add_argument(
        "--max-ngram-occurrences",
        type=int,
        default=1,
        help="block a token n-gram after this many occurrences; 0 disables",
    )
    parser.add_argument("--loop-ngram-size", type=int, default=4)
    parser.add_argument(
        "--word-loop-ngram-size",
        type=int,
        default=4,
        help="veto repeated normalized word n-grams at logits; 0 disables",
    )
    parser.add_argument("--word-max-ngram-occurrences", type=int, default=1)
    parser.add_argument(
        "--allow-thinking",
        action="store_true",
        help="allow Gemma's thinking token; disabled by default for transcription",
    )
    parser.add_argument(
        "--endpoint-rms",
        type=float,
        default=0.0,
        help="positive RMS threshold enables pause-delimited continuous turns",
    )
    parser.add_argument(
        "--vad",
        choices=("silero", "energy", "webrtc"),
        default="silero",
        help=(
            "speech admission backend; all detectors run on CPU and produce "
            "no transcript"
        ),
    )
    parser.add_argument(
        "--vad-mode",
        type=int,
        choices=range(4),
        default=2,
        help="WebRTC aggressiveness: 0 least aggressive, 3 most aggressive",
    )
    parser.add_argument(
        "--silero-model", type=Path, default=DEFAULT_SILERO_VAD
    )
    parser.add_argument("--silero-threshold", type=float, default=0.30)
    parser.add_argument(
        "--silero-release-threshold", type=float, default=0.125
    )
    parser.add_argument("--endpoint-silence-ms", type=int, default=450)
    parser.add_argument("--pre-roll-ms", type=int, default=750)
    parser.add_argument(
        "--speech-start-ms",
        type=int,
        default=120,
        help="voiced audio required before opening a neural turn",
    )
    parser.add_argument(
        "--speech-start-window-ms",
        type=int,
        default=180,
        help="window containing the required voiced start frames",
    )
    parser.add_argument(
        "--start-max-spectral-flatness",
        type=float,
        default=0.04,
        help="start-only broadband-noise rejection; 0 disables",
    )
    parser.add_argument("--max-turn-s", type=float, default=9.5)
    parser.add_argument(
        "--speaker-attribution",
        action="store_true",
        help=(
            "opt in to post-transcription CPU speaker clustering; this is "
            "not part of the direct audio-to-Gemma path"
        ),
    )
    parser.add_argument("--speaker-threshold", type=float, default=0.30)
    parser.add_argument("--speaker-merge-threshold", type=float, default=0.45)
    parser.add_argument("--mic-seconds", type=float, default=0.0)
    parser.add_argument("--input-device", default=None)
    parser.add_argument(
        "--capture-wav",
        type=Path,
        help="write the exact post-device microphone PCM used by endpointing",
    )
    parser.add_argument(
        "--turn-wav-dir",
        type=Path,
        help="audit-only directory for the exact bounded PCM sent to each neural turn",
    )
    parser.add_argument("--calibration-s", type=float, default=1.5)
    parser.add_argument("--paced", action="store_true")
    parser.add_argument(
        "--replay-config",
        type=Path,
        help=(
            "for --audio, restore endpoint/VAD settings from a prior result "
            "JSON (especially its live microphone calibration)"
        ),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.replay_config is not None:
        if args.audio is None:
            parser.error("--replay-config requires --audio")
        replay = json.loads(args.replay_config.read_text())
        scalar_settings = {
            "endpoint_rms": "endpoint_rms",
            "endpoint_silence_ms": "endpoint_silence_ms",
            "pre_roll_ms": "pre_roll_ms",
            "max_turn_s": "max_turn_s",
            "start_max_spectral_flatness": "start_max_spectral_flatness",
            "silero_threshold": "silero_threshold",
            "silero_release_threshold": "silero_release_threshold",
        }
        for report_name, argument_name in scalar_settings.items():
            if replay.get(report_name) is not None:
                setattr(args, argument_name, replay[report_name])
        packet_ms = int(replay.get("packet_ms", args.packet_ms))
        if replay.get("speech_start_frames") is not None:
            args.speech_start_ms = (
                int(replay["speech_start_frames"]) * packet_ms
            )
        if replay.get("speech_start_window_frames") is not None:
            args.speech_start_window_ms = (
                int(replay["speech_start_window_frames"]) * packet_ms
            )

    if args.self_test:
        _self_test()
        if args.audio is None:
            return 0
    if args.audio is None and not args.microphone:
        parser.error(
            "--audio or --microphone is required unless only --self-test is requested"
        )
    if args.packet_ms <= 0:
        parser.error("--packet-ms must be positive")

    print("loading direct Whisper -> adapter -> Gemma MLX pipeline", flush=True)
    waveform = _read_audio(args.audio) if args.audio is not None else None
    terminal: list[tuple[int, str]] = []

    def commit(index: int, text: str) -> None:
        terminal.append((index, text))
        print(f"COMMIT[{index:04d}] {text}", flush=True)

    def provisional(text: str) -> None:
        if text:
            print(f"LIVE {text}", flush=True)

    def factory():
        return MlxGemmaAudioTranscriber(
            args.model,
            args.adapter,
            args.audio_gguf,
            device=args.device,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            repetition_context=args.repetition_context,
            suppress_thinking=not args.allow_thinking,
            loop_ngram_size=args.loop_ngram_size,
            max_ngram_occurrences=args.max_ngram_occurrences,
            word_loop_ngram_size=args.word_loop_ngram_size,
            word_max_ngram_occurrences=args.word_max_ngram_occurrences,
        )
    calibration = None
    endpoint_rms = args.endpoint_rms
    if (
        args.microphone
        and endpoint_rms <= 0
        and args.vad in ("energy", "silero")
    ):
        print(
            f"keep quiet for {args.calibration_s:.1f}s microphone calibration",
            flush=True,
        )
        endpoint_rms, calibration = calibrate_microphone(
            frame_ms=args.packet_ms,
            seconds=args.calibration_s,
            device=args.input_device,
        )
        print(
            f"room RMS {calibration['median_rms']:.5f}; "
            f"speech threshold {endpoint_rms:.5f}",
            flush=True,
        )
    if args.microphone or endpoint_rms > 0 or args.vad in ("webrtc", "silero"):
        session = EndpointedSession(
            factory,
            frame_ms=args.packet_ms,
            rms_threshold=endpoint_rms,
            silence_ms=args.endpoint_silence_ms,
            pre_roll_ms=args.pre_roll_ms,
            speech_start_ms=args.speech_start_ms,
            speech_start_window_ms=args.speech_start_window_ms,
            start_max_spectral_flatness=args.start_max_spectral_flatness,
            max_turn_s=args.max_turn_s,
            vad_backend=args.vad,
            webrtc_mode=args.vad_mode,
            silero_model=args.silero_model,
            silero_threshold=args.silero_threshold,
            silero_release_threshold=args.silero_release_threshold,
            turn_wav_dir=args.turn_wav_dir,
            on_commit=commit,
            speaker_attribution=args.speaker_attribution,
            speaker_threshold=args.speaker_threshold,
            speaker_merge_threshold=args.speaker_merge_threshold,
        )
        schema = "gemma-mlx-direct-endpointed-v1"
    else:
        session = RealtimeTurn(
            transcriber_factory=factory,
            decode_every_s=args.decode_every,
            first_decode_s=args.first_decode,
            stable_passes=args.stable_passes,
            on_commit=commit,
            on_provisional=provisional,
        )
        schema = "gemma-mlx-direct-streaming-v1"
    session.start()
    transcriber = session.transcriber
    packet_samples = max(1, round(SAMPLE_RATE * args.packet_ms / 1000))
    capture_started = time.monotonic()
    microphone_health = None
    if args.microphone:
        capture = LivePcmCapture(
            frame_ms=args.packet_ms, device=args.input_device
        )
        capture_writer = None
        if args.capture_wav is not None:
            import soundfile as soundfile

            args.capture_wav.parent.mkdir(parents=True, exist_ok=True)
            capture_writer = soundfile.SoundFile(
                str(args.capture_wav),
                mode="w",
                samplerate=SAMPLE_RATE,
                channels=1,
                subtype="PCM_16",
            )

        def publish_microphone_packet(packet: AudioPacket) -> None:
            if capture_writer is not None:
                capture_writer.write(packet.samples)
            session.push(
                packet.samples,
                sequence=packet.sequence,
                captured_at=packet.captured_at,
            )

        print("microphone open; Ctrl-C ends the session", flush=True)
        try:
            capture.start()
            deadline = (
                capture_started + args.mic_seconds
                if args.mic_seconds > 0 else None
            )
            try:
                while deadline is None or time.monotonic() < deadline:
                    try:
                        packet = capture.read(timeout=0.5)
                    except queue.Empty:
                        continue
                    publish_microphone_packet(packet)
            except KeyboardInterrupt:
                print("microphone stop requested", flush=True)
            finally:
                capture.close()
            # stop() may leave callbacks already queued; drain them in sequence
            # so closing the device cannot make its last packets disappear.
            for packet in capture.drain():
                publish_microphone_packet(packet)
        finally:
            if capture_writer is not None:
                capture_writer.close()
        microphone_health = capture.health()
        source_samples = capture.samples_captured
        source_name = "microphone"
        paced = True
    else:
        assert waveform is not None
        for sequence, offset in enumerate(
                range(0, len(waveform), packet_samples)):
            session.push(
                waveform[offset:offset + packet_samples],
                sequence=sequence,
                captured_at=capture_started + offset / SAMPLE_RATE,
            )
            if args.paced:
                deadline = (
                    capture_started
                    + min(len(waveform), offset + packet_samples) / SAMPLE_RATE
                )
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
        source_samples = len(waveform)
        source_name = str(args.audio.resolve())
        paced = args.paced
    capture_s = time.monotonic() - capture_started
    drain_started = time.monotonic()
    report = session.finish(
        timeout=max(120.0, source_samples / SAMPLE_RATE * 6)
    )
    report.update({
        "schema": schema,
        "architecture": (
            "PCM->MiniCPM_Whisper->linear1/ReLU/AvgPool5"
            "->trained_adapter->Gemma4_MLX->text"
        ),
        "transcript_runtime_handoff": False,
        "model": str(transcriber.model_path),
        "adapter": str(transcriber.adapter_path),
        "adapter_sha256": transcriber.adapter_sha256,
        "audio_gguf": str(transcriber.audio_gguf_path),
        "audio": source_name,
        "audio_s": source_samples / SAMPLE_RATE,
        "model_load_s": transcriber.load_s,
        "repetition_penalty": args.repetition_penalty,
        "loop_ngram_size": args.loop_ngram_size,
        "max_ngram_occurrences": args.max_ngram_occurrences,
        "word_loop_ngram_size": args.word_loop_ngram_size,
        "word_max_ngram_occurrences": args.word_max_ngram_occurrences,
        "capture_wall_s": capture_s,
        "drain_s": time.monotonic() - drain_started,
        "paced": paced,
        "microphone": microphone_health,
        "capture_wav": (
            str(args.capture_wav.resolve()) if args.capture_wav else None
        ),
        "replay_config": (
            str(args.replay_config.resolve()) if args.replay_config else None
        ),
        "calibration": calibration,
        "terminal_matches_ledger": (
            " ".join(text for _, text in terminal) == report["transcript"]
        ),
        "commit_indices_contiguous": (
            [index for index, _ in terminal] == list(range(len(terminal)))
        ),
    })
    if args.reference or args.reference_text:
        reference = _compact(
            args.reference.read_text() if args.reference else args.reference_text
        )
        edits = _word_distance(_words(reference), _words(report["transcript"]))
        report["reference"] = reference
        report["word_edits"] = edits
        report["reference_words"] = len(_words(reference))
        report["wer"] = edits / max(1, len(_words(reference)))
    print("FINAL " + report["transcript"], flush=True)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print("RESULT " + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
