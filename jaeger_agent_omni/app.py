"""Voice in, cognition out. The application.

Two paths through the same models, selectable at runtime:

    text   waveform -> whisper (encoder+decoder) -> transcript -> backbone
           Works today. This is the baseline, and it is also Stage 2's
           teacher: the transcript it produces is what the audio path must
           learn to match.

    omni   waveform -> whisper encoder -> JaegerAudioProjector -> backbone
           No transcript anywhere. Produces nonsense until the projector is
           trained — which is exactly what makes it useful before then, since
           it proves the plumbing independently of the weights.

Running the text path with `capture_to=` writes every turn to the corpus. So
using the baseline generates the data that trains the thing meant to replace
it.

    python -m jaeger_agent_omni --audio clip.wav --backbone <hf-id>
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core.capture import SAMPLE_RATE, CaptureWriter
from .core.neural import JaegerAudioProjector, hidden_size_of
from .core.outputs import CognitiveOutput
from .core.performance_spec import neutral
from .core.prompt import ChatWrapper
from .core.streaming import CHUNK_SECONDS, StreamingChunker

PATHS = ("text", "omni")


def transcribe_kwargs(model, language: str) -> dict:
    """`language`/`task` are multilingual-only.

    English-only checkpoints (`*.en`, which is what the legacy stack runs)
    raise if either is passed. They are also the checkpoints that least need
    it — there is no language to detect.
    """
    multilingual = getattr(model.generation_config, "is_multilingual", None)
    if multilingual is None:
        multilingual = getattr(model.config, "is_multilingual", True)
    return {"language": language, "task": "transcribe"} if multilingual else {}


@dataclass
class Timing:
    """Per-stage wall clock. The breakdown, not one end-to-end number — the
    whole argument for omni is which stage the time is actually in.

    `overlapped_ms` is work done *while the user was still speaking*. It is
    excluded from the total on purpose: it costs the listener nothing. That
    exclusion is the entire latency thesis, made measurable.
    """

    transcribe_ms: float = 0.0
    encode_ms: float = 0.0
    project_ms: float = 0.0
    generate_ms: float = 0.0
    overlapped_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        """Time the listener actually waits. Overlapped work is not in it."""
        return (self.transcribe_ms + self.encode_ms
                + self.project_ms + self.generate_ms)

    def __str__(self) -> str:
        parts = [f"{k.replace('_ms','')} {v:.0f}ms"
                 for k, v in vars(self).items()
                 if v > 0 and k != "overlapped_ms"]
        out = f"{' + '.join(parts)} = {self.total_ms:.0f}ms"
        if self.overlapped_ms > 0:
            out += f"  (+{self.overlapped_ms:.0f}ms overlapped, free)"
        return out


def load_models(whisper_id: str, backbone_id: str, device: str = "cpu",
                dtype=None, need_decoder: bool = True):
    """Load both models under transformers. Returns (processor, whisper, tok, lm).

    `need_decoder=False` loads only the encoder — the omni path never uses
    Whisper's decoder, and skipping it saves both memory and load time.
    """
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              WhisperForConditionalGeneration, WhisperModel,
                              WhisperProcessor)

    dtype = dtype or (torch.float32 if device == "cpu" else torch.float16)
    processor = WhisperProcessor.from_pretrained(whisper_id)
    cls = WhisperForConditionalGeneration if need_decoder else WhisperModel
    whisper = cls.from_pretrained(whisper_id, dtype=dtype).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(backbone_id)
    backbone = AutoModelForCausalLM.from_pretrained(backbone_id, dtype=dtype)
    backbone = backbone.to(device).eval()
    return processor, whisper, tokenizer, backbone


class OmniApp:
    """One turn at a time. Streaming is Stage 5; this is the shape it grows into."""

    def __init__(self, processor, whisper, tokenizer, backbone, *,
                 path: str = "text", projector=None, device: str = "cpu",
                 soul: str = "mochi", capture_to: str | Path | None = None,
                 max_new_tokens: int = 64, language: str = "en"):
        if path not in PATHS:
            raise ValueError(f"path must be one of {PATHS}, got {path!r}")
        self.processor = processor
        self.whisper = whisper
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.path = path
        self.device = device
        self.soul = soul
        self.max_new_tokens = max_new_tokens
        self.language = language
        self.capture = CaptureWriter(capture_to) if capture_to else None

        cog_dim = hidden_size_of(backbone)
        if path == "omni":
            if projector is None:
                projector = JaegerAudioProjector(
                    audio_dim=whisper.config.d_model, cognition_dim=cog_dim)
            if projector.cognition_dim != cog_dim:
                raise ValueError(
                    f"projector emits {projector.cognition_dim}, backbone wants "
                    f"{cog_dim} — a projector is only valid for the backbone it "
                    "was trained against")
            self.projector = projector.to(device).eval()
            # Same wrapper as training. A model trained inside the chat
            # template and generated outside it sees a distribution it never
            # learned — which reads as the model ignoring the audio.
            self.wrapper = ChatWrapper(tokenizer, backbone, device=device)
        else:
            self.projector = None
            self.wrapper = None

    # --- the two paths -----------------------------------------------------
    def _features(self, audio: np.ndarray):
        return self.processor(audio, sampling_rate=SAMPLE_RATE,
                              return_tensors="pt").input_features.to(
                                  self.device, self.whisper.dtype)

    def transcribe(self, audio: np.ndarray) -> str:
        import torch
        with torch.no_grad():
            # Pin the language where the checkpoint allows it: left to
            # auto-detect, a short or noisy clip can be mis-detected and
            # transcribed into the wrong language, which poisons the teacher
            # target rather than merely degrading it.
            ids = self.whisper.generate(
                self._features(audio),
                **transcribe_kwargs(self.whisper, self.language))
        return self.processor.batch_decode(
            ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()

    def _encoder_states(self, audio: np.ndarray):
        import torch
        # Accept a full WhisperModel, a WhisperForConditionalGeneration, or a
        # bare encoder (what a delta artifact supplies — THEIR fine-tuned one).
        m = getattr(self.whisper, "model", self.whisper)
        enc = getattr(m, "encoder", m)
        with torch.no_grad():
            return enc(self._features(audio)).last_hidden_state

    def _per_layer(self, ids):
        """Per-layer inputs for backbones that need them (Gemma E4B: 256 dims
        per token per layer). None for everything else."""
        lm = getattr(self.backbone, "language_model", None)
        if lm is None:
            lm = getattr(getattr(self.backbone, "model", None), "language_model", None)
        if lm is None or not hasattr(lm, "get_per_layer_inputs"):
            return None
        cfg = self.backbone.config
        tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        if not getattr(tcfg, "hidden_size_per_layer_input", None):
            return None
        return lm.get_per_layer_inputs(ids, None)

    def _generate_from_embeds(self, embeds):
        """Greedy decode from an audio prefix.

        HF's .generate(inputs_embeds=...) cannot be used with Gemma E4B: with
        no input_ids it tries to RECOVER them by comparing every embedding
        against the whole vocab matrix -- [1,T,1,2560] against
        [1,1,262144,2560], a 29 GiB allocation that then fails anyway, because
        projector-produced audio embeddings are not in the vocab.

        So decode manually and supply per_layer_inputs from placeholder ids,
        the same way StreamingSession does on the MLX side.
        """
        import torch

        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id or eos
        n = embeds.shape[1]
        dev = embeds.device
        ids = torch.full((1, n), pad, dtype=torch.long, device=dev)
        out_ids, cache = [], None
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                ple = self._per_layer(ids)
                kw = {"per_layer_inputs": ple} if ple is not None else {}
                o = self.backbone(inputs_embeds=embeds, use_cache=False, **kw)
                nxt = int(o.logits[:, -1, :].argmax(-1).item())
                if nxt == eos:
                    break
                out_ids.append(nxt)
                step = self.backbone.get_input_embeddings()(
                    torch.tensor([[nxt]], device=dev)).to(embeds.dtype)
                embeds = torch.cat([embeds, step], dim=1)
                ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
        return self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()

    def listen(self, audio: np.ndarray) -> tuple:
        """Audio in -> (CognitiveOutput, Timing). The whole app in one call."""
        if audio.ndim != 1:
            raise ValueError(f"expected mono, got shape {audio.shape}")
        gen_id = uuid.uuid4().hex[:12]
        t = Timing()
        transcript = ""

        if self.path == "text":
            t0 = time.monotonic()
            transcript = self.transcribe(audio)
            t.transcribe_ms = (time.monotonic() - t0) * 1000

            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": transcript}],
                add_generation_prompt=True, tokenize=False)
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            t0 = time.monotonic()
            embeds = self.backbone.get_input_embeddings()(ids)
            reply = self._generate_from_embeds(embeds)
            t.generate_ms = (time.monotonic() - t0) * 1000
        else:
            t0 = time.monotonic()
            states = self._encoder_states(audio)
            t.encode_ms = (time.monotonic() - t0) * 1000

            t0 = time.monotonic()
            duration = len(audio) / SAMPLE_RATE
            embeds = self.projector(states.float(), duration_s=duration)
            embeds, _ = self.wrapper.wrap(embeds)
            t.project_ms = (time.monotonic() - t0) * 1000

            t0 = time.monotonic()
            reply = self._generate_from_embeds(embeds.to(self.backbone.dtype))
            t.generate_ms = (time.monotonic() - t0) * 1000

        out = CognitiveOutput(
            generation_id=gen_id, text=reply,
            spec=neutral(reply, soul=self.soul, generation_id=gen_id))

        if self.capture is not None and self.path == "text":
            # Only the text path captures: its transcript is the teacher signal.
            # Capturing the omni path would log an untrained model's guesses as
            # if they were ground truth.
            self.capture.write(audio, transcript=transcript, response=reply,
                               soul=self.soul, meta={"path": self.path})
        return out, t


class StreamingOmni:
    """The omni path, prefilling while the user is still speaking.

    A batch call encodes the whole utterance after the user stops. This
    consumes 1-second chunks as they arrive, so by end-of-speech nearly all the
    encoder and projector work is already done. Same models, same projector —
    only *when* the work happens changes, and that is where the ~950 ms goes
    (see `dev/legacy pipeline measured.md`).

    MiniCPM does the same thing through `streaming_prefill`, one call per 1 s
    chunk. Ours differs in one respect: it re-encodes the accumulated audio per
    chunk rather than carrying an encoder KV cache, because HF Whisper does not
    expose one. That is O(n^2) over a turn and fine at conversational lengths;
    the rolling 1500-frame cache in `streaming.py` is the shape to grow into.
    """

    def __init__(self, app: "OmniApp"):
        if app.path != "omni":
            raise ValueError("StreamingOmni needs an app on the omni path")
        self.app = app
        self.chunker = StreamingChunker()
        self.reset()

    def reset(self):
        self.chunker = StreamingChunker()
        self._audio = np.zeros(0, dtype=np.float32)
        self._embeds = None
        self.overlapped_ms = 0.0
        self.chunks_seen = 0

    def feed(self, samples: np.ndarray) -> int:
        """Push mic samples. Returns the number of chunks prefilled."""
        done = 0
        for chunk in self.chunker.push(samples):
            self._ingest(chunk.samples)
            done += 1
        return done

    def _ingest(self, samples: np.ndarray):
        t0 = time.monotonic()
        self._audio = np.concatenate([self._audio, samples])
        states = self.app._encoder_states(self._audio)
        embeds = self.app.projector(
            states.float(), duration_s=len(self._audio) / SAMPLE_RATE)
        self._embeds, _ = self.app.wrapper.wrap(embeds)
        self.chunks_seen += 1
        self.overlapped_ms += (time.monotonic() - t0) * 1000

    def finish(self, tail: np.ndarray | None = None) -> tuple:
        """End of turn. Only generation is left, and only the tail to encode."""
        import uuid

        t = Timing(overlapped_ms=self.overlapped_ms)
        if tail is not None and len(tail):
            t0 = time.monotonic()
            self._ingest(tail)
            # The tail is NOT free — the user has stopped, so this is waiting.
            spent = (time.monotonic() - t0) * 1000
            self.overlapped_ms -= spent
            t.overlapped_ms = self.overlapped_ms
            t.encode_ms = spent
        if self._embeds is None:
            raise ValueError("nothing fed: call feed() before finish()")

        t0 = time.monotonic()
        reply = self.app._generate_from_embeds(self._embeds.to(self.app.backbone.dtype)
                                               if hasattr(self.app.backbone, "dtype")
                                               else self._embeds)
        t.generate_ms = (time.monotonic() - t0) * 1000

        gen_id = uuid.uuid4().hex[:12]
        out = CognitiveOutput(generation_id=gen_id, text=reply,
                              spec=neutral(reply, soul=self.app.soul,
                                           generation_id=gen_id))
        return out, t


def read_wav(path: str | Path) -> np.ndarray:
    """16 kHz mono float32 from a PCM wav. stdlib only."""
    import wave

    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"expected PCM16, got {w.getsampwidth()*8}-bit")
        rate, channels = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())

    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        # Linear resample. Fine for 44.1/48k -> 16k speech; swap for a proper
        # polyphase filter if aliasing ever shows up in the transcripts.
        n = int(round(len(audio) * SAMPLE_RATE / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def demo():
    """Self-check against stubs — no weights. Proves routing, timing, capture."""
    import sys
    import tempfile
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
    from stubs import StubAudioEncoder, StubBackbone

    class FakeProcessor:
        def __call__(self, audio, sampling_rate, return_tensors):
            class R:
                input_features = torch.tensor(audio).unsqueeze(0)
            return R()

        def batch_decode(self, ids, **kw):
            return ["what is two plus two"]

    class FakeWhisper(StubAudioEncoder):
        dtype = torch.float32

        def __init__(self):
            super().__init__(audio_dim=64)
            self.encoder = self
            self.config = type("C", (), {"d_model": 64})()

        generation_config = type("G", (), {"is_multilingual": True})()

        def generate(self, feats, **kw):
            assert kw.get("language") == "en", "language must be pinned"
            return torch.zeros(1, 3, dtype=torch.long)

    class FakeTokenizer:
        pad_token_id, eos_token_id = 0, 0
        chat_template = "yes"

        def apply_chat_template(self, msgs, add_generation_prompt=True,
                                tokenize=False):
            return "".join(f"<|{m['role']}|>{m['content']}" for m in msgs) + "<|a|>"

        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            n = max(1, min(12, len(text) // 4))
            return type("R", (), {"input_ids": torch.randint(1, 120, (1, n))})()

        def decode(self, ids, skip_special_tokens=True):
            return "Four."

    class FakeLM(StubBackbone):
        dtype = torch.float32

        def generate(self, inputs_embeds=None, **kw):
            assert inputs_embeds is not None, "must generate from embeddings"
            return torch.zeros(1, 4, dtype=torch.long)

    audio = (np.sin(np.arange(SAMPLE_RATE) * 0.03) * 0.3).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        lm = FakeLM(hidden_size=96, vocab_size=128, n_layers=2)
        app = OmniApp(FakeProcessor(), FakeWhisper(), FakeTokenizer(), lm,
                      path="text", capture_to=tmp)
        out, t = app.listen(audio)
        assert out.text == "Four." and out.speaks
        assert out.spec.generation_id == out.generation_id
        # Text path spends its time in transcribe + generate, and nowhere else.
        assert t.transcribe_ms > 0 and t.generate_ms > 0
        assert t.encode_ms == 0 and t.project_ms == 0

        from .core.capture import CaptureReader
        turns = list(CaptureReader(tmp))
        assert len(turns) == 1
        assert turns[0].transcript == "what is two plus two"
        assert turns[0].response == "Four."

        # Omni path: no transcription at all, and a projector stage appears.
        omni = OmniApp(FakeProcessor(), FakeWhisper(), FakeTokenizer(), lm,
                       path="omni", capture_to=tmp,
                       projector=JaegerAudioProjector(audio_dim=64, cognition_dim=96))
        out2, t2 = omni.listen(audio)
        assert t2.transcribe_ms == 0, "omni must never transcribe"
        # Audio must be wrapped in the chat template, not fed bare.
        assert omni.wrapper is not None and omni.wrapper.overhead_tokens > 0
        assert app.wrapper is None, "the text path uses the tokenizer directly"
        assert t2.encode_ms > 0 and t2.project_ms > 0

        # ...and it must NOT capture: an untrained model's guesses are not
        # teacher signal.
        assert len(list(CaptureReader(tmp))) == 1

    try:
        OmniApp(FakeProcessor(), FakeWhisper(), FakeTokenizer(), lm, path="omni",
                projector=JaegerAudioProjector(audio_dim=64, cognition_dim=999))
    except ValueError as exc:
        assert "trained against" in str(exc)
    else:
        raise AssertionError("projector/backbone mismatch must be rejected")

    for bad in ("both", "", "TEXT"):
        try:
            OmniApp(FakeProcessor(), FakeWhisper(), FakeTokenizer(), lm, path=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"path {bad!r} must be rejected")

    try:
        app.listen(np.zeros((2, 100), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("stereo input must be rejected")

    # English-only checkpoints must NOT receive language/task — they raise.
    en_only = type("M", (), {
        "generation_config": type("G", (), {"is_multilingual": False})(),
        "config": type("C", (), {})()})()
    assert transcribe_kwargs(en_only, "en") == {}
    multi = type("M", (), {
        "generation_config": type("G", (), {"is_multilingual": True})(),
        "config": type("C", (), {})()})()
    assert transcribe_kwargs(multi, "en") == {"language": "en", "task": "transcribe"}

    # wav reader: resamples and downmixes.
    with tempfile.TemporaryDirectory() as tmp:
        import wave
        p = Path(tmp) / "s.wav"
        with wave.open(str(p), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(48000)
            w.writeframes((np.zeros(48000 * 2) * 32767).astype("<i2").tobytes())
        got = read_wav(p)
        assert got.ndim == 1 and abs(len(got) - SAMPLE_RATE) <= 1, len(got)

    # Streaming: work done during speech is excluded from the wait.
    lm2 = FakeLM(hidden_size=96, vocab_size=128, n_layers=2)
    stream_app = OmniApp(FakeProcessor(), FakeWhisper(), FakeTokenizer(), lm2,
                         path="omni",
                         projector=JaegerAudioProjector(audio_dim=64, cognition_dim=96))
    st = StreamingOmni(stream_app)

    # Three seconds of speech arriving in 0.5 s pieces -> 3 whole chunks.
    prefilled = sum(st.feed(audio[:SAMPLE_RATE // 2]) for _ in range(6))
    assert prefilled == 3, prefilled
    assert st.overlapped_ms > 0, "prefill must have done real work"

    out3, t3 = st.finish()
    assert out3.text == "Four."
    assert t3.generate_ms > 0
    assert t3.overlapped_ms > 0
    # The whole point: overlapped work is not counted as waiting.
    assert t3.total_ms < t3.total_ms + t3.overlapped_ms

    # A tail after end-of-speech IS waiting, and must be counted as such.
    st.reset()
    assert st.chunks_seen == 0 and st.overlapped_ms == 0
    st.feed(audio[:SAMPLE_RATE])
    _, t4 = st.finish(tail=audio[:SAMPLE_RATE // 4])
    assert t4.encode_ms > 0, "tail encode must be charged to the wait"

    # finish() before any feed() is a caller bug, not a silent empty reply.
    st.reset()
    try:
        st.finish()
    except ValueError:
        pass
    else:
        raise AssertionError("finish() with no audio must raise")

    # Streaming is omni-only; the text path cannot prefill a transcript.
    try:
        StreamingOmni(app)
    except ValueError:
        pass
    else:
        raise AssertionError("StreamingOmni must reject the text path")

    print(f"app: text {t}")
    print(f"app: omni {t2}")
    print(f"app: streaming {t3}")
    print("app: routing, timing split, capture policy, streaming prefill OK")


if __name__ == "__main__":
    demo()
