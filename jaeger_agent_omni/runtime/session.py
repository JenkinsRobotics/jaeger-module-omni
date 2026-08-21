"""Streaming duplex session — the LM prefills WHILE the user is still talking.

The measurement that shaped this (M1 Max, E4B 4-bit):

    audio tower encode        4 ms      never the bottleneck
    LM prefill over audio   148 ms      <- the real cost, and it is hideable
    decode                   19 ms/token

So the tower is not what we overlap; the LANGUAGE MODEL is. Each 1 s chunk is
encoded AND pushed through the LM into a persistent KV cache as it arrives. At
end-of-speech the cache already holds the whole utterance, and only the last
partial chunk remains — turning a 148 ms wait into roughly one chunk's worth.

    speech .....................|
    encode+prefill  ▁▁▁▁▁▁▁▁▁▁▁▁|          all of this is free
                                 ▔▔▔       only this is waiting
                                    decode

Measured steady state (M1 Max, E4B 4-bit): ~118 ms of work per 1000 ms of
audio — 8x realtime headroom, so the runtime keeps up with a live mic.
The FIRST chunk costs ~963 ms (MLX graph compilation); call `warmup()` before
the conversation or the user's opening sentence is the one that stutters.

Same session accepts text and (later) vision, because they all reduce to
embeddings entering one KV cache — which is what makes this an omni runtime
rather than a voice one.

    python -m jaeger_agent_omni.runtime.session      # self-check, no model
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TurnTiming:
    """Time the LISTENER waits, separated from work hidden under speech."""

    overlapped_ms: float = 0.0      # done while they were still talking: free
    tail_ms: float = 0.0            # last chunk, after they stopped
    first_token_ms: float = 0.0     # cache -> first generated token
    decode_ms: float = 0.0

    @property
    def wait_ms(self) -> float:
        """What the human experiences: everything after they stop speaking."""
        return self.tail_ms + self.first_token_ms + self.decode_ms

    def __str__(self) -> str:
        return (f"wait {self.wait_ms:.0f}ms "
                f"(tail {self.tail_ms:.0f} + first-token {self.first_token_ms:.0f} "
                f"+ decode {self.decode_ms:.0f}) | {self.overlapped_ms:.0f}ms hidden")


class StreamingSession:
    """One conversation turn's worth of streaming state.

    Holds a KV cache across chunks. `feed_audio` is safe to call from an audio
    callback; `respond` is called once the turn-end decision fires.
    """

    def __init__(self, model, processor, prefiller, cache_factory=None,
                 system: str | None = None, config: dict | None = None):
        self.model = model
        self.processor = processor
        self.prefiller = prefiller
        self._make_cache = cache_factory or self._default_cache
        self.system = system
        # Gemma brackets audio with its own markers inside the turn structure.
        # Feeding bare embeddings produces fluent garbage — the same failure the
        # delta path hit, for the same reason. Ids come from the model config.
        cfg = config or {}
        self.boa_id = cfg.get("boa_token_id")
        self.eoa_id = cfg.get("eoa_token_id")
        self.audio_id = cfg.get("audio_token_id")
        self.image_id = cfg.get("image_token_id")
        self.n_image_tokens = cfg.get("vision_soft_tokens_per_image", 280)
        self.reset()

    # -- chat scaffolding ---------------------------------------------------
    def _embed_ids(self, ids):
        """Token ids -> (embeddings, per_layer_inputs), via the model's own
        assembly so embed_scale and per-layer inputs are handled exactly as in
        a normal forward pass."""
        import mlx.core as mx

        arr = mx.array([ids])
        out = self.model.get_input_embeddings(input_ids=arr)
        if isinstance(out, tuple):
            return out[0], out[1]
        # InputEmbeddingsFeatures: per_layer_inputs is [B, T, n_layers, d] —
        # one vector per token PER LAYER. Dropping it yields fluent garbage.
        return getattr(out, "inputs_embeds", out), getattr(out, "per_layer_inputs", None)

    def _embed_audio_chunk(self, features):
        """Audio features -> (embeddings, per_layer_inputs) via the model's own
        assembly.

        The naive route — splice raw tower output into the stream — is wrong on
        E4B twice over: token embeddings are multiplied by `embed_scale`, and
        every position (audio included) needs a 256-wide `per_layer_inputs`
        vector for each of the 42 layers. Building a run of `<|audio|>`
        placeholder ids and letting `get_input_embeddings` masked-scatter the
        features in gets both right, exactly as a normal forward pass would.
        """
        import mlx.core as mx

        mel, valid = features
        # POLARITY. The feature extractor returns a VALIDITY mask (True = real
        # audio). `get_input_embeddings` expects `input_features_mask` and
        # inverts it internally (audio_mask = ~mask); the tower's `audio_mask`
        # is a PADDING mask (True = ignore). Passing validity straight into
        # audio_mask marks every frame invalid, the tower zeroes them, and the
        # model answers as if it heard silence — fluent, on-topic-for-a-chatbot,
        # and completely unrelated to the audio. Measured, not theorised.
        n = self.prefiller.n_tokens_for(mel, ~valid)
        ids = mx.array([[self.audio_id] * n])
        out = self.model.get_input_embeddings(input_ids=ids,
                                              audio_features=mel,
                                              input_features_mask=valid)
        if isinstance(out, tuple):
            return out[0], out[1]
        return (getattr(out, "inputs_embeds", out),
                getattr(out, "per_layer_inputs", None))

    def open_turn(self):
        """Prefill everything that precedes the user's audio: bos, system,
        the user turn header, and the begin-audio marker."""
        tok = self.processor.tokenizer
        msgs = ([{"role": "system", "content": self.system}] if self.system else [])
        msgs.append({"role": "user", "content": "\x00A\x00"})
        rendered = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False)
        head, tail = rendered.split("\x00A\x00", 1)
        self._turn_tail = tail
        ids = tok(head, add_special_tokens=False).input_ids
        if self.boa_id is not None:
            ids = ids + [self.boa_id]
        emb, per_layer = self._embed_ids(ids)
        self.timing.overlapped_ms += self._prefill(emb, per_layer)

    def close_turn(self):
        """End-audio marker plus the model turn header — after this the next
        token generated is the reply's first word."""
        tok = self.processor.tokenizer
        ids = ([self.eoa_id] if self.eoa_id is not None else [])
        ids += tok(self._turn_tail, add_special_tokens=False).input_ids
        emb, per_layer = self._embed_ids(ids)
        self.timing.tail_ms += self._prefill(emb, per_layer)

    @staticmethod
    def _default_cache():
        from mlx_vlm.models.cache import KVCache
        lm = None
        return KVCache()

    def warmup_vision(self, size: int = 64):
        """Compile the vision graph too. Skip this and the first frames cost
        5-13 s instead of ~2.2 s, which reads as a quadratic blowup in the
        loop's lag counter when it is only MLX compiling."""
        import numpy as np

        if self.image_id is None:
            return 0.0
        t0 = time.monotonic()
        saved = self.cache, self.n_prefilled
        self.cache, self.n_prefilled = None, 0
        try:
            self.feed_image(np.zeros((size, size, 3), dtype=np.uint8))
        finally:
            self.cache, self.n_prefilled = saved
        return (time.monotonic() - t0) * 1000

    def warmup(self, seconds: float = 1.0):
        """Compile graphs and page weights BEFORE the conversation starts.

        Measured, and it is not subtle: the first chunk costs ~963 ms of MLX
        graph compilation while every chunk after is ~118 ms. Skip this and the
        first thing a user ever says is the one turn that stutters. Runs on
        silence, then throws the cache away.
        """
        import numpy as np

        t0 = time.monotonic()
        saved = self.cache, self.n_prefilled
        self.cache, self.n_prefilled = None, 0
        try:
            emb = self.prefiller.mels(
                np.zeros(int(16000 * seconds), dtype=np.float32))
            e, pl = self._embed_audio_chunk(emb)
            self._prefill(e, pl)
        finally:
            self.cache, self.n_prefilled = saved
            self.prefiller.reset()
        return (time.monotonic() - t0) * 1000

    def reset(self):
        """New turn. Cache is per-turn; conversation history is the caller's."""
        self.cache = None
        self.timing = TurnTiming()
        self.n_prefilled = 0
        self.prefiller.reset()
        self._cancelled = False
        self._cancel_at = 0.0
        self.cancel_latency_ms = None
        self._generation_start = 0
        self._turn_tail = ""

    # -- listening ----------------------------------------------------------
    def _ensure_cache(self):
        if self.cache is None:
            n_layers = len(self.model.language_model.model.layers)
            self.cache = [self._make_cache() for _ in range(n_layers)]
        return self.cache

    def _prefill(self, embeds, per_layer=None) -> float:
        """Push embeddings through the LM into the KV cache. Returns ms.

        `per_layer` is not optional decoration on E4B: the model feeds per-layer
        embeddings alongside the main stream, and omitting them produces fluent
        garbage rather than an error. Text segments carry theirs; audio
        segments correctly have none.
        """
        import mlx.core as mx

        t0 = time.monotonic()
        out = self.model.language_model(inputs_embeds=embeds,
                                        per_layer_inputs=per_layer,
                                        cache=self._ensure_cache())
        logits = out.logits if hasattr(out, "logits") else out
        mx.eval(logits)
        self._last_logits = logits
        self.n_prefilled += embeds.shape[1]
        return (time.monotonic() - t0) * 1000

    def feed_audio(self, samples) -> int:
        """Mic samples in. Whole chunks are encoded AND prefilled immediately —
        this is the work that disappears under the user's own speech."""
        chunks_before = self.prefiller.state.n_chunks
        self.prefiller.push(samples)
        new = self.prefiller.state.n_chunks - chunks_before
        # Encode time is booked by the prefiller; prefill time is booked here.
        # Both happened while the user was speaking, so both are hidden.
        for emb in self.prefiller.state.embeds[chunks_before:]:
            e, pl = self._embed_audio_chunk(emb)
            self.timing.overlapped_ms += self._prefill(e, pl)
        return new

    @property
    def hidden_ms(self) -> float:
        """All work done under the user's speech: encoding plus LM prefill."""
        return self.timing.overlapped_ms + self.prefiller.state.overlapped_ms

    def feed_image(self, image) -> int:
        """An image into the same cache as audio and text.

        This is what makes the runtime omni rather than voice: every modality
        reduces to embeddings scattered over placeholder tokens in ONE stream,
        so vision costs a preprocessing call and nothing architectural.
        Measured: 280 soft tokens per image, same 2560-wide space audio lands
        in.

        `image` is anything the processor's image_processor accepts (HWC uint8
        array, PIL image). Frames arriving during speech are hidden work, same
        as audio chunks.
        """
        import mlx.core as mx
        import numpy as np

        if self.image_id is None:
            raise ValueError("this model declares no image_token_id")
        ip = getattr(self.processor, "image_processor", None)
        if ip is None:
            raise ValueError("processor has no image_processor")

        out = ip(images=[np.asarray(image)], return_tensors="np")
        feats = out[0] if isinstance(out, tuple) else out
        px = mx.array(feats["pixel_values"])
        ids = mx.array([[self.image_id] * self.n_image_tokens])
        emb = self.model.get_input_embeddings(input_ids=ids, pixel_values=px)
        embeds, per_layer = ((emb[0], emb[1]) if isinstance(emb, tuple)
                             else (getattr(emb, "inputs_embeds", emb),
                                   getattr(emb, "per_layer_inputs", None)))
        self.timing.overlapped_ms += self._prefill(embeds, per_layer)
        return embeds.shape[1]

    def feed_text(self, text: str) -> int:
        """Text into the same cache — omni means one stream, many modalities."""
        import mlx.core as mx

        ids = self.processor.tokenizer(text, add_special_tokens=False).input_ids
        embeds, per_layer = self._embed_ids(ids)
        self.timing.overlapped_ms += self._prefill(embeds, per_layer)
        return embeds.shape[1]

    def end_of_turn(self, close: bool = True):
        """Speech stopped: flush the tail. This part IS the wait."""
        t0 = time.monotonic()
        tail_start = len(self.prefiller.state.embeds)
        self.prefiller.flush()
        for emb in self.prefiller.state.embeds[tail_start:]:
            e, pl = self._embed_audio_chunk(emb)
            self._prefill(e, pl)
        self.timing.tail_ms += (time.monotonic() - t0) * 1000
        if close and self._turn_tail:
            self.close_turn()

    # -- speaking -----------------------------------------------------------
    def cancel(self):
        """Barge-in: stop generating. Safe to call from an audio callback.

        Cancellation lands at the next token boundary, so worst-case latency is
        one decode step (~19 ms measured) — well under the ~200 ms of speech
        the legacy stack requires before it even calls it a barge-in.
        """
        self._cancel_at = time.monotonic()
        self._cancelled = True

    def interrupt(self) -> int:
        """Abandon the half-spoken reply but KEEP the conversation.

        Rewinds the KV cache to the moment before generation started, so the
        user's original question and everything before it survive while the
        abandoned answer does not. Starting a fresh session instead — which is
        the obvious thing, and what a first attempt does — throws away the
        context, and the follow-up ("what about Japan?") arrives with nothing
        to attach to. Returns tokens discarded.

        The unspoken half of a reply is also something the model should not
        believe it said: leaving it in the cache makes the assistant act as if
        it had finished a sentence the human cut off.
        """
        drop = self.n_prefilled - self._generation_start
        if drop <= 0 or not self.cache:
            return 0
        if not all(getattr(c, "is_trimmable", lambda: False)() for c in self.cache):
            raise RuntimeError("cache is not trimmable; cannot rewind an "
                               "interrupted turn without losing history")
        for c in self.cache:
            c.trim(drop)
        self.n_prefilled -= drop
        self._cancelled = False
        return drop

    def resume_listening(self):
        """After an interrupt: open a fresh user turn on the SAME cache."""
        self.prefiller.reset()
        self.timing = TurnTiming()
        self.open_turn()

    def respond(self, max_tokens: int = 64, eos_ids=(), on_token=None) -> str:
        """Generate from the already-warm cache. First token is nearly free
        because the utterance was prefilled while it was being spoken."""
        import mlx.core as mx

        if self.n_prefilled == 0:
            raise ValueError("nothing prefilled — feed audio or text first")

        tok = mx.argmax(self._last_logits[:, -1, :], axis=-1)
        mx.eval(tok)

        out_ids, t0, first_done = [], time.monotonic(), None
        self.cancel_latency_ms = None
        # Where the reply starts, so an interrupt can rewind exactly this far.
        self._generation_start = self.n_prefilled
        for _ in range(max_tokens):
            if self._cancelled:
                # Barge-in: stop at the token boundary. The caller drops queued
                # audio; nothing further is generated for this generation_id.
                self.cancel_latency_ms = (time.monotonic() - self._cancel_at) * 1000
                break
            tid = int(tok.item())
            if tid in eos_ids:
                break
            out_ids.append(tid)
            if on_token is not None:
                on_token(tid, len(out_ids))
            if first_done is None:
                first_done = time.monotonic()
                self.timing.first_token_ms = (first_done - t0) * 1000
            step_emb, step_per_layer = self._embed_ids([int(tok.item())])
            o = self.model.language_model(inputs_embeds=step_emb,
                                          per_layer_inputs=step_per_layer,
                                          cache=self.cache)
            self.n_prefilled += 1
            logits = o.logits if hasattr(o, "logits") else o
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)

        if first_done is not None:
            self.timing.decode_ms = (time.monotonic() - first_done) * 1000
        return self.processor.tokenizer.decode(out_ids)


def demo():
    """Cache-reuse accounting and cancellation, with a stand-in model."""

    class FakeArr:
        def __init__(self, n):
            self.shape = (1, n, 8)

        def __getitem__(self, idx):        # logits[:, -1, :]
            return self

        def __iter__(self):                # never unpack an array as a tuple
            raise TypeError("FakeArr is not iterable")

        def __invert__(self):              # ~validity -> padding mask
            return self

        def __mul__(self, k):              # embed_scale
            return self

        __rmul__ = __mul__

    class FakeCache:
        def __init__(self):
            self.n = 0

    class FakeLM:
        class model:
            layers = [1, 2, 3]
            embed_scale = 1.0

            @staticmethod
            def embed_tokens(ids):
                return FakeArr(1)

        def __call__(self, inputs_embeds=None, cache=None, per_layer_inputs=None):
            assert cache is not None, "prefill must reuse the cache"
            assert hasattr(inputs_embeds, "shape"), \
                "prefill received un-embedded input — mels must go through " \
                "get_input_embeddings first"
            for c in cache:
                c.n += inputs_embeds.shape[1]
            return type("O", (), {"logits": FakeArr(inputs_embeds.shape[1])})()

    class FakeModel:
        language_model = FakeLM()

        @staticmethod
        def get_input_embeddings(input_ids=None, audio_features=None,
                                 input_features_mask=None, pixel_values=None):
            n = input_ids.shape[1] if hasattr(input_ids, "shape") else 1
            return FakeArr(n), None

    class FakePrefiller:
        def __init__(self):
            from .gemma_audio import PrefillState
            self.state = PrefillState()

        def n_tokens_for(self, mel, pad_mask=None):
            return mel.shape[1]

        def push(self, samples):
            self.state.embeds.append((FakeArr(25), FakeArr(25)))
            self.state.n_chunks += 1
            self.state.n_tokens += 25
            self.state.overlapped_ms += 4.0

        def flush(self):
            self.state.embeds.append((FakeArr(7), FakeArr(7)))
            self.state.n_tokens += 7

        def reset(self):
            self.state.reset()

    class FakeIds:
        def __init__(self, n):
            self.shape = (1, n)

    import mlx.core as _mx
    _mx.array = lambda x, **k: FakeIds(
        len(x[0]) if isinstance(x, list) and x and isinstance(x[0], list) else 1)

    s = StreamingSession(FakeModel(), None, FakePrefiller(),
                         cache_factory=FakeCache,
                         config={"audio_token_id": 1, "boa_token_id": 2,
                                 "eoa_token_id": 3})

    # Each chunk prefills into the SAME cache — that is the whole mechanism.
    assert s.feed_audio(None) == 1
    assert s.feed_audio(None) == 1
    assert s.n_prefilled == 50, s.n_prefilled
    assert len(s.cache) == 3, "one cache per layer"
    assert all(c.n == 50 for c in s.cache), "cache must accumulate across chunks"
    assert s.timing.overlapped_ms > 0

    # The tail is charged to the wait, never to the hidden column.
    hidden_before = s.timing.overlapped_ms
    s.end_of_turn(close=False)
    assert s.timing.tail_ms >= 0.0
    assert s.timing.overlapped_ms == hidden_before, "tail must not count as free"
    assert s.n_prefilled == 57
    assert all(c.n == 57 for c in s.cache)

    # wait_ms excludes overlapped work — the number a human actually feels.
    assert s.timing.wait_ms == (s.timing.tail_ms + s.timing.first_token_ms
                                + s.timing.decode_ms)

    # Encoding and prefill are both hidden under speech.
    assert s.hidden_ms > s.timing.overlapped_ms, "encode time must be counted too"

    # Cancellation stops generation immediately (barge-in).
    s._cancelled = True
    s._last_logits = FakeArr(1)
    import mlx.core as mx
    mx.argmax = lambda a, axis=None: type("T", (), {"item": lambda self: 0,
                                                    "__getitem__": lambda s, i: s})()
    mx.eval = lambda *a, **k: None

    class T:
        @staticmethod
        def decode(ids):
            return "".join(map(str, ids))

    s.processor = type("P", (), {"tokenizer": T})()
    assert s.respond(max_tokens=10) == "", "cancel must yield nothing"

    # warmup must not leave state behind — the next real turn starts clean.
    s2 = StreamingSession(FakeModel(), None, FakePrefiller(),
                          cache_factory=FakeCache)
    s2.prefiller.mels = lambda x: (FakeArr(25), FakeArr(25))
    s2.warmup(0.1)
    assert s2.cache is None and s2.n_prefilled == 0
    assert s2.prefiller.state.n_chunks == 0

    # Vision enters the SAME cache through the same placeholder mechanism —
    # that is the omni claim, so it gets asserted.
    s_v = StreamingSession(FakeModel(), None, FakePrefiller(),
                           cache_factory=FakeCache,
                           config={"audio_token_id": 1, "image_token_id": 9,
                                   "vision_soft_tokens_per_image": 4})
    s_v.processor = type("P", (), {
        "image_processor": lambda *a, **k: ({"pixel_values": [[0]]},)})()
    import numpy as _np
    n_img = s_v.feed_image(_np.zeros((4, 4, 3), dtype=_np.uint8))
    assert n_img == 4, n_img
    assert s_v.n_prefilled == 4
    s_v.feed_audio(None)
    assert s_v.n_prefilled == 29, "audio and vision must share one stream"
    assert all(c.n == 29 for c in s_v.cache)

    # A model without vision must refuse rather than silently no-op.
    s_nv = StreamingSession(FakeModel(), None, FakePrefiller(),
                            cache_factory=FakeCache, config={"audio_token_id": 1})
    try:
        s_nv.feed_image(_np.zeros((4, 4, 3), dtype=_np.uint8))
    except ValueError:
        pass
    else:
        raise AssertionError("missing image_token_id must raise")

    # Barge-in must keep the conversation: rewind the abandoned reply out of
    # the cache, keep everything the user said before it.
    class TrimCache(FakeCache):
        def is_trimmable(self):
            return True

        def trim(self, n):
            self.n -= n
            return n

    s3 = StreamingSession(FakeModel(), None, FakePrefiller(),
                          cache_factory=TrimCache,
                          config={"audio_token_id": 1, "boa_token_id": 2,
                                  "eoa_token_id": 3})
    s3.processor = type("P", (), {"tokenizer": T})()
    s3.feed_audio(None)
    context = s3.n_prefilled
    s3._generation_start = context
    s3.n_prefilled += 9                      # nine tokens of abandoned reply
    for c in s3.cache:
        c.n += 9
    assert s3.interrupt() == 9, "must discard exactly the generated tokens"
    assert s3.n_prefilled == context, "context before the reply must survive"
    assert all(c.n == context for c in s3.cache)
    assert s3.interrupt() == 0, "nothing left to discard is not an error"

    # A cache that cannot be trimmed must say so rather than silently
    # corrupting the conversation.
    s4 = StreamingSession(FakeModel(), None, FakePrefiller(),
                          cache_factory=FakeCache,
                          config={"audio_token_id": 1})
    s4.feed_audio(None)
    s4._generation_start = 0
    s4.n_prefilled += 5
    try:
        s4.interrupt()
    except RuntimeError:
        pass
    else:
        raise AssertionError("untrimmable cache must raise")

    # Reset clears cache and timing for the next turn.
    s.reset()
    assert s.cache is None and s.n_prefilled == 0
    try:
        s.respond()
    except ValueError:
        pass
    else:
        raise AssertionError("respond with an empty cache must raise")

    print("runtime.session: chunked prefill into one KV cache, "
          "wait/hidden split, cancellation OK")


if __name__ == "__main__":
    demo()
