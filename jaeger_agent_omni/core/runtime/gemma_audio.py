"""Incremental audio prefill on Gemma 4's NATIVE tower (MLX).

The finding that makes always-on duplex buildable without training an encoder:

    1 s of audio -> 25 audio tokens          (MiniCPM spends 10 for the same second)
    chunked encoding == whole-file encoding  (33 tokens either way, measured)
    tower attention_context_right = 0        (causal — no future needed)

So the tower can be driven a second at a time while the user is still talking,
and the tokens accumulate exactly as if the finished utterance had been encoded
at the end. That is incremental prefill, on weights Google already trained.

    encoder tokens are 1536-wide; embed_audio projects them to the language
    model's width (640 for E4B), which is where they join the text stream.

One measured constraint: the FEATURE EXTRACTOR silently truncates past ~130 mel
frames (~2 s). Feed it waveform chunks, never a whole utterance — a 5 s clip
yields the same 33 tokens as a 2 s one, losing the rest without error.

## What this is NOT worth (measured, M1 Max, E4B 4-bit MLX)

    audio tower encode      4 ms     <- nothing to hide behind speech
    LLM prefill over audio  148 ms   <- THIS is what overlapping saves
    decode                  19 ms/token

Hiding the tower buys ~0 ms. The value of incremental prefill is that the
LANGUAGE MODEL's prefill can be run per chunk as tokens arrive, so at
end-of-speech only the last chunk remains. Chunked KV-cached prefill is the
real target; encoding was never the bottleneck. Recorded here because the
opposite was assumed for weeks.

    python -m jaeger_agent_omni.core.runtime.gemma_audio    # self-check (no model)
"""

from __future__ import annotations

from dataclasses import dataclass, field

SAMPLE_RATE = 16000
CHUNK_S = 1.0                      # 1 s -> 25 tokens (the tower's rate, not ours)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_S)
TOKENS_PER_CHUNK = 25
FE_SAFE_SAMPLES = int(SAMPLE_RATE * 1.5)   # under the ~2 s truncation cliff


@dataclass
class PrefillState:
    """Audio tokens accumulated so far this turn, plus the timing that proves
    the work happened during speech rather than after it."""

    embeds: list = field(default_factory=list)     # projected [1, T, d_model]
    n_tokens: int = 0
    n_chunks: int = 0
    overlapped_ms: float = 0.0

    def reset(self):
        self.embeds.clear()
        self.n_tokens = 0
        self.n_chunks = 0
        self.overlapped_ms = 0.0


class GemmaAudioPrefiller:
    """Feed mic audio in; get language-model-ready embeddings out, live.

    Holds no waveform history — each chunk is encoded once and kept as tokens.
    That is what makes it O(n) over a turn, unlike re-encoding accumulated
    audio (which is what the HF-Whisper path had to do for lack of a cache).
    """

    def __init__(self, model, processor, chunk_samples: int = CHUNK_SAMPLES):
        if chunk_samples > FE_SAFE_SAMPLES:
            raise ValueError(
                f"chunk_samples {chunk_samples} exceeds the feature extractor's "
                f"~{FE_SAFE_SAMPLES/SAMPLE_RATE:.1f}s cliff; longer chunks are "
                "silently truncated")
        self.model = model
        self.fe = getattr(processor, "feature_extractor", processor)
        self.chunk_samples = chunk_samples
        self.state = PrefillState()
        self._buf = None

    # -- encoding -----------------------------------------------------------
    def mels(self, samples):
        """waveform chunk -> (mel features, VALIDITY mask: True = real audio).

        Stops at mels on purpose. The model's `get_input_embeddings` runs the
        tower AND `embed_audio` itself inside its audio scatter, so handing it
        anything further along double-encodes: 1536-wide tower output gets
        re-fed to a tower expecting 128-bin mels, and the matmul shape gives
        the game away (12288 = 8 x 1536).
        """
        import mlx.core as mx
        import numpy as np

        feats = self.fe(np.asarray(samples, dtype=np.float32),
                        sampling_rate=SAMPLE_RATE, return_tensors="np")
        keys = list(feats.keys())
        return (mx.array(np.array(feats[keys[0]])),
                mx.array(np.array(feats[keys[1]])))

    def n_tokens_for(self, mel, pad_mask=None) -> int:
        """Audio tokens a mel chunk occupies — needed to size the placeholder
        run before the model scatters features into it.

        `pad_mask` is the tower's convention: True = IGNORE this frame. That is
        the inverse of the feature extractor's validity mask.
        """
        return int(self.model.audio_tower(mel, pad_mask)[0].shape[1])

    def encode_chunk(self, samples):
        """Tower features, for callers driving the tower directly (probes, the
        gate). The streaming path uses `mels()` instead."""
        mel, valid = self.mels(samples)
        return self.model.audio_tower(mel, ~valid)[0]

    def project(self, features):
        """Raw tower features -> language-model width. Only for direct-drive
        callers; the streaming path lets the model project during scatter."""
        return self.model.embed_audio(features)

    def push(self, samples) -> int:
        """Push mic samples. Encodes whole chunks; returns chunks prefilled.

        Partial audio is buffered, never encoded short — a short chunk would
        shift the token alignment for the rest of the turn.
        """
        import time

        import numpy as np

        samples = np.asarray(samples, dtype=np.float32)
        self._buf = samples if self._buf is None else np.concatenate([self._buf, samples])

        done = 0
        while len(self._buf) >= self.chunk_samples:
            chunk, self._buf = (self._buf[:self.chunk_samples],
                                self._buf[self.chunk_samples:])
            t0 = time.monotonic()
            emb = self.mels(chunk)
            self.state.embeds.append(emb)
            self.state.n_tokens += TOKENS_PER_CHUNK
            self.state.n_chunks += 1
            # Counted as overlapped: the user is still speaking while this runs.
            self.state.overlapped_ms += (time.monotonic() - t0) * 1000
            done += 1
        return done

    def flush(self):
        """End of turn: encode the tail. This part IS latency — the user has
        stopped, so it is time they wait."""
        import time

        import numpy as np

        if self._buf is None or len(self._buf) == 0:
            return 0.0
        t0 = time.monotonic()
        emb = self.mels(self._buf)
        self.state.embeds.append(emb)
        self._buf = None
        return (time.monotonic() - t0) * 1000

    def audio_embeds(self):
        """All audio tokens for this turn, concatenated."""
        import mlx.core as mx

        if not self.state.embeds:
            raise ValueError("nothing prefilled: call push() first")
        return mx.concatenate(self.state.embeds, axis=1)

    def reset(self):
        self.state.reset()
        self._buf = None

    @property
    def pending_samples(self) -> int:
        return 0 if self._buf is None else len(self._buf)


def demo():
    """Chunking arithmetic and buffering, without loading a 4 GB model."""

    class FakeEmb:
        def __init__(self, n):
            self.shape = (1, n, 640)

    class FakeModel:
        """Stands in for the measured behaviour: ~25 tokens per second."""

        def audio_tower(self, mel, mask=None):
            secs = mel.shape[1] / 99.0            # ~99 mel frames per second
            return FakeEmb(max(1, round(25 * secs))), None

        def embed_audio(self, tokens):
            return tokens

    class FakeFE:
        def __call__(self, x, sampling_rate=None, return_tensors=None):
            import numpy as np
            n = max(1, int(len(x) / SAMPLE_RATE * 99))
            return {"mel": np.zeros((1, n, 128), dtype=np.float32),
                    "mask": np.ones((1, n), dtype=bool)}

    # mels() returns mx arrays; the stub keeps numpy, which is shape-compatible
    # for every assertion below.
    import mlx.core as _mx
    _mx.array = lambda a, **k: a

    import numpy as np
    p = GemmaAudioPrefiller.__new__(GemmaAudioPrefiller)
    p.model, p.fe = FakeModel(), FakeFE()
    p.chunk_samples, p.state, p._buf = CHUNK_SAMPLES, PrefillState(), None

    # Partial audio is buffered, not encoded short.
    assert p.push(np.zeros(8000)) == 0 and p.pending_samples == 8000
    assert p.state.n_chunks == 0

    # Completing the second yields exactly one chunk of ~25 tokens.
    assert p.push(np.zeros(8000)) == 1
    assert p.state.n_chunks == 1 and p.pending_samples == 0
    assert p.state.n_tokens == TOKENS_PER_CHUNK, p.state.n_tokens

    # Three more seconds at once -> three chunks, remainder held.
    assert p.push(np.zeros(CHUNK_SAMPLES * 3 + 500)) == 3
    assert p.state.n_chunks == 4 and p.pending_samples == 500
    assert p.state.n_tokens == 4 * TOKENS_PER_CHUNK

    # Work done while the user talked is recorded as overlapped.
    assert p.state.overlapped_ms >= 0.0

    # flush() encodes the tail and is NOT overlapped — it is real wait time.
    before = p.state.overlapped_ms
    tail_ms = p.flush()
    assert p.pending_samples == 0 and tail_ms >= 0.0
    assert p.state.overlapped_ms == before, "tail must not count as free"
    assert p.state.n_tokens == 4 * TOKENS_PER_CHUNK, \
        "flush appends mels; token count is settled at scatter time"

    # Concatenation happens over the accumulated chunks, not raw audio.
    assert len(p.state.embeds) == 5

    p.reset()
    assert p.state.n_tokens == 0 and p.pending_samples == 0
    try:
        p.audio_embeds()
    except ValueError:
        pass
    else:
        raise AssertionError("audio_embeds with nothing prefilled must raise")

    # Chunks longer than the feature extractor's cliff must be refused, not
    # silently truncated — that bug loses audio without any error.
    try:
        GemmaAudioPrefiller(FakeModel(), FakeFE(), chunk_samples=SAMPLE_RATE * 5)
    except ValueError:
        pass
    else:
        raise AssertionError("over-long chunk size must be rejected")

    print(f"runtime.gemma_audio: {CHUNK_S}s chunks -> {TOKENS_PER_CHUNK} tokens, "
          f"buffering + overlap accounting OK")


if __name__ == "__main__":
    demo()
