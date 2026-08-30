"""Stage 5 — streaming audio chunker.

Ports MiniCPM's streaming design (audition.cpp:396-397):

    const int n_audio_ctx = hparams.n_ctx;      // 1500
    const int n_iter = n_audio_ctx / n_tokens;  // 1500 / 50 = 30

Whisper's fixed 30 s window is neither padded around nor patched out. It becomes
the depth of a rolling KV cache: each 1 s chunk contributes 50 encoder frames,
thirty chunks fill the context, and older frames fall off the front. Each second
of audio costs one second of encoder compute while the model still attends over
the previous 30 seconds.

This module owns the bookkeeping only — sample buffering, chunk boundaries,
cache eviction, token accounting. It does not run the encoder; that is the
caller's job, and keeping them separate means this is testable without weights.

    python -m jaeger_agent_omni.core.streaming     # self-check
"""

from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)      # 16000
FRAMES_PER_CHUNK = 50                                  # 50 fps encoder
CACHE_FRAMES = 1500                                    # 30 s, whisper n_audio_ctx
CHUNKS_IN_CACHE = CACHE_FRAMES // FRAMES_PER_CHUNK     # 30


@dataclass
class Chunk:
    index: int
    samples: np.ndarray          # exactly CHUNK_SAMPLES, zero-padded if final
    is_last: bool
    n_valid_samples: int         # real audio in this chunk, before padding

    @property
    def n_frames(self) -> int:
        return FRAMES_PER_CHUNK

    @property
    def start_s(self) -> float:
        return self.index * CHUNK_SECONDS


class StreamingChunker:
    """Feed samples as they arrive from the mic; get whole chunks out.

    Partial audio is held back rather than emitted short — the encoder expects
    a full chunk, and a short one would silently shift the frame alignment.
    """

    def __init__(self):
        self._buf = np.zeros(0, dtype=np.float32)
        self._emitted = 0

    def push(self, samples: np.ndarray) -> list:
        if samples.ndim != 1:
            raise ValueError(f"expected mono, got shape {samples.shape}")
        self._buf = np.concatenate([self._buf, samples.astype(np.float32)])

        out = []
        while len(self._buf) >= CHUNK_SAMPLES:
            out.append(Chunk(index=self._emitted,
                             samples=self._buf[:CHUNK_SAMPLES],
                             is_last=False,
                             n_valid_samples=CHUNK_SAMPLES))
            self._buf = self._buf[CHUNK_SAMPLES:]
            self._emitted += 1
        return out

    def flush(self) -> list:
        """End of turn. Emit the tail zero-padded and marked is_last."""
        n_valid = len(self._buf)
        if n_valid == 0:
            # Nothing pending, but the caller still needs a boundary marker.
            return [Chunk(index=self._emitted, samples=np.zeros(CHUNK_SAMPLES, np.float32),
                          is_last=True, n_valid_samples=0)]
        padded = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        padded[:n_valid] = self._buf
        chunk = Chunk(index=self._emitted, samples=padded, is_last=True,
                      n_valid_samples=n_valid)
        self._buf = np.zeros(0, dtype=np.float32)
        self._emitted += 1
        return [chunk]

    @property
    def pending_samples(self) -> int:
        return len(self._buf)


class EncoderCache:
    """Rolling frame budget for the Whisper encoder, MiniCPM-style.

    Tracks how many frames are resident and how many are evicted, so the caller
    can size the real KV cache and know when context is being lost. Holds counts,
    not tensors — the accounting is what is easy to get wrong.
    """

    def __init__(self, capacity_frames: int = CACHE_FRAMES):
        self.capacity = capacity_frames
        self.resident = 0
        self.evicted = 0

    def admit(self, n_frames: int) -> int:
        """Add frames, evicting from the front. Returns frames evicted."""
        self.resident += n_frames
        if self.resident <= self.capacity:
            return 0
        drop = self.resident - self.capacity
        self.resident = self.capacity
        self.evicted += drop
        return drop

    @property
    def seconds_resident(self) -> float:
        return self.resident / (FRAMES_PER_CHUNK / CHUNK_SECONDS)

    @property
    def is_full(self) -> bool:
        return self.resident >= self.capacity

    def reset(self):
        self.resident = 0
        self.evicted = 0


def audio_tokens_for(n_chunks: int, pool_stride: int = 5) -> int:
    """LLM tokens produced by n_chunks of speech. 1 s -> 50 frames -> 10 tokens."""
    return n_chunks * (FRAMES_PER_CHUNK // pool_stride)


def demo():
    # Constants match the reference implementation.
    assert CHUNKS_IN_CACHE == 30, CHUNKS_IN_CACHE
    assert CACHE_FRAMES / FRAMES_PER_CHUNK == 30.0

    c = StreamingChunker()

    # Partial audio is held, not emitted short.
    assert c.push(np.zeros(8000, np.float32)) == []
    assert c.pending_samples == 8000

    # Completing the second half emits exactly one chunk.
    chunks = c.push(np.zeros(8000, np.float32))
    assert len(chunks) == 1 and chunks[0].index == 0
    assert chunks[0].n_valid_samples == CHUNK_SAMPLES
    assert not chunks[0].is_last
    assert c.pending_samples == 0

    # A big push emits several at once and keeps the remainder.
    chunks = c.push(np.zeros(CHUNK_SAMPLES * 3 + 500, np.float32))
    assert [k.index for k in chunks] == [1, 2, 3]
    assert c.pending_samples == 500

    # Flush pads the tail and marks it, preserving the true length.
    tail = c.flush()
    assert len(tail) == 1 and tail[0].is_last
    assert tail[0].n_valid_samples == 500
    assert len(tail[0].samples) == CHUNK_SAMPLES
    assert c.pending_samples == 0

    # Flushing an empty buffer still yields a boundary marker.
    assert c.flush()[0].is_last and c.flush()[0].n_valid_samples == 0

    # Sample-accurate content, not just shapes: a ramp must survive split and
    # reassembly in order.
    c2 = StreamingChunker()
    ramp = np.arange(CHUNK_SAMPLES * 2, dtype=np.float32)
    got = np.concatenate([k.samples for k in c2.push(ramp)])
    assert np.array_equal(got, ramp), "chunker must not reorder or drop samples"

    # Timeline is derivable from the index.
    assert c2.push(np.zeros(CHUNK_SAMPLES, np.float32))[0].start_s == 2.0

    # Cache fills after 30 chunks, then evicts one chunk's worth per chunk.
    cache = EncoderCache()
    for _ in range(CHUNKS_IN_CACHE):
        assert cache.admit(FRAMES_PER_CHUNK) == 0, "must not evict before full"
    assert cache.is_full and cache.resident == CACHE_FRAMES
    assert cache.seconds_resident == 30.0
    assert cache.admit(FRAMES_PER_CHUNK) == FRAMES_PER_CHUNK
    assert cache.resident == CACHE_FRAMES, "capacity must hold steady once full"
    assert cache.evicted == FRAMES_PER_CHUNK

    # Steady state: each new second costs exactly one second of eviction.
    for _ in range(10):
        assert cache.admit(FRAMES_PER_CHUNK) == FRAMES_PER_CHUNK
    assert cache.resident == CACHE_FRAMES

    cache.reset()
    assert cache.resident == 0 and cache.evicted == 0

    # Token budget: 10 audio tokens/s. A 5 s utterance costs ~50 tokens, which
    # is the number that has to stay small or prefill balloons.
    assert audio_tokens_for(5) == 50
    assert audio_tokens_for(CHUNKS_IN_CACHE) == 300

    print(f"streaming: {CHUNK_SECONDS:.0f}s chunks, {FRAMES_PER_CHUNK} frames each, "
          f"{CACHE_FRAMES}-frame cache = {CHUNKS_IN_CACHE}s context")
    print(f"streaming: {audio_tokens_for(1)} audio tokens/s -> "
          f"{audio_tokens_for(CHUNKS_IN_CACHE)} tokens at full context")
    print("streaming: buffering, boundaries, padding, eviction, budget OK")


if __name__ == "__main__":
    demo()
