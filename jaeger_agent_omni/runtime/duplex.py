"""Always-on duplex loop — mic and camera both streaming, continuously.

MiniCPM's omni mode interleaves modalities in TIME: a video frame, a second of
audio, a frame, a second, so the model sees the room evolving while you speak.
Sequence order carries the timeline; there is no fusion module. This reproduces
that on native Gemma.

    mic     ────────────────────────────────────────  never closes
    camera  ──┬────────┬────────┬────────┬─────────  1 frame / N seconds
              ▼        ▼        ▼        ▼
    cache   [img][aud][img][aud][img][aud]...        one interleaved stream
                                          │
                              turn-end says "my turn"
                                          ▼
                                      respond, cancellable

Two rules that make it feel live rather than batched:

  DROP, DON'T QUEUE. If encoding falls behind the clock, discard the frame or
  chunk instead of building a backlog. A backlog means the model answers
  questions about a room that has moved on — worse than missing a frame.

  VISION IS THE BUDGET. Fully warmed, M1 Max / E4B 4-bit:

      per frame          1035 ms   tower + scatter + LM prefill of 280 tokens
      per second audio    118 ms   8x realtime headroom

  Reference point, same Mac: MiniCPM-o does audio+vision prefill in 379 ms.
  It is ~2.7x faster and the reason is architectural, not tuning — its
  perceiver resampler emits 64 tokens per image where Gemma's tower emits 280.
  4.4x the tokens, 2.7x the time.

  So `frame_interval_s` is a duty-cycle dial:

      1 s   -> 104% of realtime   impossible
      5 s   ->  21%               comfortable, and the demo bar
      10 s  ->  10%               background awareness

  Resolution is NOT a dial: the processor normalises every input to 672x912,
  so a 224px frame costs the same as a 640px one. Only the interval matters.

  WARM BOTH PATHS before the loop (`warmup()` and `warmup_vision()`). Unwarmed
  vision costs 5-13 s on first frames, which reads as a quadratic blowup when
  it is only MLX compiling — that misreading cost an hour here.

    python -m jaeger_agent_omni.runtime.duplex     # self-check, no devices
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

AUDIO_CHUNK_S = 1.0
SAMPLE_RATE = 16000
FRAME_INTERVAL_S = 5.0      # ~2.2 s per frame measured; see the module note
MAX_LAG_S = 0.5             # beyond this, drop rather than queue


@dataclass
class DuplexStats:
    audio_chunks: int = 0
    audio_samples: int = 0
    audio_samples_dropped: int = 0
    endpoint_events: list = field(default_factory=list)
    frames: int = 0
    frames_dropped: int = 0
    audio_dropped: int = 0
    turns: int = 0
    interrupts: int = 0
    max_lag_s: float = 0.0

    def __str__(self) -> str:
        return (f"{self.turns} turns · {self.audio_chunks} audio chunks · "
                f"{self.frames} frames ({self.frames_dropped} dropped) · "
                f"{self.interrupts} interrupts · peak lag {self.max_lag_s*1000:.0f}ms"
                + (f" · endpoint {'/'.join(self.endpoint_events[:4])}"
                   if self.endpoint_events else ""))


class DuplexLoop:
    """Drives a StreamingSession from live sources at wall-clock rate.

    Sources are callables so the loop is testable without hardware and works
    the same against a mic, a file, or a test double:

        audio_source()  -> float32 samples since last call (or None)
        frame_source()  -> HWC image array (or None)
        turn_ended()    -> bool: has the user finished speaking?
        on_reply(text)  -> speak it
        user_speaking() -> bool: barge-in signal while we are replying
    """

    def __init__(self, session, audio_source, frame_source=None,
                 turn_ended=None, on_reply=None, user_speaking=None,
                 before_respond=None,
                 frame_interval_s: float = FRAME_INTERVAL_S,
                 max_lag_s: float = MAX_LAG_S, clock=time.monotonic):
        self.session = session
        self.audio_source = audio_source
        self.frame_source = frame_source
        # No turn_ended supplied means HANDS-FREE: detect the end of speech
        # from the audio itself. `turn_ended=False` would leave the model
        # waiting to be told a turn is over, which is not always-on at all.
        self.endpoint = None
        if turn_ended is None:
            from .endpoint import Endpoint
            self.endpoint = Endpoint()
            turn_ended = self._endpoint_says_ended
        self.turn_ended = turn_ended
        self.on_reply = on_reply or (lambda text: None)
        self.user_speaking = user_speaking or (lambda: False)
        # Runs after the final partial audio chunk is flushed but before the
        # user turn closes.  Live diarization uses this seam to append an
        # inline [speaker A] tag without moving any audio out of sequence.
        self.before_respond = before_respond
        self.frame_interval_s = frame_interval_s
        self.max_lag_s = max_lag_s
        self.clock = clock
        self.stats = DuplexStats()
        self._next_frame_at = 0.0
        self._last_drain = 0.0
        self._running = False

    def stop(self):
        self._running = False

    # -- one tick ------------------------------------------------------------
    def _lag(self) -> float:
        """How far behind the incoming stream we are.

        Measured as time since the last source read, NOT time elapsed within
        the current tick. The first version charged every source for work done
        earlier in the same tick, so one slow vision frame starved audio for
        the rest of the loop — 1 audio chunk in 18 seconds. Lag is a property
        of the stream, not of who ran first.
        """
        return self.clock() - self._last_drain

    def _pump_vision(self, now: float):
        """A frame on the interval — the timeline, not every frame available."""
        if self.frame_source is None or now < self._next_frame_at:
            return
        if self._lag() > self.max_lag_s:
            # Behind the clock: a stale frame is worse than no frame, and
            # vision is the expensive one to skip.
            self.stats.frames_dropped += 1
            self._next_frame_at = now + self.frame_interval_s
            return
        frame = self.frame_source()
        self._next_frame_at = now + self.frame_interval_s
        if frame is None:
            return
        self.session.feed_image(frame)
        self.stats.frames += 1
        self._last_drain = self.clock()

    def _endpoint_says_ended(self) -> bool:
        return bool(self.endpoint and self.endpoint.ended)

    def _watch_endpoint(self, samples):
        """Feed the endpoint the same audio the model gets, frame by frame.

        `probably_ended` is the speculation point: the reply can start here,
        ~510 ms before the turn formally closes. That gap IS the latency win
        over a per-frame LISTEN/SPEAK poll.
        """
        if self.endpoint is None:
            return
        n = int(SAMPLE_RATE * self.endpoint.frame_ms / 1000)
        for i in range(0, len(samples) - n + 1, n):
            ev = self.endpoint.push(samples[i:i + n])
            if ev:
                self.stats.endpoint_events.append(ev)

    def _pump_audio(self, now: float):
        samples = self.audio_source()
        if samples is None or len(samples) == 0:
            return
        self._watch_endpoint(samples)
        lag = self._lag()
        self.stats.max_lag_s = max(self.stats.max_lag_s, lag)
        if lag > self.max_lag_s:
            # Audio is cheap (~118 ms/s); dropping it means we are badly behind.
            self.stats.audio_dropped += 1
            self.stats.audio_samples_dropped += len(samples)
            self._last_drain = self.clock()
            return
        self.stats.audio_chunks += self.session.feed_audio(samples)
        self.stats.audio_samples += len(samples)
        self._last_drain = self.clock()

    def _respond(self):
        """Answer, watching for barge-in on every token."""
        if self.before_respond is None:
            self.session.end_of_turn()
        else:
            self.session.end_of_turn(close=False)
            self.before_respond()
            self.session.close_turn()

        def on_token(_tid, _n):
            if self.user_speaking():
                self.session.cancel()

        text = self.session.respond(max_tokens=96, eos_ids=self.eos_ids,
                                    on_token=on_token)
        self.stats.turns += 1
        interrupted = bool(self.session._cancelled)
        if interrupted:
            # Interrupted: drop the half-said reply, keep the conversation.
            self.session.interrupt()
            self.stats.interrupts += 1
        # The abandoned suffix is neither something the model completed nor
        # something the listener heard. Do not publish it as a reply or carry
        # it into summaries after the cache rewind.
        if text.strip() and not interrupted:
            self.on_reply(text)
        if self.endpoint is not None:
            self.endpoint.reset()      # else `ended` latches and every tick answers
        self.session.resume_listening()
        # Audio callbacks continue filling their lock-free queue while decode
        # runs.  Drain that capture into the newly opened turn immediately.
        # Charging it as wall-clock lag would discard precisely the speech
        # heard during our reply, even though it is fresh and already local.
        self._last_drain = self.clock()
        self._pump_audio(self._last_drain)

    def run(self, eos_ids=(), max_seconds: float | None = None,
            tick_s: float = 0.05):
        """Listen, watch, answer, repeat. Blocks until stop() or the budget."""
        self.eos_ids = eos_ids
        self._running = True
        started = self.clock()
        self.session.open_turn()
        self._next_frame_at = started
        self._last_drain = self.clock()

        while self._running:
            now = self.clock()
            if max_seconds is not None and now - started >= max_seconds:
                break
            self._pump_vision(now)
            self._pump_audio(now)
            if self.turn_ended():
                self._respond()
            elapsed = self.clock() - now
            if elapsed < tick_s:
                time.sleep(tick_s - elapsed)
        return self.stats


def demo():
    """Interleaving, drop-don't-queue, and barge-in, on a fake clock."""
    import numpy as np

    class FakeSession:
        def __init__(self):
            self.stream, self.n_prefilled = [], 0
            self._cancelled = False
            self.replies = 0

        def open_turn(self):
            self.stream.append("open")

        def feed_audio(self, s):
            self.stream.append("a")
            return 1

        def feed_image(self, f):
            self.stream.append("v")

        def end_of_turn(self):
            self.stream.append("eot")

        def respond(self, max_tokens=0, eos_ids=(), on_token=None):
            if on_token:
                on_token(1, 1)
            # Real decoding returns the prefix produced before cancellation.
            return "partial" if self._cancelled else "reply"

        def cancel(self):
            self._cancelled = True

        def interrupt(self):
            self.stream.append("rewind")
            self._cancelled = False
            return 3

        def resume_listening(self):
            self.stream.append("resume")

    # --- interleaving: vision on its interval, audio every tick ------------
    t = [0.0]
    clock = lambda: t[0]
    sess = FakeSession()
    frames = [np.zeros((2, 2, 3), np.uint8)] * 100
    loop = DuplexLoop(sess, audio_source=lambda: np.zeros(16000, np.float32),
                      frame_source=lambda: frames.pop() if frames else None,
                      frame_interval_s=2.0, clock=clock)
    loop.eos_ids = ()
    loop.session.open_turn()
    loop._next_frame_at = 0.0
    loop._last_drain = 0.0
    for _ in range(5):
        loop._last_drain = t[0]     # keeping up: never behind on a fake clock
        loop._pump_vision(t[0])
        loop._pump_audio(t[0])
        t[0] += 1.0
    # 5 seconds, frame every 2s -> frames at 0, 2, 4
    assert loop.stats.frames == 3, loop.stats.frames
    assert loop.stats.audio_chunks == 5
    assert loop.stats.audio_samples == 5 * 16000
    # Frames land at t=0,2,4 with audio every second between them — the
    # timeline is carried by sequence order, exactly as MiniCPM does it.
    assert "".join(x for x in sess.stream if x in "av") == "vaavaava", sess.stream

    # --- drop, don't queue -------------------------------------------------
    slow = [0.0]
    def slow_clock():
        slow[0] += 1.0          # every read advances a second: always "behind"
        return slow[0]
    s2 = FakeSession()
    l2 = DuplexLoop(s2, audio_source=lambda: np.zeros(16000, np.float32),
                    frame_source=lambda: np.zeros((2, 2, 3), np.uint8),
                    max_lag_s=0.5, clock=slow_clock)
    l2._last_drain = 0.0            # far behind: every read advances a second
    l2._pump_vision(slow[0])
    l2._pump_audio(slow[0])
    assert l2.stats.frames_dropped == 1, "stale frame must be dropped"
    assert l2.stats.audio_dropped == 1, "stale audio must be dropped"
    assert "v" not in s2.stream and "a" not in s2.stream

    # --- barge-in during a reply rewinds and keeps listening ---------------
    s3 = FakeSession()
    published = []
    l3 = DuplexLoop(s3, audio_source=lambda: None,
                    user_speaking=lambda: True, on_reply=published.append,
                    clock=clock)
    l3.eos_ids = ()
    l3._respond()
    assert l3.stats.interrupts == 1
    assert "rewind" in s3.stream and "resume" in s3.stream
    assert s3.stream.index("rewind") < s3.stream.index("resume")
    assert not published, "an abandoned reply prefix must not be published"

    # a quiet reply still resumes listening
    s4 = FakeSession()
    l4 = DuplexLoop(s4, audio_source=lambda: None, clock=clock)
    l4.eos_ids = ()
    l4._respond()
    assert l4.stats.interrupts == 0 and "resume" in s4.stream

    # A diarization tag lands after the audio tail and before close_turn.  It
    # must never be inserted ahead of an unflushed partial audio chunk.
    class TaggedSession(FakeSession):
        def end_of_turn(self, close=True):
            self.stream.append("flush")
            if close:
                self.stream.append("close")
        def feed_text(self, text):
            self.stream.append(text)
        def close_turn(self):
            self.stream.append("close")

    tagged = TaggedSession()
    lt = DuplexLoop(tagged, audio_source=lambda: None,
                    before_respond=lambda: tagged.feed_text("[speaker A]"),
                    clock=clock)
    lt.eos_ids = ()
    lt._respond()
    assert tagged.stream[:3] == ["flush", "[speaker A]", "close"], tagged.stream

    print(f"runtime.duplex: interleave a/v, drop-don't-queue, barge-in OK "
          f"({loop.stats})")



    # HANDS-FREE. No turn_ended callback: the loop must notice the turn ended
    # from the audio alone and answer. Without this the pipeline is push-to-talk
    # wearing an always-on label.
    class _S:
        def __init__(self): self.replies = 0; self._cancelled = False
        def open_turn(self): pass
        def feed_audio(self, x): return 1
        def end_of_turn(self): pass
        def respond(self, **kw): self.replies += 1; return "answer"
        def interrupt(self): pass
        def resume_listening(self): pass
        def cancel(self): pass

    loud, hush = [0.3] * (SAMPLE_RATE // 10), [0.0] * (SAMPLE_RATE // 10)
    src = iter([loud] * 8 + [hush] * 8)
    tick = [0.0]

    def _clock():
        tick[0] += 0.05
        return tick[0]

    sess = _S()
    hf = DuplexLoop(sess, audio_source=lambda: next(src, None), clock=_clock)
    assert hf.endpoint is not None, "no turn_ended must mean hands-free"
    hs = hf.run(eos_ids=(), max_seconds=3.0, tick_s=0.0)
    for want in ("speech_start", "probably_ended", "speech_end"):
        assert want in hs.endpoint_events, f"missing {want}: {hs.endpoint_events}"
    # speculation must precede the commit -- that gap is the latency win
    assert (hs.endpoint_events.index("probably_ended")
            < hs.endpoint_events.index("speech_end"))
    assert sess.replies >= 1, "hands-free must answer with no external signal"
    print("runtime.duplex: hands-free turn detection OK "
          f"({'/'.join(hs.endpoint_events)})")

if __name__ == "__main__":
    demo()
