"""When did the user stop talking? The last piece of hands-free operation.

Without this the model needs to be told a turn ended. With it, the pipeline is
genuinely always-on: audio streams continuously and the turn closes itself.

The design is taken from the parallel agent's measured v5 result, which beat
MiniCPM's own turn-taking by 2.28x. Worth being precise about WHY, because it
is not a better encoder -- MiniCPM emits a LISTEN/SPEAK decision per 1-second
frame, so it cannot notice a turn ended until the next frame boundary. Its
2196 ms was ~1670 ms of waiting plus ~525 ms of compute. Silence detection at
10 ms resolution collapses the first term.

    speech ─────────────┐
                        │◄── quiet_ms accumulates
                        │
         probably_ended ─┤  SEMANTIC_MIN_MS   speculate the reply here
                        │
            speech_end ──┤  SILENCE_MS        commit; reuse the speculation
                        │
      acoustic guard ────┘  GUARD_MS          only if nothing looked final

Speculating at `probably_ended` is where the latency actually goes: by the
time the endpoint confirms, the reply is already in hand. Their run recorded
`stt_reused_speculative_finals: 1` -- the speculation was used, not wasted.

FRAME ROUNDING. Thresholds must round UP to whole capture frames. Their first
run had a 500 ms threshold silently become 480 ms against 20 ms frames, which
is invisible in every log and quietly shortens every turn.

    python -m jaeger_agent_omni.runtime.endpoint      # self-check, no models
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SAMPLE_RATE = 16000
SILENCE_MS = 510.0        # quiet needed to declare the turn over
SEMANTIC_MIN_MS = 500.0   # earliest a speculative reply may start
GUARD_MS = 800.0          # never commit before this if speech was ragged
SPEECH_RMS = 0.012        # above this a frame counts as speech


def frames_up(ms: float, frame_ms: float) -> int:
    """Round a threshold UP to whole frames. Rounding DOWN is the bug: a
    500 ms threshold against 20 ms frames became 480 ms and shortened every
    turn, with nothing in any log to show for it."""
    return max(1, math.ceil(ms / frame_ms))


@dataclass
class Endpoint:
    """Feed it frames, ask it what happened. No model, no allocation."""

    frame_ms: float = 20.0
    silence_ms: float = SILENCE_MS
    semantic_min_ms: float = SEMANTIC_MIN_MS
    guard_ms: float = GUARD_MS
    speech_rms: float = SPEECH_RMS

    speaking: bool = False
    quiet_frames: int = 0
    speech_frames: int = 0
    speculated: bool = False
    ended: bool = False
    events: list = field(default_factory=list)

    def __post_init__(self):
        self.n_silence = frames_up(self.silence_ms, self.frame_ms)
        self.n_semantic = frames_up(self.semantic_min_ms, self.frame_ms)
        self.n_guard = frames_up(self.guard_ms, self.frame_ms)

    def reset(self):
        self.speaking = self.ended = self.speculated = False
        self.quiet_frames = self.speech_frames = 0
        self.events = []

    @staticmethod
    def rms(frame) -> float:
        n = len(frame)
        if not n:
            return 0.0
        return (sum(float(x) * float(x) for x in frame) / n) ** 0.5

    def push(self, frame) -> str | None:
        """One frame in; returns an event name the first time each fires.

        'speech_start' | 'probably_ended' | 'speech_end' | None
        """
        if self.ended:
            return None
        loud = self.rms(frame) >= self.speech_rms
        if loud:
            self.speech_frames += 1
            self.quiet_frames = 0
            if not self.speaking:
                self.speaking = True
                self.events.append("speech_start")
                return "speech_start"
            # Speech resumed after a pause: the speculation is now stale. This
            # is the barge-in-on-yourself case -- someone pausing mid-sentence
            # must not have their turn closed out from under them.
            self.speculated = False
            return None
        if not self.speaking:
            return None                      # silence before any speech
        self.quiet_frames += 1

        if not self.speculated and self.quiet_frames >= self.n_semantic:
            self.speculated = True
            self.events.append("probably_ended")
            return "probably_ended"          # start the reply NOW, speculatively
        # The guard only binds when speech was too short to trust -- a cough
        # or a syllable should not close a turn just because quiet followed.
        long_enough = self.speech_frames >= self.n_guard // 4
        if self.quiet_frames >= self.n_silence and long_enough:
            self.ended = True
            self.events.append("speech_end")
            return "speech_end"
        return None

    @property
    def quiet_ms(self) -> float:
        return self.quiet_frames * self.frame_ms


def demo() -> None:
    import random

    fm = 20.0
    ep = Endpoint(frame_ms=fm)
    n = int(SAMPLE_RATE * fm / 1000)
    loud = [0.3] * n
    quiet = [0.0] * n

    # rounding UP, the bug that silently shortened every turn
    assert frames_up(500.0, 20.0) == 25, frames_up(500.0, 20.0)
    assert frames_up(510.0, 20.0) == 26          # 25 frames = 500 ms, not enough
    assert frames_up(1.0, 20.0) == 1             # never zero frames

    assert ep.push(quiet) is None                # silence before speech: nothing
    assert ep.push(loud) == "speech_start"
    for _ in range(40):                          # 800 ms of speech
        ep.push(loud)

    got = [e for e in (ep.push(quiet) for _ in range(26)) if e]
    assert got == ["probably_ended", "speech_end"], got
    # speculation fires BEFORE the commit -- that is the whole latency win
    assert ep.events.index("probably_ended") < ep.events.index("speech_end")

    # a pause mid-sentence must not end the turn, and must invalidate the
    # speculative reply rather than let a stale one be committed
    ep2 = Endpoint(frame_ms=fm)
    ep2.push(loud)
    for _ in range(40):
        ep2.push(loud)
    for _ in range(25):
        ep2.push(quiet)
    assert ep2.speculated and not ep2.ended
    ep2.push(loud)
    assert not ep2.speculated, "resumed speech must invalidate the speculation"
    assert not ep2.ended

    # a cough: too little speech to close a turn on
    ep3 = Endpoint(frame_ms=fm)
    ep3.push(loud)
    for _ in range(60):
        ep3.push(quiet)
    assert not ep3.ended, "a single loud frame must not constitute a turn"

    # rms is real, not a stub
    assert Endpoint.rms([0.0] * 10) == 0.0
    assert abs(Endpoint.rms([0.5] * 10) - 0.5) < 1e-9
    assert Endpoint.rms([]) == 0.0
    random.seed(0)
    print(f"endpoint: rounding, speculate-then-commit, resume, guard OK "
          f"(silence {SILENCE_MS:.0f}ms = {ep.n_silence} frames)")


if __name__ == "__main__":
    demo()
