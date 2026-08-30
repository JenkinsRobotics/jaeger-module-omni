"""Real-time interaction events separated from conversational policy."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass


BARGE_POLICIES = ("stop", "continue")


@dataclass(frozen=True)
class InteractionEvent:
    """One detected fact and the policy action chosen for it."""

    sequence: int
    kind: str
    monotonic_s: float
    stage: str
    action: str
    evidence: dict[str, float]

    def as_dict(self) -> dict:
        return asdict(self)


class InteractionPolicy:
    """Convert notifications to actions without changing their detection."""

    def __init__(self, barge_in: str = "stop"):
        if barge_in not in BARGE_POLICIES:
            raise ValueError(
                f"barge-in policy must be one of {BARGE_POLICIES}, got "
                f"{barge_in!r}")
        self.barge_in = barge_in
        self.events: list[InteractionEvent] = []

    def speech_during_output(self, stage: str, **evidence: float) -> str:
        """Record double-talk and return ``stop`` or ``continue``."""
        action = self.barge_in
        self.events.append(InteractionEvent(
            sequence=len(self.events) + 1,
            kind="speech_during_agent_output",
            monotonic_s=time.monotonic(),
            stage=str(stage),
            action=action,
            evidence={key: float(value) for key, value in evidence.items()},
        ))
        return action

    @property
    def stop_count(self) -> int:
        return sum(event.action == "stop" for event in self.events)

    def as_dicts(self) -> list[dict]:
        return [event.as_dict() for event in self.events]


def demo() -> None:
    stop = InteractionPolicy("stop")
    keep = InteractionPolicy("continue")
    assert stop.speech_during_output("playback", clean_rms=0.1) == "stop"
    assert keep.speech_during_output("generation", clean_rms=0.2) == \
        "continue"
    assert stop.stop_count == 1 and keep.stop_count == 0
    assert keep.as_dicts()[0]["kind"] == "speech_during_agent_output"
    print("runtime.interaction: notification/action split and policies OK")


if __name__ == "__main__":
    demo()
