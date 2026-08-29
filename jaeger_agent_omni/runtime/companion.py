"""Audio-native admission and conversation state for an always-on companion.

The microphone path must not decode speech to text just to decide whether the
agent should answer.  Instead, the same Gemma pass that receives native audio
emits one small control envelope before any user-visible response::

    DECISION: RESPOND
    RESPONSE: ...

or::

    DECISION: LISTEN
    REASON: not_addressed

This is a neural semantic decision, not a wake-word string search.  The parser
is deliberately fail-closed: malformed output is never spoken by an always-on
agent.  Energy endpointing remains a cheap acoustic proposal mechanism; it is
not treated as evidence that the sound was speech or that Jaeger was addressed.

    python -m jaeger_agent_omni.runtime.companion
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field


RESPOND = "respond"
LISTEN = "listen"
VALID_ACTIONS = (RESPOND, LISTEN)

COMPANION_SYSTEM = """
You are Jaeger, an always-on companion in a room that may contain several
people, media playback, and environmental sounds. For every audio turn, first
decide whether Jaeger should answer. Return exactly one of these forms:

DECISION: RESPOND
RESPONSE: <a concise, natural answer>

DECISION: LISTEN
REASON: <not_addressed, non_speech, incomplete, self_speech, or wait>

RESPOND when Jaeger is directly addressed by name or clear conversational
intent. During an active follow-up window, also respond to a natural follow-up
from the person already conversing with Jaeger. A different speaker does not
inherit that follow-up permission unless they directly address Jaeger. LISTEN
to conversations between other people, television or recorded speech, random
environmental sounds, Jaeger's own voice, and incomplete thoughts where the
speaker plainly intends to continue. Never put an answer in a LISTEN decision.
The companion-state annotation is trusted runtime evidence; when it says known
Jaeger playback overlapped the turn, do not mistake that playback for a user.
Do not repeat these rules or describe the control format.
""".strip()


@dataclass(frozen=True)
class CompanionDecision:
    """One semantic admission result generated from native audio."""

    action: str
    response: str = ""
    reason: str = ""
    valid: bool = True
    raw: str = ""

    @property
    def should_respond(self) -> bool:
        return self.valid and self.action == RESPOND and bool(
            self.response.strip())

    def as_dict(self) -> dict:
        return asdict(self)


def parse_decision(text: str) -> CompanionDecision:
    """Parse Gemma's control envelope, suppressing malformed output.

    The parser accepts harmless whitespace and case variation, but it does not
    infer an action from free-form prose.  Precision matters more than recall
    for an always-on room agent: an uncertain turn should stay silent.
    """
    raw = str(text or "").strip()
    match = re.search(
        r"(?:^|\n)\s*DECISION\s*:\s*(RESPOND|LISTEN)\b",
        raw, flags=re.IGNORECASE)
    if not match:
        return CompanionDecision(
            LISTEN, reason="malformed_control", valid=False, raw=raw)
    action = match.group(1).lower()
    if action == RESPOND:
        response = re.search(
            r"(?:^|\n)\s*RESPONSE\s*:\s*(.*)$", raw,
            flags=re.IGNORECASE | re.DOTALL)
        content = response.group(1).strip() if response else ""
        if not content:
            return CompanionDecision(
                LISTEN, reason="missing_response", valid=False, raw=raw)
        return CompanionDecision(RESPOND, response=content, raw=raw)
    reason = re.search(
        r"(?:^|\n)\s*REASON\s*:\s*([^\n]+)", raw,
        flags=re.IGNORECASE)
    return CompanionDecision(
        LISTEN, reason=(reason.group(1).strip() if reason else "unspecified"),
        raw=raw)


def streaming_response(text: str) -> str:
    """Return only the safe-to-speak portion of an in-progress envelope."""
    raw = str(text or "")
    decision = re.search(
        r"(?:^|\n)\s*DECISION\s*:\s*(RESPOND|LISTEN)\b",
        raw, flags=re.IGNORECASE)
    if not decision or decision.group(1).lower() != RESPOND:
        return ""
    response = re.search(
        r"(?:^|\n)\s*RESPONSE\s*:\s*(.*)$", raw,
        flags=re.IGNORECASE | re.DOTALL)
    return response.group(1) if response else ""


def followup_window(turns: int, agent_asked: bool) -> float:
    """Conversation window in seconds, capped at 45 seconds.

    A new exchange gets 15 seconds.  A question from Jaeger gets 27 seconds,
    and a demonstrated multi-turn exchange gets 36 seconds.  This state is
    advisory context for the neural decision, never an automatic admission.
    """
    turns = max(0, int(turns))
    if turns >= 3:
        return 36.0
    if agent_asked:
        return 27.0
    return 15.0


@dataclass
class CompanionState:
    """Small, auditable state around the neural RESPOND/LISTEN decision."""

    clock: object = time.monotonic
    exchange_turns: int = 0
    engaged_until: float = 0.0
    speech_started_at: float | None = None
    current_speaker: str | None = None
    engaged_speaker: str | None = None
    decisions: list[dict] = field(default_factory=list)

    def speech_started(self, at: float | None = None) -> None:
        self.speech_started_at = float(self.clock() if at is None else at)

    def followup_was_open_at_speech_start(self) -> bool:
        at = self.speech_started_at
        return at is not None and at <= self.engaged_until

    def turn_metadata(self, speaker: str | None = None,
                      now: float | None = None,
                      self_speech_overlap: bool = False) -> str:
        """Non-transcript metadata appended after the native audio turn."""
        now = float(self.clock() if now is None else now)
        if self.engaged_until and now > self.engaged_until:
            self.exchange_turns = 0
        self.current_speaker = speaker
        followup = self.followup_was_open_at_speech_start()
        state = "engaged_followup" if followup else "ambient"
        who = f"; speaker={speaker}" if speaker else ""
        if not speaker or not self.engaged_speaker:
            continuity = "unknown"
        else:
            continuity = "same" if speaker == self.engaged_speaker else "different"
        return (
            f" [companion_state={state}; followup_open_at_speech_start="
            f"{'yes' if followup else 'no'}; speaker_continuity={continuity}"
            f"; known_jaeger_playback_overlap="
            f"{'yes' if self_speech_overlap else 'no'}"
            f"{who}]"
        )

    def observe(self, decision: CompanionDecision,
                at: float | None = None) -> None:
        now = float(self.clock() if at is None else at)
        record = decision.as_dict()
        record.update({
            "monotonic_s": now,
            "speaker": self.current_speaker,
            "followup_open_at_speech_start":
                self.followup_was_open_at_speech_start(),
        })
        self.decisions.append(record)
        if decision.should_respond:
            self.exchange_turns += 1
            if self.current_speaker:
                self.engaged_speaker = self.current_speaker
            asked = decision.response.rstrip().endswith("?")
            self.engaged_until = now + min(
                45.0, followup_window(self.exchange_turns, asked))
        elif now > self.engaged_until:
            self.exchange_turns = 0
            self.engaged_speaker = None
        self.speech_started_at = None


def system_prompt(extra: str = "") -> str:
    extra = str(extra or "").strip()
    return COMPANION_SYSTEM + (f"\n\nAdditional behavior: {extra}" if extra else "")


def demo() -> None:
    good = parse_decision(
        "DECISION: RESPOND\nRESPONSE: Sure, I can help with that.")
    assert good.should_respond and good.response.startswith("Sure")
    quiet = parse_decision("DECISION: LISTEN\nREASON: not_addressed")
    assert not quiet.should_respond and quiet.valid
    malformed = parse_decision("I think you should try again.")
    assert not malformed.should_respond and not malformed.valid
    assert streaming_response(
        "DECISION: RESPOND\nRESPONSE: Hello there") == "Hello there"
    assert streaming_response("DECISION: LISTEN\nREASON: non_speech") == ""
    assert [followup_window(x, False) for x in (0, 1, 3)] == [15, 15, 36]
    assert followup_window(1, True) == 27

    now = [100.0]
    state = CompanionState(clock=lambda: now[0])
    state.speech_started()
    assert "ambient" in state.turn_metadata("A")
    state.observe(good)
    assert state.engaged_speaker == "A"
    assert state.engaged_until == 115.0
    now[0] = 114.0
    state.speech_started()
    now[0] = 130.0  # ASR/generation may finish after the deadline.
    assert state.followup_was_open_at_speech_start()
    metadata = state.turn_metadata("A")
    assert ("engaged_followup" in metadata and
            "speaker_continuity=same" in metadata)
    state.observe(quiet)
    assert len(state.decisions) == 2
    print("runtime.companion: fail-closed admission, safe streaming, "
          "start-time follow-up windows OK")


if __name__ == "__main__":
    demo()
