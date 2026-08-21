"""PerformanceSpec — the one contract three consumers read.

Cognition emits one of these per utterance. Text output, the TTS slot, and the
animation slot each read the same object. Nothing else crosses between them.

    mind
     |
    PerformanceSpec
     +--> text            chat, logs, tools
     +--> tts slot        voice + style
     +--> animation slot  face + gaze

It carries INTENT, never rig or voice specifics. What an angry mochi sounds and
looks like lives in the Soul binding, not here. Adding a character is authoring
a binding; it does not touch cognition.

Today this is authored (by cognition, or a lookup). At Generation 2 a small
network predicts it. The consumers never learn the difference — that is the
whole point of pinning it now.

    python research/performance_spec.py     # self-check
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

# Ekman-6 + neutral. Not chosen for scientific completeness — chosen because
# TTS instruct modes and ARKit blendshape presets both already speak this set.
EMOTIONS = ("neutral", "happy", "sad", "angry", "surprised", "afraid", "disgusted")
DELIVERIES = ("normal", "whisper", "shout", "sung")


@dataclass
class PerformanceSpec:
    generation_id: str            # cancellation key — invalidate on barge-in
    text: str
    soul: str                     # resolves voice + rig + style bindings
    delivery: str = "normal"
    emotion: str = "neutral"
    intensity: float = 0.5        # 0..1
    pace: float = 1.0             # 0.5..2.0, multiplier on the Soul's baseline
    energy: float = 0.5           # 0..1
    emphasis: list = field(default_factory=list)   # [(start, end, strength)]
    pauses: list = field(default_factory=list)     # [(after_char, ms)]
    gaze: str = "camera"

    def __post_init__(self):
        self.validate()

    def validate(self) -> "PerformanceSpec":
        if self.emotion not in EMOTIONS:
            raise ValueError(f"emotion {self.emotion!r} not in {EMOTIONS}")
        if self.delivery not in DELIVERIES:
            raise ValueError(f"delivery {self.delivery!r} not in {DELIVERIES}")
        for name, lo, hi in (("intensity", 0.0, 1.0), ("energy", 0.0, 1.0),
                             ("pace", 0.5, 2.0)):
            v = getattr(self, name)
            if not lo <= v <= hi:
                raise ValueError(f"{name}={v} outside [{lo}, {hi}]")
        n = len(self.text)
        for start, end, strength in self.emphasis:
            if not 0 <= start < end <= n:
                raise ValueError(f"emphasis span ({start}, {end}) outside text len {n}")
            if not 0.0 <= strength <= 1.0:
                raise ValueError(f"emphasis strength {strength} outside [0, 1]")
        for after_char, ms in self.pauses:
            if not 0 <= after_char <= n:
                raise ValueError(f"pause at {after_char} outside text len {n}")
            if ms < 0:
                raise ValueError(f"pause {ms}ms is negative")
        return self

    # Wire format is plain JSON — it crosses a slot boundary, and a dict of
    # floats is debuggable in a way a tensor is not.
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "PerformanceSpec":
        d = json.loads(s)
        d["emphasis"] = [tuple(e) for e in d.get("emphasis", [])]
        d["pauses"] = [tuple(p) for p in d.get("pauses", [])]
        return cls(**d)

    def emphasis_words(self) -> list:
        """The spans as substrings — what a TTS instruct prompt wants."""
        return [self.text[s:e] for s, e, _ in self.emphasis]


def neutral(text: str, soul: str, generation_id: str) -> PerformanceSpec:
    """The lookup-table default. Cognition overrides what it cares about."""
    return PerformanceSpec(generation_id=generation_id, text=text, soul=soul)


def demo():
    spec = PerformanceSpec(
        generation_id="g1",
        text="I told you that would happen",
        soul="mochi",
        emotion="angry",
        intensity=0.8,
        pace=1.2,
        emphasis=[(2, 6, 0.9)],
        pauses=[(6, 250)],
    )
    assert spec.emphasis_words() == ["told"], spec.emphasis_words()

    # Round-trips through the wire without loss — it crosses a slot boundary.
    again = PerformanceSpec.from_json(spec.to_json())
    assert again == spec, (again, spec)

    # Validation is at the trust boundary, not decorative: bad input from
    # cognition must fail loudly rather than drive a rig to a garbage pose.
    for bad in (
        dict(emotion="furious"),                  # not in the agreed set
        dict(delivery="rapping"),
        dict(intensity=1.5),
        dict(pace=0.1),
        dict(emphasis=[(2, 200, 0.5)]),           # span past end of text
        dict(emphasis=[(2, 6, 9.0)]),             # strength out of range
        dict(pauses=[(3, -10)]),                  # negative pause
    ):
        try:
            PerformanceSpec(generation_id="g", text="hello", soul="mochi", **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should have rejected {bad}")

    # An empty span is not a span.
    try:
        PerformanceSpec(generation_id="g", text="hello", soul="m", emphasis=[(2, 2, 0.5)])
    except ValueError:
        pass
    else:
        raise AssertionError("zero-width emphasis should be rejected")

    d = neutral("hello", soul="mochi", generation_id="g2")
    assert d.emotion == "neutral" and d.pace == 1.0

    print(f"performance_spec: {len(EMOTIONS)} emotions, {len(DELIVERIES)} deliveries")
    print("performance_spec: round-trip, spans, validation OK")


if __name__ == "__main__":
    demo()
