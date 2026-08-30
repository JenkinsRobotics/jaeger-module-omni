"""The three-way output contract — what one cognitive forward pass produces.

MiniCPM-o has two outputs: text and audio. We have three, because the avatar
reads the same intermediate the voice does. Adding the third cost almost
nothing precisely *because* the intermediate is explicit rather than hidden
states going straight to a codec.

    cognition
        |
    CognitiveOutput
        +--> text          chat, logs, transcript
        +--> spec          voice + face  (PerformanceSpec)
        +--> tool_calls    structured actions

## Why there is no tool library here

`jaeger-agent` already has one, and those tools are text-in/text-out —
modality-agnostic by construction. Duplicating them would be bloat.

They also transfer for free: the backbone here **is** gemma-4-E4B, so the chat
template, the dialect parsers and their regression tests all still apply. The
backbone cannot tell whether its input arrived as tokens or projected audio.

What is Omni-specific is the *routing*: one forward pass fanning out to three
consumers, with speech never parsed back into commands.

That last rule is load-bearing. Actions travel in `tool_calls`, never by
scraping what the character said out loud. A character that says "I'll delete
that file" as dialogue must not delete a file.

## Neural tool modules — later, and not through here

`tool_calls` is the *symbolic* path: the answer genuinely is a string or a
number, and text loses nothing. Perceptual specialists (vision, episodic
memory, motion) are a different shape — they reach cognition the same way audio
does, through a trained projector, not through a call-and-parse round trip.

    a neural tool is just another projector

See the plan doc, "Tools: two paths". Nothing is reserved for it here, because
it does not route through this object at all.

    python -m jaeger_agent_omni.core.outputs     # self-check
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from .performance_spec import PerformanceSpec


@dataclass(frozen=True)
class ToolCall:
    """One structured action. Mirrors the shape jaeger-agent's dialects emit."""

    name: str
    arguments: dict
    call_id: Optional[str] = None

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("tool call needs a name")
        if not isinstance(self.arguments, dict):
            raise ValueError(f"arguments must be a dict, got {type(self.arguments)}")


@dataclass
class CognitiveOutput:
    """One turn's worth of cognition, fanned out to three consumers.

    `generation_id` is the cancellation key. On barge-in every consumer drops
    work tagged with the stale id — that is how interruption stays coherent
    across the speech and animation slots at once.
    """

    generation_id: str
    text: str
    spec: Optional[PerformanceSpec] = None
    tool_calls: list = field(default_factory=list)

    def __post_init__(self):
        if not self.generation_id:
            raise ValueError("generation_id is required — it is the cancel key")
        if self.spec is not None and self.spec.generation_id != self.generation_id:
            raise ValueError(
                f"spec.generation_id {self.spec.generation_id!r} != "
                f"{self.generation_id!r} — a spec from another generation would "
                "survive cancellation")
        for call in self.tool_calls:
            if not isinstance(call, ToolCall):
                raise ValueError(f"tool_calls must hold ToolCall, got {type(call)}")

    @property
    def speaks(self) -> bool:
        return self.spec is not None

    @property
    def acts(self) -> bool:
        return bool(self.tool_calls)

    def to_json(self) -> str:
        d = asdict(self)
        d["spec"] = json.loads(self.spec.to_json()) if self.spec else None
        return json.dumps(d, ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "CognitiveOutput":
        d = json.loads(s)
        spec = PerformanceSpec(**{
            **d["spec"],
            "emphasis": [tuple(e) for e in d["spec"].get("emphasis", [])],
            "pauses": [tuple(p) for p in d["spec"].get("pauses", [])],
        }) if d.get("spec") else None
        return cls(generation_id=d["generation_id"], text=d["text"], spec=spec,
                   tool_calls=[ToolCall(**c) for c in d.get("tool_calls", [])])


def demo():
    from .performance_spec import neutral

    spec = neutral("the time is four", soul="mochi", generation_id="g1")
    out = CognitiveOutput(generation_id="g1", text="the time is four", spec=spec,
                          tool_calls=[ToolCall("get_time", {"tz": "UTC"}, "c1")])
    assert out.speaks and out.acts
    assert CognitiveOutput.from_json(out.to_json()) == out

    # Silent action: cognition may act without speaking.
    silent = CognitiveOutput(generation_id="g2", text="",
                             tool_calls=[ToolCall("get_time", {})])
    assert silent.acts and not silent.speaks
    assert CognitiveOutput.from_json(silent.to_json()) == silent

    # A spec from a different generation must not ride along — it would outlive
    # the cancellation that was supposed to kill it.
    try:
        CognitiveOutput(generation_id="g3", text="hi",
                        spec=neutral("hi", soul="mochi", generation_id="OTHER"))
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched generation_id must be rejected")

    for bad in (
        dict(generation_id="", text="hi"),
        dict(generation_id="g", text="hi", tool_calls=[{"name": "x"}]),
    ):
        try:
            CognitiveOutput(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should have rejected {bad}")

    for bad in (("", {}), ("  ", {}), ("ok", ["not", "a", "dict"])):
        try:
            ToolCall(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should have rejected ToolCall{bad}")

    print("outputs: three-way routing, cancel-key binding, round-trip OK")


if __name__ == "__main__":
    demo()
