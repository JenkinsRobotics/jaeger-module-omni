"""JaegerAgent's tool loop over Jaeger Omni's native multimodal KV cache.

Audio and vision are already embedded in the session before this loop starts.
The adapter therefore ignores its internal native-turn sentinel, lets Gemma
generate directly from those embeddings, dispatches structured tool calls,
and appends only trusted tool results/control messages to the same cache.
There is no audio transcript handoff.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .adapters.base import ProviderAdapter
from .dialects import extract_tool_calls, strip_reasoning_channels
from .loop.jaeger_agent import JaegerAgent
from .schemas.message_types import Message
from jaeger_agent_omni.core.skill_registry import toolset_scoping
from ._vendor.jaeger_os.core.tools.tool_registry import get_tools
from ._vendor.jaeger_os.core.tools.tool_schema import ToolDef


NATIVE_MULTIMODAL_TURN = (
    "[native multimodal turn already embedded in the neural session]"
)

# These copied tools would reintroduce a decoded ASR cascade or a parallel TTS
# path. Omni hears through its selected neural ear and speaks through its
# generation-id-bound output router, so neither belongs in its action catalog.
OMNI_BLOCKED_TOOLS = frozenset({"listen", "text_to_speech"})

# The audio model pays attention over every schema token for every generated
# token.  Jaeger Agent's desktop-oriented 18-tool CORE made a five-token voice
# answer take ~7.2 seconds.  Keep the neural voice bootstrap deliberately
# small; all 95 base safe tools (plus skill-provided tools) remain registered,
# and ``load_tools`` can expose a specialist set without rebuilding the session.
OMNI_CORE_TOOLS = frozenset({
    "calculate", "get_time", "load_tools", "list_skills",
    "memory", "recall", "use_skill", "web_search",
})

_CALL_BLOCKS = (
    re.compile(r"<\|tool_call\|>\s*.*?\s*<\|/tool_call\|>", re.DOTALL),
    re.compile(r"<\|tool_call>\s*call:[^<]*<tool_call\|>", re.DOTALL),
    re.compile(r"<tool_call>\s*.*?\s*</tool_call>", re.DOTALL),
)
_BARE_GEMMA_CALL = re.compile(
    r"^\s*call:[A-Za-z_][A-Za-z0-9_]*\s*\{.*\}\s*$", re.DOTALL,
)


def omni_tool_visible(name: str) -> bool:
    """Omni's context-tight visibility rule, independent of process env."""
    if name in OMNI_BLOCKED_TOOLS:
        return False
    if name in OMNI_CORE_TOOLS:
        return True
    active = toolset_scoping.active_toolset_names() - {"core"}
    if any(name in toolset_scoping.TOOLSETS.get(group, ()) for group in active):
        return True
    if any(name in members for group, members in
           toolset_scoping._SKILL_TOOLSETS.items() if group in active):
        return True
    classified = (
        name in toolset_scoping.CORE
        or any(name in members for members in toolset_scoping.TOOLSETS.values())
        or any(name in members
               for members in toolset_scoping._SKILL_TOOLSETS.values())
    )
    return not classified


def omni_tools(*, visible_only: bool = True) -> list[ToolDef]:
    """Return available, Omni-safe tools from the copied registry."""
    result = []
    for tool in get_tools():
        if tool.name in OMNI_BLOCKED_TOOLS:
            continue
        if visible_only and not omni_tool_visible(tool.name):
            continue
        if hasattr(tool, "is_available") and not tool.is_available():
            continue
        result.append(tool)
    return result


def _function_schema(tool: ToolDef) -> dict[str, Any]:
    schema = tool.to_openai_schema()
    function = dict(schema.get("function") or {})
    function["description"] = " ".join(
        str(function.get("description") or "").split())[:96]

    # This is a generation hint, not the dispatch validator: ToolDef keeps the
    # authoritative Pydantic schema.  Retain argument names, primitive types,
    # requiredness and small enums while dropping titles/descriptions/defaults
    # that dominate a voice model's attention prefix.
    source = function.get("parameters") or {}
    properties: dict[str, Any] = {}
    for name, value in (source.get("properties") or {}).items():
        value = value if isinstance(value, dict) else {}
        prop: dict[str, Any] = {"type": value.get("type", "string")}
        enum = value.get("enum")
        if isinstance(enum, list) and len(enum) <= 8:
            prop["enum"] = enum
        items = value.get("items")
        if isinstance(items, dict) and items.get("type"):
            prop["items"] = {"type": items["type"]}
        properties[name] = prop
    function["parameters"] = {
        "type": "object",
        "properties": properties,
        "required": source.get("required") or [],
    }
    return function


def _schema_block(tools: Iterable[ToolDef]) -> str:
    values = [_function_schema(tool) for tool in tools]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class AgenticPreparation:
    """System prompt and initial schema identity fixed before audio starts."""

    system: str
    initial_tools: tuple[ToolDef, ...]


def prepare_agentic_system(base_system: str) -> AgenticPreparation:
    """Attach the small initial tool surface to an existing Omni prompt."""
    # Importing the tool package registers the copied batteries exactly once.
    from jaeger_agent_omni import tools as _tools  # noqa: F401

    toolset_scoping.reset_toolsets()
    initial = tuple(omni_tools())
    catalog = sorted(toolset_scoping.all_toolsets())
    instructions = (
        "\n\nAGENTIC RUNTIME\n"
        "The user's audio/vision is already present as native neural state. "
        "Never call a microphone transcription tool and never request a "
        "transcript. Use tools only when an action or external fact is needed. "
        "Emit a Gemma tool call exactly as "
        "<|tool_call>call:TOOL_NAME{JSON_ARGUMENTS}<tool_call|> and do not "
        "speak the markup. Tool results will arrive as trusted runtime control "
        "data. For ordinary conversation, answer normally. The available "
        "toolsets can be widened with load_tools.\n"
        f"TOOLSET NAMES: {','.join(catalog)}. Call load_tools with no name "
        "for descriptions.\n"
        f"ACTIVE TOOL SCHEMAS: {_schema_block(initial)}"
    )
    return AgenticPreparation(
        system=(base_system or "").rstrip() + instructions,
        initial_tools=initial,
    )


class OmniSessionAdapter(ProviderAdapter):
    """Incremental agent adapter for an already-prefilled Omni session."""

    name = "jaeger-omni-native"

    def __init__(self, session: Any, *, eos_ids: Iterable[int] = (),
                 max_tokens: int = 128,
                 initial_tools: Iterable[ToolDef] = ()) -> None:
        if not callable(getattr(session, "respond", None)):
            raise TypeError("Omni session needs respond()")
        if not callable(getattr(session, "continue_agentic", None)):
            raise TypeError(
                "selected session cannot append agentic tool results")
        self.session = session
        # Capture before live.py installs its output-streaming wrapper.
        self._native_respond = session.respond
        self.eos_ids = tuple(int(value) for value in eos_ids)
        self.max_tokens = int(max_tokens)
        self._message_cursor = 0
        self._known_schemas = {tool.name for tool in initial_tools}
        self._token_observer = None
        self.last_raw_response = ""

    def format_messages(self, messages: list[Message], tools: list[ToolDef],
                        system: str) -> dict[str, Any]:
        return {"messages": list(messages), "tools": list(tools),
                "system": system}

    @staticmethod
    def _control_message(messages: list[Message]) -> str:
        rows = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "tool":
                rows.append(json.dumps({
                    "tool_result": {
                        "id": message.get("tool_call_id"),
                        "name": message.get("name"),
                        "content": content or "",
                    }
                }, ensure_ascii=False, separators=(",", ":")))
            elif role == "user" and content != NATIVE_MULTIMODAL_TURN:
                # Loop backstops and steering are trusted agent-control text,
                # not decoded microphone content.
                rows.append(str(content or ""))
        return "\n".join(row for row in rows if row)

    def call(self, formatted: dict[str, Any],
             interrupt_event: threading.Event, **kwargs: Any) -> str:
        messages = formatted["messages"]
        pending = messages[self._message_cursor:]
        self._message_cursor = len(messages)
        control = self._control_message(pending)

        tools = formatted.get("tools") or []
        new_tools = [tool for tool in tools
                     if tool.name not in self._known_schemas]
        if new_tools:
            control = (control + "\nUPDATED TOOL SCHEMAS: " +
                       _schema_block(new_tools)).strip()
            self._known_schemas.update(tool.name for tool in new_tools)
        if control:
            self.session.continue_agentic(control)

        observer = self._token_observer

        def token_seen(token_id: int, count: int) -> None:
            if interrupt_event.is_set():
                self.session.cancel()
            if observer is not None:
                observer(token_id, count)

        self.last_raw_response = self._native_respond(
            max_tokens=self.max_tokens,
            eos_ids=self.eos_ids,
            on_token=token_seen,
        )
        return self.last_raw_response

    def parse_response(self, raw: Any) -> Message:
        text = strip_reasoning_channels(str(raw or ""))
        calls = extract_tool_calls(text)
        # The stock-Whisper session decodes with ``skip_special_tokens`` so
        # Gemma's <|tool_call> ... <tool_call|> boundary tokens can disappear
        # while the semantic body survives. Recover only when the *entire*
        # response is the native ``call:name{args}`` form; prose containing
        # ``call:`` remains ordinary user-facing text.
        bare_call = bool(_BARE_GEMMA_CALL.fullmatch(text))
        if not calls and bare_call:
            calls = extract_tool_calls(
                f"<|tool_call>{text.strip()}<tool_call|>"
            )
        cleaned = text
        for pattern in _CALL_BLOCKS:
            cleaned = pattern.sub("", cleaned)
        if calls and bare_call:
            cleaned = ""
        message: Message = {
            "role": "assistant",
            "content": cleaned.strip() or None,
        }
        if calls:
            message["tool_calls"] = calls
        return message

    def supports(self, feature: str) -> bool:  # noqa: ARG002
        return False


class OmniAgentRuntime:
    """One native Omni session plus the copied Jaeger agent loop."""

    def __init__(self, session: Any, preparation: AgenticPreparation,
                 *, eos_ids: Iterable[int] = (), max_tokens: int = 128,
                 max_iterations: int = 12,
                 workspace: str | Path | None = None,
                 load_skills: bool = True) -> None:
        from .workspace import ensure_bound

        layout = ensure_bound(workspace)
        if load_skills:
            from .runtime import _load_skills
            self.skills = _load_skills(layout)
        else:
            self.skills = None
        self.adapter = OmniSessionAdapter(
            session, eos_ids=eos_ids, max_tokens=max_tokens,
            initial_tools=preparation.initial_tools)
        self.agent = JaegerAgent(
            self.adapter,
            system_prompt=preparation.system,
            max_iterations=max_iterations,
            blocked_tools=OMNI_BLOCKED_TOOLS,
            tool_visibility=omni_tool_visible,
            # Native user intent is opaque to the inherited text-based
            # multi-step detector. Gemma must see each result to finish actions
            # and emit the companion control envelope when one is required.
            skip_final_tools=frozenset(),
        )
        self.session = session
        self._native_cancel = session.cancel
        # live.py uses this to buffer model tokens until it knows they are a
        # final answer rather than executable markup.
        self.session.agentic_buffering = True

    def respond_native(self, max_tokens: int = 128, eos_ids=(),
                       on_token=None, **kwargs: Any) -> str:  # noqa: ARG002
        self.adapter.max_tokens = int(max_tokens)
        self.adapter.eos_ids = tuple(int(value) for value in eos_ids)
        self.adapter._token_observer = on_token
        try:
            return self.agent.run_turn(NATIVE_MULTIMODAL_TURN)
        finally:
            self.adapter._token_observer = None

    def interrupt(self) -> None:
        self.agent.interrupt()
        self._native_cancel()


__all__ = [
    "AgenticPreparation", "NATIVE_MULTIMODAL_TURN", "OMNI_BLOCKED_TOOLS",
    "OMNI_CORE_TOOLS",
    "OmniAgentRuntime", "OmniSessionAdapter", "omni_tool_visible",
    "omni_tools", "prepare_agentic_system",
]
