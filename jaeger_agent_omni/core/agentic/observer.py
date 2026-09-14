"""Safe, UI-ready observability for the JaegerAgent execution loop.

This surface reports observable control flow: model iterations, structured
tool plans, dispatches, results, liveness, and interrupts.  The native runtime
does not request Gemma's optional thinking channel, and this observer never
turns opaque callback state into a purported chain-of-thought transcript.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .loop.callbacks import AgentCallbacks


_SENSITIVE_PARTS = (
    "authorization", "cookie", "credential", "password", "secret", "token",
    "api_key", "apikey", "private_key",
)


@dataclass(frozen=True)
class AgenticEvent:
    """One sanitized execution fact suitable for logs and operator UI."""

    kind: str
    title: str
    detail: str = ""
    phase: str = ""
    ok: bool = True
    elapsed_s: float = 0.0
    iteration: int = 0
    at_monotonic_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sensitive(key: Any) -> bool:
    normalized = str(key or "").lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_PARTS)


def _sanitize(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if _sensitive(key):
        return "<redacted>"
    if depth >= 3:
        return "…"
    if isinstance(value, dict):
        return {
            str(k): _sanitize(v, key=str(k), depth=depth + 1)
            for k, v in list(value.items())[:12]
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(v, depth=depth + 1) for v in list(value)[:8]]
    if isinstance(value, str):
        compact = " ".join(value.split())
        return compact if len(compact) <= 160 else compact[:159] + "…"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__}>"


def safe_preview(value: Any, limit: int = 260) -> str:
    """Return a bounded, secret-redacted JSON preview."""
    try:
        text = json.dumps(_sanitize(value), ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 — observability cannot break execution
        text = f"<{type(value).__name__}>"
    return text if len(text) <= limit else text[: limit - 1] + "…"


class AgenticObserver:
    """Translate low-level agent callbacks into sanitized execution events."""

    def __init__(self, emit: Callable[[AgenticEvent], None],
                 clock=time.monotonic) -> None:
        self._emit_target = emit
        self._clock = clock
        self._lock = threading.Lock()
        self._last_heartbeat_second = -1
        self._last_activity_at = -1000.0

    def emit(self, kind: str, title: str, detail: str = "", *,
             phase: str = "", ok: bool = True, elapsed_s: float = 0.0,
             iteration: int = 0) -> None:
        event = AgenticEvent(
            kind=kind, title=title, detail=detail, phase=phase, ok=ok,
            elapsed_s=round(float(elapsed_s or 0.0), 3),
            iteration=int(iteration or 0), at_monotonic_s=self._clock(),
        )
        try:
            self._emit_target(event)
        except Exception:  # noqa: BLE001 — observer never breaks the turn
            pass

    def callbacks(self) -> AgentCallbacks:
        return AgentCallbacks(
            tool_progress=self._tool_progress,
            thinking=self._thinking,
            step=self._step,
            interrupt=self._interrupt,
            tool_done=self._tool_done,
            heartbeat=self._heartbeat,
        )

    def _tool_progress(self, name: str, phase: str, data: Any) -> None:
        if phase == "start":
            self.emit("tool", f"Calling {name}", safe_preview(data),
                      phase="start")

    def _tool_done(self, name: str, args: dict[str, Any], result: Any,
                   ok: bool, error: str | None, elapsed_s: float) -> None:
        detail = safe_preview(result if ok else {"error": error, "result": result})
        self.emit("tool", f"{name} {'completed' if ok else 'failed'}", detail,
                  phase="done", ok=ok, elapsed_s=elapsed_s)

    def _step(self, index: int, message: dict[str, Any]) -> None:
        calls = message.get("tool_calls") or []
        if calls:
            for call in calls:
                name = str(call.get("name") or "unknown")
                args = call.get("arguments") or {}
                self.emit(
                    "plan", f"Plan step {index}: use {name}",
                    safe_preview(args), phase="selected", iteration=index,
                )
            return
        self.emit(
            "step", f"Iteration {index}: response candidate",
            "No tool call selected; preparing the user-visible response.",
            phase="complete", iteration=index,
        )

    def _thinking(self, _opaque_state: str) -> None:
        # Treat this callback as an opaque liveness/recovery signal. The native
        # Gemma runtime does not enable a separate thinking stream, so do not
        # label this state as model reasoning or forward its payload.
        now = self._clock()
        with self._lock:
            if now - self._last_activity_at < 1.0:
                return
            self._last_activity_at = now
        self.emit(
            "activity", "Agent recovery/activity",
            "Observable control event; no separate Gemma reasoning stream "
            "is enabled.",
            phase="active",
        )

    def _heartbeat(self, elapsed_s: float) -> None:
        second = int(elapsed_s)
        with self._lock:
            if second < 1 or second == self._last_heartbeat_second:
                return
            self._last_heartbeat_second = second
        self.emit("heartbeat", "Model working", f"{elapsed_s:.1f}s elapsed",
                  phase="active", elapsed_s=elapsed_s)

    def _interrupt(self) -> None:
        self.emit("interrupt", "Agentic turn interrupted",
                  "Generation and pending tool work were cancelled.",
                  phase="cancelled", ok=False)


__all__ = ["AgenticEvent", "AgenticObserver", "safe_preview"]
