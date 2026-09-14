"""Thread-safe capture admission for half/full-duplex voice sessions.

The model pipeline is shared.  Duplex mode only changes whether microphone
audio may enter the open turn while an assistant response is in flight.
"""

from __future__ import annotations

import threading
import time


VALID_DUPLEX_MODES = frozenset({"half", "full"})


class DuplexModePolicy:
    """Own the live duplex mode and the half-duplex response barrier.

    ``response_active`` spans cognition, synthesis, playback, and the short
    speaker-tail drain.  Full duplex ignores that barrier; half duplex rejects
    capture until :meth:`finish_response` reopens the next user turn.
    """

    def __init__(self, mode: str = "full", clock=time.monotonic) -> None:
        if mode not in VALID_DUPLEX_MODES:
            raise ValueError(f"duplex mode must be half or full, got {mode!r}")
        self._mode = mode
        self._next_response_id = 0
        self._active_response_ids: set[int] = set()
        self._clock = clock
        self._lock = threading.RLock()
        self._transitions: list[dict] = []

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def response_active(self) -> bool:
        with self._lock:
            return bool(self._active_response_ids)

    def begin_response(self) -> int:
        with self._lock:
            self._next_response_id += 1
            response_id = self._next_response_id
            self._active_response_ids.add(response_id)
            return response_id

    def finish_response(self, response_id: int) -> None:
        with self._lock:
            self._active_response_ids.discard(response_id)

    def clear_responses(self) -> None:
        """Release all barriers after queued/in-flight output is cancelled."""
        with self._lock:
            self._active_response_ids.clear()

    def accepts_capture(self) -> bool:
        with self._lock:
            return self._mode == "full" or not self._active_response_ids

    def set_mode(self, mode: str) -> tuple[str, str] | None:
        if mode not in VALID_DUPLEX_MODES:
            raise ValueError(f"duplex mode must be half or full, got {mode!r}")
        with self._lock:
            old = self._mode
            if old == mode:
                return None
            self._mode = mode
            self._transitions.append({
                "at_monotonic_s": self._clock(),
                "from": old,
                "to": mode,
                "response_active": bool(self._active_response_ids),
            })
            return old, mode

    def transitions(self) -> list[dict]:
        with self._lock:
            return [dict(item) for item in self._transitions]
