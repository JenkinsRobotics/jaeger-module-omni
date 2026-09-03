"""Text channel node: typed input. Trivial on purpose — text is already
tokens — but a DECLARED channel of the graph: typing IS addressing (skips
the wake gate), and full duplex applies to it (a typed message during
playback is an interruption under the floor policy)."""
from __future__ import annotations

import queue


class TextInput:
    def __init__(self) -> None:
        self.q: queue.Queue[str] = queue.Queue()

    def submit(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.q.put(text)

    def poll(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None


def build(cfg=None):
    return TextInput()
