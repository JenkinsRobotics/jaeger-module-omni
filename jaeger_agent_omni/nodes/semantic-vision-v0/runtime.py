"""Series vision channel: mmproj projector -> Gemma content. v0 = the same projector the cascade uses; the trained end state feeds Gemma jointly.

Ported 2026-09-02 from the multimodal modular twin (verified runtimes)."""
from __future__ import annotations
SAMPLE_RATE = 16000
STT_MODEL = "large-v3-turbo"
KOKORO_VOICE = "af_heart"
KOKORO_LANG = "a"
from pathlib import Path
LMSTUDIO = Path.home() / ".lmstudio/models/lmstudio-community"
LLM_MODEL_PATH = LMSTUDIO / "gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
MMPROJ = LMSTUDIO / "gemma-4-E4B-it-GGUF/mmproj-gemma-4-E4B-it-BF16.gguf"


import threading

import numpy as np

import sys as _sys
from pathlib import Path as _P


class VisionMmproj:
    def __init__(self) -> None:
        self.ready = False
        self.handler = None

    def load(self, want: bool, say=print):
        if not want:
            return None
        if MMPROJ.is_file():
            from llama_cpp.llama_chat_format import Gemma4ChatHandler
            say(f"loading vision projector {MMPROJ.name}…")
            self.handler = Gemma4ChatHandler(clip_model_path=str(MMPROJ),
                                             verbose=False)
            self.ready = True
        else:
            say(f"no vision projector at {MMPROJ} — text and audio only")
        return self.handler

    @staticmethod
    def content(user_text: str, image_uri):
        if image_uri is None:
            return user_text
        return [{"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": user_text or "What do you see?"}]


def build():
    return VisionMmproj()
