"""Mouth node: Kokoro, native 24 kHz chunks; device/barge belong to audio-duplex-io.

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


class MouthKokoro:
    def __init__(self) -> None:
        self.tts = None

    def load(self, say=print) -> None:
        from kokoro import KPipeline
        say("loading Kokoro…")
        self.tts = KPipeline(lang_code=KOKORO_LANG)
        list(self.tts("ready", voice=KOKORO_VOICE))

    def synth(self, text: str):
        for r in self.tts(text, voice=KOKORO_VOICE):
            if r.audio is not None:
                yield np.asarray(r.audio, dtype=np.float32)


def build():
    return MouthKokoro()
