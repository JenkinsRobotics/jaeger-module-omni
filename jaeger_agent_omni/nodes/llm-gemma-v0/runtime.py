"""Series brain node: Gemma E4B + rolling conversation log (the ONLY memory).

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


class BrainGemma:
    MAX_TOKENS = 160
    MAX_MESSAGES = 16               # 8 exchanges

    def __init__(self) -> None:
        self.llm = None
        self.history: list[dict] = []

    def load(self, chat_handler=None, say=print) -> None:
        from llama_cpp import Llama
        say(f"loading {LLM_MODEL_PATH.name}…")
        kw = dict(model_path=str(LLM_MODEL_PATH), n_ctx=4096,
                  n_gpu_layers=-1, verbose=False)
        if chat_handler is not None:
            kw["chat_handler"] = chat_handler
        self.llm = Llama(**kw)
        self.llm.create_chat_completion(
            messages=[{"role": "user", "content": "hi"}], max_tokens=1)

    def answer(self, content, system: str) -> str:
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      *self.history,
                      {"role": "user", "content": content}],
            max_tokens=self.MAX_TOKENS, temperature=0.7)
        return (out["choices"][0]["message"]["content"] or "").strip()

    def remember(self, user_text: str, reply: str) -> None:
        """There is no built-in memory: the log IS the context."""
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        if len(self.history) > self.MAX_MESSAGES:
            # Converse shares this exact list with its audio/text brain.
            del self.history[:-self.MAX_MESSAGES]

    def clear(self) -> None:
        self.history.clear()


def build():
    return BrainGemma()
