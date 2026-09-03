"""Series channel: utterance-rate transcription. Off-the-shelf turbo;
the v1 encoder+adapter node replaces this wholesale."""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "node.json").read_text()).get("config", {})


class SemanticNode:
    rate = "utterance"

    def __init__(self):
        self._model = None
        self.n = 0
        self.t_ms = 0.0

    def _load(self):
        if self._model is None:
            from pywhispercpp.model import Model
            self._model = Model(CFG.get("model", "large-v3-turbo"),
                                print_realtime=False, print_progress=False)
        return self._model

    def process(self, audio, t):
        t0 = time.perf_counter()
        segs = self._load().transcribe(
            np.asarray(audio, np.float32).reshape(-1),
            language=CFG.get("language", "en"))
        text = " ".join(s.text for s in segs).strip()
        self.t_ms += (time.perf_counter() - t0) * 1000
        self.n += 1
        return {"text": text}

    def stats(self):
        return {"utterances": self.n,
                "avg_ms": round(self.t_ms / max(1, self.n), 1)}


def build(cfg=None):
    return SemanticNode()
