"""Parallel turn-state channel: speaking envelope + end-of-turn pressure.
30 ms rate — the timeliness the serial cascade never had."""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "node.json").read_text()).get("config", {})


class TurnStateNode:
    rate = "frame"

    def __init__(self):
        self.gate = CFG.get("gate_rms", 0.02)
        self.hang = CFG.get("hangover_ms", 500)
        self.min_speech = CFG.get("min_speech_ms", 300)
        self.speech_ms = 0
        self.silence_ms = 0
        self.in_turn = False
        self.n = 0

    def process(self, frame, t):
        f = np.asarray(frame, np.float32).reshape(-1)
        loud = float(np.sqrt(np.mean(f ** 2))) >= self.gate
        self.n += 1
        if loud:
            self.speech_ms += 30
            self.silence_ms = 0
            if self.speech_ms >= self.min_speech:
                self.in_turn = True
        else:
            self.silence_ms += 30
            if not self.in_turn and self.silence_ms >= 120:
                # isolated clicks separated by silence must not sum to the
                # 300 ms minimum (review 2026-09-02, #7 — ten clicks over
                # six seconds produced one concatenated "utterance")
                self.speech_ms = 0
        eot = 0.0
        if self.in_turn and self.silence_ms > 0:
            eot = min(1.0, self.silence_ms / self.hang)
        ended = self.in_turn and eot >= 1.0
        if ended:
            self.in_turn = False
            self.speech_ms = 0
        return {"speaking": loud, "in_turn": self.in_turn or ended,
                "silence_ms": self.silence_ms, "eot_pressure": round(eot, 2),
                "turn_ended": ended}

    def stats(self):
        return {"frames": self.n}


def build(cfg=None):
    return TurnStateNode()
