"""Fusion: where the channels converge. v0 emits the fused record and the
[context:] line; the SEAM is that these records become dedicated context
tokens through a separately trained projection into Gemma."""
import datetime
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Family looks_complete lesson, ported: the trained adapter transcribes
# AMI-style — ALL CAPS, no punctuation — so "ends with . ! ?" judged EVERY
# spoken turn incomplete and every follow-up paid the 8 s hold (sim
# 2026-09-02). Punctuation completes a turn when present; when absent the
# WORDS decide: only a hanging connector/article tail marks a thought
# unfinished. This is the latency contract: complete turns reply at
# turn rate.
_HANG_TAIL = {
    "and", "or", "but", "of", "the", "a", "an", "to", "with", "in", "on",
    "at", "for", "by", "so", "because", "if", "then", "is", "are", "was",
    "were", "my", "your", "their", "his", "her", "its", "our", "that",
    "which", "who", "whose", "when", "while", "than", "versus", "about",
    "into", "from", "very", "really", "quite", "plus", "minus", "um", "uh",
}


def looks_complete(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if t.endswith((".", "!", "?")):
        return True
    if t.endswith((",", ":", ";", "-")):
        return False
    words = re.sub(r"[^a-z0-9' ]+", " ", t.lower()).split()
    return bool(words) and words[-1] not in _HANG_TAIL


class FusionNode:
    rate = "utterance"

    def __init__(self):
        self.n = 0

    def process(self, inputs, t):
        text = inputs.get("text", "")
        spk = inputs.get("speaker", {})
        turn = inputs.get("turn", {})
        wake = inputs.get("wake")
        dur = max(float(inputs.get("duration_s", 0.0)), 1e-3)
        rec = {
            "text": text,
            "speaker": spk.get("speaker", 0),
            "speaker_conf": spk.get("conf", 0.0),
            "pitch_hz": spk.get("pitch_hz", 0.0),
            "energy_db": spk.get("energy_db", 0.0),
            "wpm": round(len(text.split()) / dur * 60.0, 1),
            "still_speaking": bool(turn.get("in_turn") and
                                   not turn.get("turn_ended")),
            "wake": wake,
            "complete": looks_complete(text),
            "iso_time": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        bits = [f"speaker=S{rec['speaker']}",
                f"conf={rec['speaker_conf']:.0%}",
                f"pitch={rec['pitch_hz']:.0f}Hz"]
        if wake:
            bits.append(f"wake={wake}")
        if not rec["complete"]:
            bits.append("incomplete")
        rec["context_line"] = "[context: " + " ".join(bits) + "]"
        self.n += 1
        return rec

    def stats(self):
        return {"records": self.n}


def build(cfg=None):
    return FusionNode()
