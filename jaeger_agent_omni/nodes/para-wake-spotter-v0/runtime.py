"""Parallel wake channel: query-by-example template matching on the
WAVEFORM — no STT in the loop. enroll() is one-shot: one example
becomes a template; spotting is embedding distance. v0 embedding =
sequence stats of the hand features (speaker-biased; the trained
phrase embedding replaces embed() only)."""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "node.json").read_text()).get("config", {})


def _fingerprint(audio):
    """The measured v2 fingerprint (pitch-first), vendored from the voice
    family. Returns (vector, prosody)."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    frames = x[: len(x) // 480 * 480].reshape(-1, 480)
    if not len(frames):
        return (np.asarray([np.log2(150.0), 0.1, 1.0, 2.0, 0.0], np.float32),
                {"pitch_hz": 0.0, "energy_db": -90.0})
    rms = np.sqrt((frames ** 2).mean(axis=1))
    voiced = frames[rms > max(0.01, np.percentile(rms, 60))]
    if len(voiced) < 4:
        voiced = frames
    zc, ce, ro, pi = [], [], [], []
    for fr in voiced[:100]:
        zc.append(float(np.mean(np.abs(np.diff(np.sign(fr))) > 0)))
        spec = np.abs(np.fft.rfft(fr * np.hanning(len(fr))))
        freqs = np.fft.rfftfreq(len(fr), 1 / 16000)
        ce.append(float((spec * freqs).sum() / max(spec.sum(), 1e-9)))
        cum = np.cumsum(spec)
        ro.append(float(freqs[int(np.searchsorted(cum, 0.85 * cum[-1]))]))
        ac = np.correlate(fr, fr, "full")[len(fr) - 1:]
        lo, hi = 16000 // 400, 16000 // 70
        if hi > lo:
            pi.append(float(16000 / (lo + int(np.argmax(ac[lo:hi])))))
    p = np.asarray(pi if pi else [150.0], np.float32)
    pm = float(np.median(p))
    spread = float(np.log2(max(np.percentile(p, 90), 1.0)
                           / max(np.percentile(p, 10), 1.0)))
    v = np.asarray([np.log2(max(pm, 1.0)), float(np.mean(zc)),
                    float(np.mean(ce)) / 1000.0, float(np.mean(ro)) / 1000.0,
                    spread], np.float32)
    return v, {"pitch_hz": round(pm, 1),
               "energy_db": round(20 * float(np.log10(
                   np.sqrt((x ** 2).mean()) + 1e-9)), 1)}


_FP_SCALE = np.asarray([0.25, 0.08, 0.40, 0.80, 0.30], np.float32)
_FP_W = np.asarray([3.0, 1.0, 1.0, 0.5, 0.5], np.float32)


def _fp_dist(a, b):
    z = (a - b) / _FP_SCALE
    return float(np.sqrt(float((_FP_W * z * z).sum()) / float(_FP_W.sum())))


def embed(audio):
    """Phrase embedding v0: fingerprint stats over thirds of the clip +
    duration — crude sequence shape, one-shot by construction."""
    x = np.asarray(audio, np.float32).reshape(-1)
    parts = np.array_split(x, 3)
    vecs = [_fingerprint(p)[0] for p in parts]
    return np.concatenate(vecs + [np.asarray([len(x) / 16000.0], np.float32)])


def edist(a, b):
    sc = np.concatenate([np.tile(_FP_SCALE, 3), [0.6]])
    w = np.concatenate([np.tile(_FP_W, 3), [1.0]])
    z = (a - b) / sc
    return float(np.sqrt(float((w * z * z).sum()) / float(w.sum())))


class WakeSpotterNode:
    rate = "frame"

    def __init__(self):
        self.win = int(CFG.get("window_s", 1.0) * 16000)
        self.cad = int(CFG.get("cadence_s", 0.3) * 16000)
        self.match = CFG.get("match_dist", 0.55)
        self.refractory = CFG.get("refractory_s", 2.0)
        self.buf = np.zeros(0, np.float32)
        self.since = 0
        self.templates = {}
        self._load()
        self.last_hit_t = -1e9
        self.n_checks = 0
        self.n_hits = 0

    def _wfile(self):
        return HERE / "weights" / "templates.npz"

    def _load(self):
        if self._wfile().exists():
            z = np.load(self._wfile(), allow_pickle=False)
            self.templates = {k: z[k] for k in z.files}

    def enroll(self, name, audio):
        """ONE example -> one template. That is the whole point."""
        self.templates[name] = embed(audio)
        np.savez(self._wfile(), **self.templates)

    def process(self, frame, t):
        f = np.asarray(frame, np.float32).reshape(-1)
        self.buf = np.concatenate([self.buf, f])[-self.win:]
        self.since += f.size
        if (self.since < self.cad or self.buf.size < self.win
                or not self.templates):
            return {"wake": None}
        self.since = 0
        if float(np.sqrt(np.mean(self.buf ** 2))) < 0.01:
            return {"wake": None}
        if t - self.last_hit_t < self.refractory:
            return {"wake": None}
        self.n_checks += 1
        q = embed(self.buf)
        best, bd = None, 1e9
        for name, tpl in self.templates.items():
            d = edist(q, tpl)
            if d < bd:
                best, bd = name, d
        if bd <= self.match:
            self.last_hit_t = t
            self.n_hits += 1
            return {"wake": best, "dist": round(bd, 3)}
        return {"wake": None}

    def stats(self):
        return {"templates": len(self.templates), "checks": self.n_checks,
                "hits": self.n_hits}


def build(cfg=None):
    return WakeSpotterNode()
