"""Parallel speaker channel: rolling-window embedding + online centroids.
Centroids persist in weights/ — session-to-session speaker memory."""
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


class SpeakerNode:
    rate = "frame"

    def __init__(self):
        self.win = int(CFG.get("window_s", 1.2) * 16000)
        self.cad = int(CFG.get("cadence_s", 0.6) * 16000)
        self.dist = CFG.get("new_speaker_dist", 1.0)
        self.max_spk = CFG.get("max_speakers", 4)
        self.buf = np.zeros(0, np.float32)
        self.since = 0
        self.centroids, self.counts = [], []
        self._load()
        self.out = {"speaker": 0, "conf": 0.0, "pitch_hz": 0.0}
        self.n_updates = 0
        self.t_ms = 0.0

    def _wfile(self):
        return HERE / "weights" / "centroids.npz"

    def _load(self):
        if self._wfile().exists():
            z = np.load(self._wfile())
            self.centroids = [c.astype(np.float32) for c in z["centroids"]]
            self.counts = list(z["counts"])

    def save(self):
        if self.centroids:
            np.savez(self._wfile(), centroids=np.stack(self.centroids),
                     counts=np.asarray(self.counts))

    def process(self, frame, t):
        f = np.asarray(frame, np.float32).reshape(-1)
        self.buf = np.concatenate([self.buf, f])[-self.win:]
        self.since += f.size
        if self.since < self.cad or self.buf.size < self.win:
            return self.out
        self.since = 0
        if float(np.sqrt(np.mean(self.buf ** 2))) < 0.01:
            return self.out                       # silence: hold last
        t0 = time.perf_counter()
        v, pros = _fingerprint(self.buf)
        if not self.centroids:
            self.centroids.append(v); self.counts.append(1)
            self.out = {"speaker": 1, "conf": 1.0, **pros}
        else:
            ds = [_fp_dist(v, c) for c in self.centroids]
            b = int(np.argmin(ds))
            if ds[b] <= self.dist or len(self.centroids) >= self.max_spk:
                n = self.counts[b]
                self.centroids[b] = (self.centroids[b] * n + v) / (n + 1)
                self.counts[b] += 1
                self.out = {"speaker": b + 1,
                            "conf": round(1.0 / (1.0 + ds[b]), 3), **pros}
            else:
                self.centroids.append(v); self.counts.append(1)
                self.out = {"speaker": len(self.centroids), "conf": 1.0, **pros}
        self.t_ms += (time.perf_counter() - t0) * 1000
        self.n_updates += 1
        return self.out

    def stats(self):
        return {"updates": self.n_updates, "speakers": len(self.centroids),
                "avg_ms": round(self.t_ms / max(1, self.n_updates), 2)}


def build(cfg=None):
    return SpeakerNode()
