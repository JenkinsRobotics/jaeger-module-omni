"""Who said what? K-speaker diarization of a recorded session.

The agent is the THIRD person in the room: two people talk, it listens.
Assigning context per talker starts here — tag every stretch of the session
with which of the two speakers produced it, then merge with the whisper
transcript into a speaker-attributed script.

Embedding model: campplus.onnx — the SAME speaker embedder token2wav uses
for voice cloning, already on disk. Windows of 1.5 s are embedded, clustered
into cosine k-means centroids, and smoothed.  Use k=3 when the agent's own
uncancelled playback is the third voice in a two-person room recording.

    python research/speaker_map.py --dir /tmp/session1
    python research/speaker_map.py --dir /tmp/session1 --k 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaeger_agent_omni.runtime.speaker import (  # noqa: E402
    CAMPPLUS,
    SAMPLE_RATE as SR,
    WINDOW_S as WIN_S,
    OnlineSpeakerTracker,
    fbank80,
)

HOP_S = 0.75


def embed_windows(audio):
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(CAMPPLUS),
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    win, hop = int(WIN_S * SR), int(HOP_S * SR)
    out = []
    for start in range(0, max(1, len(audio) - win), hop):
        seg = audio[start:start + win]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < 0.02:                       # silence window: no speaker
            out.append((start / SR, None))
            continue
        feats = fbank80(seg)[None]
        emb = sess.run(None, {name: feats})[0][0]
        out.append((start / SR, emb / (np.linalg.norm(emb) + 1e-9)))
    return out


def cluster_k(points, k=2):
    """Cosine k-means with deterministic farthest-first initialisation."""
    import numpy as np

    embs = [e for _, e in points if e is not None]
    if k < 1:
        raise ValueError("k must be at least 1")
    if len(embs) < 2 * k:
        return None, None
    X = np.stack(embs)
    sims = X @ X.T
    # Start with the most globally distinct observation, then repeatedly add
    # the point least similar to its closest existing centre.  Unlike choosing
    # only the most-distant pair, this remains meaningful for k > 2.
    chosen = [int(np.argmin(sims.mean(1)))]
    while len(chosen) < k:
        nearest = sims[:, chosen].max(1)
        nearest[chosen] = np.inf
        chosen.append(int(np.argmin(nearest)))
    c = X[chosen].copy()
    for _ in range(24):
        assign = (X @ c.T).argmax(1)
        old = c.copy()
        for label in range(k):
            if (assign == label).any():
                m = X[assign == label].mean(0)
                c[label] = m / (np.linalg.norm(m) + 1e-9)
            else:
                # Re-seed an empty cluster from the point least represented by
                # any non-empty centre instead of returning a duplicate voice.
                represented = (X @ c.T).max(1)
                c[label] = X[int(np.argmin(represented))]
        if np.allclose(c, old, atol=1e-6):
            break
    pairwise = c @ c.T
    separation = [float(pairwise[i, j]) for i in range(k)
                  for j in range(i + 1, k)]
    return c, separation


def cluster_two(points):
    """Backward-compatible k=2 helper; returns the one centroid cosine."""
    centroids, pairwise = cluster_k(points, 2)
    return centroids, (pairwise[0] if pairwise else None)


def main() -> int:
    import numpy as np
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--k", type=int, default=2,
                    help="number of voices (use 3 for agent + two talkers)")
    ap.add_argument("--transcribe", action="store_true",
                    help="also whisper each speaker turn (slower)")
    a = ap.parse_args()

    wav_path = Path(a.dir) / "raw_mic.wav"
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    print(f"{wav_path.name}: {len(audio)/sr:.0f} s")

    points = embed_windows(audio)
    if not 1 <= a.k <= 26:
        ap.error("--k must be between 1 and 26")
    centroids, pairwise = cluster_k(points, a.k)
    if centroids is None:
        print("not enough speech to cluster")
        return 1
    worst = max(pairwise, default=0.0)
    pairs = ", ".join(f"{x:.3f}" for x in pairwise) or "n/a"
    print(f"{a.k} speaker centroids, pairwise cosine [{pairs}] "
          f"({'well separated' if worst < 0.6 else 'some similar voices'})\n")

    # assign + smooth (majority over 3)
    labels = []
    for t, e in points:
        if e is None:
            labels.append((t, None))
        else:
            labels.append((t, int((e @ centroids.T).argmax())))
    for i in range(1, len(labels) - 1):
        t, l = labels[i]
        prev, nxt = labels[i - 1][1], labels[i + 1][1]
        if l is not None and prev == nxt and prev is not None and l != prev:
            labels[i] = (t, prev)

    # collapse into segments
    segs, cur = [], None
    for t, l in labels:
        if l is None:
            continue
        if cur and cur[2] == l and t - cur[1] <= WIN_S:
            cur = (cur[0], t + WIN_S, l)
        else:
            if cur:
                segs.append(cur)
            cur = (t, t + WIN_S, l)
    if cur:
        segs.append(cur)

    talk = [0.0] * a.k
    for s0, s1, l in segs:
        talk[l] += s1 - s0
    for label, seconds in enumerate(talk):
        print(f"SPEAKER {chr(65 + label)}  {seconds:5.1f} s of speech")
    print()

    tx = None
    if a.transcribe:
        import torch
        from transformers import (WhisperForConditionalGeneration,
                                  WhisperProcessor)
        proc = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
        wmod = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-medium.en").eval()

        def tx(seg):
            f = proc(seg, sampling_rate=SR,
                     return_tensors="pt").input_features
            with torch.no_grad():
                ids = wmod.generate(f, max_new_tokens=64)
            return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    for s0, s1, l in segs[:24]:
        line = f"  [{chr(65 + l)}] {s0:6.1f}-{s1:5.1f}s"
        if tx:
            line += "  " + tx(audio[int(s0 * SR):int(s1 * SR)])[:70]
        print(line)
    return 0


def demo() -> None:
    import numpy as np
    f = fbank80(np.random.default_rng(0).normal(0, 0.1, SR).astype(np.float32))
    assert f.shape[1] == 80, f.shape
    assert np.isfinite(f).all()
    assert abs(f.mean(0)).max() < 1e-4, "CMN must zero the mean"
    pts = [(0, np.array([1.0, 0, 0])), (1, np.array([0.9, 0.1, 0])),
           (2, np.array([0, 1.0, 0])), (3, np.array([0.1, 0.9, 0])),
           (4, np.array([0, 0, 1.0])), (5, np.array([0, 0.1, 0.9])),
           (6, None)]
    pts = [(t, e / np.linalg.norm(e) if e is not None else None)
           for t, e in pts]
    c, pairwise = cluster_k(pts, 3)
    assert c is not None and max(pairwise) < 0.5
    tracker = OnlineSpeakerTracker.__new__(OnlineSpeakerTracker)
    tracker.k, tracker.new_speaker_cosine = 3, 0.65
    tracker.centroids, tracker.counts = [], []
    assigned = [tracker.assign_embedding(e) for _, e in pts if e is not None]
    assert assigned == ["A", "A", "B", "B", "C", "C"], assigned
    print("speaker_map: fbank+CMN, 3-means, stable online A/B/C labels OK")


if __name__ == "__main__":
    raise SystemExit(demo() if "--demo" in sys.argv else main())
