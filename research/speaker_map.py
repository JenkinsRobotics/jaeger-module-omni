"""Who said what? Two-speaker diarization of a recorded session.

The agent is the THIRD person in the room: two people talk, it listens.
Assigning context per talker starts here — tag every stretch of the session
with which of the two speakers produced it, then merge with the whisper
transcript into a speaker-attributed script.

Embedding model: campplus.onnx — the SAME speaker embedder token2wav uses
for voice cloning, already on disk. Windows of 1.5 s are embedded, clustered
into two centroids (k-means, k=2, cosine), and smoothed.

    python research/speaker_map.py --dir /tmp/session1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CAMPPLUS = (Path.home() /
            "models/MiniCPM-o-4_5-token2wav/assets/token2wav/campplus.onnx")
SR = 16000
WIN_S = 1.5
HOP_S = 0.75


def fbank80(wav, sr=SR):
    """Kaldi-style 80-dim log-mel fbank, the input campplus was trained on.
    Computed with numpy to avoid a torchaudio dependency."""
    import numpy as np

    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    # pre-emphasis + framing
    wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1]).astype(np.float32)
    n = 1 + max(0, (len(wav) - frame_len) // hop)
    frames = np.stack([wav[i * hop:i * hop + frame_len] for i in range(n)])
    frames *= np.hamming(frame_len).astype(np.float32)
    nfft = 512
    spec = np.abs(np.fft.rfft(frames, nfft)) ** 2
    # mel filterbank 80
    def hz2mel(h): return 2595 * np.log10(1 + h / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz2mel(20), hz2mel(sr / 2), 82)
    hz = mel2hz(mels)
    bins = np.floor((nfft + 1) * hz / sr).astype(int)
    fb = np.zeros((80, nfft // 2 + 1), dtype=np.float32)
    for i in range(80):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c == l:
            c = l + 1
        if r <= c:
            r = c + 1
        fb[i, l:c] = np.linspace(0, 1, c - l, endpoint=False)
        fb[i, c:r] = np.linspace(1, 0, r - c, endpoint=False)
    logmel = np.log(np.maximum(spec @ fb.T, 1e-10))
    return (logmel - logmel.mean(0)).astype(np.float32)   # CMN, as campplus


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


def cluster_two(points):
    """k-means, k=2, cosine, deterministic init from the two most-distant."""
    import numpy as np

    embs = [e for _, e in points if e is not None]
    if len(embs) < 4:
        return None, None
    X = np.stack(embs)
    sims = X @ X.T
    i, j = np.unravel_index(np.argmin(sims), sims.shape)
    c = np.stack([X[i], X[j]])
    for _ in range(12):
        assign = (X @ c.T).argmax(1)
        for k in (0, 1):
            if (assign == k).any():
                m = X[assign == k].mean(0)
                c[k] = m / (np.linalg.norm(m) + 1e-9)
    sep = float(c[0] @ c[1])
    return c, sep


def main() -> int:
    import numpy as np
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--transcribe", action="store_true",
                    help="also whisper each speaker turn (slower)")
    a = ap.parse_args()

    wav_path = Path(a.dir) / "raw_mic.wav"
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    print(f"{wav_path.name}: {len(audio)/sr:.0f} s")

    points = embed_windows(audio)
    centroids, sep = cluster_two(points)
    if centroids is None:
        print("not enough speech to cluster")
        return 1
    print(f"two speaker centroids, cosine between them {sep:.3f} "
          f"({'well separated' if sep < 0.6 else 'similar voices'})\n")

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

    talk = [0.0, 0.0]
    for s0, s1, l in segs:
        talk[l] += s1 - s0
    print(f"SPEAKER A  {talk[0]:5.1f} s of speech")
    print(f"SPEAKER B  {talk[1]:5.1f} s of speech\n")

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
        line = f"  [{'A' if l == 0 else 'B'}] {s0:6.1f}-{s1:5.1f}s"
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
    pts = [(0, np.array([1.0, 0])), (1, np.array([0.9, 0.1])),
           (2, np.array([0, 1.0])), (3, np.array([0.1, 0.9])),
           (4, None)]
    pts = [(t, e / np.linalg.norm(e) if e is not None else None)
           for t, e in pts]
    c, sep = cluster_two(pts)
    assert c is not None and sep < 0.5
    print("speaker_map: fbank shape+CMN, 2-means separation OK")


if __name__ == "__main__":
    raise SystemExit(demo() if "--demo" in sys.argv else main())
