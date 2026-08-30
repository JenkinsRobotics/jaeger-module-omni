"""Speaker embeddings and stable online labels; no transcription involved."""

from __future__ import annotations

from pathlib import Path

CAMPPLUS = (Path.home() /
            "models/MiniCPM-o-4_5-token2wav/assets/token2wav/campplus.onnx")
SAMPLE_RATE = 16000
WINDOW_S = 1.5


def fbank80(wav, sample_rate: int = SAMPLE_RATE):
    """Kaldi-style 80-dim log-mel fbank expected by CampPlus."""
    import numpy as np

    frame_len = int(0.025 * sample_rate)
    hop = int(0.010 * sample_rate)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1]).astype(np.float32)
    n = 1 + max(0, (len(wav) - frame_len) // hop)
    frames = np.stack(
        [wav[i * hop:i * hop + frame_len] for i in range(n)])
    frames *= np.hamming(frame_len).astype(np.float32)
    nfft = 512
    spec = np.abs(np.fft.rfft(frames, nfft)) ** 2

    def hz2mel(h):
        return 2595 * np.log10(1 + h / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mels = np.linspace(hz2mel(20), hz2mel(sample_rate / 2), 82)
    bins = np.floor((nfft + 1) * mel2hz(mels) / sample_rate).astype(int)
    bank = np.zeros((80, nfft // 2 + 1), dtype=np.float32)
    for i in range(80):
        left, centre, right = bins[i], bins[i + 1], bins[i + 2]
        if centre == left:
            centre = left + 1
        if right <= centre:
            right = centre + 1
        bank[i, left:centre] = np.linspace(
            0, 1, centre - left, endpoint=False)
        bank[i, centre:right] = np.linspace(
            1, 0, right - centre, endpoint=False)
    logmel = np.log(np.maximum(spec @ bank.T, 1e-10))
    return (logmel - logmel.mean(0)).astype(np.float32)


class OnlineSpeakerTracker:
    """Assign stable A/B/... labels to completed live turns."""

    def __init__(self, k: int = 2, new_speaker_cosine: float = 0.65,
                 model_path: str | Path = CAMPPLUS):
        import onnxruntime as ort

        if not 1 <= k <= 26:
            raise ValueError("k must be between 1 and 26")
        self.k = k
        self.new_speaker_cosine = float(new_speaker_cosine)
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.centroids = []
        self.counts = []

    def embedding(self, audio):
        import numpy as np

        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        window = int(WINDOW_S * SAMPLE_RATE)
        if len(wav) < int(0.4 * SAMPLE_RATE):
            return None
        # The last window before an endpoint is normally one talker even in a
        # continuous room.  Pad short completed turns to the training window.
        wav = wav[-window:]
        if len(wav) < window:
            wav = np.pad(wav, (window - len(wav), 0))
        if float(np.sqrt(np.mean(wav ** 2))) < 0.01:
            return None
        emb = self.session.run(
            None, {self.input_name: fbank80(wav)[None]})[0][0]
        return emb / (np.linalg.norm(emb) + 1e-9)

    def assign_embedding(self, emb):
        import numpy as np

        e = np.asarray(emb, dtype=np.float32)
        e /= np.linalg.norm(e) + 1e-9
        if not self.centroids:
            label = 0
            self.centroids.append(e.copy())
            self.counts.append(1)
        else:
            sims = np.asarray(self.centroids) @ e
            best = int(np.argmax(sims))
            if (len(self.centroids) < self.k and
                    float(sims[best]) < self.new_speaker_cosine):
                label = len(self.centroids)
                self.centroids.append(e.copy())
                self.counts.append(1)
            else:
                label = best
                n = self.counts[label]
                mean = (self.centroids[label] * n + e) / (n + 1)
                self.centroids[label] = mean / (np.linalg.norm(mean) + 1e-9)
                self.counts[label] = n + 1
        return chr(65 + label)

    def identify(self, audio):
        embedding = self.embedding(audio)
        return self.assign_embedding(embedding) if embedding is not None else None


def demo() -> None:
    import numpy as np

    tracker = OnlineSpeakerTracker.__new__(OnlineSpeakerTracker)
    tracker.k, tracker.new_speaker_cosine = 3, 0.65
    tracker.centroids, tracker.counts = [], []
    vectors = [np.array([1, 0, 0]), np.array([.9, .1, 0]),
               np.array([0, 1, 0]), np.array([.1, .9, 0]),
               np.array([0, 0, 1]), np.array([0, .1, .9])]
    assert [tracker.assign_embedding(x) for x in vectors] == \
        ["A", "A", "B", "B", "C", "C"]
    feats = fbank80(np.random.default_rng(0).normal(
        0, 0.1, SAMPLE_RATE).astype(np.float32))
    assert feats.shape[1] == 80 and abs(feats.mean(0)).max() < 1e-4
    print("runtime.speaker: fbank+CMN and stable online A/B/C labels OK")


if __name__ == "__main__":
    demo()
