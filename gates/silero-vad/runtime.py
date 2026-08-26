"""Gate-local Silero ONNX streaming VAD."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Engine:
    def __init__(self, threshold=0.30):
        import numpy as np
        import onnxruntime as ort

        self.np = np
        self.threshold = float(threshold)
        self.session = ort.InferenceSession(
            str(ROOT / "weights/silero_vad_16k_op15.onnx"),
            providers=["CPUExecutionProvider"])
        self.state = np.zeros((2, 1, 128), dtype=np.float32)

    def reset(self):
        self.state.fill(0)

    def probability(self, frame):
        x = self.np.asarray(frame, dtype=self.np.float32).reshape(1, -1)
        if x.shape[1] != 480:
            raise ValueError("Silero gate requires exactly 480 samples")
        probability, self.state = self.session.run(None, {
            "input": x, "state": self.state,
            "sr": self.np.array(16000, dtype=self.np.int64)})
        return float(probability[0, 0])

    def admitted(self, frame):
        return self.probability(frame) >= self.threshold


def build(*, threshold=0.30):
    return Engine(threshold=threshold)
