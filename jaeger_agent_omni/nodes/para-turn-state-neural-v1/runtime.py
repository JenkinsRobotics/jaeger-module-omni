"""Independent neural speech endpoint; no Whisper/adapter/text dependency."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


class TurnStateNode:
    rate = 'frame'

    def __init__(self, probability=None):
        from jaeger_agent_omni.core.runtime.endpoint import Endpoint
        if probability is None:
            import onnxruntime as ort
            from jaeger_agent_omni.node_registry import load_package, verify_weights
            verify_weights(load_package(ROOT))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self.engine = ort.InferenceSession(str(ROOT/'weights/silero_vad_16k_op15.onnx'),
                sess_options=options, providers=['CPUExecutionProvider'])
            self.state = np.zeros((2, 1, 128), np.float32)
            probability = self._probability
        self.probability = probability
        self.last_probability = 0.0
        self.endpoint = Endpoint(frame_ms=30, speech_probability=self._observe)
        self.frames = 0

    def _probability(self, frame):
        value, self.state = self.engine.run(None, {
            'input': np.asarray(frame, np.float32).reshape(1, 480),
            'state': self.state, 'sr': np.array(16000, np.int64)})
        return float(value[0, 0])

    def _observe(self, frame):
        self.last_probability = float(self.probability(frame))
        return self.last_probability

    def process(self, frame, timestamp):
        frame = np.asarray(frame, np.float32).reshape(-1)
        if frame.size != 480:
            raise ValueError('neural turn node requires 480 samples at 16 kHz')
        event = self.endpoint.push(frame)
        self.frames += 1
        ended = event == 'speech_end'
        in_turn = self.endpoint.speaking
        threshold = .125 if in_turn else .30
        result = {'speaking': self.last_probability >= threshold,
                  'in_turn': in_turn or ended, 'silence_ms': self.endpoint.quiet_ms,
                  'eot_pressure': min(1.0, self.endpoint.quiet_ms/510) if in_turn else 0.0,
                  'turn_ended': ended, 'speech_probability': self.last_probability}
        if ended:
            self.endpoint.reset()
        return result

    def stats(self):
        return {'frames': self.frames, 'last_probability': self.last_probability,
                'classifier': 'frozen Silero ONNX', 'input': 'waveform; no adapter features',
                'scope': 'speech presence and acoustic endpoint, not overlap or addressed intent'}


def build(cfg=None):
    return TurnStateNode()
