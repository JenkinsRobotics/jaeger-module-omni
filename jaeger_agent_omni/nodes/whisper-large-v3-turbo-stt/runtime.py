"""Node-owned stock Whisper-large-v3-turbo STT runtime."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Engine:
    def __init__(self, device="auto"):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "mps" else torch.float32
        model = ROOT / "weights/model"
        self.processor = WhisperProcessor.from_pretrained(str(model))
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(model), dtype=self.dtype).to(self.device).eval()

    def transcribe(self, audio):
        batch = self.processor(
            audio, sampling_rate=16_000, return_tensors="pt",
            return_attention_mask=True)
        with self.torch.inference_mode():
            ids = self.model.generate(
                batch.input_features.to(self.device, dtype=self.dtype),
                attention_mask=batch.attention_mask.to(self.device),
                language="en", task="transcribe")
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def build(*, device="auto"):
    return Engine(device=device)
