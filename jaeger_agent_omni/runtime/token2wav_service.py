"""Repository-owned JSON service for the StepAudio Token2Wav neural decoder.

This is intentionally a process boundary, not a text boundary. The isolated
environment receives integer speech-code chunks and returns waveform chunks;
no language string crosses the boundary.
"""

from __future__ import annotations

import contextlib
import json
import os
import struct
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def _log(message: str) -> None:
    print(f"[token2wav] {message}", file=sys.stderr, flush=True)


class Token2WavService:
    def __init__(self) -> None:
        self.model = None
        self.stream_cache = None
        self.hift_cache = None
        self.stream_cache_base = None
        self.hift_cache_base = None
        self.reference: str | None = None

    @staticmethod
    def _clone(value):
        import torch

        if isinstance(value, dict):
            return {key: Token2WavService._clone(item)
                    for key, item in value.items()}
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, (list, tuple)):
            return type(value)(Token2WavService._clone(item) for item in value)
        return value

    def init(self, model_dir: str, device: str = "cpu",
             float16: bool = False, n_timesteps: int = 5) -> dict:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                from stepaudio2 import Token2wav

                self.model = Token2wav(
                    model_dir, float16=bool(float16),
                    n_timesteps=int(n_timesteps))
            _log(f"initialized {model_dir} on {device}")
            return {"status": "ok", "message": "Token2Wav initialized"}
        except Exception as exc:  # pragma: no cover - model/environment path
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(exc)}

    def set_ref_audio(self, ref_audio_path: str) -> dict:
        if self.model is None:
            return {"status": "error", "message": "not initialized"}
        if not Path(ref_audio_path).is_file():
            return {"status": "error", "message": "reference audio missing"}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                caches = self.model.set_stream_cache(ref_audio_path)
            self.stream_cache, self.hift_cache = caches
            self.stream_cache_base = self._clone(self.stream_cache)
            self.hift_cache_base = self._clone(self.hift_cache)
            self.reference = str(ref_audio_path)
            return {"status": "ok", "message": "reference voice embedded"}
        except Exception as exc:  # pragma: no cover - model/environment path
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(exc)}

    def reset(self) -> dict:
        if self.stream_cache_base is None:
            return {"status": "error", "message": "reference voice not set"}
        self.stream_cache = self._clone(self.stream_cache_base)
        self.hift_cache = self._clone(self.hift_cache_base)
        return {"status": "ok", "message": "stream reset"}

    @staticmethod
    def _write_wav(path: str, audio: np.ndarray, sample_rate: int = 24000):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(np.asarray(audio).reshape(-1), -1, 1) * 32767).astype(
            np.int16)
        data = pcm.tobytes()
        with output.open("wb") as handle:
            handle.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
            handle.write(b"fmt " + struct.pack(
                "<IHHIIHH", 16, 1, 1, sample_rate,
                sample_rate * 2, 2, 16))
            handle.write(b"data" + struct.pack("<I", len(data)) + data)

    def process(self, tokens: list[int], last_chunk: bool,
                output_path: str) -> dict:
        if self.stream_cache is None or self.reference is None:
            return {"status": "error", "message": "reference voice not set"}
        if not tokens:
            return {"status": "error", "message": "empty speech-code chunk"}
        try:
            self.model.stream_cache = self.stream_cache
            self.model.hift_cache_dict = self.hift_cache
            started = time.monotonic()
            with contextlib.redirect_stdout(sys.stderr):
                audio = self.model.stream(
                    generated_speech_tokens=[int(token) for token in tokens],
                    prompt_wav=self.reference,
                    last_chunk=bool(last_chunk),
                    return_waveform=True,
                )
            self.stream_cache = self.model.stream_cache
            self.hift_cache = self.model.hift_cache_dict
            elapsed = (time.monotonic() - started) * 1000
            if audio is None or not np.asarray(audio).size:
                return {"status": "ok", "message": "no audio yet",
                        "output_path": None}
            self._write_wav(output_path, audio)
            return {
                "status": "ok", "message": "waveform generated",
                "output_path": output_path,
                "num_samples": int(np.asarray(audio).size),
                "sample_rate": 24000, "inference_time_ms": elapsed,
            }
        except Exception as exc:  # pragma: no cover - model/environment path
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(exc)}

    def dispatch(self, command: dict) -> dict:
        name = command.get("cmd")
        if name == "init":
            return self.init(
                command["model_dir"], command.get("device", "cpu"),
                command.get("float16", False), command.get("n_timesteps", 5))
        if name == "set_ref_audio":
            return self.set_ref_audio(command["ref_audio_path"])
        if name == "process":
            return self.process(command["tokens"], command.get("last_chunk", True),
                                command["output_path"])
        if name == "reset":
            return self.reset()
        if name == "quit":
            return {"status": "ok", "message": "bye"}
        return {"status": "error", "message": f"unknown command: {name}"}


def main() -> int:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    service = Token2WavService()
    print(json.dumps({"status": "ready"}), flush=True)
    for line in sys.stdin:
        command = {}
        try:
            command = json.loads(line)
            result = service.dispatch(command)
        except Exception as exc:  # protocol errors must not kill the service
            result = {"status": "error", "message": str(exc)}
        print(json.dumps(result), flush=True)
        if command.get("cmd") == "quit":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
