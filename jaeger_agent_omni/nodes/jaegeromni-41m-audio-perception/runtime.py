"""Entrypoint for the JaegerOmni 41M neural audio-perception candidate."""

from pathlib import Path

from jaeger_agent_omni.core.runtime.stock_whisper import StockWhisperGemmaSession

NODE_ROOT = Path(__file__).resolve().parent


def build(*, model_path, system=None, device="auto"):
    return StockWhisperGemmaSession(
        NODE_ROOT, model_path=model_path, system=system, device=device)
