"""Entrypoint for the JaegerOmni 41M Medium room/multidomain node."""

from pathlib import Path

from jaeger_agent_omni.core.runtime.stock_whisper import StockWhisperGemmaSession

NODE_ROOT = Path(__file__).resolve().parent


def build(*, model_path, system=None, device="auto"):
    """Build this immutable encoder/adapter pairing."""
    return StockWhisperGemmaSession(
        NODE_ROOT, model_path=model_path, system=system, device=device)
