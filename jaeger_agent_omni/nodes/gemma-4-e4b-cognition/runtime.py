"""Node-owned Gemma 4 E4B MLX loader."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build():
    from mlx_lm import load

    return load(str(ROOT / "weights/model"), lazy=False)
