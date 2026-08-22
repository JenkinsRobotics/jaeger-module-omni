#!/usr/bin/env python3
"""Launch the Jaeger Agent Omni full-duplex desktop demo.

The application source is kept in ``Full DuplexDemo App`` because that is the
donated UI this repository adopted. This launcher gives it a stable entry:

    python demo.py
    python demo.py --check
    python demo.py --selftest-jaeger research/cases12/q00.wav --seconds 20
"""

from __future__ import annotations

import runpy
from pathlib import Path


APP = (Path(__file__).resolve().parent / "Full DuplexDemo App" /
       "omni_duplex_gui.py")


def main() -> int:
    if not APP.exists():
        raise SystemExit(f"full-duplex demo source missing: {APP}")
    namespace = runpy.run_path(str(APP), run_name="jaeger_full_duplex_demo")
    return int(namespace["main"]())


if __name__ == "__main__":
    raise SystemExit(main())
