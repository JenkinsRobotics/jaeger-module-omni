#!/usr/bin/env python3
"""CPU check: heads-free causal port agrees across training and runtime."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "jaeger_agent_omni/core"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mlx.utils import tree_flatten
    from gemmaomni_adapter import build_context_adapter, load_any_adapter
    from training.cleanroom_context_ce import _build_adapter, _validate_saved_checkpoint

    mx.random.seed(7)
    bidirectional = build_context_adapter(hidden=8)
    values = {name: np.asarray(v) for name, v in tree_flatten(bidirectional.parameters())}
    values.update(input_dim=np.array(1024), hidden=np.array([8]), pool_step=np.array(2),
                  arch=np.frombuffer(b"context_causal_v1", dtype=np.uint8),
                  adapter_arch=np.array("cleanroom_context_causal_v1"), causal=np.array([1]))
    trainer = _build_adapter(values, causal=True)
    x = mx.array(np.random.default_rng(7).normal(size=(80, 1024)).astype(np.float32))
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "port.npz"
        np.savez(checkpoint, **values)
        _validate_saved_checkpoint(checkpoint, causal=True, base=values)
        try:
            _validate_saved_checkpoint(checkpoint, causal=True, base={})
        except ValueError:
            pass
        else:
            raise AssertionError("checkpoint topology mismatch was accepted")
        runtime, architecture = load_any_adapter(checkpoint)
        outputs = [np.asarray(fn(x)) for fn in (trainer, runtime)]
        prefix = np.asarray(runtime(x[:40]))
        agreement = float(np.max(np.abs(outputs[0]-outputs[1])))
        drift = float(np.max(np.abs(outputs[1][:20]-prefix)))
        control = float(np.max(np.abs(np.asarray(bidirectional(x))[:20]
                                      - np.asarray(bidirectional(x[:40])))))
        assert agreement < 1e-5 and drift < 1e-5 and control > 1e-3
        assert architecture == "context_causal_v1"
        assert not any(name.startswith(("head_", "self_proj")) for name in values)
    report = {"command": [sys.executable,*sys.argv], "device": "cpu",
              "training_runtime_max_difference": agreement,
              "causal_prefix_max_drift": drift, "bidir_control_drift": control,
              "passed": True, "scope": "adapter only; Whisper still re-encodes growing prefixes"}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
