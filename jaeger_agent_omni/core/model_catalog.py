"""Canonical audio-ear identities shared by every Jaeger entry point.

The catalog resolves a user-facing model name to the low-level runtime seam
(``tower`` or ``stock``) and its immutable node directory.  Keeping this in
one model-free module prevents the GUI, live CLI, and benchmarks from silently
assigning the same result to different checkpoints.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
NODES_ROOT = PACKAGE_ROOT / "nodes"
DEFAULT_FRONT_END = "jaegeromni-41m-medium"


@dataclass(frozen=True)
class FrontEndSpec:
    """One stable product identity and its exact neural input node."""

    model_id: str
    label: str
    engine: str
    node: Path | None
    status: str
    production_candidate: bool = True


@dataclass(frozen=True)
class FrontEndSelection:
    """Resolved catalog identity, including an optional explicit override."""

    spec: FrontEndSpec
    node: Path | None
    checkpoint: str
    custom: bool = False

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def engine(self) -> str:
        return self.spec.engine


def _node(env: str, relative: str) -> Path:
    return Path(os.environ.get(env, NODES_ROOT / relative)).expanduser()


FRONT_END_SPECS = (
    FrontEndSpec(
        "gemma-native", "Gemma Native", "tower", None,
        "duplex-validated fallback"),
    FrontEndSpec(
        "gemmaomni-41m-medium", "GemmaOmni 41M Medium", "stock",
        _node("JAEGER_CLEANROOM_NODE",
              "stock-whisper-medium-cleanroom-ctx93h-step2000"),
        "best general accuracy; five-minute gate pending"),
    FrontEndSpec(
        "gemmaomni-41m-final4000", "GemmaOmni 41M Final 4000", "stock",
        _node("JAEGER_GEMMAOMNI_FINAL4000_NODE",
              "gemmaomni-41m-medium-final4000"),
        "imported causal meeting specialist; validation pending",
        production_candidate=False),
    FrontEndSpec(
        "jaegeromni-41m-medium", "JaegerOmni 41M Medium", "stock",
        _node("JAEGER_41M_MEDIUM_NODE", "jaegeromni-41m-medium"),
        "broad-corpus winner; five-minute gate pending"),
    FrontEndSpec(
        "jaegeromni-41m-causal-balanced",
        "JaegerOmni 41M Causal Balanced", "stock",
        _node("JAEGER_41M_CAUSAL_NODE", "jaegeromni-41m-causal-balanced"),
        "causal multidomain candidate; perception heads pending",
        production_candidate=False),
    FrontEndSpec(
        "jaegeromni-41m-audio-perception",
        "JaegerOmni 41M Audio Perception", "stock",
        _node("JAEGER_41M_PERCEPTION_NODE",
              "jaegeromni-41m-audio-perception"),
        "scene validated; conversation-event heads shadow only",
        production_candidate=False),
    FrontEndSpec(
        "jaegeromni-50m-medium", "JaegerOmni 50M Medium", "stock",
        _node("JAEGER_STOCK_NODE", "stock-whisper-medium-cekl-blend50"),
        "fastest viable adapter"),
    FrontEndSpec(
        "jaegeromni-50m-large", "JaegerOmni 50M Large", "stock",
        _node("JAEGER_50M_LARGE_NODE", "stock-whisper-large-v3-turbo50m"),
        "transfer failed; experimental", production_candidate=False),
)
FRONT_ENDS = {spec.model_id: spec for spec in FRONT_END_SPECS}
FRONT_END_ALIASES = {
    "tower": "gemma-native",
    "stock": "jaegeromni-50m-medium",
    "cleanroom": "gemmaomni-41m-medium",
}


def canonical_id(value: str) -> str:
    """Resolve a compatibility alias to one canonical identity."""
    model_id = FRONT_END_ALIASES.get(value, value)
    if model_id not in FRONT_ENDS:
        raise ValueError(f"unknown Jaeger model: {value}")
    return model_id


def checkpoint_label(node: Path | None) -> str:
    """Return a short manifest identity without hashing or loading weights."""
    if node is None:
        return "built-in"
    try:
        manifest = json.loads((node / "node.json").read_text())
        return str(manifest["adapter"]["sha256"])[:12]
    except (OSError, KeyError, TypeError, ValueError):
        return "manifest-error"


def resolve_front_end(value: str, adapter_override=None) -> FrontEndSelection:
    """Resolve a CLI/UI value to its runtime engine and paired node.

    Explicit adapter overrides remain supported for research, but only on a
    stock-Whisper identity.  The returned selection is marked custom so its
    result cannot masquerade as the catalog checkpoint.
    """
    spec = FRONT_ENDS[canonical_id(value)]
    node = spec.node
    custom = False
    if adapter_override is not None and spec.engine != "stock":
        raise ValueError(
            f"--adapter cannot be used with {spec.model_id}; select a "
            "stock-Whisper identity")
    if adapter_override is not None:
        node = Path(adapter_override).expanduser()
        custom = node.resolve() != spec.node.resolve() if spec.node else True
    label = checkpoint_label(node) if node is None or node.is_dir() else \
        f"custom:{node.name}"
    if custom:
        spec = replace(spec, label=f"{spec.label} (custom adapter)")
    return FrontEndSelection(spec=spec, node=node, checkpoint=label,
                             custom=custom)


def front_end_choices(include_aliases: bool = True) -> tuple[str, ...]:
    """Ordered argparse choices, canonical identities first."""
    values = tuple(spec.model_id for spec in FRONT_END_SPECS)
    return values + (tuple(FRONT_END_ALIASES) if include_aliases else ())


def demo() -> None:
    """Model-free catalog contract used by the CPU check suite."""
    assert len(FRONT_END_SPECS) == 8
    assert len(FRONT_ENDS) == 8
    assert canonical_id("tower") == "gemma-native"
    assert canonical_id("stock") == "jaegeromni-50m-medium"
    assert canonical_id("cleanroom") == "gemmaomni-41m-medium"
    assert DEFAULT_FRONT_END == "jaegeromni-41m-medium"
    assert resolve_front_end("gemma-native").checkpoint == "built-in"
    try:
        resolve_front_end("gemma-native", "adapter.npz")
    except ValueError:
        pass
    else:
        raise AssertionError("native ear must reject a silent adapter override")
    assert not FRONT_ENDS["jaegeromni-50m-large"].production_candidate
    assert not FRONT_ENDS["gemmaomni-41m-final4000"].production_candidate
    assert not FRONT_ENDS["jaegeromni-41m-causal-balanced"].production_candidate
    assert not FRONT_ENDS["jaegeromni-41m-audio-perception"].production_candidate
    print("model_catalog: eight canonical ears, aliases, and identities OK")


if __name__ == "__main__":
    demo()
