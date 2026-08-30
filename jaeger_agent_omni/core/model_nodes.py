"""Versioned manifests for inseparable audio-encoder/adapter pairings."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST = "node.json"
SCHEMA_VERSION = 1
KIND = "jaeger_audio_node"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AudioNode:
    """Resolved node metadata plus its exact adapter checkpoint."""

    root: Path
    manifest: dict[str, Any]
    adapter: Path

    @property
    def node_id(self) -> str:
        return str(self.manifest["id"])

    @property
    def encoder_id(self) -> str:
        return str(self.manifest["encoder"]["id"])

    @property
    def encoder_path(self) -> Path:
        return (self.root / self.manifest["encoder"]["path"]).resolve()


def load_audio_node(path: str | Path) -> AudioNode:
    """Load a node directory and verify the immutable artifact reference."""
    root = Path(path).expanduser().resolve()
    manifest_path = root / MANIFEST
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError(f"{path}: expected a node directory containing {MANIFEST}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{manifest_path}: unsupported schema_version")
    if manifest.get("kind") != KIND:
        raise ValueError(f"{manifest_path}: kind must be {KIND!r}")
    if manifest.get("runtime") != "stock_whisper_gemma":
        raise ValueError(f"{manifest_path}: unsupported runtime")
    for key in ("id", "encoder", "adapter", "backbone", "status"):
        if key not in manifest:
            raise ValueError(f"{manifest_path}: missing {key}")
    adapter_meta = manifest["adapter"]
    adapter = (root / adapter_meta["path"]).resolve()
    if not adapter.is_file():
        raise ValueError(f"{manifest_path}: missing adapter {adapter}")
    expected_hash = str(adapter_meta.get("sha256", ""))
    if len(expected_hash) != 64 or _sha256(adapter) != expected_hash:
        raise ValueError(f"{manifest_path}: adapter SHA-256 mismatch")
    node = AudioNode(root=root, manifest=manifest, adapter=adapter)
    encoder_weights = node.encoder_path / "model.safetensors"
    if not node.encoder_path.is_dir() or not encoder_weights.is_file():
        raise ValueError(f"{manifest_path}: incomplete local encoder weights")
    return node


def verify_audio_node(node: AudioNode) -> None:
    """Perform the slower full encoder-weight integrity check on demand."""
    expected = str(node.manifest["encoder"].get("weights_sha256", ""))
    weights = node.encoder_path / "model.safetensors"
    if len(expected) != 64 or _sha256(weights) != expected:
        raise ValueError(f"{node.root / MANIFEST}: encoder SHA-256 mismatch")


def load_node_factory(node: AudioNode):
    """Load the node-owned runtime entrypoint declared in its manifest."""
    implementation = str(node.manifest.get("implementation", ""))
    try:
        relative_file, symbol = implementation.split(":", 1)
    except ValueError as error:
        raise ValueError(
            f"{node.root / MANIFEST}: implementation must be file.py:symbol") \
            from error
    runtime_file = (node.root / relative_file).resolve()
    if runtime_file.parent != node.root or not runtime_file.is_file():
        raise ValueError(f"{node.root / MANIFEST}: invalid runtime file")
    module_name = "_jaeger_node_" + node.node_id.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, runtime_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"{runtime_file}: cannot load node runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, symbol, None)
    if not callable(factory):
        raise ValueError(f"{runtime_file}: missing callable {symbol}")
    return factory


def resolve_audio_node(path: str | Path) -> tuple[Path, AudioNode | None]:
    """Accept a legacy checkpoint file or a strongly paired node folder."""
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        node = load_audio_node(candidate)
        return node.adapter, node
    return candidate, None


def validate_node_checkpoint(node: AudioNode, values: dict[str, Any]) -> None:
    """Reject a manifest whose encoder boundary differs from its checkpoint."""
    encoder = node.manifest["encoder"]
    adapter = node.manifest["adapter"]
    checks = {
        "encoder id": (node.encoder_id, str(values["encoder"])),
        # Older valid medium checkpoints predate an explicit input_dim field;
        # validate_checkpoint derives it from the pinned encoder ID first.
        "input dimension": (int(encoder["output_dim"]),
                            int(values.get("input_dim", encoder["output_dim"]))),
        "input boundary": (str(encoder["boundary"]),
                           str(values["input_boundary"])),
        "adapter architecture": (str(adapter["architecture"]),
                                 str(values["adapter_arch"])),
        "pool step": (int(adapter["pool_step"]), int(values["pool_step"])),
    }
    mismatches = [
        f"{name}: manifest={expected!r}, checkpoint={actual!r}"
        for name, (expected, actual) in checks.items() if expected != actual
    ]
    if mismatches:
        raise ValueError(f"{node.root / MANIFEST}: incompatible node pairing; "
                         + "; ".join(mismatches))
