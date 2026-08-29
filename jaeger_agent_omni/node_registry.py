"""Discover, verify, and select flat self-contained Jaeger neural nodes."""

from __future__ import annotations

import hashlib
import importlib.util
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_NODE_KINDS = frozenset({"jaeger_audio_node", "jaeger_neural_node"})
SUPPORTED_GATE_KINDS = frozenset({"jaeger_neural_gate", "jaeger_signal_gate"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class NodePackage:
    root: Path
    manifest: dict[str, Any]

    @property
    def node_id(self) -> str:
        return str(self.manifest["id"])

    @property
    def slot(self) -> str:
        return str(self.manifest["slot"])

    def local_path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise ValueError(f"{self.node_id}: path escapes node folder")
        return path


def load_package(
    root: str | Path,
    *,
    manifest_name: str = "node.json",
    supported_kinds: frozenset[str] = SUPPORTED_NODE_KINDS,
) -> NodePackage:
    root = Path(root).expanduser().resolve()
    manifest_path = root / manifest_name
    if not manifest_path.is_file():
        raise ValueError(f"{root}: missing {manifest_name}")
    manifest = json.loads(manifest_path.read_text())
    required = {
        "schema_version", "kind", "id", "slot", "status",
        "enabled_by_default", "implementation", "consumes", "produces"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"{manifest_path}: missing {sorted(missing)}")
    if manifest["schema_version"] != 1:
        raise ValueError(f"{manifest_path}: unsupported schema version")
    if manifest["kind"] not in supported_kinds:
        raise ValueError(f"{manifest_path}: unsupported package kind")
    package = NodePackage(root, manifest)
    runtime_name, separator, symbol = str(manifest["implementation"]).partition(":")
    if not separator or not symbol:
        raise ValueError(f"{manifest_path}: implementation must be file.py:symbol")
    runtime = package.local_path(runtime_name)
    if not runtime.is_file():
        raise ValueError(f"{manifest_path}: missing runtime {runtime_name}")
    for section in ("model", "library"):
        if section in manifest:
            declared = package.local_path(str(manifest[section]["path"]))
            if not declared.exists():
                raise ValueError(f"{manifest_path}: missing {section} weights")
    if "adapter" in manifest:
        adapter = package.local_path(str(manifest["adapter"]["path"]))
        if not adapter.is_file():
            raise ValueError(f"{manifest_path}: missing adapter weights")
    if "encoder" in manifest and "path" in manifest["encoder"]:
        encoder = package.local_path(str(manifest["encoder"]["path"]))
        if not encoder.is_dir():
            raise ValueError(f"{manifest_path}: missing encoder weights")
    for asset in manifest.get("assets", []):
        path = package.local_path(str(asset["path"]))
        if not path.is_file():
            raise ValueError(f"{manifest_path}: missing asset {asset['path']}")
    return package


def discover(
    root: str | Path,
    *,
    manifest_name: str = "node.json",
    supported_kinds: frozenset[str] = SUPPORTED_NODE_KINDS,
) -> dict[str, NodePackage]:
    """Discover direct child folders only; registries stay flat."""
    root = Path(root).expanduser().resolve()
    packages = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not (child / manifest_name).is_file():
            continue
        package = load_package(
            child,
            manifest_name=manifest_name,
            supported_kinds=supported_kinds,
        )
        if package.node_id in packages:
            raise ValueError(f"duplicate node id: {package.node_id}")
        packages[package.node_id] = package
    return packages


def load_factory(package: NodePackage):
    relative, symbol = str(package.manifest["implementation"]).split(":", 1)
    runtime = package.local_path(relative)
    name = "_jaeger_package_" + package.node_id.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, runtime)
    if spec is None or spec.loader is None:
        raise ValueError(f"{runtime}: cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, symbol, None)
    if not callable(factory):
        raise ValueError(f"{runtime}: missing callable {symbol}")
    return factory


def verify_weights(package: NodePackage) -> None:
    """Hash every node's primary immutable model, adapter, or library."""
    manifest = package.manifest
    checks: list[tuple[Path, str]] = []
    for section in ("model", "library"):
        if section not in manifest:
            continue
        item = manifest[section]
        path = package.local_path(str(item["path"]))
        if path.is_dir():
            path /= str(item["weights_file"])
        checks.append((path, str(item.get("weights_sha256") or item.get("sha256"))))
    if "adapter" in manifest:
        item = manifest["adapter"]
        checks.append((package.local_path(str(item["path"])), str(item["sha256"])))
    if "encoder" in manifest and manifest["encoder"].get("weights_sha256"):
        item = manifest["encoder"]
        checks.append((package.local_path(str(item["path"])) / "model.safetensors",
                       str(item["weights_sha256"])))
    for item in manifest.get("assets", []):
        checks.append((package.local_path(str(item["path"])), str(item["sha256"])))
    for path, expected in checks:
        if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
            raise ValueError(f"{package.node_id}: weight hash mismatch for {path}")


def validate_profiles(root: str | Path,
                      packages: dict[str, NodePackage]) -> dict[str, Any]:
    document = json.loads((Path(root) / "profiles.json").read_text())
    if document.get("schema_version") != 1:
        raise ValueError("profiles.json: unsupported schema")
    profiles = document.get("profiles", {})
    if document.get("default") not in profiles:
        raise ValueError("profiles.json: missing default profile")
    all_ids = set(packages)
    for name, profile in profiles.items():
        enabled = profile.get("enabled", [])
        disabled = profile.get("disabled", [])
        if len(enabled) != len(set(enabled)) or set(enabled) & set(disabled):
            raise ValueError(f"profile {name}: duplicate or conflicting nodes")
        unknown = (set(enabled) | set(disabled)) - all_ids
        if unknown:
            raise ValueError(f"profile {name}: unknown nodes {sorted(unknown)}")
        if set(enabled) | set(disabled) != all_ids:
            raise ValueError(f"profile {name}: every node must be explicitly enabled/disabled")
        slots = [packages[node].slot for node in enabled]
        if len(slots) != len(set(slots)):
            raise ValueError(f"profile {name}: more than one node fills a slot")
        if not profile.get("allow_rejected", False):
            rejected = [node for node in enabled
                        if str(packages[node].manifest["status"]).startswith("rejected")]
            if rejected:
                raise ValueError(f"profile {name}: rejected nodes enabled {rejected}")
    return document


def select_profile(
    root: str | Path,
    name: str | None = None,
    *,
    manifest_name: str = "node.json",
    supported_kinds: frozenset[str] = SUPPORTED_NODE_KINDS,
) -> dict[str, NodePackage]:
    packages = discover(
        root,
        manifest_name=manifest_name,
        supported_kinds=supported_kinds,
    )
    document = validate_profiles(root, packages)
    selected_name = name or str(document["default"])
    if selected_name not in document["profiles"]:
        raise ValueError(f"unknown node profile: {selected_name}")
    return {node_id: packages[node_id]
            for node_id in document["profiles"][selected_name]["enabled"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parent / "nodes")
    parser.add_argument("--profile", default=None)
    parser.add_argument(
        "--all", action="store_true",
        help="validate profiles and inspect every package, not only one profile",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.all:
        selected = discover(args.root)
        validate_profiles(args.root, selected)
    else:
        selected = select_profile(args.root, args.profile)
    for package in selected.values():
        if args.verify:
            verify_weights(package)
        print(f"{package.slot:18s} {package.node_id:42s} "
              f"{package.manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
