"""Discover, verify, and select flat Jaeger signal/admission gates."""

from __future__ import annotations

import argparse
from pathlib import Path

from jaeger_agent_omni.node_registry import (
    SUPPORTED_GATE_KINDS,
    NodePackage,
    discover,
    select_profile,
    validate_profiles,
    verify_weights,
)


def discover_gates(root: str | Path) -> dict[str, NodePackage]:
    return discover(
        root,
        manifest_name="gate.json",
        supported_kinds=SUPPORTED_GATE_KINDS,
    )


def select_gate_profile(
    root: str | Path, name: str | None = None
) -> dict[str, NodePackage]:
    return select_profile(
        root,
        name,
        manifest_name="gate.json",
        supported_kinds=SUPPORTED_GATE_KINDS,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "gates",
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument(
        "--all", action="store_true",
        help="validate profiles and inspect every gate, not only one profile",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.all:
        selected = discover_gates(args.root)
        validate_profiles(args.root, selected)
    else:
        selected = select_gate_profile(args.root, args.profile)
    for package in selected.values():
        if args.verify:
            verify_weights(package)
        print(
            f"{package.slot:18s} {package.node_id:42s} "
            f"{package.manifest['status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
