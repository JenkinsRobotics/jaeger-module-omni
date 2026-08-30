# Vendored agent runtime provenance

The agent loop in this directory was copied from the read-only sibling
`~/GITHUB/jaeger-agent` at commit:

```text
a3696e9e78d64e1b1996bacc036902b60eafe30e
```

The minimum JaegerOS framework code needed by that loop was copied beneath
`_vendor/jaeger_os` from the read-only sibling at commit:

```text
e05dc24eec3ff09af6132b7b0e4fca96d059c494
```

Both copies retain their upstream Apache-2.0 license files. Git metadata,
caches, mutable instance state, build products, hardware firmware, and model
weights were excluded. Imports were namespaced under
`jaeger_agent_omni.core.agentic` so this repository has no runtime link to the
sibling repositories.

Omni-specific changes include:

- a provider adapter over an already-prefilled native multimodal Gemma cache;
- an explicit `listen`/`text_to_speech` denylist for the native path;
- dynamic schema delivery after `load_tools`;
- interrupt propagation into the active generation;
- standard-JSON-first parsing for Gemma tool arguments;
- a preserved tool denylist when the registry refreshes;
- lazy bus imports so the native runtime does not require pyzmq.

The public copied capability surfaces live at sibling package paths:
`jaeger_agent_omni/tools`, `skills`, `memory`. Their registry implementation
lives at `jaeger_agent_omni/core/skill_registry`.
