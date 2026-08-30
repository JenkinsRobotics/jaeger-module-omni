# Agentic core

This is the copied and Omni-adapted Jaeger Agent control plane. It parses
structured calls, applies permissions and loop limits, dispatches registered
tools, loads skills, and persists memory.

Public extensions do not belong here:

- neural capabilities: `jaeger_agent_omni/nodes/`
- symbolic actions: `jaeger_agent_omni/tools/`
- procedures: `jaeger_agent_omni/skills/`
- durable state implementation: `jaeger_agent_omni/memory/`

`omni_runtime.py` is the native bridge. It starts from audio/vision embeddings
already present in Gemma's cache; it does not request or synthesize an input
transcript. See the repository-level `MULTIMODAL_AGENT.md` and this directory's
`ORIGIN.md`.
