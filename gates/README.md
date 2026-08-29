# Signal and admission gates

Gates condition or admit PCM around the neural matrix. They do not perform a
learned semantic transformation and must not introduce a transcript handoff.
Each gate owns its runtime, immutable assets, typed ports, and `gate.json`.

- `silero-vad` is a small neural speech detector, but its role is an admission
  gate: it emits speech probability and a policy decision, not semantic state.
- `speex-aec` is adaptive DSP. It subtracts the known playback reference from
  microphone PCM so the neural matrix can continue listening while it speaks.

Keeping these components outside `jaeger_agent_omni/nodes/` makes the boundary explicit: neural
nodes carry learned perception, identity, cognition, or generation state;
gates only clean, time, or admit the signal. A deterministic DSP gate is not
made better merely by replacing it with a trained network.

`profiles.json` explicitly controls the two gates independently of the neural
node profile.
