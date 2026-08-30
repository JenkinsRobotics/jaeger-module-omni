"""Causal multi-task acoustic controller and typed companion contract.

The controller is pre-Whisper and never consumes or produces text. It accepts
the microphone plus Jaeger's exact playback reference, predicts the echo to
remove, and emits typed acoustic probabilities for later neural/policy stages.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "weights/controller.pt"
ARCHITECTURE = "causal_frame_gru_multitask_v1"
SAMPLE_RATE = 16_000
HOP_SAMPLES = 160
HIDDEN = 128
RECURRENT_LAYERS = 2
SCENES = ("silence", "environment", "self", "speech")
EVENTS = ("external_speech", "self_speech", "music", "noise", "overlap")
TURN_STATES = ("forming", "committed")


class ControllerOutput:
    """One streaming call's waveform plus typed frame-level tensors."""

    def __init__(self, *, cleaned_pcm, embedding, voice_probability,
                 scene_probabilities, event_probabilities,
                 turn_probabilities, wake_probability):
        self.cleaned_pcm = cleaned_pcm
        self.embedding = embedding
        self.voice_probability = voice_probability
        self.scene_probabilities = scene_probabilities
        self.event_probabilities = event_probabilities
        self.turn_probabilities = turn_probabilities
        self.wake_probability = wake_probability

    def as_dict(self) -> dict:
        return {
            "cleaned_pcm": self.cleaned_pcm,
            "embedding": self.embedding,
            "voice_probability": self.voice_probability,
            "scene_probabilities": self.scene_probabilities,
            "event_probabilities": self.event_probabilities,
            "turn_probabilities": self.turn_probabilities,
            "wake_probability": self.wake_probability,
        }


class CausalAcousticController(nn.Module):
    """Small shared causal trunk with waveform and typed-control heads."""

    def __init__(self, hidden: int = HIDDEN,
                 recurrent_layers: int = RECURRENT_LAYERS):
        super().__init__()
        self.hidden = int(hidden)
        self.recurrent_layers = int(recurrent_layers)
        frame_width = HOP_SAMPLES * 2
        self.input_norm = nn.LayerNorm(frame_width)
        self.frame_encoder = nn.Linear(frame_width, self.hidden)
        self.recurrent = nn.GRU(
            self.hidden, self.hidden, num_layers=self.recurrent_layers,
            batch_first=True)
        self.echo_head = nn.Linear(self.hidden, HOP_SAMPLES)
        self.scene_head = nn.Linear(self.hidden, len(SCENES))
        self.event_head = nn.Linear(self.hidden, len(EVENTS))
        self.turn_head = nn.Linear(self.hidden, len(TURN_STATES))
        self.wake_head = nn.Linear(self.hidden, 1)
        # An untrained node must be waveform-safe even though none of its
        # probabilities are calibrated. Zero echo prediction is exact
        # passthrough and gives training a stable residual objective.
        nn.init.zeros_(self.echo_head.weight)
        nn.init.zeros_(self.echo_head.bias)

    def forward(self, microphone, playback_reference, recurrent_state=None):
        if microphone.ndim != 3 or microphone.shape[-1] != HOP_SAMPLES:
            raise ValueError("microphone must be [batch, frames, 160]")
        if playback_reference.shape != microphone.shape:
            raise ValueError("playback reference shape must match microphone")
        features = torch.cat((microphone, playback_reference), dim=-1)
        features = torch.nn.functional.gelu(
            self.frame_encoder(self.input_norm(features)))
        embedding, next_state = self.recurrent(features, recurrent_state)
        estimated_echo = torch.tanh(self.echo_head(embedding))
        cleaned = torch.clamp(microphone - estimated_echo, -1.0, 1.0)
        events = self.event_head(embedding)
        external = torch.sigmoid(events[..., EVENTS.index("external_speech")])
        own = torch.sigmoid(events[..., EVENTS.index("self_speech")])
        voice = torch.maximum(external, own)
        return {
            "cleaned": cleaned,
            "embedding": embedding,
            "scene_logits": self.scene_head(embedding),
            "event_logits": events,
            "turn_logits": self.turn_head(embedding),
            "wake_logits": self.wake_head(embedding)[..., 0],
            "voice_probability": voice,
            "recurrent_state": next_state,
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class AcousticController:
    """Stateful NumPy-facing streaming wrapper around the causal network."""

    def __init__(self, model: CausalAcousticController, *, device="cpu",
                 use_fp16: bool | None = None):
        self.device = torch.device(device)
        if use_fp16 is None:
            use_fp16 = self.device.type == "mps"
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.model = model.to(self.device, dtype=self.dtype).eval()
        self._state = None

    def reset(self) -> None:
        self._state = None

    def hear(self, output: ControllerOutput, threshold: float = 0.5):
        return hear(output, threshold)

    def categorise(self, heard):
        return categorise(heard)

    def state_of(self, categorised, semantic_commit_probability=None):
        return state_of(categorised, semantic_commit_probability)

    def addressed_to_me(self, turn, *, semantic_address_probability,
                        conversation_probability=0.0, threshold=0.5):
        return addressed_to_me(
            turn,
            semantic_address_probability=semantic_address_probability,
            conversation_probability=conversation_probability,
            threshold=threshold)

    def decide(self, context):
        return decide(context)

    def process(self, microphone, playback_reference) -> ControllerOutput:
        mic = np.asarray(microphone, dtype=np.float32).reshape(-1)
        reference = np.asarray(playback_reference, dtype=np.float32).reshape(-1)
        if mic.shape != reference.shape:
            raise ValueError("microphone and playback reference lengths differ")
        if not len(mic) or len(mic) % HOP_SAMPLES:
            raise ValueError("streaming blocks must be a non-empty multiple of 160")
        frames = len(mic) // HOP_SAMPLES
        mic_tensor = torch.from_numpy(mic.reshape(1, frames, HOP_SAMPLES)).to(
            self.device, dtype=self.dtype)
        ref_tensor = torch.from_numpy(
            reference.reshape(1, frames, HOP_SAMPLES)).to(
                self.device, dtype=self.dtype)
        with torch.inference_mode():
            raw = self.model(mic_tensor, ref_tensor, self._state)
        self._state = raw["recurrent_state"]

        def array(value):
            return value[0].float().cpu().numpy()

        scenes = torch.softmax(raw["scene_logits"], dim=-1)
        events = torch.sigmoid(raw["event_logits"])
        turns = torch.softmax(raw["turn_logits"], dim=-1)
        return ControllerOutput(
            cleaned_pcm=array(raw["cleaned"]).reshape(-1),
            embedding=array(raw["embedding"]),
            voice_probability=array(raw["voice_probability"]),
            scene_probabilities={
                name: array(scenes[..., index])
                for index, name in enumerate(SCENES)},
            event_probabilities={
                name: array(events[..., index])
                for index, name in enumerate(EVENTS)},
            turn_probabilities={
                name: array(turns[..., index])
                for index, name in enumerate(TURN_STATES)},
            wake_probability=array(torch.sigmoid(raw["wake_logits"])),
        )


class Heard:
    def __init__(self, output: ControllerOutput, is_voiced: bool):
        self.output = output
        self.is_voiced = bool(is_voiced)


class Categorised:
    def __init__(self, heard: Heard, category: str, probabilities: dict):
        self.heard = heard
        self.category = category
        self.probabilities = probabilities


class TurnState:
    def __init__(self, categorised: Categorised, state: str,
                 commit_probability: float):
        self.categorised = categorised
        self.state = state
        self.commit_probability = float(commit_probability)


class AddressState:
    def __init__(self, turn: TurnState, addressed: bool,
                 probability: float):
        self.turn = turn
        self.addressed = bool(addressed)
        self.probability = float(probability)


class Action:
    def __init__(self, context: AddressState, action: str):
        self.context = context
        self.action = action


def _last_mean(values, frames: int = 3) -> float:
    value = np.asarray(values, dtype=np.float32).reshape(-1)
    return float(np.mean(value[-frames:])) if len(value) else 0.0


def hear(output: ControllerOutput, threshold: float = 0.5) -> Heard:
    """HEAR: neural voice evidence, not an RMS/energy test."""
    return Heard(output, _last_mean(output.voice_probability) >= threshold)


def categorise(heard: Heard) -> Categorised:
    """CATEGORISE: silence/environment/self/speech from acoustic tensors."""
    probabilities = {
        name: _last_mean(values)
        for name, values in heard.output.scene_probabilities.items()}
    category = max(probabilities, key=probabilities.get)
    return Categorised(heard, category, probabilities)


def state_of(categorised: Categorised,
             semantic_commit_probability: float | None = None) -> TurnState:
    """STATE: combine acoustic and later neural semantic completion.

    The optional semantic score comes from JaegerOmni/Gemma embeddings, never
    a decoded transcript. When absent, the acoustic head remains advisory.
    """
    acoustic = _last_mean(
        categorised.heard.output.turn_probabilities["committed"])
    probability = (acoustic if semantic_commit_probability is None else
                   acoustic * float(semantic_commit_probability))
    return TurnState(
        categorised, "committed" if probability >= 0.5 else "forming",
        probability)


def addressed_to_me(turn: TurnState, *, semantic_address_probability: float,
                     conversation_probability: float = 0.0,
                     threshold: float = 0.5) -> AddressState:
    """CONTEXT: fuse later neural addressee and ongoing-conversation scores."""
    wake = _last_mean(turn.categorised.heard.output.wake_probability)
    probability = max(
        float(semantic_address_probability), wake,
        float(conversation_probability))
    return AddressState(turn, probability >= threshold, probability)


def decide(context: AddressState) -> Action:
    """RESPOND: answer only committed, external, addressed speech."""
    category = context.turn.categorised.category
    answer = (context.addressed and context.turn.state == "committed" and
              context.turn.categorised.heard.is_voiced and
              category == "speech")
    return Action(context, "answer" if answer else "log_only")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(*, checkpoint: str | Path | None = None, device="cpu",
          initialize_untrained: bool = False) -> AcousticController:
    """Build the disabled node or fail closed when weights are absent."""
    path = Path(checkpoint).expanduser().resolve() if checkpoint else \
        DEFAULT_CHECKPOINT
    model = CausalAcousticController()
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("architecture") != ARCHITECTURE:
            raise ValueError(f"{path}: incompatible controller architecture")
        if int(saved.get("sample_rate", 0)) != SAMPLE_RATE:
            raise ValueError(f"{path}: incompatible sample rate")
        model.load_state_dict(saved["state_dict"], strict=True)
    elif not initialize_untrained:
        raise RuntimeError(
            "neural-acoustic-controller has no promoted checkpoint; "
            "use initialize_untrained=True only for CPU contract tests")
    return AcousticController(model, device=device)
