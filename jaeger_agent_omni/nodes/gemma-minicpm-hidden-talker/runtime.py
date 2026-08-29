"""Gemma tensor state to cloned speech; no decoded-text handoff."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Engine:
    def __init__(self, talker_device="mps", vocoder_device="cpu"):
        import torch

        from jaeger_agent_omni.neural.speech import (
            GemmaQwenHiddenBridge,
            TokenIdBridge,
        )
        from research.load_minicpm_talker import load as load_talker

        self.torch = torch
        self.vocoder_device = vocoder_device
        self.token_bridge = TokenIdBridge.load(
            ROOT / "weights/gemma_to_qwen_token_ids.npz")
        checkpoint = torch.load(
            ROOT / "weights/gemma_qwen_hidden_rank128.pt",
            map_location="cpu", weights_only=False)
        self.hidden_bridge = GemmaQwenHiddenBridge.from_checkpoint(checkpoint)
        self.talker = load_talker(
            ROOT / "weights/talker/MiniCPM-o-4_5-tts-F16.gguf")
        if talker_device == "mps":
            self.talker = self.talker.to(device="mps", dtype=torch.float16)
        elif talker_device != "cpu":
            raise ValueError("talker_device must be cpu or mps")
        dtype = next(self.talker.parameters()).dtype
        device = next(self.talker.parameters()).device
        self.hidden_bridge = self.hidden_bridge.to(device=device, dtype=dtype)
        self.client = None
        self.reference = None

    def condition(self, token_ids, hidden_states):
        """Return inherited Talker conditions from Gemma tensors only."""
        torch = self.torch
        hidden = torch.as_tensor(hidden_states)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != 2560:
            raise ValueError("hidden_states must be [tokens, 2560] or [1, tokens, 2560]")
        qwen_ids, aligned = self.token_bridge.expand(token_ids, hidden)
        device = next(self.talker.parameters()).device
        dtype = next(self.talker.parameters()).dtype
        qwen_ids = qwen_ids.to(device)
        aligned = aligned.to(device=device, dtype=dtype)
        with torch.no_grad():
            qwen_hidden = self.hidden_bridge(aligned)
            return (self.talker.emb_text(qwen_ids) +
                    self.talker.proj_semantic(qwen_hidden))

    def set_reference_voice(self, ref_audio):
        from research.t2w_client import Token2WavClient, _clip_reference

        if self.client is None:
            self.client = Token2WavClient()
            self.client.init(
                model_dir=ROOT / "weights/token2wav",
                device=self.vocoder_device)
        reference = _clip_reference(ref_audio)
        self.client.set_ref_audio(reference)
        self.reference = str(reference)

    def synthesize(self, token_ids, hidden_states, out_path, *,
                   ref_audio=None, seed=4, max_codes=180):
        """Stream speech codes into the cloned-voice neural decoder."""
        from research.t2w_client import synthesize_code_stream_with_client

        if ref_audio is not None and str(ref_audio) != self.reference:
            self.set_reference_voice(ref_audio)
        if self.client is None:
            raise ValueError("set a reference voice before synthesis")
        condition = self.condition(token_ids, hidden_states)
        self.torch.manual_seed(int(seed))
        with self.torch.no_grad():
            codes = self.talker.iter_code_chunks(
                [(condition, True)], max_codes=int(max_codes))
            output, metrics = synthesize_code_stream_with_client(
                self.client, codes, out_path)
        return output, metrics

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None


def build(*, talker_device="mps", vocoder_device="cpu"):
    return Engine(talker_device=talker_device, vocoder_device=vocoder_device)
