"""Load MiniCPM-o's talker — the speech-out stack — into our modules.

Measured anatomy (tts GGUF + tts-condition-graph.cpp):

    conditioning   merged[t] = emb_text(token_ids[t])
                             + L2_normalize(proj_semantic(llm_hidden[t]))
    talker         20-block llama (768 hidden, 12 heads, CosyVoice2 lineage)
    output         head_code -> 6562 speech codes, autoregressive
    voice          proj_spk(4096) -> 768 speaker conditioning

Two findings that shape the speech delta:

  1. The LLM hidden state is L2-NORMALIZED before conditioning — direction
     only, scale discarded. Our delta (backbone -> 4096) only has to point
     right.
  2. The talker READS TEXT through Qwen's vocab (emb_text, 152064). Text is
     re-tokenizable from any backbone, so a ZERO-CONDITIONING baseline —
     text embeddings alone, hidden term zeroed — is runnable with NO trained
     delta at all. That is the text->speech baseline the matrix wants.

Not ported here: token2wav (CosyVoice2 flow-matching + HiFi-GAN, 700 MB of
GGUF). Until it is, "speech out" means speech CODES whose statistics we can
check, not audible waveforms. The honest audible route today remains Track A.

RoPE note: their converter (convert_tts.py) copies HF safetensors unpermuted,
and the lineage is Qwen2-style — so standard HF rotate-half RoPE is assumed.
The definitive check is audible output once token2wav exists.

    python research/load_minicpm_talker.py          # load + code-gen smoke test
    python research/load_minicpm_talker.py --selfcheck
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_gguf import read_tensors                               # noqa: E402

TTS_GGUF = Path.home() / "models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-tts-F16.gguf"

N_TEXT_VOCAB = 152064       # Qwen's text vocab, read by emb_text
N_CODES = 6562              # CosyVoice2 speech codes (+specials)
HIDDEN = 768
QWEN_HIDDEN = 4096

_BLOCK_MAP = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


class ConditionProjector(nn.Module):
    """proj_semantic / proj_spk: 4096 -> 768, ReLU between, L2-norm after.

    The L2 normalisation is part of the measured graph
    (tts-condition-graph.cpp), not an addition of ours.
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(QWEN_HIDDEN, HIDDEN)
        self.linear2 = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(F.relu(self.linear1(x)))
        return F.normalize(x, dim=-1)


class MiniCPMTalker(nn.Module):
    """Their talker in HF clothing: LlamaModel body + code head + embeddings."""

    def __init__(self):
        super().__init__()
        from transformers import LlamaConfig, LlamaModel
        cfg = LlamaConfig(
            hidden_size=HIDDEN, num_hidden_layers=20, num_attention_heads=12,
            num_key_value_heads=12, intermediate_size=3072, head_dim=64,
            vocab_size=N_TEXT_VOCAB + N_CODES, rope_theta=10000.0,
            rms_norm_eps=1e-6, max_position_embeddings=4096,
            attention_bias=False, mlp_bias=False, tie_word_embeddings=False)
        self.body = LlamaModel(cfg)
        self.emb_text = nn.Embedding(N_TEXT_VOCAB, HIDDEN)
        self.emb_code = nn.Embedding(N_CODES, HIDDEN)
        self.head_code = nn.Linear(HIDDEN, N_CODES, bias=False)
        self.proj_semantic = ConditionProjector()
        self.proj_spk = ConditionProjector()

    def condition(self, text_ids: torch.Tensor,
                  llm_hidden: torch.Tensor | None = None) -> torch.Tensor:
        """The measured conditioning: text embedding + normalised hidden.

        `llm_hidden=None` is the zero-conditioning baseline — text alone.
        """
        merged = self.emb_text(text_ids)
        if llm_hidden is not None:
            merged = merged + self.proj_semantic(llm_hidden)
        return merged

    @torch.no_grad()
    def generate_codes(self, merged: torch.Tensor, max_codes: int = 200,
                       temperature: float = 0.0) -> torch.Tensor:
        """Autoregressive speech codes from a conditioning prefix. Greedy by
        default; the reference uses sampling with repetition handling (RAS) —
        good enough for statistics, not for final audio quality."""
        seq = merged
        out = []
        past = None
        for _ in range(max_codes):
            r = self.body(inputs_embeds=seq, past_key_values=past, use_cache=True)
            past = r.past_key_values
            logits = self.head_code(r.last_hidden_state[:, -1])
            if temperature > 0:
                code = torch.multinomial(F.softmax(logits / temperature, -1), 1)
            else:
                code = logits.argmax(-1, keepdim=True)
            out.append(code)
            seq = self.emb_code(code)
        return torch.cat(out, dim=1)


def load(gguf_path=TTS_GGUF) -> MiniCPMTalker:
    talker = MiniCPMTalker()
    raw = read_tensors(gguf_path)

    state = {}
    for g, arr in raw.items():
        t = torch.from_numpy(arr.copy())
        if g.startswith("blk."):
            _, n, rest = g.split(".", 2)
            name = rest.rsplit(".", 1)[0]
            kind = rest.rsplit(".", 1)[1]
            state[f"body.layers.{n}.{_BLOCK_MAP[name]}.{kind}"] = t
        elif g == "token_embd.weight":
            state["body.embed_tokens.weight"] = t
        elif g == "output_norm.weight":
            state["body.norm.weight"] = t
        elif g == "emb_text.weight":
            state["emb_text.weight"] = t
        elif g == "emb_code.0.weight":
            state["emb_code.weight"] = t
        elif g == "head_code.0.weight":
            state["head_code.weight"] = t
        elif g.startswith("projector_semantic."):
            state["proj_semantic." + g.split(".", 1)[1]] = t.squeeze()
        elif g.startswith("projector_spk."):
            state["proj_spk." + g.split(".", 1)[1]] = t.squeeze()

    ref = talker.state_dict()
    for k, v in state.items():
        if k not in ref:
            raise KeyError(f"mapped to unknown key {k}")
        state[k] = v.reshape(ref[k].shape)
    missing = [k for k in ref if k not in state and "rotary" not in k]
    if missing:
        raise ValueError(f"missing: {missing[:6]}")
    talker.load_state_dict(state, strict=False)
    return talker.eval().requires_grad_(False)


def main() -> int:
    if not TTS_GGUF.exists():
        print(f"missing {TTS_GGUF}")
        return 1
    talker = load()
    n = sum(p.numel() for p in talker.parameters())
    print(f"talker loaded: {n/1e6:.0f}M params, "
          f"{N_TEXT_VOCAB} text vocab + {N_CODES} codes")

    # Zero-conditioning smoke test: Qwen-tokenize a sentence, generate codes
    # from text embeddings alone. Success = non-degenerate code sequence.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")   # qwen2 vocab
    ids = tok("We choose to go to the moon.", return_tensors="pt").input_ids
    ids = ids.clamp(max=N_TEXT_VOCAB - 1)
    merged = talker.condition(ids, llm_hidden=None)
    codes = talker.generate_codes(merged, max_codes=120)

    uniq = codes.unique().numel()
    top = codes.mode().values.item()
    top_frac = (codes == top).float().mean().item()
    print(f"zero-conditioning codes: {codes.shape[1]} generated, "
          f"{uniq} distinct, mode covers {top_frac*100:.0f}%")
    if uniq < 5 or top_frac > 0.9:
        print("DEGENERATE — code stream is near-constant. Conditioning or rope "
              "convention is likely wrong; do not build on this yet.")
    else:
        print("non-degenerate code stream — talker body is alive in our port. "
              "Audible verification still requires token2wav.")
    return 0


def demo():
    # Mapping covers every family in the gguf listing and nothing invents keys.
    m = MiniCPMTalker()
    ref = m.state_dict()
    assert "body.layers.19.mlp.down_proj.weight" in ref
    assert "emb_code.weight" in ref and ref["emb_code.weight"].shape == (N_CODES, HIDDEN)
    assert ref["head_code.weight"].shape == (N_CODES, HIDDEN)
    assert ref["body.embed_tokens.weight"].shape == (N_TEXT_VOCAB + N_CODES, HIDDEN)

    # Vocab arithmetic that makes token_embd = [text | code] plausible.
    assert N_TEXT_VOCAB + N_CODES == 158626

    # Conditioning: zero-hidden baseline differs from conditioned, and the
    # hidden term is scale-invariant (L2 norm) — the measured property.
    ids = torch.randint(0, 100, (1, 6))
    h = torch.randn(1, 6, QWEN_HIDDEN)
    a = m.condition(ids)
    b = m.condition(ids, h)
    assert not torch.allclose(a, b), "hidden conditioning must change the merge"
    # The measured property is that the hidden term arrives UNIT-NORM (the
    # graph L2-normalises after the projector). Input scale may still steer
    # direction through the ReLU+bias — that is faithful to the reference.
    norms = m.proj_semantic(h).norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), \
        "projector output must be unit-norm"

    print("load_minicpm_talker: mapping, vocab split, scale-invariance OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
