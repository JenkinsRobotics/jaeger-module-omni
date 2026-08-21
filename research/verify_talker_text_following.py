"""Does the talker follow text when given REAL Qwen hidden states?

The decisive speech-out experiment. Zero-conditioning babbles (measured twice);
this substitutes the one missing ingredient — genuine Qwen3-8B hidden states,
the distribution the talker was trained on — and asks whisper whether the
output finally says the requested words.

    verdict A: whisper hears the sentence  -> talker port fully verified, and
               Qwen states become the output-delta teacher. Speech-out reduces
               to training backbone->Qwen-space, same trick as the input delta.
    verdict B: still babble               -> our conditioning sequence differs
               from theirs in some further way; back to omni.cpp.

Teacher-forcing approximation: their runtime conditions on hidden states of
text the LLM is *generating* after <|tts_bos|>; we teacher-force the same
tokens through the same model and read the states. Same tokens, same model,
same positions — the states are equivalent up to sampling context.

RAM note: Qwen3-8B fp16 needs ~16 GB. Run AFTER the MPS training budget ends,
not beside it — unified memory.

    python research/verify_talker_text_following.py [--say "..."] [--keep-qwen]
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QWEN_ID = "Qwen/Qwen3-8B"


def qwen_states_for(text: str) -> tuple:
    """(token_ids [1,T], hidden [1,T,4096]) for the spoken text, then free Qwen."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN_ID)
    lm = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.float16,
                                              low_cpu_mem_usage=True)
    lm.eval().requires_grad_(False)

    # Their template: the assistant's spoken text follows <|tts_bos|>. We
    # teacher-force exactly that layout and read states at the text positions.
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": f"Say this sentence aloud: {text}"}],
        add_generation_prompt=True, tokenize=False) + "<|tts_bos|>"
    prefix_ids = tok(prompt, return_tensors="pt",
                     add_special_tokens=False).input_ids
    text_ids = tok(text, return_tensors="pt",
                   add_special_tokens=False).input_ids

    full = torch.cat([prefix_ids, text_ids], dim=1)
    with torch.no_grad():
        out = lm(input_ids=full, output_hidden_states=True)
    hidden = out.hidden_states[-1][:, prefix_ids.shape[1]:, :].float()

    del lm, out
    gc.collect()
    return text_ids, hidden


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--say", default="We choose to go to the moon.")
    ap.add_argument("--out", default="training/runs/t2w_conditioned.wav")
    ap.add_argument("--ref-audio", default="training/samples/jfk_moon.wav")
    args = ap.parse_args()

    print(f"1/4 Qwen hidden states for {args.say!r} (fp16, ~16 GB, one forward)")
    text_ids, hidden = qwen_states_for(args.say)
    print(f"    {text_ids.shape[1]} text tokens, hidden {tuple(hidden.shape)}")

    print("2/4 talker: condition = emb_text + L2norm(proj(hidden)), + audio_bos")
    from load_minicpm_talker import load as load_talker
    talker = load_talker()
    # One-shot proof: this is the final text chunk, so OpenBMB's generator
    # appends text_eos before audio_bos.  Omitting text_eos was the reason the
    # earlier port ran to 600 codes and ignored the requested sentence.
    merged = talker.condition(text_ids, llm_hidden=hidden, text_finished=True)
    codes = talker.generate_codes(merged, max_codes=600)
    print(f"    {codes.shape[1]} codes, {codes.unique().numel()} distinct")

    print("3/4 token2wav")
    from t2w_client import synthesize
    out = synthesize(codes[0].tolist(), args.out, args.ref_audio)

    print("4/4 whisper round-trip")
    from jaeger_agent_omni.app import OmniApp, load_models, read_wav
    p, w, t, b = load_models("openai/whisper-tiny",
                             "HuggingFaceTB/SmolLM2-135M-Instruct", device="cpu")
    app = OmniApp(p, w, t, b, path="text", device="cpu")
    heard = app.transcribe(read_wav(out))

    print(f"\nasked : {args.say!r}")
    print(f"heard : {heard!r}")
    asked = set(args.say.lower().replace('.', '').split())
    got = set(heard.lower().replace('.', '').replace(',', '').split())
    overlap = len(asked & got) / max(1, len(asked))
    print(f"word overlap {overlap*100:.0f}%")
    if overlap >= 0.6:
        print("VERDICT A — the talker follows text with real conditioning. "
              "Port verified; Qwen is the output-delta teacher.")
    else:
        print("VERDICT B — conditioning still differs from theirs somewhere. "
              "Next: diff our condition sequence against omni.cpp simplex path.")
    return 0


def demo():
    print("verify_talker_text_following: run manually — needs 16 GB for Qwen")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
