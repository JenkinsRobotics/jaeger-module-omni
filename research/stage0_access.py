"""Stage 0 — raw access check.

Proves we can pull tensors out of both models under `transformers`, and reports
the real memory cost. Nothing is trained here.

    pip install torch transformers accelerate numpy
    python research/stage0_access.py

Gemma is a gated repo — accept the license on HF and `huggingface-cli login`
first, or point GEMMA_ID at a local snapshot directory.
"""

import os
import numpy as np
import torch

# ponytail: verify GEMMA_ID against the actual repo before the first run — the
# exact Gemma 4 E4B id is not guessed correctly by this script's author.
WHISPER_ID = os.environ.get("WHISPER_ID", "openai/whisper-medium")
GEMMA_ID = os.environ.get("GEMMA_ID", "google/gemma-4-E4B-it")

# Whisper-medium encoder: 24 layers, d_model 1024, always 1500 frames (30s pad).
WHISPER_FRAMES = 1500
WHISPER_DIM = 1024


def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        # ponytail: fp16 on MPS — bf16 support is partial and silently falls
        # back. Switch to bf16 when training moves to CUDA.
        return "mps", torch.float16
    return "cpu", torch.float32


def mem_gb(device):
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**3
    if device == "mps":
        return torch.mps.current_allocated_memory() / 1024**3
    return float("nan")


def check_whisper(device, dtype):
    from transformers import WhisperProcessor, WhisperModel

    proc = WhisperProcessor.from_pretrained(WHISPER_ID)
    model = WhisperModel.from_pretrained(WHISPER_ID, dtype=dtype).to(device).eval()

    # Plumbing check only — random audio is enough to prove shapes.
    audio = np.random.randn(16000 * 5).astype(np.float32) * 0.01
    feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        hidden = model.encoder(feats.to(device, dtype)).last_hidden_state

    print(f"  whisper encoder out : {tuple(hidden.shape)}  {hidden.dtype}")
    print(f"  5s of audio         : {WHISPER_FRAMES} frames — padded to 30s regardless")
    assert hidden.shape[1] == WHISPER_FRAMES, hidden.shape
    assert hidden.shape[2] == WHISPER_DIM, hidden.shape
    return hidden.shape[2]


def check_gemma(device, dtype):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(GEMMA_ID)
    model = AutoModelForCausalLM.from_pretrained(GEMMA_ID, dtype=dtype).to(device).eval()

    ids = tok("what is two plus two", return_tensors="pt").input_ids.to(device)
    embed = model.get_input_embeddings()(ids)
    with torch.no_grad():
        out = model(inputs_embeds=embed, output_hidden_states=True)

    print(f"  input embeddings    : {tuple(embed.shape)}  {embed.dtype}")
    print(f"  hidden layers       : {len(out.hidden_states)}")
    print(f"  hidden size         : {model.config.hidden_size}")
    assert embed.shape[-1] == model.config.hidden_size
    # The projector must emit exactly this — it is the target space.
    return model.config.hidden_size


def main():
    device, dtype = device_and_dtype()
    print(f"device {device} / dtype {dtype}\n")

    print(f"whisper  {WHISPER_ID}")
    audio_dim = check_whisper(device, dtype)
    print(f"  peak mem            : {mem_gb(device):.1f} GB\n")

    print(f"gemma    {GEMMA_ID}")
    cog_dim = check_gemma(device, dtype)
    print(f"  peak mem (both)     : {mem_gb(device):.1f} GB\n")

    print(f"JaegerAudioProjector must map {audio_dim} -> {cog_dim}")
    print(f"and compress {WHISPER_FRAMES} frames (50/s) down to ~6-12 tokens/s.")
    print("\nStage 0 passed.")


if __name__ == "__main__":
    main()
