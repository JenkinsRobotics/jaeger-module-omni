"""Can gemma-4-E4B train the delta on this Mac? Measure, do not assume.

The backbone is frozen but gradients must still flow through all 42 layers to
reach the delta at the input, so its activations are live. This loads the real
model, runs one forward+backward with a realistic prefix, and reports peak RSS.

    python research/e4b_fit.py
"""
from __future__ import annotations

import os, resource, time

import torch

MODEL = "google/gemma-4-E4B-it"


def rss_gb() -> float:
    """Process RSS. NOTE: MPS allocations live in Metal's unified-memory pool
    and do NOT appear here -- a 15 GB model reads as ~1 GB RSS. Use mps_gb()
    for the number that actually decides whether a run fits."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def mps_gb() -> float:
    try:
        return torch.mps.driver_allocated_memory() / 2**30
    except Exception:
        return float("nan")


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    m = m.to(dev).eval().requires_grad_(False)
    print(f"loaded {time.perf_counter()-t0:.1f}s  peak {rss_gb():.1f} GB  dev={dev}")

    h = m.config.get_text_config().hidden_size
    print(f"hidden_size {h}  layers {m.config.get_text_config().num_hidden_layers}")

    # A delta emits audio-slot embeddings; they must carry grad into the stack.
    n_audio, n_text = 40, 24
    audio = torch.randn(1, n_audio, h, device=dev, dtype=torch.bfloat16,
                        requires_grad=True)
    ids = tok(" The capital of France is Paris.", return_tensors="pt").to(dev)
    txt = m.get_input_embeddings()(ids["input_ids"]).to(torch.bfloat16)
    inputs = torch.cat([audio, txt], dim=1)
    labels = torch.cat([torch.full((1, n_audio), -100, device=dev),
                        ids["input_ids"]], dim=1)

    # PER_LAYER_INPUTS. E4B feeds a 256-wide vector per token PER LAYER (42 of
    # them) alongside the ordinary embedding; without it the model is fluent
    # and wrong. Passing inputs_embeds with input_ids=None makes transformers
    # try to RECOVER the ids by comparing every embedding against the whole
    # vocab matrix -- [1,T,1,2560] vs [1,1,262144,2560], which asks for 29 GiB
    # and then fails anyway, because delta-produced audio embeddings are not
    # in the vocab by construction.
    #
    # So compute them ourselves from PLACEHOLDER ids (audio token in the audio
    # slots, real ids for text) and pass them in. forward() accepts
    # per_layer_inputs directly; it only refuses it alongside input_ids.
    audio_id = getattr(m.config, "audio_token_id", None) or tok.pad_token_id
    slot_ids = torch.cat([torch.full((1, n_audio), audio_id, device=dev,
                                     dtype=ids["input_ids"].dtype),
                          ids["input_ids"]], dim=1)
    lm = m.language_model if hasattr(m, "language_model") else m.model.language_model
    per_layer = lm.get_per_layer_inputs(slot_ids, None)
    print(f"per_layer_inputs {tuple(per_layer.shape)}")

    t0 = time.perf_counter()
    out = m(inputs_embeds=inputs, per_layer_inputs=per_layer, labels=labels)
    fwd = time.perf_counter() - t0
    print(f"forward  {fwd:.2f}s  loss {out.loss.item():.3f}  peak {rss_gb():.1f} GB")

    t0 = time.perf_counter()
    out.loss.backward()
    bwd = time.perf_counter() - t0
    ok = audio.grad is not None and torch.isfinite(audio.grad).all()
    print(f"backward {bwd:.2f}s  grad_reaches_delta={bool(ok)}  mps {mps_gb():.1f} GB")

    # The first step pays Metal kernel compilation; only steady state predicts
    # how long a real run takes.
    times = []
    for _ in range(5):
        audio.grad = None
        t0 = time.perf_counter()
        o = m(inputs_embeds=inputs, per_layer_inputs=per_layer, labels=labels)
        o.loss.backward()
        torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    steady = sorted(times)[len(times) // 2]
    print(f"steady   {steady:.2f}s/step  (first was {fwd+bwd:.2f}s -- Metal compile)")
    print(f"memory   mps {mps_gb():.1f} GB   rss {rss_gb():.1f} GB (rss excludes MPS)")
    for n in (500, 2000):
        print(f"         {n} steps ~ {n*steady/60:.0f} min")
    print(f"\nVERDICT: {'FITS' if mps_gb() < 24 else 'TIGHT'} at {mps_gb():.1f} GB, "
          f"{steady:.2f}s/step (seq {n_audio+n_text})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
