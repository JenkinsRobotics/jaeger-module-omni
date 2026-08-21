"""Fit Gemma token embeddings to MiniCPM Talker's lexical conditions.

Thousands of shared English words supervise the lexical branch before the
small contextual activation corpus is used. Runtime remains tensor-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def words(n: int) -> list[str]:
    source = Path("/usr/share/dict/words")
    values = [w.strip().lower() for w in source.read_text().splitlines()
              if 2 <= len(w.strip()) <= 12 and w.strip().isalpha()]
    values = sorted(set(values))
    rng = np.random.default_rng(0)
    rng.shuffle(values)
    return [" " + w for w in values[:n]]


def build_pairs(model_path: str, talker_table: Path, n: int,
                batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    from mlx_vlm.utils import load
    from transformers import AutoTokenizer

    model, processor = load(model_path)
    gemma_tok = processor.tokenizer
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if qwen_tok.pad_token_id is None:
        qwen_tok.pad_token = qwen_tok.eos_token
    table = np.load(talker_table, mmap_mode="r")
    xs, ys = [], []
    values = words(n)
    for start in range(0, len(values), batch_size):
        text = values[start:start + batch_size]
        g = gemma_tok(text, add_special_tokens=False, padding=True,
                      return_tensors="np")
        ids = mx.array(g.input_ids)
        embedded = model.get_input_embeddings(input_ids=ids)
        embedded = embedded[0] if isinstance(embedded, tuple) else getattr(
            embedded, "inputs_embeds", embedded)
        mask = mx.array(g.attention_mask).astype(mx.float32)[..., None]
        mean = (embedded.astype(mx.float32) * mask).sum(1) / mask.sum(1)
        mx.eval(mean)
        xs.append(np.array(mean, copy=False))

        q = qwen_tok(text, add_special_tokens=False, padding=True,
                     return_tensors="np")
        qm = q.attention_mask[..., None].astype(np.float32)
        y = (table[q.input_ids] * qm).sum(1) / qm.sum(1)
        ys.append(np.asarray(y, dtype=np.float32))
        if start == 0 or (start // batch_size) % 10 == 0:
            print(f"embedded {min(start + batch_size, len(values))}/{len(values)}",
                  flush=True)
    return (torch.from_numpy(np.concatenate(xs)),
            torch.from_numpy(np.concatenate(ys)))


def rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=str(
        Path.home() / "models/gemma-4-e4b-it-mlx-full"))
    ap.add_argument("--talker-table", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--words", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()
    x, y = build_pairs(args.model, args.talker_table, args.words, args.batch)
    idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))
    cut = int(len(idx) * 0.9)
    tr, va = idx[:cut], idx[cut:]
    x = rms(x)
    layer = nn.Linear(x.shape[1], y.shape[1])
    opt = torch.optim.AdamW(layer.parameters(), lr=args.lr, weight_decay=0.01)
    for epoch in range(1, args.epochs + 1):
        order = tr[torch.randperm(len(tr))]
        for start in range(0, len(order), args.batch):
            take = order[start:start + args.batch]
            pred = layer(x[take])
            loss = F.mse_loss(pred, y[take]) + 0.05 * (
                1 - F.cosine_similarity(pred, y[take], dim=-1).mean())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            pred = layer(x[va])
            cos = float(F.cosine_similarity(pred, y[va], dim=-1).mean())
            mse = float(F.mse_loss(pred, y[va]))
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d} holdout mse {mse:.5f} cosine {cos:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "jaeger-lexical-bridge-v1",
                "state": layer.state_dict(), "holdout_cosine": cos,
                "holdout_mse": mse, "words": len(x)}, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
