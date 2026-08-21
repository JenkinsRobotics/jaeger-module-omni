"""Train the Gemma-to-Qwen hidden bridge on captured teacher activations."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from jaeger_agent_omni.neural.speech import (  # noqa: E402
    GemmaQwenHiddenBridge, HiddenBridgeConfig)


def load_matrix(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        rows, width = struct.unpack("<ii", f.read(8))
        raw = np.frombuffer(f.read(), dtype="<f4").copy()
    return torch.from_numpy(raw.reshape(rows, width))


def resample(x: torch.Tensor, n: int) -> torch.Tensor:
    if len(x) == n:
        return x
    return F.interpolate(x.T[None], size=n, mode="linear",
                         align_corners=True)[0].T.contiguous()


def load_pairs(pair_path: Path, teacher: Path) -> list[dict]:
    source = torch.load(pair_path, map_location="cpu", weights_only=False)
    items = []
    for i, item in enumerate(source["items"]):
        chunks = sorted((teacher / f"round_{i:03d}/llm_debug").glob(
            "chunk_*/llm_hidden_states.bin"))
        target = torch.cat([load_matrix(p) for p in chunks])
        target = resample(target, len(item["hidden_states"]))
        items.append({"case": item["case"],
                      "source": item["hidden_states"].float(),
                      "target": target.float()})
    return items


def stack(items: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.cat([x["source"] for x in items]),
            torch.cat([x["target"] for x in items]))


@torch.no_grad()
def metrics(model, data, device) -> dict:
    source, target = (x.to(device) for x in data)
    pred = model(source)
    return {"loss": float(model.loss(pred, target).cpu()),
            "cosine": float(F.cosine_similarity(pred, target, dim=-1).mean().cpu()),
            "mse": float(F.mse_loss(pred, target).cpu())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=500)
    args = ap.parse_args()
    items = load_pairs(args.pairs, args.teacher)
    val = [x for i, x in enumerate(items) if i % 5 == 0]
    train = [x for i, x in enumerate(items) if i % 5 != 0]
    train_data, val_data = stack(train), stack(val)
    target_norm = float(train_data[1].norm(dim=-1).mean())
    device = torch.device(args.device)
    torch.manual_seed(0)
    model = GemmaQwenHiddenBridge(HiddenBridgeConfig(
        source_dim=train_data[0].shape[-1], target_dim=train_data[1].shape[-1],
        rank=args.rank, target_norm=target_norm)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    source, target = (x.to(device) for x in train_data)
    before = {"train": metrics(model, train_data, device),
              "val": metrics(model, val_data, device)}
    best, best_state, best_step, stale = -1.0, None, 0, 0
    for step in range(1, args.steps + 1):
        model.train()
        pred = model(source)
        loss = model.loss(pred, target)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % args.report_every == 0 or step == 1:
            vm = metrics(model.eval(), val_data, device)
            print(f"step {step:4d} train {float(loss.detach().cpu()):.4f} "
                  f"val-cos {vm['cosine']:.4f} val-loss {vm['loss']:.4f}",
                  flush=True)
            if vm["cosine"] > best:
                best, best_step, stale = vm["cosine"], step, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                stale += args.report_every
            if stale >= args.patience:
                print(f"early stop: no holdout gain for {stale} steps")
                break
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    after = {"train": metrics(model, train_data, "cpu"),
             "val": metrics(model, val_data, "cpu")}
    ck = model.checkpoint()
    ck.update({"best_step": best_step, "metrics_before": before,
               "metrics_after": after,
               "train_cases": [x["case"] for x in train],
               "val_cases": [x["case"] for x in val]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.out)
    print(json.dumps({"best_step": best_step, "before": before,
                      "after": after}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
