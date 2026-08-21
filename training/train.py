"""The Stage 3/4 training loop.

Trains `JaegerAudioProjector` to land audio where the text teacher already
lands. Whisper frozen, backbone frozen, projector trainable — gradients still
flow *through* the backbone, which is why VRAM stays high even though only ~6M
parameters are being optimised.

    python training/train.py --stub                  # end-to-end on stubs, ~10s
    python training/train.py --backbone <hf-id>      # a real run

`--stub` is not a toy: it exercises the identical loop, optimiser, loss and
checkpoint path against tiny random models. If the loop is broken, it fails
here in seconds rather than an hour into a rented GPU.

ponytail: a plain loop, not Trainer/Lightning/accelerate. One model is
trainable and one process is enough; reach for a framework when distributed
training is actually needed.
"""

import argparse
import json
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))

from jaeger_agent_omni.neural import JaegerAudioProjector      # noqa: E402
from jaeger_agent_omni.prompt import (ChatWrapper, DEFAULT_SYSTEM,  # noqa: E402
                                      WARMUP_SYSTEM)
from config import TrainConfig, distill as distill_cfg, warmup  # noqa: E402
from dataset import IGNORE_INDEX  # noqa: E402
from distill import DistillWeights, agreement_rate, total_loss  # noqa: E402


def pick_device(requested: str | None = None) -> tuple:
    """Device + dtype. bf16 on CUDA; fp16 on MPS, where bf16 falls back."""
    if requested == "cpu" or (requested is None and not torch.cuda.is_available()
                              and not torch.backends.mps.is_available()):
        return "cpu", torch.float32
    if requested == "cuda" or (requested is None and torch.cuda.is_available()):
        return "cuda", torch.bfloat16
    return "mps", torch.float16


class Trainer:
    def __init__(self, cfg: TrainConfig, encoder, backbone, projector,
                 device: str = "cpu", wrapper=None):
        self.cfg = cfg
        self.device = device
        self.encoder = encoder.to(device).requires_grad_(False).eval()
        self.backbone = backbone.to(device).requires_grad_(False).eval()
        self.projector = projector.to(device)

        # Fail loudly on a dim mismatch. Broadcasting one of these silently is
        # how a run burns a day producing noise.
        want = _hidden_size(backbone)
        if projector.cognition_dim != want:
            raise ValueError(
                f"projector emits {projector.cognition_dim}, backbone wants {want}. "
                "cognition_dim must be read from the backbone config, never hardcoded.")

        self.dtype = next(backbone.parameters()).dtype
        # Audio goes where the user's words would go, inside the chat template.
        # Bare embeddings leave an instruct backbone with no idea it is in a
        # conversation, and it falls back to generic instruction-following.
        self.wrapper = wrapper
        trainable = [p for p in projector.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(trainable, lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.opt, max_lr=cfg.lr, total_steps=cfg.max_steps,
            pct_start=cfg.warmup_steps / cfg.max_steps)
        self.weights = (DistillWeights.warmup() if cfg.stage == "stage3_warmup"
                        else DistillWeights.distill())
        self.step = 0
        self.history = []

    def teacher(self, token_ids):
        """Text path: same frozen weights, transcript instead of audio."""
        with torch.no_grad():
            return self.backbone(input_ids=token_ids, output_hidden_states=True)

    def audio_prefix(self, waveform, durations=None) -> torch.Tensor:
        """waveform/features -> [B, n_audio_tokens, cognition_dim]."""
        with torch.no_grad():
            audio = self.encoder(waveform).last_hidden_state
        # Trim when durations are known: untrimmed, every clip costs the token
        # budget of a full 30 s window regardless of its real length.
        if durations is not None and len(set(durations)) == 1:
            embeds = self.projector(audio.float(), duration_s=durations[0])
        else:
            embeds = self.projector(audio.float())
        # self.dtype, not backbone.dtype — nn.Module has no .dtype and the stub
        # backbone is a plain nn.Module.
        return embeds.to(self.dtype)

    def _per_layer(self, n_prefix, labels):
        """Gemma E4B feeds a 256-wide vector per token PER LAYER (42 of them)
        beside the ordinary embedding. Returns None for backbones without it.

        It matters how this is obtained. Calling forward() with inputs_embeds
        and no input_ids makes transformers try to RECOVER the ids by comparing
        every embedding against the whole vocab matrix -- [1,T,1,2560] against
        [1,1,262144,2560], a 29 GiB allocation that then fails anyway, because
        projector-produced audio embeddings are not in the vocab by
        construction. That failure is what made E4B look untrainable here and
        sent us to a 135M stand-in, whose scores measured the stand-in.

        So build them from PLACEHOLDER ids -- the audio token across the audio
        prefix, real ids after -- and pass them in explicitly. forward()
        accepts per_layer_inputs; it only refuses it alongside input_ids.
        """
        lm = getattr(self.backbone, "language_model", None)
        if lm is None:
            lm = getattr(getattr(self.backbone, "model", None), "language_model", None)
        if lm is None or not hasattr(lm, "get_per_layer_inputs"):
            return None                      # stub, or a plain text backbone
        cfg = self.backbone.config
        text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        if not getattr(text_cfg, "hidden_size_per_layer_input", None):
            return None

        audio_id = getattr(cfg, "audio_token_id", None)
        if audio_id is None:
            audio_id = getattr(text_cfg, "pad_token_id", 0) or 0
        if labels is not None:
            dev, dt = labels.device, labels.dtype
        else:
            dev, dt = next(self.backbone.parameters()).device, torch.long
        ids = torch.full(tuple(n_prefix), audio_id, dtype=dt, device=dev)
        if labels is not None:
            ids = torch.cat([ids, labels.clamp(min=0)], dim=1)
        return lm.get_per_layer_inputs(ids, None)

    def student(self, waveform, labels=None, want_hidden: bool = False,
                durations=None):
        """Audio prefix + teacher-forced target, LLaVA-style.

        The audio prefix and the target have different lengths, so they cannot
        be compared position-for-position. They are concatenated instead:

            inputs   [ audio embeds ][ target embeds ]
            labels   [ -100 ...     ][ target ids    ]

        The model predicts the target autoregressively while attending to the
        audio, and the prefix is masked out of the loss — without that mask the
        model is scored on "predicting" its own audio conditioning, which is
        both meaningless and easy.
        """
        prefix = self.audio_prefix(waveform, durations)
        if self.wrapper is not None:
            prefix, _ = self.wrapper.wrap(prefix)
        if labels is None:
            ple = self._per_layer(prefix.shape[:2], None)
            kw = {"per_layer_inputs": ple} if ple is not None else {}
            return self.backbone(inputs_embeds=prefix,
                                 output_hidden_states=want_hidden, **kw), None

        embed = self.backbone.get_input_embeddings()
        # clamp: IGNORE_INDEX is not a valid vocabulary index. It stays in the
        # label tensor, it just must not reach the embedding table.
        target = embed(labels.clamp(min=0)).to(self.dtype)
        inputs = torch.cat([prefix, target], dim=1)

        mask = torch.full(prefix.shape[:2], IGNORE_INDEX,
                          dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([mask, labels], dim=1)

        ple = self._per_layer(prefix.shape[:2], labels)
        kw = {"per_layer_inputs": ple} if ple is not None else {}
        out = self.backbone(inputs_embeds=inputs,
                            output_hidden_states=want_hidden, **kw)
        return out, full_labels

    def train_step(self, batch) -> dict:
        self.projector.train()
        self.opt.zero_grad(set_to_none=True)

        need_teacher = self.weights.hidden > 0 or self.weights.kl > 0
        out, full_labels = self.student(
            batch["waveform"], labels=batch["labels"],
            want_hidden=self.weights.hidden > 0, durations=batch.get("durations"))
        tea = self.teacher(batch["token_ids"]) if need_teacher else None

        loss, parts = total_loss(
            self.weights,
            student_hidden=out.hidden_states if self.weights.hidden > 0 else None,
            teacher_hidden=tea.hidden_states if tea is not None else None,
            student_logits=out.logits,
            teacher_logits=tea.logits if tea is not None else None,
            labels=full_labels)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.projector.parameters() if p.requires_grad],
            self.cfg.grad_clip)
        self.opt.step()
        self.sched.step()
        self.step += 1

        return {"step": self.step, "loss": loss.item(),
                "grad_norm": float(grad_norm), "lr": self.sched.get_last_lr()[0],
                **parts}

    @torch.no_grad()
    def evaluate(self, batch) -> dict:
        """Two metrics, and they answer different questions.

        `agreement`  does the audio path's next token match the text path's?
                     Stage 4's headline number.
        `target_acc` of the target tokens, how many does the audio path get
                     right under teacher forcing? Readable during warmup, when
                     agreement is still noise.
        """
        self.projector.eval()
        prefix_out, _ = self.student(batch["waveform"], durations=batch.get("durations"))
        tea = self.teacher(batch["token_ids"])
        metrics = {"agreement": agreement_rate(prefix_out.logits, tea.logits)}

        out, full_labels = self.student(batch["waveform"], labels=batch["labels"],
                                        durations=batch.get("durations"))
        pred = out.logits[:, :-1].argmax(-1)
        gold = full_labels[:, 1:]
        scored = gold != IGNORE_INDEX
        metrics["target_acc"] = ((pred == gold) & scored).sum().item() / max(
            1, scored.sum().item())
        return metrics

    def save(self, path: Path, resumable: bool = True):
        """Checkpoint the projector plus the identity of what it was trained
        against. A projector is only valid for that exact pairing, so shipping
        the weights without the pairing is shipping a trap.

        `resumable` also stores optimiser and scheduler state. Without it an
        overnight run that resumes restarts Adam's moments and the LR schedule
        from scratch, which shows up as a loss spike at every resume boundary.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {"projector": self.projector.state_dict(),
                "config": asdict(self.cfg),
                "step": self.step,
                "cognition_dim": self.projector.cognition_dim,
                "embed_norm": float(self.projector.embed_norm),
                "modality": self.projector.modality,
                "chat_wrapped": self.wrapper is not None,
                "system_prompt": getattr(self.wrapper, "system", None),
                "history": self.history}
        if resumable:
            blob["optimizer"] = self.opt.state_dict()
            blob["scheduler"] = self.sched.state_dict()
        # Write-then-rename: a checkpoint half-written when the machine sleeps
        # would otherwise replace the last good one.
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(blob, tmp)
        tmp.replace(path)

    def load(self, path: Path) -> int:
        """Resume. Returns the step resumed from."""
        ck = torch.load(path, map_location=self.device, weights_only=False)
        prev = ck["config"]
        if prev["backbone_id"] != self.cfg.backbone_id:
            raise ValueError(
                f"checkpoint was trained against {prev['backbone_id']!r}, this run "
                f"uses {self.cfg.backbone_id!r} — a projector does not transfer "
                "between backbones")
        if ck["cognition_dim"] != self.projector.cognition_dim:
            raise ValueError(
                f"checkpoint dim {ck['cognition_dim']} != {self.projector.cognition_dim}")
        self.projector.load_state_dict(ck["projector"])
        if "optimizer" in ck:
            self.opt.load_state_dict(ck["optimizer"])
            prev_total = ck.get("scheduler", {}).get("total_steps")
            if prev_total == self.cfg.max_steps:
                self.sched.load_state_dict(ck["scheduler"])
            else:
                # The run was extended (or shortened): restoring the old
                # schedule verbatim would crash one step past the old horizon.
                # Keep the fresh schedule built for the NEW horizon and
                # fast-forward it to the resumed step — cheap, it is only math.
                for _ in range(ck["step"]):
                    self.sched.step()
                print(f"train: schedule horizon {prev_total} -> "
                      f"{self.cfg.max_steps}; rebuilt and fast-forwarded")
        self.step = ck["step"]
        self.history = ck.get("history", [])
        return self.step


def _hidden_size(backbone) -> int:
    """Multimodal configs (Gemma4Config) keep hidden_size on the TEXT config;
    plain causal LMs keep it at the top. Reading the wrong one is silent until
    the projector emits the wrong width."""
    cfg = backbone.config
    if hasattr(cfg, "get_text_config"):
        return cfg.get_text_config().hidden_size
    return cfg.hidden_size


def real_setup(cfg: TrainConfig, corpus: str, device: str):
    """Load real models + corpus. Returns (encoder_fn, backbone, projector, loader).

    Whisper's decoder is never loaded — the omni path does not use it, and
    skipping it saves both memory and load time.
    """
    from transformers import (AutoModelForCausalLM, AutoTokenizer, WhisperModel,
                              WhisperProcessor)

    from dataset import CaptureDataset, loader as make_loader

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[cfg.dtype]

    processor = WhisperProcessor.from_pretrained(cfg.whisper_id)

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_id)
    backbone = AutoModelForCausalLM.from_pretrained(cfg.backbone_id, dtype=dtype)
    backbone = backbone.to(device).eval()
    emb_norm = backbone.get_input_embeddings().weight.norm(dim=-1).mean().item()

    if cfg.arch == "minicpm_delta":
        # THEIR fine-tuned encoder + THEIR projector, both frozen. Only the
        # delta trains — collapse of the front end is structurally impossible.
        from load_minicpm import load_front_end
        from jaeger_agent_omni.neural import DeltaAdapter, MiniCPMDeltaPath
        encoder, front = load_front_end()
        encoder = encoder.to(device, dtype).eval()
        delta = DeltaAdapter(source_dim=front.cognition_dim,
                             cognition_dim=_hidden_size(backbone),
                             embed_norm=emb_norm)
        projector = MiniCPMDeltaPath(front, delta).to(device)
        print(f"arch: minicpm_delta — their encoder (fine-tuned, frozen) + "
              f"their projector (21M, frozen)")
        print(f"delta: {delta.n_params()/1e3:.0f}K trainable, "
              f"{front.cognition_dim} -> {backbone.config.hidden_size}, "
              f"embed_norm {emb_norm:.2f}")
    else:
        whisper = WhisperModel.from_pretrained(cfg.whisper_id, dtype=dtype)
        encoder = whisper.encoder.to(device).eval()
        projector = JaegerAudioProjector.for_backbone(
            backbone, audio_dim=whisper.config.d_model,
            stride=cfg.pool_stride, pool_last=cfg.pool_last).to(device)
        print(f"projector: {projector.n_params()/1e3:.0f}K params, "
              f"embed_norm pinned to {float(projector.embed_norm):.2f}")

    # Stage-matched system prompt: warmup repeats, distill answers. Training
    # "repeat" under an "answer" prompt manufactures refusal boilerplate.
    system = WARMUP_SYSTEM if cfg.stage == "stage3_warmup" else DEFAULT_SYSTEM
    wrapper = ChatWrapper(tokenizer, backbone, system=system, device=device)
    print(f"prompt: audio spliced into the chat template "
          f"({wrapper.overhead_tokens} template tokens; "
          f"system={'warmup/repeat' if system is WARMUP_SYSTEM else 'assistant'})")

    common = dict(stage=cfg.stage, min_confidence=cfg.min_confidence,
                  max_audio_s=cfg.max_audio_s, val_fraction=cfg.val_fraction)
    # Prefixes in training only. Validation stays whole-utterance so the
    # reported number means the same thing across runs and is comparable to
    # the text path, which always sees a finished utterance.
    train_ds = CaptureDataset(corpus, processor, tokenizer, split="train",
                              prefix_fraction=cfg.prefix_fraction,
                              seed=cfg.seed, **common)
    val_ds = CaptureDataset(corpus, processor, tokenizer, split="val", **common)

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
        else tokenizer.eos_token_id
    return (encoder, backbone, projector,
            make_loader(train_ds, cfg.batch_size, pad),
            make_loader(val_ds, cfg.batch_size, pad, shuffle=False),
            train_ds, val_ds, wrapper)


def stub_setup(cfg: TrainConfig, device: str):
    from stubs import StubAudioEncoder, StubBackbone
    audio_dim, cog_dim, vocab = 64, 96, 128
    enc = StubAudioEncoder(audio_dim=audio_dim)
    lm = StubBackbone(hidden_size=cog_dim, vocab_size=vocab, n_layers=2)
    proj = JaegerAudioProjector(audio_dim=audio_dim, cognition_dim=cog_dim,
                                stride=cfg.pool_stride, pool_last=cfg.pool_last)
    return enc, lm, proj, vocab


def stub_batch(cfg: TrainConfig, vocab: int, device: str, seed: int = 0):
    # Generate on CPU then move: a CPU generator cannot seed an MPS/CUDA
    # tensor directly, and seeding on CPU keeps the batch identical across
    # devices, which is what makes a stub run reproducible.
    g = torch.Generator().manual_seed(seed)
    n = cfg.batch_size
    return {"waveform": torch.randn(n, 16000, generator=g).to(device),
            "token_ids": torch.randint(0, vocab, (n, 7), generator=g).to(device),
            "labels": torch.randint(0, vocab, (n, 10), generator=g).to(device)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stub", action="store_true",
                    help="run the identical loop on tiny random models")
    ap.add_argument("--backbone", default="", help="HF id for a real run")
    ap.add_argument("--corpus", default=None,
                    help="capture corpus directory (required for a real run)")
    ap.add_argument("--whisper", default="openai/whisper-medium")
    ap.add_argument("--arch", default="minicpm_delta",
                    choices=["scratch", "minicpm_delta"])
    ap.add_argument("--stage", default="stage3_warmup",
                    choices=["stage3_warmup", "stage4_distill"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/run",
                    help="checkpoint prefix, resolved against the repo root")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the checkpoint at --out if it exists")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly after N hours — for overnight runs")
    args = ap.parse_args()

    if not args.stub and not args.backbone:
        ap.error("need --backbone <hf-id>, or --stub to dry-run the loop")

    device, _dtype = pick_device(args.device)
    make = warmup if args.stage == "stage3_warmup" else distill_cfg
    cfg = make(args.backbone or "stub://tiny",
               **({"arch": "scratch", "whisper_id": "stub"} if args.stub else {}))
    if args.steps:
        cfg.max_steps = args.steps
        cfg.warmup_steps = min(cfg.warmup_steps, max(1, args.steps // 10))
        cfg.__post_init__()

    if args.stub:
        cfg.batch_size = 4
        enc, lm, proj, vocab = stub_setup(cfg, device)
        dl, wrapper = None, None
    else:
        if not args.corpus:
            ap.error("a real run needs --corpus (see training/make_corpus.py)")
        cfg.whisper_id = args.whisper
        cfg.arch = args.arch
        cfg.__post_init__()
        cfg.dtype = "fp32" if device in ("cpu", "mps") else cfg.dtype
        (enc, lm, proj, dl, val_dl, ds, val_ds,
         wrapper) = real_setup(cfg, args.corpus, device)
        print(f"corpus: {len(ds)} train / {len(val_ds)} val turns from {args.corpus}")
        print(f"        prefix_fraction {cfg.prefix_fraction} "
              f"(mid-utterance audio in-distribution; val stays whole)")

    trainer = Trainer(cfg, enc, lm, proj, device=device, wrapper=wrapper)
    n_train = sum(p.numel() for p in proj.parameters() if p.requires_grad)
    print(f"train: {cfg.stage} on {device}, {cfg.max_steps} steps, "
          f"{n_train/1e3:.0f}K trainable, {proj.tokens_per_second:.0f} tok/s")

    def batches():
        """Stub batches are fixed; real ones cycle the corpus."""
        if dl is None:
            b = stub_batch(cfg, vocab, device)
            while True:
                yield b
        while True:
            for b in dl:
                yield {"waveform": b["features"].to(device, trainer.dtype),
                       "token_ids": b["teacher_ids"].to(device),
                       "labels": b["labels"].to(device),
                       "durations": b["durations"]}

    stream = batches()

    # Evaluate on held-out turns. Evaluating on the training stream would
    # report memorisation as accuracy, and a small corpus memorises fast.
    if dl is not None:
        vb = next(iter(val_dl))
        eval_batch = {"waveform": vb["features"].to(device, trainer.dtype),
                      "token_ids": vb["teacher_ids"].to(device),
                      "labels": vb["labels"].to(device),
                      "durations": vb["durations"]}
    else:
        eval_batch = stub_batch(cfg, vocab, device, 99)

    # Resolve against the repo, not the CWD — a training script should not
    # care whether it was invoked from the root or from training/.
    repo = Path(__file__).resolve().parent.parent
    base = Path(args.out)
    base = base if base.is_absolute() else repo / "training" / base
    out = base.with_name(f"{base.name}_{cfg.stage}.pt")

    orphan = out.with_suffix(out.suffix + ".tmp")
    if orphan.exists():
        # A .tmp with no .pt means the process died mid-write before any
        # complete checkpoint existed. Say so — silently starting from zero
        # after an overnight run is the worst possible failure mode.
        print(f"WARNING found partial checkpoint {orphan.name} "
              f"({orphan.stat().st_size/1e6:.1f} MB) with no complete "
              f"checkpoint beside it. It is truncated and unusable; removing.")
        orphan.unlink()

    if args.resume:
        if out.exists():
            resumed = trainer.load(out)
            print(f"train: resumed from {out} at step {resumed}")
        else:
            print(f"train: --resume given but {out.name} does not exist; "
                  f"starting from scratch")

    print(f"train: checkpointing every {cfg.checkpoint_every} steps")

    # Stop cleanly on Ctrl-C, on SIGTERM, and on the machine going down. A
    # killed overnight run should lose one interval, not the night.
    stopping = {"flag": False}

    def request_stop(signum, _frame):
        stopping["flag"] = True
        print(f"\ntrain: signal {signum} — finishing this step, then saving")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    deadline = (time.monotonic() + args.max_hours * 3600) if args.max_hours else None
    if deadline:
        print(f"train: budget {args.max_hours:.1f}h — will stop cleanly before then")

    t0 = time.monotonic()
    first = None
    m = {"loss": float("nan"), "step": trainer.step}
    log_every = max(1, cfg.max_steps // 10)
    reason = "completed"

    while trainer.step < cfg.max_steps:
        m = trainer.train_step(next(stream))
        first = first if first is not None else m["loss"]

        if trainer.step % log_every == 0:
            ev = trainer.evaluate(eval_batch)
            trainer.history.append({**m, **ev, "elapsed_s": round(time.monotonic()-t0, 1)})
            print(f"  step {m['step']:>5}  loss {m['loss']:.4f}  "
                  f"grad {m['grad_norm']:.2f}  lr {m['lr']:.2e}  "
                  f"target_acc {ev['target_acc']:.2f}  "
                  f"agreement {ev['agreement']:.2f}")

        if trainer.step % cfg.checkpoint_every == 0:
            trainer.save(out)

        if stopping["flag"]:
            reason = "interrupted"
            break
        if deadline and time.monotonic() > deadline:
            reason = "time budget reached"
            break

    trainer.save(out)
    out.with_suffix(".json").write_text(
        json.dumps({"config": asdict(cfg), "history": trainer.history,
                    "stopped": reason, "step": trainer.step}, indent=2))

    print(f"train: {reason} at step {trainer.step}/{cfg.max_steps} — "
          f"{first:.4f} -> {m['loss']:.4f} in {time.monotonic()-t0:.1f}s")
    print(f"train: saved {out}")
    if reason != "completed":
        print(f"train: resume with --resume (optimiser and schedule restored)")
    return 0


def demo():
    """Self-check: the loop runs, loss falls, checkpoint round-trips."""
    import tempfile
    from stubs import StubAudioEncoder, StubBackbone

    torch.manual_seed(0)
    cfg = warmup("stub://tiny", max_steps=120, warmup_steps=10, batch_size=4)
    enc, lm, proj, vocab = stub_setup(cfg, "cpu")
    trainer = Trainer(cfg, enc, lm, proj, device="cpu")

    batch = stub_batch(cfg, vocab, "cpu")
    first = trainer.train_step(batch)
    for _ in range(cfg.max_steps - 1):
        last = trainer.train_step(batch)

    assert last["loss"] < first["loss"], (first["loss"], last["loss"])
    assert trainer.step == cfg.max_steps
    assert last["grad_norm"] <= cfg.grad_clip + 1e-3, "clipping is not applied"
    # OneCycle warms up then decays — the end lr must be below the peak.
    assert last["lr"] < cfg.lr, (last["lr"], cfg.lr)

    ev = trainer.evaluate(batch)
    assert 0.0 <= ev["agreement"] <= 1.0
    assert 0.0 <= ev["target_acc"] <= 1.0

    # The audio prefix must be masked out of the loss. Without the mask the
    # model is scored on reproducing its own conditioning.
    out, full = trainer.student(batch["waveform"], labels=batch["labels"])
    n_audio = trainer.audio_prefix(batch["waveform"]).shape[1]
    assert full.shape[1] == n_audio + batch["labels"].shape[1]
    assert (full[:, :n_audio] == IGNORE_INDEX).all(), "audio prefix not masked"
    assert (full[:, n_audio:] == batch["labels"]).all(), "labels were altered"
    assert out.logits.shape[1] == full.shape[1], "logits/labels must align"

    # Backbone stayed frozen; the projector did not.
    assert all(p.grad is None for p in trainer.backbone.parameters())
    assert proj.linear1.weight.grad.abs().sum() > 0

    # A projector emitting the wrong width must be caught at construction, not
    # broadcast into silence.
    try:
        Trainer(cfg, enc, lm, JaegerAudioProjector(audio_dim=64, cognition_dim=999))
    except ValueError as exc:
        assert "cognition_dim" in str(exc)
    else:
        raise AssertionError("dim mismatch must be rejected")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        trainer.save(path)

        # Resume restores step, optimiser and schedule — not just weights.
        # Without the optimiser, every resume boundary shows up as a loss spike.
        enc2, lm2, proj2, _ = stub_setup(cfg, "cpu")
        t2 = Trainer(cfg, enc2, lm2, proj2, device="cpu")
        assert t2.step == 0
        assert t2.load(path) == cfg.max_steps
        assert t2.step == cfg.max_steps
        assert t2.opt.state_dict()["state"], "optimiser state not restored"
        assert len(t2.history) == len(trainer.history)
        for a, b in zip(proj2.parameters(), trainer.projector.parameters()):
            assert torch.equal(a, b), "weights not restored"

        # Extending a run: resume with a LARGER max_steps must rebuild the
        # schedule, not crash one step past the old horizon (hit for real
        # when the corpus quadrupled mid-day).
        cfg_long = TrainConfig(**{**cfg.to_dict(), "max_steps": cfg.max_steps * 3})
        enc3, lm3, proj3, _ = stub_setup(cfg_long, "cpu")
        t_ext = Trainer(cfg_long, enc3, lm3, proj3, device="cpu")
        assert t_ext.load(path) == cfg.max_steps
        for _ in range(cfg.max_steps + 5):       # step PAST the old horizon
            t_ext.train_step(batch)
        assert t_ext.step == cfg.max_steps * 2 + 5

        # A checkpoint from another backbone must be refused, not loaded.
        other = TrainConfig(**{**cfg.to_dict(), "backbone_id": "someone/else"})
        t3 = Trainer(other, enc2, lm2, JaegerAudioProjector(audio_dim=64,
                                                            cognition_dim=96))
        try:
            t3.load(path)
        except ValueError as exc:
            assert "does not transfer" in str(exc)
        else:
            raise AssertionError("cross-backbone resume must be refused")

        ck = torch.load(path, weights_only=False)
        # The checkpoint must carry what it was trained against — a projector
        # is only valid for that exact encoder/backbone pairing.
        assert ck["config"]["backbone_id"] == "stub://tiny"
        assert ck["cognition_dim"] == proj.cognition_dim
        assert ck["modality"] == "audio"
        assert ck["step"] == cfg.max_steps
        fresh = JaegerAudioProjector(audio_dim=64, cognition_dim=96)
        fresh.load_state_dict(ck["projector"])

    print(f"train: {cfg.max_steps} steps {first['loss']:.3f} -> {last['loss']:.3f}, "
          f"agreement {ev['agreement']:.2f}")
    print("train: loop, clipping, schedule, freezing, checkpoint identity OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
